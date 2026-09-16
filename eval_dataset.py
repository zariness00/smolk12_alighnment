import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from rejected_responses import ResponseGenerator

load_dotenv()

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ALIGNED_MODEL = "zariness00/smolk12_dpo"
DPO_DATASET = "zariness00/dpo-dataset"
BASE_MODEL_API_URL = os.getenv("ENDPOINT_VLLM")
HF_TOKEN = os.getenv("HF_TOKEN")

DATA_DIR = Path("data/alignment")
EVAL_PROMPTS_PATH = DATA_DIR / "eval_prompts.csv"
BASE_MODEL_RESPONSES_PATH = DATA_DIR / "evals_base_model_responses.csv"
ALIGNED_MODEL_RESPONSES_PATH = DATA_DIR / "evals_aligned_model_responses.csv"

NUM_SAMPLES = 100
RANDOM_SEED = 42
CHECKPOINT_EVERY = 5
ENDPOINT_WORKERS = 10


def normalize_prompt(value: str) -> str:
    return str(value).strip().strip('"')


def load_or_sample_eval_prompts() -> list[str]:
    """Create the fixed evaluation sample once, then reuse it."""
    if EVAL_PROMPTS_PATH.exists():
        prompts = pd.read_csv(EVAL_PROMPTS_PATH)["prompt"].astype(str).tolist()
        print(f"Loaded {len(prompts)} fixed prompts from {EVAL_PROMPTS_PATH}")
        return prompts

    dataset = load_dataset(path=DPO_DATASET, split="train")
    dataframe = dataset.to_pandas()
    dataframe["prompt"] = dataframe["chosen"].apply(
        lambda messages: normalize_prompt(messages[0]["content"])
    )
    prompts = (
        dataframe[["prompt"]]
        .drop_duplicates()
        .sample(
            n=min(NUM_SAMPLES, dataframe["prompt"].nunique()),
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(EVAL_PROMPTS_PATH, index=False)
    print(f"Saved {len(prompts)} fixed prompts to {EVAL_PROMPTS_PATH}")
    return prompts["prompt"].tolist()


def local_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def extract_response(output: list[dict]) -> str:
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return str(generated[-1]["content"]).strip()
    return str(generated).strip()


def generate_model_responses(
    prompts: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Generate locally and checkpoint results so an interrupted run can resume."""
    if output_path.exists():
        results = pd.read_csv(output_path).dropna(subset=["model_responses"])
        results = results[
            results["model_responses"].astype(str).str.strip().ne("")
        ]
    else:
        results = pd.DataFrame(columns=["user_prompts", "model_responses"])

    completed = {
        normalize_prompt(prompt) for prompt in results["user_prompts"].astype(str)
    }
    pending = [
        prompt for prompt in prompts if normalize_prompt(prompt) not in completed
    ]
    if not pending:
        print(f"All {len(prompts)} responses already exist in {output_path}")
        return

    device = local_device()
    print(f"Loading {model_name} on {device} ({len(pending)} prompts remaining)")
    response_generator = ResponseGenerator(model_name=model_name)
    generator = response_generator.pipe
    if device != -1:
        target_device = torch.device("cuda:0" if device == 0 else device)
        generator.model.to(target_device)
        generator.device = target_device

    for number, prompt in enumerate(pending, start=1):
        messages = [{"role": "user", "content": normalize_prompt(prompt)}]
        try:
            output = generator(messages)
            response = extract_response(output)
        except Exception as error:
            print(f"Generation failed for prompt {number}: {error}")
            response = ""

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {"user_prompts": [prompt], "model_responses": [response]}
                ),
            ],
            ignore_index=True,
        )
        if number % CHECKPOINT_EVERY == 0 or number == len(pending):
            results.to_csv(output_path, index=False)
            print(f"Saved {len(results)}/{len(prompts)} responses to {output_path}")

def generate_endpoint_responses(prompts: list[str], output_path: Path) -> None:
    """Generate base responses through the existing HF endpoint."""
    if not BASE_MODEL_API_URL:
        raise ValueError(
            "Add BASE_MODEL_API_URL or ENDPOINT_VLLM to .env before running base."
        )
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing from .env")

    if output_path.exists():
        results = pd.read_csv(output_path).dropna(subset=["model_responses"])
        results = results[
            results["model_responses"].astype(str).str.strip().ne("")
        ]
    else:
        results = pd.DataFrame(columns=["user_prompts", "model_responses"])

    completed = {
        normalize_prompt(prompt) for prompt in results["user_prompts"].astype(str)
    }
    pending = [
        prompt for prompt in prompts if normalize_prompt(prompt) not in completed
    ]
    if not pending:
        print(f"All {len(prompts)} responses already exist in {output_path}")
        return

    client = InferenceClient(base_url=BASE_MODEL_API_URL, token=HF_TOKEN)

    def generate_one(prompt: str) -> str:
        messages = [{"role": "user", "content": normalize_prompt(prompt)}]
        try:
            output = client.chat_completion(
                messages=messages,
                model=BASE_MODEL,
                seed=42,
            )
            return str(output.choices[0].message.content).strip()
        except Exception as error:
            print(f"Generation failed: {error}")
            return ""

    for start in range(0, len(pending), ENDPOINT_WORKERS):
        batch = pending[start : start + ENDPOINT_WORKERS]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            responses = list(executor.map(generate_one, batch))
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {"user_prompts": batch, "model_responses": responses}
                ),
            ],
            ignore_index=True,
        )
        results.to_csv(output_path, index=False)
        print(f"Saved {len(results)}/{len(prompts)} responses to {output_path}")


def main() -> None:
    prompts = load_or_sample_eval_prompts()
    generate_endpoint_responses(prompts, BASE_MODEL_RESPONSES_PATH)
    generate_model_responses(
        prompts, ALIGNED_MODEL, ALIGNED_MODEL_RESPONSES_PATH
    )


if __name__ == "__main__":
    main()
