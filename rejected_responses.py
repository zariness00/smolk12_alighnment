import time
import pandas as pd
from huggingface_hub import InferenceClient
from transformers import pipeline
import csv
import os
from dotenv import load_dotenv
from taming_utils import ParallelEvaluator

load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
SYSTEM_PROMPT = "Keep the response short"
MAX_NEW_TOKENS = 500
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


class ResponseGenerator:
    """Generates responses from a base unaligned LLM using either local
    transformers or remote inference."""
    def __init__(self, model_name=None, api_url=None):
        """Initialize with either local model name or API endpoint URL."""
        self.model_name = model_name
        self.api_url = api_url
        if model_name:
            self.pipe = pipeline("text-generation",
        model=model_name,
        max_new_tokens=MAX_NEW_TOKENS)
        if api_url:
            self.client = InferenceClient(
                base_url=api_url,
                token=os.getenv("HF_TOKEN"),
            )

    def generate_responses(self, prompts: list[str]) -> pd.DataFrame:
        """Generate responses for a DataFrame of prompts.
        Args:
        prompts_df: DataFrame with 'user_prompts' columnsave_to_csv: Optional filepath to save responses
        Returns:
        DataFrame with prompts and generated responses
        """
        responses = []
        for prompt in prompts:
        # Remove enclosing quotes if present
            prompt = prompt.strip('"')
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                if self.model_name:
                # Local generation
                #print(" Generating response from local model...")
                    output = self.pipe(messages)
                    response = output[0]['generated_text'][1]['content']
                elif self.api_url:
                    output = self.client.chat_completion(
                        messages,
                        model=MODEL_NAME,
                        max_tokens=MAX_NEW_TOKENS,
                    )
                    response = output.choices[0].message.content
                responses.append(response)
                # Add delay to respect rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error generating response for prompt: {prompt}")
                print(f"Error: {str(e)}")
                responses.append("")
        results_df = pd.DataFrame({
            "user_prompts": prompts,
            "model_responses": responses
        })
        return results_df

if __name__ == "__main__":
    USER_PROMPTS_PATH = "data/alignment/user_prompts.csv"
    API_URL = os.getenv("ENDPOINT_VLLM")
    NUM_CHUNKS = 10
    DATASET_SIZE = 5000
    MAX_NEW_RESPONSES_PER_RUN = 1000
    CHECKPOINT_BATCH_SIZE = 50
    REJECTED_RESPONSES_PATH = "data/alignment/rejected_responses.csv"

    if not API_URL:
        raise ValueError("ENDPOINT is missing from .env")
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN is missing from .env")

    evaluator = ResponseGenerator(api_url=API_URL)
    user_prompts_df = pd.read_csv(USER_PROMPTS_PATH)
    user_prompts = user_prompts_df.iloc[:DATASET_SIZE, 0].astype(str).tolist()

    parallel_evaluator = ParallelEvaluator(evaluator)

    if os.path.exists(REJECTED_RESPONSES_PATH):
        rejected_responses = pd.read_csv(REJECTED_RESPONSES_PATH)
        rejected_responses = rejected_responses.dropna(subset=["model_responses"])
        rejected_responses = rejected_responses[
            rejected_responses["model_responses"].astype(str).str.strip().ne("")
        ]
        # Discard the earlier test output that contained the full chat template.
        rejected_responses = rejected_responses[
            ~rejected_responses["model_responses"].astype(str).str.contains(
                "<|im_start|>", regex=False
            )
        ]
    else:
        rejected_responses = pd.DataFrame(
            columns=["user_prompts", "model_responses"]
        )

    def prompt_key(value: str) -> str:
        return str(value).strip().strip('"')

    completed_prompts = {
        prompt_key(prompt) for prompt in rejected_responses["user_prompts"]
    }
    pending_prompts = [
        prompt for prompt in user_prompts
        if prompt_key(prompt) not in completed_prompts
    ][:MAX_NEW_RESPONSES_PER_RUN]

    run_target = len(rejected_responses) + len(pending_prompts)
    print(
        f"Starting with {len(rejected_responses)} saved responses; "
        f"this run targets {run_target}/{DATASET_SIZE}."
    )

    for start in range(0, len(pending_prompts), CHECKPOINT_BATCH_SIZE):
        prompts_batch = pending_prompts[start:start + CHECKPOINT_BATCH_SIZE]
        batch_results = parallel_evaluator.evaluate(
            prompts=prompts_batch,
            n_parts=min(NUM_CHUNKS, len(prompts_batch)),
        )

        # Failed calls return empty responses and will be retried next run.
        batch_results = batch_results.dropna(subset=["model_responses"])
        batch_results = batch_results[
            batch_results["model_responses"].astype(str).str.strip().ne("")
        ]
        rejected_responses = pd.concat(
            [rejected_responses, batch_results], ignore_index=True
        )
        rejected_responses["_prompt_key"] = rejected_responses[
            "user_prompts"
        ].map(prompt_key)
        rejected_responses = rejected_responses.drop_duplicates(
            subset=["_prompt_key"], keep="last"
        ).drop(columns="_prompt_key")
        rejected_responses.to_csv(
            REJECTED_RESPONSES_PATH,
            quoting=csv.QUOTE_ALL,
            index=False,
        )
        print(f"Saved {len(rejected_responses)}/{DATASET_SIZE} responses.")
