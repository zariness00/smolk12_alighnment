import os
import json
from dotenv import load_dotenv
load_dotenv()
from taming_utils import OpenAIBatchProcessor
from pathlib import Path
import pandas as pd
import asyncio

MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
async def process_aligned_responses(
    policy: str,
    company_name: str,
    user_prompts: list,
    model: str = MODEL_NAME,
    api_url: str = "",
    ) -> dict:
    """Process a batch of prompts to generate aligned responses.
    Args:
    policy: Content of the policy to enforce
    company_name: Name of the company whose policy is being enforced
    user_prompts: List of user prompts to process
    Returns:
    Dictionary containing the batch processing results
    """
    ALIGNED_SUFFIX = f"""I'm sorry, but I can't help with that since
    it violates {company_name}'s policy."""
    system_prompt = f"""
    ROLE: You are a helpful assistant.
    TASK: Generate a response to the user prompt
    REQUIREMENTS:
    - Response must comply with the provided POLICY
    - If chat is not aligned with POLICY, you MUST return your
    response with the following suffix:
    "{ALIGNED_SUFFIX}" and explain why it violates the POLICY.
    - Keep the response short
    POLICY: {policy}
    """
    requests = [
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
        for prompt in user_prompts
    ]
    processor = OpenAIBatchProcessor(
        api_key=os.getenv("HF_TOKEN"),
        api_url=f"{api_url.rstrip('/')}/v1/chat/completions",
        max_requests_per_minute=1500,
        max_tokens_per_minute=125000,
        # Failed prompts are omitted and retried on the next script run.
        max_retries=0,
    )
    results = await processor.process_batch(
        requests=requests,
    )
    print(f"Total requests: {processor.total_requests}")
    print(f"Successful requests: {processor.successful_requests}")
    print(f"Failed requests: {processor.failed_requests}")
    print(f"Rate limit errors: {processor.rate_limit_errors}")
    return results


def split_thinking(content: str) -> tuple[str, str]:
    """Split Qwen thinking from the final answer."""
    content = content or ""
    if "<think>" not in content:
        return "", content.strip()
    if "</think>" not in content:
        # An unfinished reasoning block has no usable chosen response.
        return content.removeprefix("<think>").strip(), ""

    thinking, final_answer = content.split("</think>", maxsplit=1)
    thinking = thinking.removeprefix("<think>").strip()
    return thinking, final_answer.strip()


def parse_qwen_result(prompt: str, result: dict) -> dict | None:
    """Create one clean dataset row from a raw vLLM response."""
    if not isinstance(result, dict) or result.get("error"):
        return None

    choices = result.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")

    if reasoning is None:
        reasoning, chosen_response = split_thinking(raw_content)
    else:
        chosen_response = raw_content.strip()

    if not chosen_response:
        return None

    return {
        "user_prompt": str(prompt).strip().strip('"'),
        "reasoning_content": reasoning or "",
        "chosen_response": chosen_response,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main() -> None:
    CHOSEN_RESPONSES_JSONL_PATH = Path("data/alignment/chosen_responses.jsonl")
    USER_PROMPTS_PATH = "data/alignment/user_prompts.csv"
    POLICY_PATH = "k_12_policy.md"
    COMPANY_NAME = "Acme Inc."
    API_URL = os.getenv("ENDPOINT_QWEN")
    DATASET_SIZE = 5000
    MAX_NEW_RESPONSES_PER_RUN = 1000
    CHECKPOINT_BATCH_SIZE = 100

    if not API_URL:
        raise ValueError("ENDPOINT_QWEN is missing from .env")
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN is missing from .env")

    policy = Path(POLICY_PATH).read_text(encoding="utf-8")
    user_prompts_df = pd.read_csv(USER_PROMPTS_PATH)
    user_prompts = user_prompts_df.iloc[:DATASET_SIZE, 0].astype(str).tolist()

    chosen_records = load_jsonl(CHOSEN_RESPONSES_JSONL_PATH)
    for record in chosen_records:
        record.pop("raw_response", None)
    chosen_records = [
        record for record in chosen_records
        if str(record.get("chosen_response", "")).strip()
    ]
    completed_prompts = {
        str(record["user_prompt"]).strip().strip('"') for record in chosen_records
    }
    pending_prompts = [
        prompt for prompt in user_prompts
        if str(prompt).strip().strip('"') not in completed_prompts
    ][:MAX_NEW_RESPONSES_PER_RUN]

    run_target = len(chosen_records) + len(pending_prompts)
    print(
        f"Starting with {len(chosen_records)} saved chosen responses; "
        f"this run targets {run_target}/{DATASET_SIZE}."
    )

    for start in range(0, len(pending_prompts), CHECKPOINT_BATCH_SIZE):
        prompts_batch = pending_prompts[start:start + CHECKPOINT_BATCH_SIZE]
        results = await process_aligned_responses(
            policy=policy,
            company_name=COMPANY_NAME,
            user_prompts=prompts_batch,
            model=MODEL_NAME,
            api_url=API_URL,
        )

        parsed_records = [
            parsed
            for prompt, result in zip(prompts_batch, results)
            if (parsed := parse_qwen_result(prompt, result)) is not None
        ]
        chosen_records.extend(parsed_records)

        # Keep one successful response per normalized prompt.
        unique_records = {}
        for record in chosen_records:
            key = str(record["user_prompt"]).strip().strip('"')
            unique_records[key] = record
        chosen_records = list(unique_records.values())

        save_jsonl(CHOSEN_RESPONSES_JSONL_PATH, chosen_records)
        print(f"Saved {len(chosen_records)}/{DATASET_SIZE} chosen responses.")


if __name__ == "__main__":
    asyncio.run(main())
