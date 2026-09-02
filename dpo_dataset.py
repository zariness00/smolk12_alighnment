from datasets import Dataset
import pandas as pd

def generate_dpo_dataset(
        chosen_responses_path,
        rejected_responses_path,
        repo_id: str = None
    ):
    push_to_hub = repo_id is not None
    chosen_responses = pd.read_json(
        chosen_responses_path,
        lines=True
    ).rename(columns={
        "user_prompt": "user_prompts",
        "chosen_response": "model_responses"
    })
    chosen_responses = chosen_responses[["user_prompts", "model_responses"]]
    rejected_responses = pd.read_csv(rejected_responses_path)
    # Join the chosen and rejected responses on user_prompts
    dpo_dataset = pd.merge(
        chosen_responses,
        rejected_responses,
        on="user_prompts",
        suffixes=('_chosen', '_rejected')
    )
    def transform_row(row):
        # Format the 'chosen' column as per the chat structure
        chosen = [
        {"content": row["user_prompts"], "role": "user"},
        {"content": row["model_responses_chosen"], "role": "assistant"}
        ]
        # Format the 'rejected' column in the same way
        rejected = [
            {"content": row["user_prompts"], "role": "user"},
            {"content": row["model_responses_rejected"], "role": "assistant"}
        ]
        return pd.Series([chosen, rejected], index=["chosen", "rejected"])
    dpo_dataset[["chosen", "rejected"]] = dpo_dataset.apply(
        transform_row,
        axis=1)
    dpo_dataset = dpo_dataset.drop(columns=["user_prompts",
        "model_responses_chosen",
        "model_responses_rejected"])
    hf_dpo_dataset = Dataset.from_pandas(dpo_dataset)
    if push_to_hub:
        hf_dpo_dataset.push_to_hub(repo_id, private=True)
    return hf_dpo_dataset

CHOSEN_RESPONSES_PATH = "data/alignment/chosen_responses.jsonl"
REJECTED_RESPONSES_PATH = "data/alignment/rejected_responses.csv"

dpo_dataset = generate_dpo_dataset(
    CHOSEN_RESPONSES_PATH,
    REJECTED_RESPONSES_PATH,
    repo_id="zariness00/dpo-dataset"
)
