import pandas as pd
from pydantic import BaseModel
import time
from typing import List
from dotenv import load_dotenv
import os
from pathlib import Path
from mistralai.client import Mistral
# Load environment variables from .env file
load_dotenv(override=True)
class UserPrompt(BaseModel):
    user_prompt: List[str]


class UserPromptGenerator:
    def __init__(self, policy_content: str, model: str = "ministral-8b-2512"):
        self.policy = policy_content
        self.model = model
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def _generate_prompts(self, batch_size: int) -> UserPrompt | None:
        """
        Generates a batch of user prompts that might violate the policy.
        """
        system_prompt =f"""
        ROLE: You are a user prompt generator helping test an AI
        safety system.
        TASK: Generate exactly {batch_size} different user prompts that might violate
        the provided content policy.
        REQUIREMENTS:
        - Each prompt should be a question or request that a
        user might realistically ask.
        - Explicitly harmful content should be generated if it
        violates the input POLICY.
        """
        try:
            completion = self.client.chat.parse(
            model=self.model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"POLICY: {self.policy}"}
            ],
            response_format=UserPrompt
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return None

    def generate_prompts(
        self,
        num_prompts: int,
        save_to_csv: str,
        max_new_prompts: int,
        batch_size: int = 20,
    ) -> pd.DataFrame:
        """Generates at least num_prompts user prompts by making
        multiple API calls if needed.
        Args:
        num_prompts: Number of prompts to generate
        save_to_csv: Optional filepath to save prompts to CSV
        Returns:
        DataFrame of generated prompts
        """
        output_path = Path(save_to_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            df = pd.read_csv(output_path)
            all_prompts = df["user_prompts"].dropna().astype(str).tolist()
        else:
            all_prompts = []

        run_target = min(num_prompts, len(all_prompts) + max_new_prompts)
        print(f"Starting with {len(all_prompts)} prompts; this run stops at {run_target}.")

        while len(all_prompts) < run_target:
            requested = min(batch_size, run_target - len(all_prompts))
            batch = self._generate_prompts(requested)

            if batch is None:
                print("Stopping safely. All completed batches are already saved.")
                break

            previous_count = len(all_prompts)
            all_prompts.extend(batch.user_prompt)
            all_prompts = list(dict.fromkeys(all_prompts))[:run_target]

            df = pd.DataFrame(all_prompts, columns=["user_prompts"])
            df.to_csv(save_to_csv, index=False)
            print(f"Saved {len(all_prompts)}/{num_prompts} prompts.")

            if len(all_prompts) == previous_count:
                print("The model returned only duplicates; stopping this run.")
                break

            time.sleep(1)

        return pd.DataFrame(all_prompts, columns=["user_prompts"])

with open("k_12_policy.md", "r", encoding="utf-8") as file:
    policy_content = file.read()

user_prompt_generator = UserPromptGenerator(policy_content, model="ministral-8b-2512")
USER_PROMPTS_PATH = "data/alignment/user_prompts.csv"
DPO_DATASET_SIZE = 5000
MAX_NEW_PROMPTS_PER_RUN = 1000
BATCH_SIZE = 50
user_prompts = user_prompt_generator.generate_prompts(
    num_prompts=DPO_DATASET_SIZE,
    save_to_csv=USER_PROMPTS_PATH,
    max_new_prompts=MAX_NEW_PROMPTS_PER_RUN,
    batch_size=BATCH_SIZE,
)
