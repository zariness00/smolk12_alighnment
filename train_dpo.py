from pathlib import Path
from datetime import timedelta
from time import monotonic

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ALIGNED_MODEL = "zariness00/smolk12_dpo"

DATASET_PATH = Path("data/alignment/combined_dpo_dataset")
DPO_CONFIG_PATH = Path("data/alignment/dpo_config.yaml")
TRAINING_RESULTS_DIR = Path("smolk12_dpo_output")
FINAL_MODEL_DIR = TRAINING_RESULTS_DIR / "final"

MAX_STEPS = 2 # 200 
WARMUP_STEPS = 50
PUSH_TO_HUB = False
FINETUNE_TAGS = ["from_SmolLM2-360M-Instruct"]


class TrainingTimerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = monotonic()
        self.start_step = state.global_step
        print(f"Training started: {state.max_steps - self.start_step} steps remaining.")

    def on_step_end(self, args, state, control, **kwargs):
        completed_steps = state.global_step - self.start_step
        total_steps = state.max_steps - self.start_step
        if completed_steps == 0:
            return

        if completed_steps == 1 or completed_steps % 5 == 0 or completed_steps == total_steps:
            elapsed_seconds = monotonic() - self.started_at
            seconds_per_step = elapsed_seconds / completed_steps
            remaining_steps = total_steps - completed_steps
            eta_seconds = seconds_per_step * remaining_steps

            elapsed = str(timedelta(seconds=int(elapsed_seconds)))
            eta = str(timedelta(seconds=int(eta_seconds)))
            print(
                f"Step {state.global_step}/{state.max_steps} | "
                f"elapsed {elapsed} | ETA {eta} | "
                f"{seconds_per_step:.1f} sec/step"
            )


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_training_args() -> DPOConfig:
    with DPO_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    training_config.update({
        "hub_model_id": ALIGNED_MODEL,
        "hub_private_repo": True,
        "output_dir": str(TRAINING_RESULTS_DIR),
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "push_to_hub": False,
    })

    supports_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    if training_config.get("bf16") and not supports_bf16:
        training_config["bf16"] = False
        print("bf16 is not supported on this device; using float32 training.")

    return DPOConfig(**training_config)


def main() -> None:
    dataset = load_from_disk(str(DATASET_PATH))
    required_columns = {"chosen", "rejected"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    device_name = get_device_name()
    print(f"Training device: {device_name}")
    print(f"Training examples: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = load_training_args()
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[TrainingTimerCallback()],
    )

    trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"Saved the final model to {FINAL_MODEL_DIR}")

    if PUSH_TO_HUB:
        trainer.push_to_hub(tags=FINETUNE_TAGS)


if __name__ == "__main__":
    main()
