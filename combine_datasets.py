from datasets import load_dataset, concatenate_datasets


DPO_DATASET = "zariness00/dpo-dataset"
COMBINED_DATASET_PATH = "data/alignment/combined_dpo_dataset"

dataset_k12 = load_dataset(path=DPO_DATASET, split="train")
dataset_ultra = load_dataset(path="trl-lib/ultrafeedback_binarized", split='train[:10%]')
dataset_ultra = dataset_ultra.remove_columns(['score_chosen', 'score_rejected'])
dataset = concatenate_datasets(
    [dataset_ultra,
     dataset_k12]).shuffle(seed=42
                          )

dataset.save_to_disk(COMBINED_DATASET_PATH)
print(f"Saved {len(dataset)} examples to {COMBINED_DATASET_PATH}")
