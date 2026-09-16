# SmolLM2 K-12 Alignment with DPO

I decided to do this project while reading the book *Large Language Models:
The Hard Parts* while being unemployed :D

Before that, I worked as an Instructional Designer for 3+ years. I was very
committed to what I used to do, and in three years I produced more than 200
lessons in different disciplines, mainly for the Computing Vertical. The
quality of those lessons was also developing together with my experience.

One day, we used the OpenAI platform and, instead of creating a LLM for our
students, we attached many, many lessons to the agent, wrote the instructions,
and connected it to the learning platform.

Two years later, I decided to go deeper and create a small language model for
an imaginary company that produces computing courses for K-12 students. The
model should help students learn while following an explicit classroom content
safety policy.

The project is based on
`HuggingFaceTB/SmolLM2-360M-Instruct`. I generated synthetic preference data,
fine-tuned the model with Direct Preference Optimization (DPO), performed a
manual vibe check, and then compared the base and aligned models using
`gpt-4o-mini` as an LLM judge.

And yes, the evaluation humbled me a little :D

## What I built

The full pipeline contains the following steps:

1. Define the K-12 safety policy.
2. Generate synthetic user prompts that may violate the policy.
3. Generate policy-aligned `chosen` responses.
4. Generate base-model `rejected` responses.
5. Build and review a DPO preference dataset.
6. Combine the K-12 data with a subset of UltraFeedback.
7. Fine-tune SmolLM2-360M-Instruct with DPO on a rented GPU.
8. Inspect training dynamics with TensorBoard.
9. Run a qualitative vibe check.
10. Generate paired base/aligned evaluation responses.
11. Evaluate both models with an LLM-as-a-judge pipeline.

## Synthetic preference data

The policy is stored in [`k_12_policy.md`](k_12_policy.md). It defines acceptable
educational assistance, prohibited K-12 content, and the expected refusal
protocol. (It is super light and simple)

I generated:

- 5,000 potentially policy-violating prompts with `ministral-8b-2512`;
- aligned `chosen` responses with `Qwen/Qwen3-8B-AWQ`;
- `rejected` responses with the base SmolLM2-360M-Instruct model;
- a 40-row sample for manual review before training.

The K-12 preference dataset was combined with 10% of
`trl-lib/ultrafeedback_binarized`. The final shuffled training dataset contained
11,204 `chosen/rejected` pairs.

The generation scripts save intermediate progress. This mattered because API
calls failed from time to time, and I did not want one SSL error to destroy
several hours of generated data.

## DPO fine-tuning

I fine-tuned the model on Vast.ai using an NVIDIA RTX 3080 Ti. This was a full
DPO fine-tune, not a LoRA adapter.

Main training parameters:

| Parameter | Value |
|---|---:|
| Training examples available | 11,204 |
| Maximum optimizer steps | 200 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 |
| Warm-up steps | 50 |
| Learning rate | `5e-5` |
| Scheduler | cosine |
| Maximum sequence length | 1,536 |
| DPO beta | 0.1 |
| Checkpoint interval | 50 steps |

With 200 optimizer steps and an effective batch size of 16, the run processed
approximately 3,200 training examples rather than a complete epoch over all
11,204 pairs.

TensorBoard showed that the model was learning the preference signal:

- training loss moved from approximately `0.693` to a last-20-step average of
  `0.378`;
- the last-20-step chosen reward averaged approximately `+2.51`;
- the rejected reward averaged approximately `-1.18`;
- the reward margin reached approximately `3.69`;
- reward accuracy over the final 20 steps was approximately `79%`.

The final model was published as a private model repository:
[`zariness00/smolk12_dpo`](https://huggingface.co/zariness00/smolk12_dpo).


## Evaluation

### Vibe check

I first compared a few hand-written safety prompts. The aligned model produced
some good refusals, but it also failed critical requests and occasionally gave
unsafe instructions. It also over-refused some harmless creative-writing
requests. This was the first sign that good training curves do not automatically
mean a safe model.

### LLM as a judge

I sampled 100 prompts and generated responses from both models. One base-model
request failed with an SSL error, so the final clean evaluation contained 99
paired responses. `gpt-4o-mini` judged each response against the K-12 policy
using this scale:

- `0.1` — Not Aligned;
- `0.5` — Somewhat Aligned;
- `1.0` — Aligned.

| Alignment category | Base model | DPO-aligned model |
|---|---:|---:|
| Not Aligned | 81 (81.8%) | 55 (55.6%) |
| Somewhat Aligned | 17 (17.2%) | 19 (19.2%) |
| Aligned | 1 (1.0%) | 25 (25.3%) |
| **Mean score** | **0.178** | **0.404** |

The mean judge score improved by approximately 2.3x, and the percentage of fully
aligned responses increased from 1.0% to 25.3%. So DPO definitely changed the
model in the intended direction.

However, 55.6% of the aligned model responses were still classified as not
aligned. I would not release this version as a public K-12 assistant...

## What did not work perfectly

This project is an experiment, not a claim that the safety problem is solved.
The main limitations are:

- the evaluation prompts came from the same DPO prompt pool, so this is an
  in-distribution evaluation rather than a clean held-out test;
- synthetic `chosen` responses were generated by an open-weight teacher model,
  and some training pairs may contain weak or inconsistent supervision;
- the 200-step run saw only part of the available dataset;
- SmolLM2-360M has limited capacity for consistently following a detailed
  safety policy;
- the judge results come from one model, one policy rubric, and a relatively
  small evaluation sample.

The aligned model demonstrated both **under-refusal** on unsafe requests and
**over-refusal** on acceptable fictional or educational requests.

## Next steps

1. Audit rows where `score_aligned == 0.1` and trace them back to their original
   `chosen/rejected` training pairs.
2. Regenerate `chosen` responses using the open source model.
3. Validate synthetic preference pairs with a judge before training.
4. Create a genuinely held-out safety and regression test set.
5. Train for longer and compare checkpoints instead of trusting one run.
6. Package inference with FastAPI and Docker, then deploy a private demo with
   authentication, logging, monitoring, and a rollback plan.

## Running the project

Install the dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

The scripts follow the pipeline order:

```bash
python user_prompts.py
python chosen_responses.py
python rejected_responses.py
python dpo_dataset.py
python combine_datasets.py
python train_dpo.py
python eval_dataset.py
python judge.py
```

API credentials and endpoint URLs are loaded from `.env`. The project uses the
following variables depending on the stage:

```text
MISTRAL_API_KEY
HF_TOKEN
ENDPOINT_QWEN
ENDPOINT_VLLM
OPENAI_API_KEY
```

The `.env` file, model weights, caches, and local training outputs must not be
committed to Git.
