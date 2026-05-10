---
tags: [experiment, finetune, plaid]
status: completed
date: 2026-04-14
wandb_run_id: 5gepdez9
result: best-so-far
related: [[plaid-finetuned-v3b]] [[plaid-finetune-history]]
---

# Exp: conditional v3b (best)

Our current best fine-tune run. Full details on the produced checkpoint: [[plaid-finetuned-v3b]].

## Config

`configs/plaid_xsum_conditional_8gpu_v3b.yaml`

```yaml
weights_path: "models/plaid/plaid1b_weights"
dim: 2048
embed_dim: 16
n_blocks: 24
n_heads: 32
vocab_size: 32768
gamma_0: -3.0
gamma_1: 6.0

data_dir: "datasets/glge-released-dataset/easy/xsum_data/org_data"
seq_len: 256
max_summary_len: 64
tokenizer_path: "models/plaid/plaid1b_weights/tokenizer.json"

batch_size: 4
learning_rate: 1.0e-6
weight_decay: 4.0e-5
betas: [0.9, 0.99]
num_epochs: 100
warmup_steps: 500
bias_warmup_steps: 5000
target_bias_scale: 1.0
self_cond_prob: 0.25
clip_quantile: 0.95
num_workers: 0
lr_schedule: "cosine"

training_mode: "conditional"
strategy: "ddp"
num_gpus: 8
precision: "bf16-mixed"
gradient_checkpointing: true

sampling_timesteps: 256
score_temp: 0.9
sampler: "inpainting"
num_eval_samples: 5

use_wandb: true
wandb_project: "plaid-xsum"
wandb_run_name: "cond-256-lr1e6-cosine-100ep"
log_interval: 50
output_dir: "./experiments/plaid_xsum_v3b"

infra:
  version: "1"
  folder: "./experiments/cache/plaid_xsum_v3b"
  cluster: slurm
  gpus_per_node: 8
  tasks_per_node: 8
  cpus_per_task: 4
  mem_gb: 320
  timeout_min: 2800
  slurm_partition: plgrid-gpu-a100
  slurm_use_srun: true
```

## Slurm jobs

- Initial run: job 2517798, started 2026-04-12.
- Resume attempt: job 2505405 / 2506663 (cancelled — see [[plaid-finetune-history]]).

## Wandb

- Run ID: `5gepdez9`
- URL: https://wandb.ai/jedrasowicz/plaid-xsum/runs/5gepdez9

## Checkpoints

```
experiments/plaid_xsum_v3b/checkpoints/cond-256-lr1e6-cosine-100ep/
├── best-epoch85.ckpt    # best val loss → use this one
├── last-v1.ckpt         # epoch 100 (last, superseded)
└── last.ckpt            # slightly older than last-v1.ckpt
```

In Lightning: `last.ckpt` becomes `last-v1.ckpt` when a new last is saved, so `last-v1.ckpt` is the final one.

## Outcome

Training loss dropped sharply in first ~500 steps then decreased smoothly. Val loss still decreasing at epoch 100 but lr reached ~0 from cosine. Subjectively the best model we've produced — coherent summaries with occasional hallucinations.

## Evaluation

See [[exp-eval-v3b-inference]] for 100 XSum dev+test generations.

## Why seq_len=256 won over 1024

See [[decision-seq-len-256]]. Short version: seq_len=1024 was never necessary for XSum (summaries are 64 tokens max) and the longer context made gradient signal sparse.

## What we did after

- Used this checkpoint for all downstream SAE analysis.
- Attempted to resume with lr=1e-7 linear tail — abandoned.
- Used it as the checkpoint in activation collection for [[exp-sae-finetuned-v3b-v1]] and subsequent SAE runs.
