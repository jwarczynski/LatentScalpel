---
tags: [experiment, sae, plaid, finetuned]
status: completed-but-flawed
date: 2026-05-03
slurm_job: 2559785
related: [[bug-sae-stuck-loss]] [[exp-sae-finetuned-v3b-v3]]
---

# Exp: SAEs on fine-tuned v3b — v2 (proper 3-split)

Second attempt. Fixed the split policy from v1 (introduced [[decision-split-policy]] matching GENIE), but **loss was flat from step 0** — we had hit [[bug-sae-stuck-loss]].

## Setup

Collected train + test splits specifically for this:
- `configs/plaid_finetuned_collection_train.yaml`: 10k XSum train samples → job 2547972 (~2h50m).
- `configs/plaid_finetuned_collection_test.yaml`: 3k XSum test samples → job 2547973 (~1h8m).
- (Val split from v1 reused — 3k dev samples.)

Trained SAEs with:

```yaml
activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/train"
val_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/validation"
test_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/test"
layer_idx: 0
expansion_factor: 16
k_target: 64
k_start_multiplier: 4.0      # <-- still had k-warmup bug
max_samples: 5000000
learning_rate: 3e-4
normalize_inputs: false      # <-- no normalization
save_best: true
monitor_metric: "val/mse"
```

Config at `infra.version=2`. Job array 2559785.

## Outcome

- 6 jobs started on 2 nodes (t0017 for 0+4, t0031 for 10+14+20+23).
- Wandb showed `train/mse_loss` plateaued at a tiny value from the first logged step. No downward curve.
- Top-K sparsity was near 256 (the initial k before annealing) → SAE was essentially outputting near-mean with no specificity.

## Root cause found

[[bug-sae-stuck-loss]] — combination of:
1. `k_start_multiplier=4.0` → initial k too high, trivial to reconstruct.
2. `normalize_inputs=false` → small-norm Plaid activations make MSE numerically meaningless.
3. `expansion_factor=16` → too much capacity for the available signal.

## Action

Cancelled job 2559785 before it finished. Bumped config to `infra.version=3` with the fixes (see [[exp-sae-finetuned-v3b-v3]]).

## Lesson

Monitor `train/mse_loss` AND `train/active_feat_frac` in the first few hundred steps. If loss is flat and active_feat_frac is near 1.0, your k is too high or your initial reconstruction is too easy.
