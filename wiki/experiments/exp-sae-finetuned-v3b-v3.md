---
tags: [experiment, sae, plaid, finetuned, done]
status: done
date: 2026-05-03
slurm_job: 2559832
related: [[bug-sae-stuck-loss]] [[decision-sae-hyperparams]] [[exp-sae-finetuned-v3b-v4]]
---

# Exp: SAEs on fine-tuned v3b — v3

Third attempt. Fixes all issues from v1 and v2: proper 3-split, plus normalization + no k-warmup + smaller expansion + more data.

## Setup

Same activation data as v2. Config at `infra.version=3` with:

```yaml
activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/train"
val_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/validation"
test_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/test"

expansion_factor: 8              # 16384 dict (was 32768 in v1/v2)
k_target: 64
k_start_multiplier: 1.0          # NO warmup, k=64 from step 0
k_anneal_fraction: 0.1
normalize_inputs: true           # NEW

learning_rate: 3e-4
batch_size: 4096
max_epochs: 15
max_samples: 50000000            # 10x more data

dead_feature_strategy: "resample"
dead_feature_window: 1_000_000
save_best: true
monitor_metric: "val/mse"

infra:
  version: "3"
```

## New features added to codebase

Commit ``92a7752``:

1. **`normalize_inputs` switch in `SAELightningModule`** — optional per-dim z-score using train mean/std buffers.
2. **`ActivationStore.compute_layer_std()`** — Welford online std with disk cache.
3. **`train/val/test active_feat_frac` metric** — fraction of features that fire in at least one example in batch.
4. **Updated `training_config.py`** — passes mean/std to Lightning module when `normalize_inputs=true`.

## Slurm

Job array 2559832 — 6 tasks (one per layer). Submitted 2026-05-03 11:04. All completed by ~14:15.

## Wandb

Project: `plaid-sae-finetuned`. One run per layer.

| Layer | wandb run       | train MSE | val MSE | val FVE | dead frac | active feat frac (val) |
|-------|-----------------|-----------|---------|---------|-----------|------------------------|
| 00    | hejbmbm6        | 0.006     | 1.71    | **−4.83** | 0.20    | 0.03                   |
| 04    | 743yhy6q        | 0.106     | 0.90    | **−0.34** | 0.82    | 0.44                   |
| 10    | kmen6rxk        | 0.220     | 0.38    | 0.50    | 0.54      | 0.45                   |
| 14    | a1pxff8x        | 0.330     | 0.38    | 0.51    | 0.55      | 0.41                   |
| 20    | n9kgatc0        | 0.354     | 0.32    | **0.63** | 0.32    | 0.43                   |
| 23    | bv9d07xr        | 0.369     | 0.48    | 0.28    | 0.59      | 0.43                   |

## Results interpretation

Three tiers:

- **Usable (layers 10, 14, 20):** val FVE ≥ 0.5, moderate dead fraction, active feature fractions in healthy range. Layer 20 is the best (val FVE 0.63). Proceed with downstream analysis using these layers.
- **Marginal (layer 23):** val FVE 0.28 is below GENIE-era baselines but features fire. Worth collecting top examples to see if anything meaningful emerges.
- **Failed (layers 00, 04):** huge train/val gap. Layer 00 is near-noise features (1% activation, massive overfit). Layer 04 memorizes train but is worse than the mean on val/test. These layers overfit because the input distribution at early layers is closer to raw embeddings — the model has not yet built a generalizable representation.

Hypothesis: early layers (0, 4) encode token identity which is very high-entropy and low-structure, so a TopK SAE with k=64 cannot compress the diversity without overfitting. Later layers (20) have built task-relevant structure that compresses well.

## Next steps

- **Downstream analysis on layers 10, 14, 20**, then optionally 23 for comparison.
  - `find-top-examples` on test split
  - `trajectory` collection
  - `interpret` with vLLM
- **For layers 0 and 4**, skip downstream. Optionally try a v4 for these with reduced expansion (e.g., 4) or larger k (128, 256) to see if FVE recovers.
- **Important**: use the `*_best.ckpt` versions (from step ~5k–10k) rather than the final epoch checkpoints — val monitor is MSE, which catches overfitting early.

