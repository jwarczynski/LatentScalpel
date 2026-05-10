---
tags: [experiment, sae, plaid, finetuned, current]
status: running
date: 2026-05-03
slurm_job: 2559832
related: [[bug-sae-stuck-loss]] [[decision-sae-hyperparams]]
---

# Exp: SAEs on fine-tuned v3b — v3 (current)

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

Job array 2559832 — 6 tasks (one per layer). Submitted 2026-05-03 11:04.

## Wandb

Project: `plaid-sae-finetuned`. New runs named per layer. Monitor metrics:

- `train/mse_loss` — should spike then decrease.
- `train/fve` — should climb to 0.7–0.9.
- `train/active_feat_frac` — should be in [0.3, 0.8].
- `val/mse`, `val/fve` — should track train.

## What to do if it still doesn't work

See [[bug-sae-stuck-loss]] "What to watch next" section. Next fallbacks:
- Increase `k_target` to 128 or 256 (current 64 might be too sparse for 2048-dim).
- Increase `learning_rate` to 1e-3.
- Check `data_mean` and `data_std` values — if std is near-zero in some dims, normalization divides by ~0 and produces huge inputs. Clamp `input_std.clamp(min=1e-3)` to be safe.

## Next stages after SAE completes

1. Re-run find-top-examples on **test** activations (not val, not train) — proper held-out set.
2. Re-run trajectory collection with new SAEs.
3. Re-run interpretation on new top-examples.
4. Compare feature quality vs v1 interpretations.
