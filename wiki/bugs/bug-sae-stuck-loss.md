---
tags: [bug, sae, training]
status: fixed
date: 2026-05-03
severity: high
fix_commit: 92a7752
related: [[topk-sae]] [[exp-sae-finetuned-v3b-v2]] [[exp-sae-finetuned-v3b-v3]]
---

# Bug: SAE training loss flat from step 0

## Symptom

First SAE training run on fine-tuned Plaid v3b activations (job 2559785, `infra.version=2`) showed train/mse_loss plateaued at a small value from the very first step with no decrease. The learned features produced near-identical interpretations ("direct quotes from named individuals") across thousands of features. See [[exp-sae-finetuned-v3b-v1]] and [[exp-sae-finetuned-v3b-v2]].

## Root causes (compound)

Several design choices conspired to give a trivial solution:

### 1. `k_start_multiplier=4.0` with `k_anneal_fraction=0.1`

Initial `k = 4 × k_target = 256` (on a 32768-dict). At this k, a random sparse code already achieves near-zero MSE because 256 active features out of 32768 is enough to cover the input manifold trivially. The SAE never has to learn specific sparse features before the k ramp-down kicks in, and by then the weights are stuck in a bad basin.

### 2. No input normalization

PLAID block outputs have layer-dependent scale. Layer 0 activations have very different norm statistics than layer 23. Without normalization:
- The loss magnitude varies by orders of magnitude across layers.
- A fixed `learning_rate=3e-4` is wrong for most layers.
- A near-zero-output SAE is easy (low input variance → low MSE).

### 3. `expansion_factor=16` too aggressive

Dictionary size = `2048 × 16 = 32768`. With only ~46M activations total (3K samples × 256 tokens × 6 layers × 10 timesteps = validation split), each feature sees ~1400 activations if uniformly active. With TopK pushing most features to zero, live features see even less. Not enough signal to differentiate.

### 4. `max_samples=5_000_000` too small

With 153M train activations available, we used only 3%. Not enough diversity.

## Fix (combined in ``92a7752``)

1. **Input normalization toggle** — `normalize_inputs: true`. Computes per-dim mean/std from train activations (cached like the mean). Subtracts mean, divides by std before encoding. Reconstruction loss is computed in the normalized space.

2. **Kill k-warmup** — `k_start_multiplier: 1.0`. Train at `k_target=64` from step 0. No ramp-down period where the problem is trivial.

3. **Reduce expansion_factor** — `expansion_factor: 8` (dictionary=16384). Less overcompleteness, more training signal per feature.

4. **Increase training data** — `max_samples: 50_000_000`. 10× more.

5. **New metric**: `active_feat_frac` logged in train/val/test. Tells us live vs dead feature counts in real time.

## Implementation details

Added to `SAELightningModule`:
- Buffers `input_mean`, `input_std` (shape `(activation_dim,)`).
- `_maybe_normalize(x)` helper applied in `training_step`, `validation_step`, `test_step`.
- `active_feat_frac` metric: `(z != 0).any(dim=0).float().mean()` — fraction of features that fire on at least one example in the batch.

Added to `ActivationStore`:
- `compute_layer_std(layer_idx, *, max_samples=None)` using Welford's online algorithm, cached to `layer_XX/std_NNNk.pt`.

Added to `SAETrainingConfig`:
- Field `normalize_inputs: bool = False`.
- Computes std when enabled, passes mean+std to `SAELightningModule`.

## Detection

- `train/mse_loss` flat from step 0 (no initial spike, no decrease) → something's wrong.
- `train/active_feat_frac` stuck near 1.0 (all features always fire) → k is too high.
- `train/active_feat_frac` stuck near 0.01 (only a few features ever fire) → dead features dominate, probably bad init or too-sparse k.

Healthy Top-K SAE training:
- Initial spike in `train/mse_loss`, then clean log-shaped decrease.
- `train/active_feat_frac` ∈ [0.3, 0.8] during training — most features have some chance to fire.
- `train/l0_sparsity` approximately equals `k_target` throughout (hard TopK constraint).
- `train/fve` ramps to 0.7–0.95 by end of training.

## What to watch next

Job 2559832 (v3) started at 2026-05-03 11:04. Monitor wandb at [plaid-sae-finetuned](https://wandb.ai/jedrasowicz/plaid-sae-finetuned) for these signals. If loss is still flat, next suspects are:
- Learning rate (try 1e-4 or 1e-3 for normalized inputs).
- `k_target` too small — try 128 or 256.
- Bad init from `initialize_from_data(data_mean)` — it sets `b_dec = data_mean`; with normalized inputs, mean is zero, so b_dec is zero.
