---
tags: [decision, sae, hyperparams]
status: adopted
date: 2026-05-03
related: [[exp-sae-finetuned-v3b-v3]] [[bug-sae-stuck-loss]]
---

# Decision: SAE hyperparameters for Plaid v3b

Settled on expansion=8, k=64, normalize_inputs=true, no k-warmup after v1/v2 showed problems.

## Current settings

```yaml
expansion_factor: 8             # dict_size = 2048 * 8 = 16384
k_target: 64                    # 3% sparsity (64/16384)
k_start_multiplier: 1.0         # NO warmup — train with k=64 from step 0
k_anneal_fraction: 0.1
normalize_inputs: true
learning_rate: 3e-4
batch_size: 4096
max_epochs: 15
max_samples: 50_000_000
dead_feature_strategy: "resample"
dead_feature_window: 1_000_000
```

## What each change fixed

### expansion_factor: 16 → 8

Reduces dictionary size from 32k to 16k. With only 47M val activations previously, we had ~23 train examples per feature — not enough to differentiate. With 50M train samples and 16k features, we have ~3000 per feature.

### k_start_multiplier: 4.0 → 1.0

Kills the k-warmup. With k=256 initially and 32k dict, near-random features could reconstruct the input well (26% firing rate). SAE was finding a trivial solution before the k ramp-down kicked in. With k=64 fixed, the task is genuinely hard from step 0.

### normalize_inputs: false → true

Plaid block outputs have layer-dependent scale. Without z-score normalization, MSE magnitude varies wildly by layer, and a fixed lr works for some layers and fails for others. Normalization also makes the loss numerically meaningful — you can read MSE values as "fraction of variance remaining."

### max_samples: 5M → 50M

We had 153M train activations available. Using only 3% was leaving most of the signal on the table.

## Why this combination might still not work

If v3 fails, next suspects to try:

1. **k_target too small**. With 2048-dim activations and 16k dictionary, 64 features may be too sparse. Try k=128 or k=256.
2. **Learning rate**. For normalized inputs, 3e-4 might be too low. Try 1e-3.
3. **Initial `b_dec = data_mean`**. With normalized inputs, `data_mean ≈ 0`, so `b_dec ≈ 0`. Probably fine but worth checking.
4. **Dead feature strategy**. `resample` kicks in every 1M tokens. For 50M samples in 15 epochs = 750M tokens, that's 750 resamples — reasonable. But if resample fires too aggressively it can destabilize. Consider `none` or longer window.

## Why NOT other architectures

- **L1-penalty SAE**: more hyperparameters to tune (sparsity coefficient), harder to compare across layers.
- **Gated SAE**: newer, less tooling in the codebase.
- **Jump ReLU SAE**: similar to gated, requires implementation.

TopK is the path of least resistance and has worked in our GENIE pipeline.

## Related

- [[bug-sae-stuck-loss]] — full diagnosis of why v2 failed.
- [[topk-sae]] — architecture details.
- [[exp-sae-finetuned-v3b-v3]] — current run using these settings.
