---
tags: [pipeline, sae]
related: [[topk-sae]] [[activation-collection]] [[bug-sae-stuck-loss]]
---

# Stage 2: SAE Training

Trains a Top-K Sparse Autoencoder per layer on collected activations.

## Command

```bash
# Single layer
uv run python main.py train-sae configs/train_sae_plaid_finetuned_v3b.yaml --layer_idx=0

# Job array (one Slurm job per layer)
uv run python main.py train-sae configs/train_sae_plaid_finetuned_v3b.yaml \
    --layers 0 4 10 14 20 23 --submit --infra.cluster=slurm
```

Config class: `SAETrainingConfig` in `geniesae/configs/training_config.py`.

## SAE architecture (Top-K)

`TopKSAE` in `geniesae/sae.py`:

- Encoder: `W_enc ∈ R^{dict_size × activation_dim}`, no bias.
- Decoder: `W_dec ∈ R^{activation_dim × dict_size}`, `b_dec` bias.
- Forward: `z = TopK_k(W_enc @ (x - b_dec))`, `x_hat = W_dec @ z + b_dec`.
- Decoder column norms kept unit-normalized after each backward via `normalize_decoder_()`.

See [[topk-sae]] for the math.

## Key config fields

| Field | Description |
|---|---|
| `activation_dir` | Train activations. |
| `val_activation_dir`, `test_activation_dir` | Held-out splits. |
| `layer_idx` | Which layer to train. |
| `expansion_factor` | Dictionary size = `activation_dim × expansion_factor`. |
| `k_target` | Number of active features per token after annealing. |
| `k_start_multiplier` | Initial k = `k_target × k_start_multiplier`. |
| `k_anneal_fraction` | Fraction of total steps over which k ramps down. |
| `normalize_inputs` | Z-score inputs using train mean/std. |
| `max_samples` | Cap training samples (balanced across timesteps). |
| `dead_feature_strategy` | `none`, `resample` (Anthropic-style), or `aux_loss`. |
| `dead_feature_window` | Tokens between dead-feature checks. |
| `resume_from` | Checkpoint file or directory (for layer-aware resume). |

## Hyperparameters (current Plaid v3b setting)

See [[decision-sae-hyperparams]] for rationale.

```yaml
expansion_factor: 8        # dict_size = 2048 * 8 = 16384
k_target: 64               # 3% sparsity
k_start_multiplier: 1.0    # no warmup — train at k=64 from step 0
k_anneal_fraction: 0.1
normalize_inputs: true
learning_rate: 3e-4
batch_size: 4096
max_epochs: 15
max_samples: 50_000_000
dead_feature_strategy: resample
dead_feature_window: 1_000_000
```

Previous (failed) v2 had `expansion_factor=16, k_start_multiplier=4.0, normalize_inputs=false, max_samples=5M` — see [[bug-sae-stuck-loss]].

## Metrics logged to wandb

**Train:**
- `train/mse_loss` — reconstruction loss in normalized space (if normalizing).
- `train/fve` — fraction of variance explained.
- `train/l0_sparsity` — average number of active features per token.
- `train/active_feat_frac` — fraction of dictionary with at least one activation in the batch.
- `train/dead_feature_fraction` — reported every `dead_feature_window` tokens.

**Val / test:**
- `val/mse`, `val/fve`, `val/l0_sparsity`, `val/active_feat_frac` (similar for test).

Healthy training signatures:

- `mse_loss` shows initial spike then monotonic decrease.
- `fve` climbs toward 0.7–0.95.
- `active_feat_frac` settles in [0.3, 0.8] — most features periodically fire.
- `l0_sparsity` ≈ `k_target` throughout (hard constraint).

Unhealthy signatures (diagnosis in [[bug-sae-stuck-loss]]):
- Flat `mse_loss` from step 0.
- `active_feat_frac` stuck near 1.0 (k too high) or 0.01 (dead dominates).
- `fve < 0.3` after full training.

## Output

```
experiments/sae_checkpoints/<model>/
├── layer_00.ckpt           # last checkpoint (final epoch)
├── layer_00_best.ckpt      # best val/mse
├── layer_04.ckpt
└── ...
```

## Resume behavior

Pass `--resume_from=<path>`:
- If path is a file → load that specific checkpoint.
- If path is a directory → look for `layer_XX.ckpt` matching current `layer_idx` (prefers `layer_XX.ckpt` over `layer_XX_best.ckpt` because the former has more training).

This makes the job array work with `--resume_from=./experiments/sae_checkpoints/genie` automatically.

## Input normalization

When `normalize_inputs=true`:
1. `ActivationStore.compute_layer_std()` computes per-dim std (Welford's algorithm, cached).
2. `SAELightningModule._maybe_normalize()` subtracts mean and divides by std in every step (train/val/test).
3. The saved checkpoint has `input_mean` and `input_std` buffers so inference code can apply the same normalization.

See [[bug-sae-stuck-loss]] for why this matters for Plaid (but apparently not for GENIE — GENIE activations happened to already be well-scaled).

## Dead feature handling

Three strategies:

- **`none`** — just track and report dead fraction to wandb.
- **`resample`** (Anthropic-style) — every `dead_feature_window` tokens, re-initialize dead neurons' encoder rows and decoder columns to point at high-loss examples from the last batch. Resets their Adam state.
- **`aux_loss`** — add auxiliary reconstruction loss using only dead features' pre-activations; encourages them to produce signal.

We use `resample` by default for Plaid (as in the original Anthropic paper).

## Known issues

- Layer 0 historically has highest dead-feature fraction (~80%+ at init). Resample helps but many features stay dead. Consider separate hyperparams for this layer.
- Large val batches are slow due to no-worker dataloader (num_workers=0 to avoid RAM duplication — each timestep file is 20 GB).
