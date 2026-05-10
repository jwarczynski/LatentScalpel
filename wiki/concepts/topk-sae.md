---
tags: [concept, sae]
related: [[sae-training]] [[bug-sae-stuck-loss]]
---

# Top-K Sparse Autoencoder

Sparse autoencoder where sparsity is enforced by keeping only the top-K highest activations per sample, zeroing the rest. Introduced by Bussmann et al. / Anthropic Interpretability team.

## Architecture

- Encoder: $z = \text{TopK}_k(W_{\text{enc}}(x - b_{\text{dec}}))$
  - $x \in \mathbb{R}^{\text{activation\_dim}}$
  - $W_{\text{enc}} \in \mathbb{R}^{\text{dict\_size} \times \text{activation\_dim}}$
  - $\text{TopK}_k$ zeros all but the top-k values per row.
  - No encoder bias (b_enc removed — was a subtle bug; see implementation).
- Decoder: $\hat{x} = W_{\text{dec}} z + b_{\text{dec}}$
  - $W_{\text{dec}} \in \mathbb{R}^{\text{activation\_dim} \times \text{dict\_size}}$
  - Decoder columns kept unit-norm after each gradient step.

## Training

**Loss** = $\|\hat{x} - x\|^2$ (MSE). No sparsity penalty — sparsity is guaranteed by the TopK operator.

**k-annealing**: start with $k = k_{\text{start}} = k_{\text{target}} \times k_{\text{start\_multiplier}}$, ramp down linearly to $k_{\text{target}}$ over `k_anneal_steps`. Our current default: `k_start_multiplier=1.0` (no warmup) — the warmup was causing [[bug-sae-stuck-loss]].

**Decoder normalization**: after each `.backward()`, `sae.normalize_decoder_()` projects each column of $W_{\text{dec}}$ back to unit norm. Without this, columns could grow unboundedly (since scaling a column + shrinking a row compensates in the forward pass).

**Dead feature handling**:
- Track which features ever activate over a sliding window.
- Features that never fire get re-initialized (Anthropic "resample" trick): pick high-loss examples from the batch, center them, normalize, and place them as new $W_{\text{dec}}$ columns and $W_{\text{enc}}$ rows. Reset the Adam state for those params.

## Why Top-K over L1-penalty SAEs?

- No dead-feature explosion at init — TopK guarantees exactly K features fire per input.
- Simpler loss (no sparsity coefficient to tune).
- Easier to compare SAEs across sparsity levels (just change K).

Downsides:
- Hard constraint can reject useful non-top-k features at edge cases.
- Sparsity is fixed per sample, not absorbing (some inputs might need 20 features, others 200).

## Key hyperparameters

| Param | Meaning | Good range |
|---|---|---|
| `expansion_factor` | dict_size / activation_dim | 4–32 |
| `k_target` | Final sparsity level | activation_dim × 0.01 to 0.1 |
| `k_start_multiplier` | Initial k / target k | 1 (no warmup) or 2–4 (with warmup) |
| `learning_rate` | Adam lr | 1e-4 to 3e-4 |
| `batch_size` | Activations per batch | 2048–8192 |

For Plaid v3b (activation_dim=2048):
- expansion_factor=8 → dict_size=16384.
- k_target=64 → 3% sparsity.
- Normalized inputs (see [[sae-training]]).

## Metrics

- **FVE** (fraction of variance explained) — typical good: 0.7–0.95.
- **L0 sparsity** — should ≈ k_target.
- **Active feature fraction** — fraction of dictionary firing on at least one example in batch. Should be in [0.3, 0.8] during training.
- **Dead feature fraction** — over a window, how many features never fired. Resample strategy aims to keep this low.

## Implementation

`geniesae/sae.py::TopKSAE`:
- Forward: encode → TopK → decode.
- `initialize_from_data(data_mean)` sets `b_dec = data_mean` for a useful starting point.
- `set_k(k)` for the annealing schedule.
- `normalize_decoder_()` for the post-backward step.

`geniesae/sae_lightning.py::SAELightningModule` wraps this with Lightning training loop logic.
