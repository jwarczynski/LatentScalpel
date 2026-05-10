---
tags: [bug, plaid, inference, training]
status: fixed
date: 2026-03-22
severity: critical
fix_commit: d06f5dd
related: [[mup-scaling]] [[plaid-1b]] [[bug-rotary-values]]
---

# Bug: muP output_mult = 0.125 instead of 1.0

## Symptom

Both unconditional and conditional generation produced gibberish — random tokens, no coherent text. Fine-tuning loss curves looked plausible but saved weights generated nonsense.

## Root cause

In `geniesae/plaid_model.py`, the function `_apply_mup_shapes` set:

```python
width_mult = dim / base_dim          # 2048 / 256 = 8.0   ✅ correct
output_mult = base_dim / dim         # 256 / 2048 = 0.125 ❌ wrong
```

The original Plaid uses `mup.MuReadout` whose default is `output_mult = 1.0` (see `mup/layer.py`). Its forward does:

```python
# MuReadout.forward
return super().forward(self.output_mult * x / self.width_mult())
```

So the effective pre-linear scaling factor is `output_mult / width_mult`:

| Setting | Correct | Our bug |
|---|---|---|
| output_mult | 1.0 | 0.125 |
| width_mult | 8.0 | 8.0 |
| scaling | 1/8 = 0.125 | 0.125/8 = 0.015625 |

Our logits were **8× too small**. A softmax over 8×-compressed logits is near-uniform → argmax picks essentially random tokens → gibberish.

## How we found it

After cloning the original Plaid repo and running our reimplementation side-by-side on the same seed, our samples were gibberish while the original produced readable English. We loaded both into the same process and compared `output_linear.output_mult` and `width_mult()`. Ours were 0.125/8.0; original mup-set values were 1.0/8.0. See [[exp-original-code-comparison]].

## Fix

`geniesae/plaid_model.py`:

```python
def _apply_mup_shapes(model: DiffusionModel, dim: int, base_dim: int = 256) -> None:
    width_mult = dim / base_dim   # 2048/256 = 8.0
    output_mult = 1.0             # MuReadout default — NOT base_dim/dim
    model.output_linear._output_mult.fill_(output_mult)
    model.output_linear._width_mult.fill_(width_mult)
```

Fixed in commit ``d06f5dd``.

## Why this bug mattered for fine-tuning too

The bug wasn't only inference — fine-tuning used the same `DiffusionModel.forward` path. Every gradient was computed through logits that were 8× too small:

- Reconstruction CE loss: computed on crushed logits → wrong gradient magnitude.
- Diffusion MSE loss: `x_reconst` is computed via `softmax(logits) @ embedding_matrix`; a near-uniform softmax collapses to the mean embedding → the model optimized a degenerate target.

All fine-tuning runs prior to ``d06f5dd`` (mid-March) suffer from this. See [[plaid-finetune-history]].

## Detection checklist

If logits have `std < 0.5` at inference time across the batch+sequence, something's wrong with scaling. Healthy range for Plaid logits is `std ∈ [1.0, 2.5]`.

```python
# Quick diagnostic
with torch.no_grad():
    logits, _ = model(z, gamma, embedding_matrix, bias_scale=1.0,
                      x_selfcond=torch.zeros_like(z))
    print(f"logits std: {logits.float().std().item():.3f}")
```
