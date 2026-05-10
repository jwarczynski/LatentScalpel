---
tags: [bug, plaid, inference, sampling]
status: fixed
date: 2026-03-22
severity: high
fix_commit: 205c301
related: [[bug-mup-scaling]] [[bug-rotary-values]] [[vdm-loss]]
---

# Bug: samplers used gamma=0 for final decode

## Symptom

Even after fixing the muP and rotary bugs, generated tokens had subtle drift at the very last step. Unconditional sampling was mostly OK, but conditional/guidance paths produced slightly off text — often with misspelled words near the final positions.

## Root cause

`InpaintingSampler`, `TokenGuidanceSampler`, and `GradientGuidanceSampler` in `geniesae/plaid_samplers.py` all ended with:

```python
# buggy final decode
logits, _ = self.model(
    z=z_final,
    gamma=torch.zeros(B, device=device),  # ❌
    embedding_matrix=embedding_matrix,
    bias_scale=1.0,
    x_selfcond=x_selfcond,
    selfcond_mask=torch.ones(B, device=device),
)
```

The original Plaid `sample.py` uses `gamma_t` from the **last iteration of the reverse chain**:

```python
# original
logits, _ = modules['model'](
    z=z.float(),
    gamma=gamma_t.float(),   # last value, NOT zero
    ...
)
```

The value of `gamma_t` at the end of sampling is `gamma(t=1/T)` — i.e. `gamma_0 ≈ -3`, **not** 0.

Why this matters in the model forward pass (`DiffusionModel.forward`):

```python
alpha_squared = torch.sigmoid(-gamma)[:,None,None]
sigma_squared = torch.sigmoid(gamma)[:,None,None]
alpha = alpha_squared.sqrt()

# z_variance depends on gamma
z_variance = (alpha_squared / self.embed_dim) + sigma_squared
x = z / z_variance.sqrt().float()

# gamma also enters bias scaling
z_scaled_for_bias = bias_scale * (alpha / sigma_squared).float() * z
```

With `gamma=0`: `alpha² = sigmoid(0) = 0.5`, `sigma² = 0.5`, `z_variance = 0.5/embed_dim + 0.5 ≈ 0.53`.
With `gamma=-3`: `alpha² = sigmoid(3) ≈ 0.95`, `sigma² ≈ 0.05`, `z_variance ≈ 0.06`.

So `z / sqrt(z_variance)` differs by roughly `sqrt(0.53/0.06) ≈ 3×` between the two. Same for `z_scaled_for_bias`. The model was never trained at gamma=0, so it produced corrupted logits at that unseen input scaling.

## Fix

All three samplers now use `gamma_t` from the final iteration (still in scope after the loop):

```python
# final decode — use gamma_t from last step, not literal zero
logits, _ = self.model(
    z=z_final,
    gamma=gamma_t.float(),  # ✅ matches training distribution
    ...
)
```

Fixed in commit ``205c301``. Three separate locations in `geniesae/plaid_samplers.py`.

## Why the standalone script avoided this

`scripts/plaid_original_xsum_generate.py` copies the original sample.py structure directly and naturally uses the loop's final `gamma_t`. That's why the original-code comparison run looked correct while ours didn't. See [[exp-original-code-comparison]].

## Detection

Compare the last iteration's gamma with what gets passed to the final decode:

```python
# Add before final decode
print(f"last gamma_t: {gamma_t.mean().item():.3f}")
print(f"gamma passed to decode: {final_gamma.mean().item():.3f}")
# These should be equal (both ≈ -3 for default gamma_0)
```
