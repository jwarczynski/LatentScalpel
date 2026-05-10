---
tags: [bug, training, dtype]
status: fixed
date: 2026-03-24
severity: low
fix_commits: [34c4f6e, 16a3477]
related: [[bug-selfcond-detachment]]
---

# Bug: torch.lerp dtype mismatch

## Symptom

Fine-tuning job failed after ~1 minute during validation step:

```
RuntimeError: expected dtype double for `weight` but got dtype float
  File "geniesae/plaid_xsum_training.py", line 252, in _compute_vlb_loss
    gamma_t = torch.lerp(gamma_t, gamma_t.detach(), sc_float)
```

## Root cause

`torch.lerp(input, end, weight)` requires `weight.dtype == input.dtype`.

Our code used `.float()` for the mask:

```python
sc_float = sc_mask.float()
gamma_t = torch.lerp(gamma_t, gamma_t.detach(), sc_float)  # gamma_t is float64
```

But `gamma_t` is float64 (produced by `NoiseSchedule.forward` which uses `.double()` internally). Similarly `alpha_1`/`sigma_1` are float64. So weight must be float64 too.

At the same time, `x_embed` is float32 (produced by `embedding_matrix[token_ids]` where embedding matrix is `.float()`). So for that lerp, weight must stay float32.

## Fix

Use separate casts for float32 and float64 inputs:

```python
if sc_mask.any():
    sc_double = sc_mask.double()   # for gamma_t, gamma_prime (float64)
    sc_float = sc_mask.float()     # for x_embed (float32)
    gamma_t = torch.lerp(gamma_t, gamma_t.detach(), sc_double)
    gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), sc_double)
    x_embed = torch.lerp(x_embed, x_embed.detach(), sc_float[:, None, None])

# And in the prior loss:
alpha_1_masked = torch.lerp(
    alpha_1.expand(B), alpha_1.detach().expand(B), selfcond_mask.double()
)[:, None, None]
```

Fixed across ``34c4f6e`` and ``16a3477``.

## Why it only showed up during validation

Training step had the same code but different sampling made `sc_mask.any()` sometimes False. Validation always hit the lerp because the mask sampling is deterministic enough. A smoke test with `overfit_batches=10` on 1 GPU caught it in 1m44s before wasting an 8-GPU run.

## Takeaway

Always do a 1-GPU smoke test before submitting multi-GPU jobs. Cluster queue waits can be hours; a 1-GPU smoke test costs ~2 min and catches 90% of pre-training errors. Pattern:

```yaml
# configs/plaid_xsum_v2_1gpu_test.yaml
overfit_batches: 10     # Lightning trick: same 10 batches every epoch
num_gpus: 1
max_epochs: 1
timeout_min: 15
```

See [[exp-conditional-v2]] for the full story.
