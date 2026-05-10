---
tags: [bug, plaid, training, loss]
status: fixed
date: 2026-03-24
severity: medium
fix_commit: 0042bcc
related: [[bug-selfcond-detachment]] [[vdm-loss]]
---

# Bug: prior loss reduction mismatch

## Symptom

Minor numerical drift in the prior loss component relative to the original Plaid. Not catastrophic on its own but compounded with other bugs during v1/v2 training.

## Root cause

Our original prior loss was:

```python
prior_kl = (-sigma_1.log() + 0.5 * sigma_1**2
            + 0.5 * alpha_1**2 * x_embed**2 - 0.5)  # (B, S, E)
prior_kl = prior_kl * loss_mask[:, :, None]
valid_counts_all = loss_mask.sum(dim=1).clamp(min=1)
prior_loss = (prior_kl.sum(dim=1) / valid_counts_all[:, None]).sum(dim=1).mean()
```

The original `train.py`:

```python
alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:, None, None]
sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:, None, None]
prior_loss = lib.ops.gaussian_kl(
    (alpha_1_masked * x_embed), sigma_1_masked,
    torch.tensor(0., device='cuda'), torch.tensor(1., device='cuda')
).sum(dim=2).mean()
```

Two differences:
1. Original detaches `alpha_1`/`sigma_1` for selfcond examples (see [[bug-selfcond-detachment]]).
2. Original reduction is `.sum(dim=2).mean()` — sum over embed_dim, mean over (B × S). Ours had a more complex masked reduction.

## Fix

Match original reduction; add masking for the conditional/template case where padding exists:

```python
# Detach alpha_1, sigma_1 for selfcond examples
alpha_1_masked = torch.lerp(
    alpha_1.expand(B), alpha_1.detach().expand(B), selfcond_mask.double()
)[:, None, None]
sigma_1_masked = torch.lerp(
    sigma_1.expand(B), sigma_1.detach().expand(B), selfcond_mask.double()
)[:, None, None]

prior_kl = (
    -sigma_1_masked.log()
    + 0.5 * sigma_1_masked ** 2
    + 0.5 * (alpha_1_masked * x_embed_d) ** 2
    - 0.5
)  # (B, S, E)

# Sum over embed_dim → (B, S)
prior_kl_per_example = prior_kl.sum(dim=2)

# For conditional/template: mean over valid (non-padding) positions only
if self.training_mode in ("conditional", "template") and boundary_idx is not None:
    prior_kl = prior_kl * loss_mask[:, :, None]
    valid_counts_all = loss_mask.sum(dim=1).clamp(min=1)
    prior_loss = (prior_kl_per_example.sum(dim=1) / valid_counts_all).mean()
else:
    prior_loss = prior_kl_per_example.mean()
```

Fixed together with [[bug-selfcond-detachment]] in commit ``0042bcc``.
