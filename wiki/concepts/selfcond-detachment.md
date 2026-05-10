---
tags: [concept, training, selfcond]
related: [[vdm-loss]] [[bug-selfcond-detachment]]
---

# Self-conditioning gradient detachment

When self-conditioning is applied, specific tensors must be detached from the computation graph to avoid destabilizing the noise schedule and embedding matrix.

## The setup

Self-conditioning runs the model twice per step for a subset of examples (25% by default):

1. **First pass** (under `torch.no_grad()`): compute `x_selfcond` (a denoised embedding prediction).
2. **Second pass** (with gradients): feed `x_selfcond` as additional input. Compute the loss.

The backward of the second pass sends gradients through everything that was used in the final forward graph — including $\gamma(t)$, the embedding matrix, and $\gamma_1$-derived quantities.

## Why detachment is needed

Consider two example classes in a batch:

- **Non-selfcond examples**: The gradient through $\gamma(t)$ tells the noise schedule "this is the right shape for the standard VLB objective."
- **Selfcond examples**: The gradient through $\gamma(t)$ tells the noise schedule "this is the right shape so the model's reconstruction of its own first-pass prediction is easier."

These two signals can **point in different directions**. Without detachment, the noise schedule gets tugged both ways and drifts. Same for the embedding matrix.

## What original Plaid detaches

From `lib/ddp.py` / `train.py`:

```python
# Detach for selfcond examples (weighted lerp)
gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)
x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:, None, None])

# Also for the prior loss
alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:, None, None]
sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:, None, None]
```

`torch.lerp(x, x.detach(), w)` is a trick to get a partially-detached tensor: when `w=0` the output is `x` (fully differentiable); when `w=1` the output is `x.detach()` (no gradient). With `w = selfcond_mask`, selfcond examples have $w=1$ and get the detached version.

## Downstream effect

With detachment: gradients from selfcond examples flow only through the **model weights** (the quantities we want to train), not through $\gamma(t)$, $x_{\text{embed}}$, $\alpha_1$, $\sigma_1$.

Without detachment: the noise schedule can change by more than a percent in the first 1000 training steps, which is always a bad sign for a pretrained model being fine-tuned.

## Implementation in our code

Must be done **before** constructing $z_t$, because $z_t = \alpha_t \cdot x_{\text{embed}} + \sigma_t \cdot \epsilon$ would otherwise carry ungated gradients into $\gamma$ via $\alpha_t$ and $\sigma_t$.

See [[bug-selfcond-detachment]] for the full restructuring we did to fix this.

## Related

- [[vdm-loss]] — where these terms enter.
- [[bug-selfcond-detachment]] — the bug we hit.
- [[bug-lerp-dtype]] — followup issue (dtype mismatch in `torch.lerp`).
