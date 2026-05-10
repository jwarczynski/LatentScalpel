---
tags: [bug, plaid, training, selfcond]
status: fixed
date: 2026-03-24
severity: critical
fix_commits: [759231c, 0042bcc]
related: [[selfcond-detachment]] [[vdm-loss]] [[plaid-finetune-history]]
---

# Bug: selfcond detachment missing in VLB loss

## Symptom

Training loss oscillated wildly — dropped enormously in the first ~500 steps, then oscillated between 2 and 4 without further decrease. Samples from the trained model were low-quality summaries with hallucinated details even after all inference-path bugs were fixed.

## Root cause

In `geniesae/plaid_xsum_training.py::_compute_vlb_loss`, self-conditioning examples did NOT detach `gamma`, `gamma_prime`, `x_embed`, or the `gamma_1`-derived quantities (`alpha_1`/`sigma_1`).

The original Plaid `train.py` does:

```python
# Detach noise-schedule gradients for selfcond examples
gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)
x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:, None, None])

# And for the prior loss
alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:, None, None]
sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:, None, None]
```

Why this matters: self-conditioned examples run the model twice — once with `torch.no_grad()` to produce `x_selfcond`, and again with gradients. The selfcond forward pass should not push gradients into the noise schedule, the embedding matrix, or the gamma_bounds. Otherwise those modules receive **conflicting gradient signals**:

1. Non-selfcond examples: gradient through `gamma(t)` telling the schedule what it should be for the standard VLB loss.
2. Selfcond examples: gradient through `gamma(t)` telling the schedule what makes the second forward pass' reconstruction better.

These two signals can point in opposite directions. Result: the noise schedule drifts, the embedding matrix drifts, and training destabilizes.

## Fix

Restructure `_compute_vlb_loss` so selfcond detachment happens **before** constructing `z_t`:

```python
# 1. Compute gamma and gamma_prime normally (through autograd)
gamma_t = ...
gamma_prime = torch.autograd.grad(gamma_t.sum(), t, create_graph=True)[0]

# 2. Build loss_mask for conditional/template modes

# 3. Selfcond detachment BEFORE constructing z_t
selfcond_mask = torch.zeros(B, device=device)
sc_mask = torch.rand(B, device=device) < self.self_cond_prob
if sc_mask.any():
    sc_double = sc_mask.double()
    sc_float = sc_mask.float()
    gamma_t = torch.lerp(gamma_t, gamma_t.detach(), sc_double)
    gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), sc_double)
    x_embed = torch.lerp(x_embed, x_embed.detach(), sc_float[:, None, None])

# 4. Derived quantities use the (possibly detached) gamma_t
alpha = torch.sigmoid(-gamma_t).sqrt()
sigma = torch.sigmoid(gamma_t).sqrt()
snr_prime = -torch.exp(-gamma_t) * gamma_prime

# 5. Build z_t from possibly-detached alpha, sigma, x_embed
noise = torch.randn(B, S, E, device=device, dtype=torch.float32)
z_t = (alpha[:, None, None] * x_embed.double()
       + sigma[:, None, None] * noise.double()).float()

# 6. Prior loss uses detached alpha_1, sigma_1 for selfcond examples
alpha_1_masked = torch.lerp(
    alpha_1.expand(B), alpha_1.detach().expand(B), selfcond_mask.double()
)[:, None, None]
sigma_1_masked = torch.lerp(
    sigma_1.expand(B), sigma_1.detach().expand(B), selfcond_mask.double()
)[:, None, None]
```

Fixed across commits ``759231c`` and ``0042bcc``. See [[bug-lerp-dtype]] for the dtype followup.

## Also fixed at the same time

**Prior loss reduction**: We were normalizing by masked valid counts in a custom way; original does `.sum(dim=2).mean()` (sum over embed_dim, mean over B×S). For conditional/template modes with padding mask, we still divide by valid counts to be correct. See [[bug-prior-loss-reduction]].

## Detection

If the noise schedule `NoiseSchedule.W1`/`W2` drifts away from its pretrained values in early training steps (e.g. >10% relative change in first 1000 steps), suspect missing detachment. Pretrained Plaid has a well-calibrated noise schedule; fine-tuning should barely move it.

```python
# Snapshot pre-training
import copy
ns_initial = copy.deepcopy(module.noise_schedule.state_dict())

# After some training steps
for k, v in module.noise_schedule.state_dict().items():
    delta = (v - ns_initial[k]).abs().max().item()
    if delta > 0.01:
        print(f"WARNING: noise_schedule.{k} changed by {delta}")
```
