---
tags: [concept, diffusion, training]
related: [[diffusion-language-models]] [[selfcond-detachment]]
---

# VDM Loss (Variational Diffusion Model)

The training objective used by Plaid. Combines three terms: reconstruction, diffusion, and prior.

## Setup

- Continuous time $t \in [0, 1]$.
- Noise schedule $\gamma(t)$: learned, monotonically increasing from $\gamma_0 \approx -3$ to $\gamma_1 = 6$.
- $\alpha_t = \sqrt{\text{sigmoid}(-\gamma_t)}$, $\sigma_t = \sqrt{\text{sigmoid}(\gamma_t)}$. Satisfies $\alpha_t^2 + \sigma_t^2 = 1$.
- SNR = $\alpha_t^2 / \sigma_t^2 = \exp(-\gamma_t)$.
- Forward: $z_t = \alpha_t \cdot x_{\text{embed}} + \sigma_t \cdot \epsilon$ where $\epsilon \sim N(0, I)$.

## Three loss components

### 1. Reconstruction loss (at $t = 0$)

Only applied to a subset of examples (`reconst_bs`). Cross-entropy between the model's logits and the ground-truth token IDs. This is the standard LM loss — teaches the model to decode embeddings back to tokens.

```python
reconst_loss = F.cross_entropy(logits[:reconst_bs], token_ids[:reconst_bs])
```

### 2. Diffusion loss (at $t > 0$)

Applied to the remaining examples. MSE between ground-truth embedding and the model's reconstruction, weighted by the SNR derivative:

$$L_{\text{diff}} = -\frac{1}{2} \cdot \text{SNR}'(t) \cdot \|x_{\text{embed}} - x_{\text{reconst}}\|^2$$

where $\text{SNR}'(t) = -\exp(-\gamma_t) \cdot \gamma'(t)$. Since $\gamma$ is increasing in $t$, $\gamma'(t) > 0$, so $\text{SNR}' < 0$, and $-0.5 \cdot \text{SNR}' > 0$.

The MSE reduction is **mean over sequence positions, sum over embed_dim** — sensitive to choice! Original `train.py`:

```python
diffusion_loss = (x_embed - x_reconst).pow(2)  # (B, S, E)
diffusion_loss = diffusion_loss.mean(dim=1).double().sum(dim=1)  # mean over seq, sum over E
diffusion_loss = -0.5 * (snr_prime * diffusion_loss)  # (B,)
```

For our conditional/template modes with masked loss positions (only summary tokens contribute), we normalize by valid positions instead of `.mean(dim=1)`.

### 3. Prior loss (at $t = 1$)

KL between $q(z_1 | x)$ and the prior $N(0, I)$:

$$L_{\text{prior}} = D_{\text{KL}}(N(\alpha_1 \cdot x_{\text{embed}}, \sigma_1^2) \| N(0, 1))$$

$$= -\log(\sigma_1) + \frac{\sigma_1^2 + (\alpha_1 \cdot x_{\text{embed}})^2}{2} - \frac{1}{2}$$

Reduction: sum over embed_dim, mean over batch and sequence.

## Total loss

$$L = L_{\text{reconst}} + L_{\text{diff}} + L_{\text{prior}}$$

All three contribute in every training step — some examples get reconst, some get diffusion, and prior applies to all.

## Low-discrepancy timestep sampling

For the diffusion loss, the batch is split: first `reconst_bs` examples get $t = 0$, remaining get timesteps evenly distributed in $(0, 1]$ via stratified sampling:

```python
n_diff = B - reconst_bs
t[reconst_bs:] = (torch.arange(n_diff) + torch.rand(1)) / n_diff
```

This gives lower variance than uniform random sampling.

## Gradient subtleties

- $\gamma(t)$ is learned, so timestep sampling contributes gradients via $\gamma'(t)$ to the noise schedule. This is why `t.requires_grad_(True)` and `torch.autograd.grad(gamma.sum(), t, create_graph=True)` to get $\gamma'$.
- Selfcond examples **must detach** $\gamma$, $\gamma'$, $x_{\text{embed}}$, $\alpha_1$, $\sigma_1$ — see [[selfcond-detachment]] and [[bug-selfcond-detachment]].
- `reconst_bs` examples have $t = 0$, so the diffusion loss contribution is zero. But the reconstruction loss uses the same forward pass output.

## Our implementation

`geniesae/plaid_xsum_training.py::_compute_vlb_loss`. Returns `{"loss", "reconst_loss", "diffusion_loss", "prior_loss"}` — all logged to wandb.

## Common bugs

- Missing selfcond detachment → unstable noise schedule → shaky training. See [[bug-selfcond-detachment]].
- Prior loss reduction mismatch → subtle loss curve differences. See [[bug-prior-loss-reduction]].
- Conditional mode + wrong loss mask → learning on padding or article tokens. Always verify the mask for the first few batches.
