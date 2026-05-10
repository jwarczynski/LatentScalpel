---
tags: [concept, diffusion]
related: [[plaid-1b]] [[genie]] [[vdm-loss]]
---

# Diffusion language models

Text models that generate by iterative denoising of a continuous latent. Our focus: **Plaid** (Gulrajani & Hashimoto, 2023) and **GENIE** as baseline.

## The setup

- Tokens → discrete space.
- Training data: token IDs → embedded to `embed_dim`-dim continuous vectors via a learned embedding matrix (constrained to unit row norm).
- Diffusion happens **in the continuous embedding space**, not over tokens directly.
- At generation time, the denoised embeddings are mapped back to tokens via argmax over logits.

For Plaid 1B:
- `embed_dim = 16` (tiny compared to model dim=2048).
- Model body operates at dim=2048, projecting in from 16 at input and out to 2048+2·16 (for "bias" columns) at output.

## Forward process (VDM-style)

$z_t = \alpha_t \cdot x_{\text{embed}} + \sigma_t \cdot \epsilon$

where $\alpha_t = \sqrt{\text{sigmoid}(-\gamma_t)}$, $\sigma_t = \sqrt{\text{sigmoid}(\gamma_t)}$. The noise schedule $\gamma(t)$ is a small MLP learned jointly. Bounds are learnable too: $\gamma_0 = -3$, $\gamma_1 = 6$.

## Reverse process

Given $z_t$, predict $x_{\text{reconst}}$ (the denoised embedding). From this and $z_t$, derive $\epsilon_{\text{pred}}$. Then:

- **Stochastic (default)**: compute transition $z_{t-\Delta t}$ using closed-form VDM formula (Appendix A.4 eqn 33).
- **DDIM**: deterministic variant.

## Self-conditioning

During training, with probability ~0.25, run the model twice: first pass (no_grad) produces `x_reconst_sc`, which is fed as `x_selfcond` to the main pass. This lets the model use its own prediction as context for the next iteration — critical for continuous diffusion to work well.

Gradient subtlety: for selfcond examples, $\gamma$, $x_{\text{embed}}$, and $\sigma_1$/$\alpha_1$ must be detached from the computation graph so the noise schedule doesn't receive conflicting gradients. See [[selfcond-detachment]] and [[bug-selfcond-detachment]].

## Loss: Variational Lower Bound

Three components:

$$L = L_{\text{reconst}} + L_{\text{diffusion}} + L_{\text{prior}}$$

- **Reconstruction (at $t=0$)**: cross-entropy between model logits and ground-truth token IDs, on a subset (`reconst_bs`) of the batch.
- **Diffusion (at $t>0$)**: $-0.5 \cdot \text{SNR}'(t) \cdot \text{MSE}(x_{\text{embed}}, x_{\text{reconst}})$. SNR = exp(-γ), SNR' < 0, so `-0.5 * SNR' > 0` → loss is positive.
- **Prior**: KL($\alpha_1 \cdot x_{\text{embed}}, \sigma_1 \|\| 0, 1$). Tiny in magnitude.

See [[vdm-loss]] for full derivation.

## Output layer quirk

The final linear combines three sources:

```python
W = torch.cat([
    output_linear.weight.T,       # standard readout
    embedding_matrix.T,           # direct embedding dot-product
    embedding_matrix.T.detach()   # detached copy for selfcond-masked examples
], dim=0)
```

This gives the model flexibility to use a trained readout OR the frozen embedding matrix directly. The last component is used when `selfcond_mask=1` (selfcond applied).

## Why continuous, not discrete?

Plaid authors argue continuous diffusion is better-suited to likelihood-based training than discrete. Quote from paper: "Continuous diffusion enables the use of the variational lower bound for principled likelihood estimation."

Practical advantage: the model can output gradients in a well-defined space. Downside: the embedding matrix has to be sharp enough that argmax over logits recovers the right token.

## Comparison

| Property | Plaid | GENIE |
|---|---|---|
| Denoising space | 16-dim continuous | BERT-style hidden state |
| Timesteps | 4096 continuous (sampled) | 2000 discrete |
| Noise schedule | Learned sqrt-based | Fixed sqrt |
| Base architecture | Decoder-only transformer | Encoder-decoder BERT |
| Parameterization | muP | Standard |
| Pretraining | OpenWebText2 | XSum (ours) or large LM corpus |

## Further reading

- Gulrajani & Hashimoto 2023, *Likelihood-Based Diffusion Language Models* — the Plaid paper.
- Kingma et al. 2021, *Variational Diffusion Models* — VDM loss.
- Ho et al. 2020, *Denoising Diffusion Probabilistic Models* — original diffusion setup.
- Li et al. 2022, *Diffusion-LM* — earlier discrete diffusion for text.
