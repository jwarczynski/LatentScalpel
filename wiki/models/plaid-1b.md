---
tags: [model, plaid, pretrained]
date: 2026-03-14
related: [[diffusion-language-models]] [[mup-scaling]] [[rotary-embeddings]]
---

# Plaid 1B — Pretrained

Gulrajani & Hashimoto's continuous-latent diffusion LM, trained on OpenWebText2.

- Paper: [Likelihood-Based Diffusion Language Models](https://arxiv.org/abs/2305.18619)
- Code: [igul222/plaid](https://github.com/igul222/plaid)

## Architecture

| Property | Value |
|---|---|
| dim (model) | 2048 |
| n_blocks | 24 |
| n_heads | 32 |
| head_dim | 64 |
| embed_dim (continuous latent) | 16 |
| vocab_size | 32768 |
| parameters | 1.28 B |
| seq_len (training) | 1024 (Plaid 1B) |
| tokenizer | OpenWebText2 BPE 32k |
| muP base_dim | 256 |

### Notable features

- **Continuous latent diffusion**: tokens are embedded into a 16-dim continuous space; the model denoises that space, not tokens directly. See [[diffusion-language-models]].
- **muP parameterization**: `output_linear` is a `mup.MuReadout`. Output scaling is `output_mult / width_mult = 1.0 / 8.0 = 0.125`. See [[mup-scaling]] and [[bug-mup-scaling]].
- **Rotary embeddings** (RoPE) on q/k only — v is identity. See [[rotary-embeddings]] and [[bug-rotary-values]].
- **Self-conditioning**: standard practice for continuous diffusion; ~25% of training examples use a second forward pass' output as input to the main forward pass.
- **Noise schedule**: small MLP mapping `t ∈ [0,1]` to normalized gamma; trained jointly. Combined with learnable `gamma_0=-3`, `gamma_1=6` bounds.

### Key constants

- `gamma_0 = -3.0` (low-noise side: `sigmoid(-γ) ≈ 0.95`)
- `gamma_1 = 6.0` (high-noise side: `sigmoid(-γ) ≈ 0.0025`)
- `sampling_timesteps` default: 4096 (original), 256 (our default for inference)
- `score_temp = 0.9` (standard)

## Weights

- Download: release tag `v1.0.0` on the original repo.
- Local path (remote): `$SCRATCH/GenieSAE/models/plaid/plaid1b_weights/`
- Files: `model.pt`, `embedding_matrix.pt`, `noise_schedule.pt`, `gamma_bounds.pt`.

## Reimplementation in this repo

`geniesae/plaid_model.py` replaces flash-attn and apex with pure PyTorch equivalents so we can load the pretrained weights without compiling CUDA kernels on Athena (where flash-attn builds failed):

- `apex.normalization.FusedRMSNorm` → custom `RMSNorm`.
- `flash_attn.flash_attn_interface.flash_attn_unpadded_qkvpacked_func` → `F.scaled_dot_product_attention`.
- `flash_attn.ops.fused_dense.FusedMLP` → simple 2-layer MLP with GELU.
- `flash_attn.layers.rotary.apply_rotary_emb_qkv_` → `apply_rotary_pos_emb` (torchscript).

Standalone script `scripts/plaid_original_xsum_generate.py` uses the same replacement strategy but keeps all the original Plaid structure intact — this is our reference implementation used for cross-validation. See [[related-plaid-original]].

## Bugs found in our reimplementation

See [[bugs-summary]]. The critical ones for the model forward pass:
- [[bug-mup-scaling]] — output_mult was 0.125 instead of 1.0.
- [[bug-rotary-values]] — v was being rotated.
- [[bug-sampler-gamma-zero]] — final decode used gamma=0.

## Inference modes

The original sample.py supports several guidance modes via `guidance_tokens`:

- **Unconditional**: empty guidance list, just argmax over generated logits.
- **Prefix completion**: guide each prefix token at its position with weight ≈ 2.0.
- **Infilling**: guide prefix at positions `[0..bi]` and suffix at `[bi+infill..end]`; middle is free.
- **Any-position** (`position='any'`): token must appear somewhere.
- **All-positions** (`position='all'`): token encouraged at every position.
- **Negation** (`complement=True`): uses `log(1 - p(y|x))` to suppress a token.

We use prefix-completion for XSum summarization: tokenize `"<article>\n\nTL;DR:"` and guide every token. See [[prompt-format-comparison]].

## Evaluation benchmarks (from paper)

Plaid 1B reports ppl on wikitext-103 around 22 (contest competitive with GPT-2 774M at the time). We haven't reproduced these numbers locally.

## Known quirks

- **Heavy use of float64** for numerical stability in the noise schedule and VDM loss. The model itself runs in bf16-mixed, but gamma/alpha/sigma computations are always float64.
- **`torch.set_default_dtype(torch.float64)` must be set globally** when loading the noise schedule in the standalone script — the `NoiseSchedule` module constructs scalar tensors via `torch.tensor([0.])` which inherit the default dtype.
- **`set_default_device('cuda')`** is similarly assumed by much of the original code.
