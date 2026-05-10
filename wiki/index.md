---
tags: [meta, index]
last_updated: 2026-05-03
---

# Index

This is the catalog of all wiki pages. Start here when exploring. See [[schema]] for conventions and [[log]] for a chronological record.

## Models

- [[plaid-1b]] — Pretrained Plaid 1B diffusion LM from Gulrajani & Hashimoto (2023)
- [[plaid-finetuned-v3b]] — Current best fine-tuned checkpoint: conditional, seq_len=256, lr=1e-6, 100 epochs
- [[plaid-finetune-history]] — v1, v2, v3a/b/c and resume runs with outcomes
- [[genie]] — Earlier diffusion LM; baseline for SAE pipeline
- [[t5]] — T5 used as additional baseline in some analyses

## Pipeline

- [[pipeline-overview]] — End-to-end SAE analysis pipeline (collect → train → top → interp → intervene)
- [[activation-collection]] — Running model on dataset, saving per-layer activations
- [[sae-training]] — Top-K SAE training on collected activations
- [[top-examples]] — Mining dataset examples that maximally activate each feature
- [[trajectory-collection]] — Full denoising trajectory with SAE activations recorded
- [[temporal-classification]] — Classifying features by when they activate (early/late/midpoint)
- [[interpretation]] — LLM-as-judge explanations and interpretability scores
- [[intervention]] — Causal tests: amplify or suppress features during generation
- [[evaluation]] — ROUGE/BLEU/BERTScore on generated summaries

## Concepts

- [[diffusion-language-models]] — Continuous-latent diffusion over token embeddings
- [[mup-scaling]] — Maximal update parameterization (and our bug)
- [[rotary-embeddings]] — RoPE in Plaid; why values must not be rotated
- [[vdm-loss]] — Variational Diffusion Model loss (VLB) derivation
- [[topk-sae]] — Top-K sparse autoencoder architecture and training
- [[prompt-format-comparison]] — TL;DR vs SUMMARY vs "article can be summarized"
- [[conditional-vs-template-training]] — Training modes for XSum fine-tuning
- [[selfcond-detachment]] — Why gamma/x_embed must be detached for selfcond examples

## Bugs

- [[bug-mup-scaling]] — output_mult was 0.125 instead of 1.0 → logits 8x too small
- [[bug-rotary-values]] — v was being rotated instead of identity pass-through
- [[bug-sampler-gamma-zero]] — Final decode used gamma=0 instead of gamma_t
- [[bug-selfcond-detachment]] — Training: gamma/x_embed not detached for selfcond
- [[bug-lerp-dtype]] — torch.lerp required matching dtypes
- [[bug-sae-stuck-loss]] — SAE loss flat from step 0; fixed by normalization + no k-warmup
- [[bug-prior-loss-reduction]] — Reduction mismatch vs original plaid
- [[bugs-summary]] — One-liner overview of every bug we found

## Experiments

- [[exp-conditional-v1]] — First conditional fine-tune runs (before bugs fixed)
- [[exp-conditional-v2]] — After rotary+mup fixes; still shaky loss
- [[exp-template-v2]] — Template mode counterpart of v2
- [[exp-conditional-v3a]] — seq_len=1024, lr=1e-6, cosine, 100 ep
- [[exp-conditional-v3b]] — **Best**: seq_len=256, lr=1e-6, cosine, 100 ep
- [[exp-template-v3c]] — Template mode with v3 hyperparams
- [[exp-resume-1e7]] — Resume v2 checkpoint with lr=1e-7 linear tail
- [[exp-sae-pretrained-plaid]] — SAEs on pretrained Plaid (OpenWebText activations)
- [[exp-sae-finetuned-v3b-v1]] — SAEs on fine-tuned v3b (val-only split, poor features)
- [[exp-sae-finetuned-v3b-v2]] — SAEs with proper train/val/test split
- [[exp-sae-finetuned-v3b-v3]] — SAEs with normalization + better config
- [[exp-eval-v3b-inference]] — 100 XSum dev+test generations from v3b
- [[exp-original-code-comparison]] — Standalone script using Plaid's original code
- [[midpoint-features]] — GENIE midpoint-feature intervention experiments (legacy)

## Infrastructure

- [[cluster-ares]] — Athena/Ares cluster: SSH, partitions, time limits, disk
- [[exca-workflows]] — Exca submit patterns, version bumping, cache invalidation
- [[git-workflow]] — Repo, remote sync, commit attribution
- [[wandb-projects]] — Wandb projects list with run IDs
- [[disk-usage]] — Where large artifacts live, sizes, retention
- [[uv-environment]] — uv package manager, lock file strategy

## Decisions

- [[decision-split-policy]] — Train/val/test split for SAE (matching GENIE)
- [[decision-sae-hyperparams]] — expansion=8, k=64, normalize=True for Plaid
- [[decision-prompt-choice]] — Why we went back to TL;DR
- [[decision-seq-len-256]] — Why short seq_len works better than 1024
- [[decision-lr-1e-6]] — Conservative lr for fine-tuning 1.3B model
- [[decision-no-mup-adam]] — Using plain AdamW instead of MuAdam
- [[decision-data-format]] — Local GLGE XSum .src files vs HuggingFace

## Related projects

- [[related-saescope]] — Interactive SAE feature explorer dashboard
- [[related-shortcutfm]] — Sister flow-matching project; wandb table pattern source
- [[related-plaid-original]] — Original igul222/plaid repo, used for validation

## Meta

- [[schema]] — Conventions, page types, workflows
- [[log]] — Chronological event log
- [[roadmap]] — What's next

## Quick links

- Repo: [jwarczynski/LatentScalpel](https://github.com/jwarczynski/LatentScalpel)
- Wandb org: [jedrasowicz](https://wandb.ai/jedrasowicz)
- Cluster scratch: `/net/tscratch/people/plgjentker/GenieSAE`
