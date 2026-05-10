---
tags: [experiment, finetune, plaid]
status: completed
date: 2026-04-12
result: inferior-to-v3b
related: [[exp-conditional-v3b]] [[decision-seq-len-256]]
---

# Exp: conditional v3a (seq_len=1024)

Sibling of [[exp-conditional-v3b]] with seq_len=1024 instead of 256.

## Config

`configs/plaid_xsum_conditional_8gpu_v3a.yaml`

Differences from v3b:
- `seq_len: 1024` (vs 256)
- `max_article_tokens: 900` (not relevant — this is for token-guidance inference, not training)

Same: lr=1e-6, cosine, weight_decay=4e-5, betas=(0.9, 0.99), 100 epochs, conditional mode, batch_size=4, 8 GPUs.

## Slurm

Job 2517797, submitted 2026-04-12 with v3b and v3c.

## Wandb

Run name: `cond-1024-lr1e6-cosine-100ep`

## Outcome

Subjectively worse than v3b:
- Gradient signal sparse (only ~6% of positions contribute to loss: the ~64 summary tokens out of 1024).
- Training was slower per-example-seen.
- Final summaries comparable to v2 — not clearly better than the shorter seq_len version.

## Why we kept it

Useful data point for [[decision-seq-len-256]]. Longer seq_len might help for different tasks where summaries or continuations are longer, but for XSum's short summaries it's wasted compute.

## Result

Superseded by [[exp-conditional-v3b]].
