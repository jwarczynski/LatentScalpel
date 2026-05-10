---
tags: [decision, finetune, hyperparams]
status: adopted
date: 2026-03-29
related: [[exp-conditional-v2]] [[exp-conditional-v3b]]
---

# Decision: learning rate 1e-6 for fine-tuning

Use `lr = 1e-6` with cosine schedule for Plaid XSum fine-tuning.

## Why this low

v2 runs at `lr = 1e-5` with linear schedule showed oscillating loss (drops ~500 steps, then oscillates between 2 and 4). A classic symptom of lr at the edge of stability.

For reference:
- Plaid paper: `lr = 1.4e-3` with MuAdam for pretraining from scratch.
- Typical LLM fine-tuning of 1B+ models: 1e-6 to 5e-6.
- Our v1: `lr = 1e-4` (way too high for fine-tuning, though the muP/rotary bugs masked it).

## Why cosine over linear

Linear decays to exactly 0 at the end, which makes resuming painful — the scheduler says lr=0 on restart unless you override (hence the `resume_lr` field we added).

Cosine stays non-zero for most of training, bottoms out gently at the end. Easier to extend if val loss is still improving.

## Downsides

With a very conservative lr, training is slow — v3b needed 100 epochs to make meaningful progress. v2 at lr=1e-5 showed fast progress in first 500 steps but then destabilized. There's a sweet spot we haven't fully characterized; 1e-6 is safe but possibly too cautious.

Future experiments could try:
- `lr = 3e-6` with cosine, 60 epochs — might reach similar quality faster.
- `lr = 1e-5` with proper warmup (2500 steps instead of 500) — might avoid early oscillation.

## Related

- [[decision-lr-1e-6]] — this page, describing lr choice.
- [[exp-resume-1e7]] — an attempted resume at even lower lr (1e-7) that was abandoned.
- [[bug-selfcond-detachment]] — another factor contributing to v2 oscillation.
