---
tags: [decision, finetune, plaid]
status: adopted
date: 2026-04-12
related: [[exp-conditional-v3a]] [[exp-conditional-v3b]]
---

# Decision: seq_len=256 for fine-tuning

Use seq_len=256 instead of 1024 for XSum fine-tuning.

## Why

XSum summaries are ≤64 tokens by design (dataset's `max_summary_len`). With seq_len=1024 and article prefix held clean, we're wasting ~900 positions on article context. That's:

1. **Wasted compute**: bf16 attention is O(n²) — the difference between 256² and 1024² is 16×.
2. **Sparse gradient signal**: only the summary positions contribute to loss (~6% of 1024). With shorter seq_len, a larger fraction of each forward pass contributes to the gradient.
3. **Memory**: batch_size=4 at seq_len=1024 OOMs on A100 40GB at certain points in training. batch_size=4 at seq_len=256 is comfortable.

## Empirical evidence

Ran v3a (seq_len=1024) and v3b (seq_len=256) in parallel with otherwise identical hyperparams. v3b produced subjectively better summaries and was the run we continued with.

## Tradeoffs

- **Article truncation**: XSum articles average ~400 words ≈ ~500 BPE tokens, so many get truncated. We lose the long-distance context. For summarization of news leads (which is basically what XSum is), the first ~250 tokens usually contain the main point anyway.
- **Generalization to other datasets**: if we ever move to CNN/DailyMail (longer articles, longer summaries), we'd need to revisit this. For XSum, 256 is fine.

## Downstream

- SAE analysis uses the same seq_len=256. Activations were collected at this length.
- Inference (`scripts/plaid_xsum_inference.py`) uses seq_len=256 for the v3b checkpoint.
