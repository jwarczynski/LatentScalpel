---
tags: [experiment, validation, plaid]
status: completed
date: 2026-03-14
related: [[related-plaid-original]] [[bugs-summary]]
---

# Exp: Original code comparison

Cross-validated our Plaid reimplementation against the original `igul222/plaid` source to isolate bugs.

## Motivation

After multiple training runs produced gibberish, we suspected our reimplementation had bugs. To pin them down, we needed a reference implementation that:
1. Uses the same weights (`plaid1b_weights/`).
2. Produces known-good output (readable text).
3. Runs on our cluster without the `flash-attn` / `apex` install headaches.

## Approach

Cloned `igul222/plaid` into `$SCRATCH/plaid`. It depends on flash-attn and apex which are hard to build on Athena. Solution: write a standalone `scripts/plaid_original_xsum_generate.py` that:

- Replaces `apex.FusedRMSNorm` with pure PyTorch RMSNorm.
- Replaces `flash_attn.flash_attn_interface` with `F.scaled_dot_product_attention`.
- Replaces `flash_attn.ops.fused_dense.FusedMLP` with a plain 2-layer MLP.
- Keeps everything else — model structure, noise schedule, sampling, loss — verbatim.

This gave us a reference implementation that runs on bare PyTorch, loads the same weights, and produces coherent output.

## Methodology

For each suspected bug, we:

1. Ran both implementations on the same seed.
2. Diffed intermediate tensor values at specific forward-pass points.
3. Isolated the exact layer/op where outputs diverged.
4. Read the original code to understand the correct behavior.
5. Fixed our reimplementation to match.

## Bugs found

Full list in [[bugs-summary]]. The three critical ones for Plaid itself:

- [[bug-mup-scaling]] — our `output_mult` was 0.125, original's is 1.0.
- [[bug-rotary-values]] — our rotary cos/sin was the wrong shape; values were rotated.
- [[bug-sampler-gamma-zero]] — our samplers used gamma=0 at the final decode.

## Output

`scripts/plaid_original_xsum_generate.py` is kept in the repo as a reference implementation. Future cross-validation should diff against its output.

Also produced side-by-side generation logs that confirmed our fine-tuned model produced the same sample quality as the original pretrained model when fed the same prompts.

## Lesson

**Always have a reference implementation you trust.** Without this script we would have spent more time debugging training loops instead of isolating the specific forward-pass bugs.
