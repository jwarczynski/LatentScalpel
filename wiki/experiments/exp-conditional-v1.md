---
tags: [experiment, finetune, plaid, broken]
status: failed
date: 2026-03-22
result: gibberish
related: [[bug-mup-scaling]] [[bug-rotary-values]]
---

# Exp: conditional v1 (broken)

First conditional fine-tune attempts. All suffered from the un-fixed [[bug-mup-scaling]] and [[bug-rotary-values]], producing gibberish output regardless of training config.

## Config snapshot

Various — tried several combinations before realizing the model itself was broken:

| lr | wd | schedule | batch | seq_len | epochs |
|---|---|---|---|---|---|
| 1e-4 | 0.01 | cosine | 4 | 1024 | 10 |
| 1e-4 | 0.01 | cosine | 4 | 1024 | 30 |

## Symptom

Samples during validation produced totally nonsensical text — no real English, no summarization behavior. Initially thought it was a fine-tuning issue but the same gibberish appeared from the **pretrained** model via our standalone inference script → it's the model forward pass, not the training loop.

## Resolution

Fixed the muP scaling and rotary bugs in commit ``d06f5dd``. All v1 checkpoints discarded. Started v2 series with the fixed model code.

## Lesson

Before investing in fine-tuning runs, always verify the pretrained model produces sensible unconditional samples end-to-end. A simple `python scripts/plaid_original_xsum_generate.py --weights_path=... --tokenizer_path=...` catches bugs like these in minutes.

## Related

- [[bug-mup-scaling]], [[bug-rotary-values]] — the two critical bugs.
- [[exp-original-code-comparison]] — methodology used to isolate the bugs.
