---
tags: [decision, prompting, xsum]
status: adopted
date: 2026-04-14
related: [[prompt-format-comparison]]
---

# Decision: TL;DR for inference prompting

For zero-shot summarization with pretrained Plaid, use `"<article>\n\nTL;DR:"` as the prefix for token guidance.

## Choices considered

See [[prompt-format-comparison]] for the full table. Summary:

| Candidate | Works? |
|---|---|
| `"<article>\n\nTL;DR:"` | ✅ best zero-shot option |
| `"<article>\n\nSummary:"` | ✅ probably similar (not tested rigorously) |
| `"<article>\n\nThe article can be summarized as follows:"` | ❌ model interprets as web UI cue |
| No suffix, just article | ❌ model continues the article |

## Why TL;DR

OpenWebText2 (Plaid's pretraining corpus) contains Reddit-style TL;DR markers as a genuine semantic signal. The model learned this pattern and produces short, informal summaries when it sees it.

## For fine-tuned model

For v3b (conditional fine-tuning), prompting is implicit — the model learned `[article | SEP | summary]` as the format. At inference, we use `InpaintingSampler(prefix_mode="clean")` with the article + SEP as the fixed prefix.

For v3c (template fine-tuning), use `"ARTICLE: <article>\n\nSUMMARY:"` — matches the training format.

## What we'll need if we ever fine-tune with TL;DR format

Would match pretraining-time natural text better than our current `[article | SEP | summary]` format. Unclear if benefit is worth the retraining cost. Not currently planned.
