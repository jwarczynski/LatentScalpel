---
tags: [concept, prompting, xsum]
related: [[exp-eval-v3b-inference]] [[decision-prompt-choice]]
---

# Prompt format comparison

When using the pretrained Plaid model for zero-shot summarization, the prompt format matters a lot. Plaid was trained unconditionally on OpenWebText2, so we rely on the model seeing natural-text patterns that imply summarization.

## Formats tried

| Format | Used in | Result |
|---|---|---|
| `article` + prefix guidance (no cue) | Early tests | Model continues the article, no summarization. |
| `"<article>\n\nTL;DR:"` | Default pre-v3 | Works reasonably well. |
| `"<article>\n\nSummary:"` | Not tested systematically | Similar behavior expected. |
| `"<article>\n\nThe article can be summarized as follows:"` | v3a-v3c experiments | **Poor** — model interprets as a web UI cue ("click here to view"), produces empty or near-empty outputs. |
| `"ARTICLE: <article>\n\nSUMMARY: <summary>"` | Template fine-tuning (v3c) | Depends on training setup — with fine-tuning works because model learned the pattern. |

## Observations

**TL;DR works because**: OpenWebText2 contains genuine Reddit-style TL;DR markers. The model learned this as a semantic signal for "short summary follows."

**"The article can be summarized as follows" fails because**: This phrasing is rare in OpenWebText2 and tends to appear in web UI / academic contexts where it's often followed by boilerplate or formatting, not a natural summary. With it the model outputs "(empty)" or `"1/3"`-style placeholder text.

**`"Summary:"`**: Probably works similarly to TL;DR. Not systematically tested because TL;DR already worked and we didn't want to introduce yet another variable.

## For token-guidance inference

`PlaidTokenGuidanceConfig.apply()` uses:

```python
prompt = article_text + self.prompt_suffix
prompt_ids = tokenizer.encode(prompt).ids
# Truncate if too long
if len(prompt_ids) > self.max_article_tokens:
    prompt_ids = prompt_ids[: self.max_article_tokens]
```

With `prompt_suffix = "\n\nTL;DR:"` by default (changed from `"\n\nThe article can be summarized as follows:"` after seeing the failure pattern).

## For fine-tuning

Template mode (v3c) uses:

```python
# XSumDataset with format_mode="template"
prefix_ids = tokenizer.encode("ARTICLE:").ids
suffix_ids = tokenizer.encode("\n\nSUMMARY:").ids
sequence = prefix_ids + article_ids + suffix_ids + summary_ids
boundary_idx = len(prefix_ids) + len(article_ids) + len(suffix_ids)
```

Loss is on summary tokens only. At inference, tokenize `"ARTICLE: <article>\n\nSUMMARY:"` and use guidance.

Conditional mode (v3a/b) uses `[article | SEP | summary]` where SEP is token 0. No natural-language cues — the model is trained to treat SEP as the boundary and to fill in summary tokens after it. Works because fine-tuning teaches the model the format explicitly.

## Key insight

For a pretrained model, **pretend you're an author writing prose that would be followed by a summary**. Markers like `TL;DR:`, `Summary:`, `In short:` work because these phrases are ubiquitous in training data. Academic-style phrases less so.

For a fine-tuned model, any consistent format works as long as the model was trained on it.

## Related

- [[conditional-vs-template-training]] — the two training-time format choices.
- [[decision-prompt-choice]] — why we landed on TL;DR.
