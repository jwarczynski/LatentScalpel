---
tags: [concept, training, xsum]
related: [[plaid-finetune-history]] [[prompt-format-comparison]]
---

# Conditional vs template training mode

Two ways we've structured fine-tuning on XSum. Both supported by `PlaidXSumTrainingModule`.

## Conditional mode (`training_mode="conditional"`)

Format: `[article | SEP | summary | PAD...]` where SEP is token 0.

During training:
1. Entire sequence is embedded and noised normally.
2. **Article prefix is replaced with clean embeddings** (no noise) — `z_t[b, :bi] = x_embed[b, :bi]`.
3. Loss is computed only on positions `[bi .. real_len]` (summary tokens) via `loss_mask`.

The model learns to denoise summary tokens given clean article context. This is essentially "inpainting with fixed prefix."

At inference: use `InpaintingSampler(prefix_mode="clean")`. The article prefix is held clean throughout the reverse diffusion; only the summary part is denoised.

**Pros:**
- Model gets strong conditioning signal — article is fixed, summary has to be consistent with it.
- Simpler inference — no guidance weight to tune.

**Cons:**
- Diverges from how the model was pretrained (unconditional continuous diffusion).
- Can fail on long articles if the summary positions are limited.

This is the mode used by **v3b (our best)**.

## Template mode (`training_mode="template"`)

Format: `"ARTICLE: <article_text>\n\nSUMMARY: <summary_text>"` as a single sequence, fully noised and denoised. Loss is on summary tokens only.

During training:
1. Construct the natural-text template via `XSumDataset(format_mode="template")`.
2. `boundary_idx` marks the first summary token (after `"\n\nSUMMARY:"`).
3. Normal noising of the entire sequence.
4. `loss_mask` only counts summary positions.

At inference: tokenize `"ARTICLE: <article>\n\nSUMMARY:"` and use it as a guidance prefix (token guidance) or inpainting prefix.

**Pros:**
- Matches the pretrained model's "continue this text" paradigm.
- Natural-language format — no special tokens needed.

**Cons:**
- Weaker conditioning — the article isn't "held fixed" during training, so the model has to learn from scratch that summary ≈ article.
- Requires the model to handle the full sequence (article + summary) under diffusion noise simultaneously.

v3c is the template-mode run at the same hyperparameters as v3b. Results were subjectively worse than v3b (conditional).

## Loss mask construction

Both modes share the loss mask builder:

```python
if self.training_mode in ("conditional", "template") and boundary_idx is not None:
    loss_mask = torch.zeros(B, S, device=device, dtype=torch.float64)
    for b in range(B):
        bi = boundary_idx[b].item()
        loss_mask[b, bi:] = attention_mask[b, bi:].double()
```

The difference is only in whether `z_t[b, :bi]` gets replaced with clean embeddings (conditional) or left noised (template).

## When to pick which

- **Conditional** — default for supervised tasks where you have aligned (input, output) pairs and want strong conditioning. Our current choice for XSum.
- **Template** — when matching pretraining distribution matters more than conditioning strength. Good for tasks where the "article" and "summary" are both just free text and the boundary is fuzzy.
- **Unconditional** — if you want to purely fine-tune on summary text without conditioning. Not used here.

## Config

```yaml
training_mode: "conditional"   # or "template" or "unconditional"
```

Propagated from `PlaidXSumConfig` to `PlaidXSumTrainingModule`.

Data module (`XSumDataModule`) picks `format_mode` based on training mode:

```python
format_mode = "template" if self.training_mode == "template" else "sep"
data_module = XSumDataModule(..., format_mode=format_mode)
```

## Related

- [[plaid-finetune-history]] — runs comparing these modes.
- [[prompt-format-comparison]] — related discussion for inference-time prompts.
