---
tags: [decision, training, mup]
status: adopted
related: [[mup-scaling]] [[decision-lr-1e-6]]
---

# Decision: standard AdamW instead of MuAdam for fine-tuning

Original Plaid uses `mup.MuAdam` which applies per-layer lr scaling to preserve the hyperparameter transfer property. For our fine-tuning we use plain `torch.optim.AdamW`.

## Why

1. **Fine-tuning scale is small**. Plaid paper used MuAdam for pretraining at batch=256, lr=1.4e-3. Our fine-tuning uses batch=32, lr=1e-6 — already conservative. MuAdam's per-layer scaling matters less at these scales because all layers are moving slowly.

2. **Integration complexity**. MuAdam requires passing `impl=` to pick the underlying Adam. Our training uses pytorch_lightning's `configure_optimizers()`, and mixing MuAdam with Lightning's expected API is awkward.

3. **Lightning scheduler compatibility**. Our linear/cosine schedules are implemented as `LambdaLR` wrappers around AdamW. Swapping in MuAdam would require re-validating the scheduler behavior.

## What we lose

- The theoretical hyperparameter transfer guarantee (tune on dim=256 proxy, apply to dim=2048 main) doesn't hold.
- Per-layer lr scaling is absent — all params get the same lr.

For us this is fine because we're not tuning hyperparameters via a proxy. We tune directly on the 1.3B model.

## What we keep

- The `MuReadout` scaling (`output_mult / width_mult`) at inference time — this is handled in `_apply_mup_shapes()` independently of the optimizer choice. See [[mup-scaling]].

## If we ever pretrain from scratch

Revisit this. Pretraining from scratch at batch=256 is when MuAdam is most valuable.
