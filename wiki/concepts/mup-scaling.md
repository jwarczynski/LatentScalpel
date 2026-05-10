---
tags: [concept, mup, plaid]
related: [[plaid-1b]] [[bug-mup-scaling]] [[decision-no-mup-adam]]
---

# muP (Maximal Update Parameterization)

A principled way to scale model initialization and learning rates so hyperparameters transfer across model sizes.

## Why Plaid uses it

Plaid was designed so that a small "proxy" model trained at dim=256 can tune hyperparameters that then apply directly to the full 1B model at dim=2048. muP makes this work by adjusting certain layers' initialization and forward-pass scaling as a function of `width_mult = dim / base_dim`.

Plaid specifically uses `mup.MuReadout` for the output projection (logits layer). Normal `Linear` would have its output magnitudes grow with width; `MuReadout` compensates.

## The `MuReadout` forward pass

From `mup/layer.py`:

```python
class MuReadout(nn.Linear):
    def __init__(self, ..., output_mult=1.0):
        self.output_mult = output_mult
        ...
    
    def forward(self, x):
        return super().forward(self.output_mult * x / self.width_mult())
```

Key point: `width_mult()` is set by `mup.set_base_shapes(main, base, delta)`. It equals `dim / base_dim`. For Plaid 1B: `2048 / 256 = 8.0`.

Combined with `output_mult=1.0` (default), the effective pre-linear scaling is **`1.0 / 8.0 = 0.125`**.

## Our reimplementation

We don't depend on the mup library at inference; we replicate the scaling manually. From `plaid_model.py`:

```python
class MuReadout(nn.Linear):
    """mup readout — just a Linear with scaling attributes."""
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer("_output_mult", torch.tensor(1.0))
        self.register_buffer("_width_mult", torch.tensor(1.0))

    @property
    def output_mult(self) -> float:
        return float(self._output_mult)

    def width_mult(self) -> float:
        return float(self._width_mult)
```

And the setter:

```python
def _apply_mup_shapes(model, dim, base_dim=256):
    width_mult = dim / base_dim   # 8.0
    output_mult = 1.0             # MuReadout default
    model.output_linear._output_mult.fill_(output_mult)
    model.output_linear._width_mult.fill_(width_mult)
```

This is **called once** in `load_plaid_modules()` after weight loading.

## The bug we hit

Initial implementation had `output_mult = base_dim / dim = 0.125`. This gave effective scaling `0.125 / 8.0 = 0.015625` — 8× too small. All logits crushed. See [[bug-mup-scaling]].

## muP vs standard AdamW for fine-tuning

For **training from scratch**, Plaid uses `mup.MuAdam` which applies per-layer lr scaling so the effective lr matches the hyperparameters tuned on the proxy model.

For **fine-tuning**, we use standard AdamW. Reasons:

1. The fine-tuning dataset is small enough that we don't need mup's hyperparameter transfer.
2. MuAdam adds complexity to our Lightning-based training loop.
3. We're using `lr=1e-6` which is low enough that the per-layer scaling doesn't matter much — at this scale, all layers update slowly regardless.

See [[decision-no-mup-adam]] for the full reasoning.

## Summary

- muP is about training-time hyperparameter transfer.
- At **inference**, the only remaining muP thing is the `output_mult / width_mult` scaling in `MuReadout.forward`.
- Our implementation mimics this with `_output_mult=1.0, _width_mult=8.0` (for 1B).
- Don't confuse `output_mult` with `base_dim/dim`; they happen to have inverse relationships in Plaid 1B but aren't equal in general.
