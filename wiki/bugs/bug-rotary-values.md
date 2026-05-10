---
tags: [bug, plaid, inference, training, rotary]
status: fixed
date: 2026-03-22
severity: critical
fix_commit: d06f5dd
related: [[rotary-embeddings]] [[plaid-1b]] [[bug-mup-scaling]]
---

# Bug: rotary embeddings applied to value vectors

## Symptom

Generation was gibberish even after fixing [[bug-mup-scaling]]. Attention outputs looked statistically off (inspection showed unusually large activation magnitudes in the MLP input).

## Root cause

`Rotary.forward` in our reimplementation produced cos/sin of shape `(1, seq_len, 1, dim)`:

```python
# our buggy version
self._cos_cached = emb.cos()[None, :, None, :]     # (1, seq, 1, dim)
self._sin_cached = emb.sin()[None, :, None, :]
```

The original Plaid uses shape `(1, seq_len, 3, 1, dim)` — an extra `3` axis for q/k/v — and zeroes out the rotation for the v slot:

```python
# original lib/rotary.py
self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
# v slot: no rotation
self.cos_cached[:, :, 2, :, :].fill_(1.)
self.sin_cached[:, :, 2, :, :].fill_(0.)
```

In our version, `apply_rotary_pos_emb(qkv, cos, sin)` broadcast a single rotation matrix across all three of q, k, v — so v was rotated along with q and k. Rotating values corrupts attention outputs because values are supposed to be position-invariant content vectors.

## Fix

`geniesae/plaid_model.py`:

```python
class Rotary(nn.Module):
    def forward(self, x):
        ...
        self._cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        self._sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        self._cos_cached[:, :, 2, :, :].fill_(1.)
        self._sin_cached[:, :, 2, :, :].fill_(0.)
        return self._cos_cached, self._sin_cached


def apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """cos/sin have shape (1, s, 3, 1, d) with v slot as identity."""
    cos = cos[:, :qkv.shape[1]]
    sin = sin[:, :qkv.shape[1]]
    return (qkv * cos) + (_rotate_half(qkv) * sin)
```

Fixed in commit ``d06f5dd``.

## Why the shape must be (1, s, 3, 1, d)

The qkv tensor has shape `(batch, seq, 3, heads, head_dim)`. Broadcasting `(1, s, 3, 1, d)` onto it applies a different rotation per q/k/v slot — which is exactly what we need to leave v unchanged while rotating q and k.

## Detection

Attention output norms should be comparable in magnitude to input norms after the residual. If attention outputs have suspiciously high `std` compared to other layers, rotary is likely rotating v.

```python
# Inspect block 0 attention output before fix
attn_out_std = x_after_attn.float().std()   # > 10 was typical with bug
                                            # ~1-3 is healthy
```

## Test written

`scripts/compare_forward_pass.py` includes a smoke test that checks:
1. Rotary cos/sin shape is `(1, seq, 3, 1, head_dim)`.
2. `cos[:,:,2,:,:] == 1.0` and `sin[:,:,2,:,:] == 0.0`.
3. `v` values pass through `apply_rotary_pos_emb` unchanged.

Passes after fix, fails before.
