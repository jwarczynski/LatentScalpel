---
tags: [concept, rotary, plaid]
related: [[plaid-1b]] [[bug-rotary-values]]
---

# Rotary embeddings (RoPE) in Plaid

Positional information encoded by rotating pairs of dimensions in the q/k vectors as a function of position. In Plaid, **values are NOT rotated** — they pass through unchanged.

## The math

For a position `m` and a pair of dimensions `(x1, x2)` at even/odd index in the head:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

with $\theta = 1 / 10000^{2i/d}$. Applied independently to q and k at each head, but NOT to v.

In practice this is vectorized as:

$$q' = q \odot \cos(\text{emb}) + \text{rotate\_half}(q) \odot \sin(\text{emb})$$

where `rotate_half((x1,...,xd/2, xd/2+1,...,xd)) = (-xd/2+1,...,-xd, x1,...,xd/2)`.

## Plaid's specific implementation

The qkv tensor has shape `(batch, seq, 3, heads, head_dim)`. Plaid's rotary cache has shape `(1, seq, 3, 1, head_dim)` — the extra `3` axis is there so the rotation matrix can be **different per q/k/v slot**. And the v slot is set to identity:

```python
# cos/sin already computed, shape (1, seq, 3, 1, d)
self.cos_cached[:, :, 2, :, :].fill_(1.)  # v slot cos = 1
self.sin_cached[:, :, 2, :, :].fill_(0.)  # v slot sin = 0
```

After `apply_rotary_pos_emb(qkv, cos, sin) = qkv * cos + rotate_half(qkv) * sin`:
- q is rotated.
- k is rotated.
- v is unchanged (cos=1, sin=0 → identity).

## Why values are not rotated

Values carry content, not position. Attention's softmax(qk/√d) assigns weights based on query-key similarity (which should respect positional context via rotated q and k). The weighted sum of values then aggregates content. If you rotate v, the aggregated content becomes position-dependent in a way the model didn't train for.

## Our bug

We initially produced cos/sin of shape `(1, seq, 1, dim)` — missing the `3` axis. When broadcast against qkv of shape `(batch, seq, 3, heads, head_dim)`, the single rotation matrix applied to ALL of q, k, v. The model was quietly corrupting its values.

See [[bug-rotary-values]] for the fix and detection strategy.

## Implementation detail

Our `apply_rotary_pos_emb` is decorated with `@torch.jit.script` for speed. Jit doesn't like branching on non-tensor conditions, so the function keeps things simple:

```python
@torch.jit.script
def apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (qkv * cos) + (rotate_half(qkv) * sin)
```

It relies on the caller pre-shaping cos/sin so v is identity.
