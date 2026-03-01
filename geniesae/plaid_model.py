"""PLAID Diffusion Language Model — vendored for SAE analysis.

Adapted from https://github.com/igul222/plaid with pure-PyTorch
replacements for fused CUDA kernels (FlashAttention, Apex RMSNorm,
fused MLP). This lets us load the pretrained 1B weights without
compiling custom CUDA extensions.

The transformer blocks are at ``DiffusionModel.blocks`` — a
``nn.ModuleList`` of ``TransformerBlock``.  Each block outputs a
tensor of shape ``(batch, seq_len, dim)`` which we hook for SAE work.

Architecture (1B config):
    dim=2048, embed_dim=16, n_blocks=24, n_heads=32, vocab_size=32768
    Learned noise schedule (continuous gamma), self-conditioning,
    muP parameterization, rotary embeddings.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Rotary Embeddings (vendored from plaid/lib/rotary.py)
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
        return self._cos_cached, self._sin_cached


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to q and k in a (b, s, 3, h, d) tensor."""
    cos = cos[:, :qkv.shape[1]]
    sin = sin[:, :qkv.shape[1]]
    # qkv shape: (b, s, 3, h, d) — apply to q (idx 0) and k (idx 1)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return torch.stack([q, k, v], dim=2)


# ---------------------------------------------------------------------------
# Pure-PyTorch replacements for fused kernels
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm — drop-in replacement for apex.normalization.FusedRMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.weight.float()).to(dtype)


class LayerNorm(nn.Module):
    """LayerNorm matching PLAID's custom implementation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(
    x: torch.Tensor, W: torch.Tensor, x_skip: torch.Tensor, residual_scale: float
) -> torch.Tensor:
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale,
    ).view(*x.shape[:-1], dim_out)


# ---------------------------------------------------------------------------
# Transformer Block (pure PyTorch attention instead of FlashAttention)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """PLAID transformer block with self-attention and MLP.

    Uses standard PyTorch attention instead of FlashAttention, and a
    plain two-layer MLP instead of the fused MLP. Weight shapes are
    identical so pretrained weights load directly.
    """

    def __init__(self, dim: int, n_heads: int, residual_scale: float) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.residual_scale = residual_scale
        self.head_dim = dim // n_heads

        self.rmsnorm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rmsnorm2 = RMSNorm(dim)
        # Replace FusedMLP with plain two-layer MLP (same weight shapes)
        self.mlp = PlainMLP(dim, 4 * dim)

    def forward(
        self, x: torch.Tensor, rotary_cos_sin: tuple, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Self-attention
        x_skip = x
        x_norm = self.rmsnorm1(x)
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

        half_dtype = qkv.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(half_dtype), sin.to(half_dtype))

        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # q,k,v: (b, s, h, d) -> (b, h, s, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

        x = residual_linear(attn_out, self.attn_out.weight, x_skip, self.residual_scale)

        # Feedforward
        x_skip = x
        x_norm = self.rmsnorm2(x)
        x_mlp = self.mlp(x_norm)
        x = torch.add(x_skip, x_mlp, alpha=self.residual_scale)

        return x


class PlainMLP(nn.Module):
    """Two-layer MLP matching flash_attn.ops.fused_dense.FusedMLP weight layout.

    FusedMLP(dim, 4*dim, bias1=False, bias2=False) has:
        fc1.weight: (4*dim, dim)
        fc2.weight: (dim, 4*dim)
    with GELU activation between them.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# ---------------------------------------------------------------------------
# PLAID model components
# ---------------------------------------------------------------------------

class EmbeddingMatrix(nn.Module):
    """Normalized embedding matrix."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.matrix.data /= self.matrix.data.norm(p=2, dim=1, keepdim=True)

    def forward(self) -> torch.Tensor:
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        return self.matrix / (norm + 1e-8)


class NoiseSchedule(nn.Module):
    """Learned noise schedule: maps t in [0,1] to gamma in [0,1]."""

    def __init__(self) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())

        def gamma_tilde(t_: torch.Tensor) -> torch.Tensor:
            h = t_.double()[:, None] - 0.5
            h = (h @ W1.T) + self.b1[None, :].double()
            h = torch.tanh(h)
            h = (h @ W2.T)[:, 0]
            return h

        gamma_tilde_0 = gamma_tilde(torch.tensor([0.0], device=t.device))
        gamma_tilde_1 = gamma_tilde(torch.tensor([1.0], device=t.device))
        gamma_tilde_t = gamma_tilde(t)
        return (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)


class GammaBounds(nn.Module):
    """Learnable gamma_0 and gamma_1 bounds."""

    def __init__(self, gamma_0: float = -3.0, gamma_1: float = 6.0) -> None:
        super().__init__()
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gamma_0.clone().double(), self.gamma_1.clone().double()


class MuReadout(nn.Linear):
    """muP readout layer — just a Linear with scaling attributes.

    In the original code this comes from the mup library. For inference
    we only need the weight/bias and the two scaling factors that are set
    by mup.set_base_shapes(). We store them as buffers so they survive
    state_dict loading.
    """

    def __init__(self, in_features: int, out_features: int, **kwargs: Any) -> None:
        super().__init__(in_features, out_features, **kwargs)
        # These get overwritten by _apply_mup_shapes after loading weights
        self.register_buffer("_output_mult", torch.tensor(1.0))
        self.register_buffer("_width_mult", torch.tensor(1.0))

    @property
    def output_mult(self) -> float:
        return float(self._output_mult)

    def width_mult(self) -> float:
        return float(self._width_mult)


class DiffusionModel(nn.Module):
    """PLAID's main diffusion transformer.

    Forward signature:
        model(z, gamma, embedding_matrix, bias_scale, x_selfcond,
              selfcond_mask=None, cu_seqlens=None)
        -> (logits, x_reconst)

    The transformer blocks we hook for SAE analysis are at ``self.blocks``.
    """

    def __init__(
        self,
        dim: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: int,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self.input_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)
        self.gamma_linear = nn.Linear(64, dim, bias=False)

        self.rotary_emb = Rotary(dim // n_heads)

        residual_scale = float(1.0 / np.sqrt(n_blocks))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, residual_scale) for _ in range(n_blocks)]
        )

        self.output_norm = LayerNorm(dim)
        self.output_linear = MuReadout(dim, vocab_size)

        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
        embedding_matrix: torch.Tensor,
        bias_scale: float,
        x_selfcond: torch.Tensor,
        selfcond_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], device=z.device)

        alpha_squared = torch.sigmoid(-gamma)[:, None, None]
        sigma_squared = torch.sigmoid(gamma)[:, None, None]
        alpha = alpha_squared.sqrt()

        # Rescale input to stdev 1
        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()

        x = self.input_linear(x)
        x = x + self.selfcond_linear(x_selfcond * float(np.sqrt(self.embed_dim)))

        # Gamma embedding (sinusoidal)
        gamma_embed = torch.linspace(-5.0, 5.0, 64 // 2, device=z.device)
        gamma_embed = gamma_embed.exp()[None, :] * gamma[:, None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:, None, :]
        x = x + gamma_embed

        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, cu_seqlens=cu_seqlens)

        x = self.output_norm(x.float())

        x *= self.output_linear.output_mult / self.output_linear.width_mult()

        W = torch.cat(
            [self.output_linear.weight.T, embedding_matrix.T, embedding_matrix.T.detach()],
            dim=0,
        )
        z_scaled_for_bias = bias_scale * (alpha / sigma_squared).float() * z
        x = torch.cat(
            [
                x,
                z_scaled_for_bias * (1 - selfcond_mask.float()[:, None, None]),
                z_scaled_for_bias * selfcond_mask.float()[:, None, None],
            ],
            dim=2,
        )
        logits = torch.addmm(
            self.output_linear.bias.view(1, self.vocab_size),
            x.view(-1, self.dim + 2 * self.embed_dim),
            W.view(self.dim + 2 * self.embed_dim, self.vocab_size),
        ).view(x.shape[0], x.shape[1], self.vocab_size)

        # Categorical reparameterization
        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat(
            [embedding_matrix, embedding_matrix.detach()], dim=1
        )
        x_reconst_a = x_reconst[:, :, : self.embed_dim]
        x_reconst_b = x_reconst[:, :, self.embed_dim :]
        lerp_weight = selfcond_mask[:, None, None].to(x_reconst_a.dtype)
        x_reconst = torch.lerp(x_reconst_a, x_reconst_b, lerp_weight)

        return logits, x_reconst


# ---------------------------------------------------------------------------
# Weight loading and muP shape application
# ---------------------------------------------------------------------------

def _remap_fused_mlp_keys(state_dict: dict) -> dict:
    """Remap FusedMLP weight keys to PlainMLP keys.

    FusedMLP stores weights as:
        blocks.X.mlp.fc1.weight  (hidden_dim, dim)
        blocks.X.mlp.fc2.weight  (dim, hidden_dim)
    Our PlainMLP uses the same key names, so no remapping needed
    if the original checkpoint uses fc1/fc2.

    However, FusedMLP might store them differently. Let's handle both.
    """
    new_sd = {}
    for k, v in state_dict.items():
        # FusedMLP uses fc1/fc2 naming already — no change needed
        new_sd[k] = v
    return new_sd


def _remap_rmsnorm_keys(state_dict: dict) -> dict:
    """Remap apex FusedRMSNorm keys to our RMSNorm.

    Apex FusedRMSNorm stores weight as just 'weight', same as ours.
    """
    return state_dict


def _apply_mup_shapes(model: DiffusionModel, dim: int, base_dim: int = 256) -> None:
    """Apply muP scaling factors to the output readout layer.

    In the original code, mup.set_base_shapes() computes these from
    base/delta models. For inference we just need the ratio.
    """
    width_mult = dim / base_dim
    output_mult = base_dim / dim  # muP readout scaling

    model.output_linear._output_mult.fill_(output_mult)
    model.output_linear._width_mult.fill_(width_mult)


def load_plaid_modules(
    weights_path: str,
    dim: int = 2048,
    embed_dim: int = 16,
    n_blocks: int = 24,
    n_heads: int = 32,
    vocab_size: int = 32768,
    gamma_0: float = -3.0,
    gamma_1: float = 6.0,
    device: str = "cuda:0",
) -> dict[str, nn.Module]:
    """Load all four PLAID modules from a weights directory.

    Args:
        weights_path: Directory containing model.pt, noise_schedule.pt,
            gamma_bounds.pt, embedding_matrix.pt.
        dim, embed_dim, n_blocks, n_heads, vocab_size: Architecture params.
        gamma_0, gamma_1: Noise schedule bounds.
        device: Target device.

    Returns:
        Dict with keys: 'model', 'noise_schedule', 'gamma_bounds', 'embedding_matrix'.
    """
    weights_dir = Path(weights_path)

    modules: dict[str, nn.Module] = {
        "noise_schedule": NoiseSchedule().float(),
        "gamma_bounds": GammaBounds(gamma_0, gamma_1).float(),
        "embedding_matrix": EmbeddingMatrix(vocab_size, embed_dim).float(),
        "model": DiffusionModel(dim, embed_dim, n_blocks, n_heads, vocab_size).float(),
    }

    for name, module in modules.items():
        ckpt_path = weights_dir / f"{name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing weight file: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _remap_fused_mlp_keys(state_dict)
        state_dict = _remap_rmsnorm_keys(state_dict)

        # Load with strict=False to handle any minor key mismatches
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (our custom buffers)
            real_missing = [k for k in missing if not k.endswith(("_output_mult", "_width_mult"))]
            if real_missing:
                print(f"[PLAID] {name}: missing keys: {real_missing}")
        if unexpected:
            print(f"[PLAID] {name}: unexpected keys: {unexpected}")

        module.to(device)
        module.eval()

    # Apply muP scaling
    _apply_mup_shapes(modules["model"], dim)

    return modules


class PlaidDiffusionHelper:
    """Diffusion sampling helper for PLAID's continuous VDM formulation.

    Unlike GENIE's discrete timestep schedule, PLAID uses a learned
    continuous noise schedule parameterized by t in [0, 1] mapped to
    gamma (log-SNR) via a neural network.

    Implements the reverse sampling from Appendix A.4 of the VDM paper.
    """

    def __init__(
        self,
        modules: dict[str, nn.Module],
        # NOTE: Default standardised to 256 steps across all PLAID pipelines
        # (evaluation, trajectory, intervention, schedule experiments).
        sampling_timesteps: int = 256,
        score_temp: float = 0.9,
    ) -> None:
        self.model = modules["model"]
        self.noise_schedule = modules["noise_schedule"]
        self.gamma_bounds = modules["gamma_bounds"]
        self.embedding_matrix_module = modules["embedding_matrix"]
        self.sampling_timesteps = sampling_timesteps
        self.score_temp = score_temp

    def get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous t to gamma using learned schedule + bounds."""
        gamma_0, gamma_1 = self.gamma_bounds()
        gamma_normalized = self.noise_schedule(t).double()
        return gamma_0 + (gamma_1 - gamma_0) * gamma_normalized

    @torch.no_grad()
    def q_sample(
        self, x_embed: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at continuous time t.

        Returns (z_t, gamma_t).
        """
        gamma = self.get_gamma(t)
        alpha_squared = torch.sigmoid(-gamma)[:, None, None]
        sigma_squared = torch.sigmoid(gamma)[:, None, None]
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()

        noise = torch.randn_like(x_embed)
        z_t = alpha * x_embed + sigma * noise
        return z_t, gamma

    @torch.no_grad()
    def p_sample_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        x_selfcond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single reverse diffusion step from time t to time s.

        Args:
            z: Current noisy latent.
            t: Current time (scalar tensor).
            s: Target time (t - 1/T).
            x_selfcond: Self-conditioning input from previous step.

        Returns:
            (z_s, x_reconst) — denoised latent and reconstruction for self-cond.
        """
        embedding_matrix = self.embedding_matrix_module()

        gamma_s = self.get_gamma(s)
        gamma_t = self.get_gamma(t)

        alpha_squared_s = torch.sigmoid(-gamma_s)
        alpha_squared_t = torch.sigmoid(-gamma_t)
        sigma_squared_t = torch.sigmoid(gamma_t)
        sigma_t = sigma_squared_t.sqrt()
        alpha_t = alpha_squared_t.sqrt()

        _, x_reconst = self.model(
            z=z.float(),
            gamma=gamma_t.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.0,
            x_selfcond=x_selfcond,
        )

        x_selfcond_out = x_reconst.clone().detach()
        x_reconst = x_reconst.double()

        # Score temperature
        epsilon_pred = (z.double() - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
        epsilon_pred /= self.score_temp
        x_reconst = (z.double() - sigma_t[:, None, None] * epsilon_pred) / alpha_t[:, None, None]

        # VDM reverse step (stochastic)
        if t.item() > 0:
            c = -torch.expm1(gamma_s - gamma_t)
            z_new = (1 - c) * alpha_squared_s.sqrt()[:, None, None] / alpha_squared_t.sqrt()[:, None, None] * z.double()
            z_new += c * (alpha_squared_s.sqrt()[:, None, None] * x_reconst)
            z_new += (c * (1 - alpha_squared_s)).sqrt()[:, None, None] * torch.randn_like(z).double()
            z = z_new.float()

        return z, x_selfcond_out
