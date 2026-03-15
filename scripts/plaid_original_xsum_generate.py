"""
Conditional text generation using the ORIGINAL Plaid 1B code + pretrained weights.

This script replaces flash-attention and apex with pure PyTorch equivalents
so we can run inference without compiling custom CUDA kernels.

It uses the guidance-based conditional generation from the original sample.py:
given an article prefix, it guides the diffusion process to produce text
that starts with the article, then generates a summary continuation.

Usage (via Slurm):
    python scripts/plaid_original_xsum_generate.py \
        --weights_path /path/to/plaid1b_weights \
        --tokenizer_path /path/to/owt2_tokenizer.json \
        --xsum_src_path /path/to/test.src \
        --xsum_tgt_path /path/to/dev.tgt \
        --output_path /path/to/output.jsonl \
        --num_samples 50 \
        --max_article_tokens 200 \
        --max_summary_tokens 80 \
        --sampling_timesteps 256 \
        --guidance_weight 2.0 \
        --score_temp 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import tqdm
from einops import rearrange
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Pure-PyTorch replacements for apex / flash-attention
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Drop-in replacement for apex.normalization.FusedRMSNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.weight.float()).to(dtype)


class SimpleMLP(nn.Module):
    """Drop-in replacement for flash_attn.ops.fused_dense.FusedMLP."""
    def __init__(self, in_features: int, hidden_features: int,
                 bias1: bool = False, bias2: bool = False, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def sdpa_attention(qkv: torch.Tensor, seq_len: int, batch_size: int,
                   causal: bool = False) -> torch.Tensor:
    """Replace flash_attn_unpadded_qkvpacked_func with PyTorch SDPA."""
    # qkv: (B*S, 3, H, D)
    qkv = rearrange(qkv, '(b s) three h d -> three b h s d', b=batch_size, s=seq_len)
    q, k, v = qkv[0], qkv[1], qkv[2]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = rearrange(out, 'b h s d -> (b s) h d')
    return out


# ---------------------------------------------------------------------------
# Rotary embeddings (from original lib/rotary.py, with torchscript fallback)
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (qkv * cos) + (rotate_half(qkv) * sin)


# ---------------------------------------------------------------------------
# Model components (from original lib/models.py, with pure-PyTorch ops)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, causal: bool, residual_scale: float):
        super().__init__()
        self.causal = causal
        self.dim = dim
        self.n_heads = n_heads
        self.residual_scale = residual_scale

        self.rmsnorm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rmsnorm2 = RMSNorm(dim)
        self.mlp = SimpleMLP(dim, 4 * dim, bias1=False, bias2=False)

    def forward(self, x: torch.Tensor, rotary_cos_sin, cu_seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Self-attention
        x_skip = x
        x = self.rmsnorm1(x)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                         three=3, h=self.n_heads)
        half_dtype = qkv.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(half_dtype), sin.to(half_dtype))
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')

        x = sdpa_attention(qkv, seq_len, batch_size, causal=self.causal)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = residual_linear(x, self.attn_out.weight, x_skip, self.residual_scale)

        # Feedforward
        x_skip = x
        x = self.rmsnorm2(x)
        x = self.mlp(x)
        x = torch.add(x_skip, x, alpha=self.residual_scale)
        return x


class EmbeddingMatrix(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.matrix.data /= self.matrix.data.norm(p=2, dim=1, keepdim=True)

    def forward(self):
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        return self.matrix / (norm + 1e-8)


class NoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())

        def gamma_tilde(t):
            h = t[:, None] - 0.5
            h = (h @ W1.T) + self.b1[None, :].double()
            h = torch.tanh(h)
            h = (h @ W2.T)[:, 0]
            return h

        gamma_tilde_0 = gamma_tilde(torch.tensor([0.], device='cuda'))
        gamma_tilde_1 = gamma_tilde(torch.tensor([1.], device='cuda'))
        gamma_tilde_t = gamma_tilde(t)
        return (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)


class GammaBounds(nn.Module):
    def __init__(self, gamma_0: float, gamma_1: float):
        super().__init__()
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))

    def forward(self):
        return self.gamma_0.clone().double(), self.gamma_1.clone().double()


class DiffusionModel(nn.Module):
    def __init__(self, dim: int, embed_dim: int, n_blocks: int,
                 n_heads: int, vocab_size: int):
        super().__init__()
        self.input_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear.weight.data.zero_()
        self.gamma_linear = nn.Linear(64, dim, bias=False)
        self.gamma_linear.weight.data.zero_()

        self.rotary_emb = Rotary(dim // n_heads)

        residual_scale = float(1. / np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, False, residual_scale)
            for _ in range(n_blocks)
        ])

        self.output_norm = LayerNorm(dim)
        # MuReadout replacement: just a linear with output_mult/width_mult
        self.output_linear = nn.Linear(dim, vocab_size)
        self.output_linear.weight.data.zero_()
        self.output_linear.bias.data.zero_()
        # Store mup scaling factors (will be set after weight loading)
        self._output_mult = 1.0
        self._width_mult = 1.0

        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, z, gamma, embedding_matrix, bias_scale, x_selfcond,
                selfcond_mask=None, cu_seqlens=None):
        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], device='cuda')

        alpha_squared = torch.sigmoid(-gamma)[:, None, None]
        sigma_squared = torch.sigmoid(gamma)[:, None, None]
        alpha = alpha_squared.sqrt()

        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()
        x = self.input_linear(x)
        x = x + self.selfcond_linear(x_selfcond * float(np.sqrt(self.embed_dim)))

        gamma_embed = torch.linspace(-5., 5., 64 // 2, device='cuda')
        gamma_embed = gamma_embed.exp()[None, :] * gamma[:, None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:, None, :]
        x = x + gamma_embed

        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, cu_seqlens=cu_seqlens)

        x = self.output_norm(x.float())

        # mup scaling: output_mult / width_mult
        x *= self._output_mult / self._width_mult

        W = torch.cat([
            self.output_linear.weight.T,
            embedding_matrix.T,
            embedding_matrix.T.detach()
        ], dim=0)
        z_scaled_for_bias = bias_scale * (alpha / sigma_squared).float() * z
        x = torch.cat([
            x,
            z_scaled_for_bias * (1 - selfcond_mask.float()[:, None, None]),
            z_scaled_for_bias * selfcond_mask.float()[:, None, None]
        ], dim=2)
        logits = torch.addmm(
            self.output_linear.bias.view(1, self.vocab_size),
            x.view(-1, self.dim + 2 * self.embed_dim),
            W.view(self.dim + 2 * self.embed_dim, self.vocab_size)
        ).view(x.shape[0], x.shape[1], self.vocab_size)

        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat([
            embedding_matrix, embedding_matrix.detach()], dim=1)
        x_reconst = torch.lerp(
            x_reconst[:, :, :self.embed_dim],
            x_reconst[:, :, self.embed_dim:],
            selfcond_mask.float()[:, None, None]
        )
        return logits, x_reconst


# ---------------------------------------------------------------------------
# Weight loading with mup parameter remapping
# ---------------------------------------------------------------------------

def load_modules(weights_path: str, dim: int = 2048, embed_dim: int = 16,
                 n_blocks: int = 24, n_heads: int = 32, vocab_size: int = 32768,
                 gamma_0: float = -3., gamma_1: float = 6.):
    """Load pretrained plaid1b weights into our pure-PyTorch modules.

    The original code uses mup.MuReadout for output_linear, which stores
    extra attributes (output_mult, width_mult). We handle this by loading
    the state dict with strict=False and manually setting the scaling.
    """
    modules = {
        'noise_schedule': NoiseSchedule().float(),
        'gamma_bounds': GammaBounds(gamma_0, gamma_1).float(),
        'embedding_matrix': EmbeddingMatrix(vocab_size, embed_dim).float(),
        'model': DiffusionModel(dim, embed_dim, n_blocks, n_heads, vocab_size).float(),
    }

    for name, module in modules.items():
        ckpt_path = os.path.join(weights_path, f'{name}.pt')
        state_dict = torch.load(ckpt_path, map_location='cuda')

        # Handle mup MuReadout: the checkpoint has output_linear with extra keys
        if name == 'model':
            # Extract mup scaling factors if present
            output_mult = state_dict.pop('output_linear.output_mult', None)
            width_mult = state_dict.pop('output_linear.width_mult', None)
            # Remove any other mup-specific keys
            keys_to_remove = [k for k in state_dict if 'infshapes' in k
                              or '_base_shape' in k or '_mup' in k]
            for k in keys_to_remove:
                state_dict.pop(k)

            # Map FusedRMSNorm weight names to our RMSNorm
            # Original: rmsnorm1.weight -> our rmsnorm1.weight (same name, OK)
            # But original also has output_norm as FusedRMSNorm in AutoregressiveModel
            # For DiffusionModel, output_norm is LayerNorm (same in original)

            module.load_state_dict(state_dict, strict=False)

            # Set mup scaling
            if output_mult is not None:
                module._output_mult = float(output_mult)
            else:
                # Default mup scaling for dim=2048: output_mult = 1/width_mult
                # width_mult = dim / base_dim = 2048 / 256 = 8
                module._output_mult = 1.0
            if width_mult is not None:
                module._width_mult = float(width_mult)
            else:
                module._width_mult = dim / 256.0  # base_dim = 256

            print(f"  Model mup scaling: output_mult={module._output_mult}, "
                  f"width_mult={module._width_mult}")
        else:
            module.load_state_dict(state_dict)

        module.cuda()
        module.eval()
        print(f"  Loaded {name} from {ckpt_path}")

    return modules


# ---------------------------------------------------------------------------
# Sampling (from original sample.py)
# ---------------------------------------------------------------------------

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Computes log(1-exp(-|x|))"""
    x = -x.abs()
    return torch.where(
        x > -0.693,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x))
    )


@torch.no_grad()
def generate_samples(
    modules: dict,
    guidance_tokens: list,
    n_samples: int = 4,
    seq_len: int = 1024,
    sampling_timesteps: int = 256,
    score_temp: float = 0.9,
    initial_noise_scale: float = 1.0,
    ddim_sampler: bool = False,
    embed_dim: int = 16,
):
    """
    Generate samples using the original PLAID sampling procedure.

    guidance_tokens: list of (token_id, weight, position, complement)
        - token_id: vocab index
        - weight: guidance weight
        - position: int (sequence position), 'any', or 'all'
        - complement: if True, guide on log(1-p(y|x))
    """
    embedding_matrix = modules['embedding_matrix']()
    gamma_0, gamma_1 = modules['gamma_bounds']()

    z = torch.randn((n_samples, seq_len, embed_dim), device='cuda') * initial_noise_scale
    x_selfcond = torch.zeros_like(z).float()

    for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., sampling_timesteps),
                                     desc="Sampling")):
        t = t[None].cuda()
        s = t - 1. / sampling_timesteps
        gamma_s = modules['noise_schedule'](s).double()
        gamma_t = modules['noise_schedule'](t).double()
        gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
        gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
        alpha_squared_s = torch.sigmoid(-gamma_s)
        alpha_squared_t = torch.sigmoid(-gamma_t)
        alpha_s = alpha_squared_s.sqrt()
        alpha_t = alpha_squared_t.sqrt()
        sigma_squared_s = torch.sigmoid(gamma_s)
        sigma_squared_t = torch.sigmoid(gamma_t)
        sigma_s = sigma_squared_s.sqrt()
        sigma_t = sigma_squared_t.sqrt()

        if len(guidance_tokens) > 0:
            with torch.enable_grad():
                z_grad = z.detach().requires_grad_(True)
                logits, x_reconst = modules['model'](
                    z=z_grad.to(torch.float32),
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.,
                    x_selfcond=x_selfcond
                )
                logprobs = F.log_softmax(logits.float(), dim=2)
                logprobs_any = logprobs.logsumexp(dim=1) - float(seq_len)

                sum_logp = 0.
                for token, weight, position, complement in guidance_tokens:
                    if position == 'any':
                        logp = logprobs_any[:, token]
                    elif position == 'all':
                        logp = logprobs[:, :, token]
                    else:
                        logp = logprobs[:, position, token]
                    if complement:
                        logp = log1mexp(logp)
                    sum_logp = sum_logp + weight * logp.sum()

                guidance_grad = autograd.grad(sum_logp, [z_grad])[0]

            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()
            epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
            epsilon_pred /= score_temp
            x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
            x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
            epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
        else:
            _, x_reconst = modules['model'](
                z=z.to(torch.float32),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()
            epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
            epsilon_pred /= score_temp
            x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t

        if t > 0:
            if ddim_sampler:
                z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
            else:
                c = -torch.expm1(gamma_s - gamma_t)
                z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

    # Final step: get logits at t=0
    logits, _ = modules['model'](
        z=z.float(),
        gamma=gamma_t.float(),
        embedding_matrix=embedding_matrix,
        bias_scale=1.,
        x_selfcond=x_selfcond
    )
    x_samples = logits.argmax(dim=-1)
    return x_samples


# ---------------------------------------------------------------------------
# XSum conditional generation
# ---------------------------------------------------------------------------

def load_xsum_articles(src_path: str, tgt_path: str | None = None,
                       max_examples: int = 50):
    """Load XSum source articles and optional reference summaries."""
    with open(src_path, 'r') as f:
        articles = [line.strip() for line in f.readlines()]
    references = None
    if tgt_path and os.path.exists(tgt_path):
        with open(tgt_path, 'r') as f:
            references = [line.strip() for line in f.readlines()]
    articles = articles[:max_examples]
    if references:
        references = references[:max_examples]
    return articles, references


def conditional_generate_xsum(
    modules: dict,
    tokenizer: Tokenizer,
    articles: list[str],
    references: list[str] | None,
    output_path: str,
    max_article_tokens: int = 900,
    n_samples_per_article: int = 1,
    sampling_timesteps: int = 256,
    guidance_weight: float = 2.0,
    score_temp: float = 0.9,
    seq_len: int = 1024,
    embed_dim: int = 16,
):
    """Generate conditional summaries for XSum articles.

    Strategy: Build a prompt "article_text\\n\\nTL;DR:" and guide all prompt
    tokens at their positions. The model was trained on OpenWebText2 which
    contains natural web text with TL;DR patterns, so this steers it toward
    summarization. The generated text after the prompt is the summary,
    truncated at EOT (token 0) or double newline.

    This matches the approach in geniesae/configs/plaid_token_guidance_config.py.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    results = []

    for idx, article in enumerate(articles):
        print(f"\n{'='*60}")
        print(f"Article {idx+1}/{len(articles)}")

        # Build prompt: article + \n\nTL;DR:
        # The model was trained on OpenWebText2 which contains natural
        # web text. TL;DR is a common summarization prompt in web text.
        prompt = article + "\n\nTL;DR:"
        prompt_ids = tokenizer.encode(prompt).ids

        # Truncate if needed (leave room for generation)
        if len(prompt_ids) > max_article_tokens:
            prompt_ids = prompt_ids[:max_article_tokens]

        prefix_len = len(prompt_ids)

        # Build guidance tokens: guide each prompt position
        guidance_tokens = [
            (token_id, guidance_weight, pos, False)
            for pos, token_id in enumerate(prompt_ids)
        ]

        print(f"  Prompt tokens: {prefix_len}, seq_len: {seq_len}")
        t_start = time.time()

        # Generate full seq_len sequence with guidance
        x_samples = generate_samples(
            modules=modules,
            guidance_tokens=guidance_tokens,
            n_samples=n_samples_per_article,
            seq_len=seq_len,
            sampling_timesteps=sampling_timesteps,
            score_temp=score_temp,
            embed_dim=embed_dim,
        )

        elapsed = time.time() - t_start
        print(f"  Generation took {elapsed:.1f}s")

        # Decode
        for sample_idx in range(n_samples_per_article):
            all_ids = x_samples[sample_idx].tolist()

            # The generated text after the prompt is our "summary"
            gen_suffix_ids = all_ids[prefix_len:]

            # Truncate at EOT (token 0 in PLAID tokenizer)
            gen_clean: list[int] = []
            for tid in gen_suffix_ids:
                if tid == 0:
                    break
                gen_clean.append(tid)

            # Decode and take first paragraph only
            gen_text = tokenizer.decode(gen_clean) if gen_clean else ""
            if "\n\n" in gen_text:
                gen_text = gen_text[:gen_text.index("\n\n")]
            gen_text = gen_text.strip()

            full_text = tokenizer.decode(all_ids)
            prompt_text = tokenizer.decode(all_ids[:prefix_len])
            ref = references[idx] if references else None

            result = {
                "idx": idx,
                "sample_idx": sample_idx,
                "article": article[:500],
                "reference_summary": ref,
                "generated_summary": gen_text,
                "generated_full": full_text,
                "prompt_text": prompt_text[:300],
                "prefix_tokens": prefix_len,
                "gen_tokens": len(gen_clean),
                "sampling_timesteps": sampling_timesteps,
                "guidance_weight": guidance_weight,
                "score_temp": score_temp,
            }
            results.append(result)

            print(f"  --- Sample {sample_idx} ---")
            print(f"  PROMPT (last 100 chars): ...{prompt_text[-100:]}")
            print(f"  GENERATED SUMMARY: {gen_text[:300]}")
            if ref:
                print(f"  REFERENCE: {ref[:200]}")

    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nSaved {len(results)} results to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Conditional XSum generation with original Plaid 1B weights"
    )
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to plaid1b_weights directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to owt2_tokenizer.json")
    parser.add_argument("--xsum_src_path", type=str, required=True,
                        help="Path to xsum test.src or dev.src")
    parser.add_argument("--xsum_tgt_path", type=str, default=None,
                        help="Path to xsum dev.tgt (optional, for reference)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of articles to process")
    parser.add_argument("--n_samples_per_article", type=int, default=1,
                        help="Number of samples per article")
    parser.add_argument("--max_article_tokens", type=int, default=900,
                        help="Max tokens for article+TL;DR prompt (leaves room for summary)")
    parser.add_argument("--sampling_timesteps", type=int, default=1024,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--guidance_weight", type=float, default=2.0,
                        help="Classifier-free guidance weight for prefix")
    parser.add_argument("--score_temp", type=float, default=0.9,
                        help="Score temperature for sampling")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--n_blocks", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--gamma_0", type=float, default=-3.)
    parser.add_argument("--gamma_1", type=float, default=6.)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float64)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Loading pretrained Plaid 1B weights...")
    modules = load_modules(
        args.weights_path,
        dim=args.dim, embed_dim=args.embed_dim,
        n_blocks=args.n_blocks, n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        gamma_0=args.gamma_0, gamma_1=args.gamma_1,
    )

    # Quick sanity check: unconditional generation of a few tokens
    print("\n--- Sanity check: unconditional generation (short) ---")
    x_uncond = generate_samples(
        modules, guidance_tokens=[], n_samples=2, seq_len=64,
        sampling_timesteps=64, score_temp=args.score_temp,
        embed_dim=args.embed_dim,
    )
    for i in range(2):
        text = tokenizer.decode(x_uncond[i].tolist())
        print(f"  Uncond sample {i}: {text[:200]}")

    # Load XSum data
    print(f"\nLoading XSum articles from {args.xsum_src_path}...")
    articles, references = load_xsum_articles(
        args.xsum_src_path, args.xsum_tgt_path, args.num_samples
    )
    print(f"  Loaded {len(articles)} articles")

    # Generate conditional summaries
    conditional_generate_xsum(
        modules=modules,
        tokenizer=tokenizer,
        articles=articles,
        references=references,
        output_path=args.output_path,
        max_article_tokens=args.max_article_tokens,
        n_samples_per_article=args.n_samples_per_article,
        sampling_timesteps=args.sampling_timesteps,
        guidance_weight=args.guidance_weight,
        score_temp=args.score_temp,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
    )


if __name__ == '__main__':
    main()
