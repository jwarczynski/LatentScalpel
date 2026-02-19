"""GENIE Diffusion Language Model.

Vendored from https://github.com/microsoft/ProphetNet/tree/master/GENIE/model
with minimal modifications for standalone use (removed relative imports,
consolidated into a single file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal)
# ---------------------------------------------------------------------------

def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of N indices (may be fractional).
        dim: Dimension of the output embedding.
        max_period: Controls the minimum frequency.

    Returns:
        Tensor of shape ``(N, dim)``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# ---------------------------------------------------------------------------
# Cross-Attention Transformer components
# ---------------------------------------------------------------------------


class GEGLU(nn.Module):
    """Gated linear unit with GELU activation (https://arxiv.org/abs/2002.05202)."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """Approximate GELU (https://arxiv.org/abs/1606.08415)."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    """Feed-forward block with configurable activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu":
            act = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act = ApproximateGELU(dim, inner_dim)
        else:
            raise ValueError(f"Unknown activation_fn: {activation_fn}")

        self.net = nn.ModuleList([act, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):
    """Multi-head (cross-)attention layer."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        return tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)

    def reshape_batch_dim_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)

    def forward(
        self, hidden_states: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and feed-forward."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
            dropout=dropout, bias=attention_bias,
        )
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.attn2 = CrossAttention(
            query_dim=dim, cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads, dim_head=attention_head_dim,
            dropout=dropout, bias=attention_bias,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden_states: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# GENIE Diffusion Language Model (unconditional / LM)
# ---------------------------------------------------------------------------


class Diffusion_LM(nn.Module):
    """GENIE diffusion language model with a BERT-based transformer backbone.

    The "decoder" layers that we hook for SAE analysis live at
    ``self.input_transformers.layer`` — a ``nn.ModuleList`` of BERT encoder
    layers.

    Args:
        in_channels: Word embedding dimension.
        model_channels: Base channel count (used for timestep embedding).
        out_channels: Output dimension (same as ``in_channels`` unless
            learning sigma).
        dropout: Dropout probability.
        config_name: HuggingFace BERT config name (e.g. ``'bert-base-uncased'``).
        vocab_size: Vocabulary size.
        init_pretrained: Whether to initialise the transformer from a
            pretrained BERT checkpoint.
        logits_mode: 1 for linear head, 2 for distance-based scoring.
        token_emb_type: ``'pretrain'`` to copy BERT embeddings, ``'random'``
            for random init.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        dropout: float = 0,
        config_name: str = "bert-base-uncased",
        vocab_size: int | None = None,
        init_pretrained: bool = True,
        logits_mode: int = 1,
        token_emb_type: str = "pretrain",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.logits_mode = logits_mode
        self.init_pretrained = init_pretrained
        self.token_emb_type = token_emb_type

        config = BertConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = dropout

        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, in_channels)

        # Position embedding
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        if token_emb_type == "pretrain":
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding.weight = temp_bert.embeddings.word_embeddings.weight
            self.position_embeddings.weight = temp_bert.embeddings.position_embeddings.weight
        elif token_emb_type == "random":
            pass
        else:
            raise NotImplementedError(f"Unknown token_emb_type: {token_emb_type}")

        # LM head (weight-tied with word embedding)
        if logits_mode == 2:
            self.lm_head = nn.Linear(in_channels, vocab_size, bias=True)
        else:
            self.lm_head = nn.Linear(in_channels, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # Input projection
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Layer norm + dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)

        # Transformer backbone — this is the "decoder" we hook for SAE work.
        # Layers are at ``self.input_transformers.layer``.
        if init_pretrained:
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
        else:
            temp_bert = BertModel(config)
            self.input_transformers = temp_bert.encoder

        # Output projection
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels),
        )

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr: torch.Tensor) -> torch.Tensor:
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(
                self.lm_head.weight, text_emb_t
            )
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1)
            )
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        src_ids: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs = self.dropout_layer(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=attention_mask
        ).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h


# ---------------------------------------------------------------------------
# GENIE Cross-Attention Diffusion LM (seq2seq / summarisation)
# ---------------------------------------------------------------------------


class CrossAttention_Diffusion_LM(nn.Module):
    """GENIE seq2seq diffusion model with cross-attention transformer blocks.

    Used for conditional generation tasks like XSum summarisation.  The
    "decoder" layers that we hook for SAE analysis live at
    ``self.transformer_blocks`` — a ``nn.ModuleList`` of
    :class:`BasicTransformerBlock`.

    Args:
        in_channels: Word embedding dimension.
        model_channels: Base channel count (used for timestep embedding).
        out_channels: Output dimension.
        dropout: Dropout probability.
        config_name: HuggingFace BERT config name.
        vocab_size: Vocabulary size.
        init_pretrained: Whether to initialise from pretrained BERT.
        logits_mode: 1 for linear head, 2 for distance-based scoring.
        token_emb_type: ``'pretrain'`` or ``'random'``.
        fix_encoder: Whether to freeze the passage encoder.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        dropout: float = 0,
        config_name: str = "bert-base-uncased",
        vocab_size: int | None = None,
        init_pretrained: bool = True,
        logits_mode: int = 1,
        token_emb_type: str = "pretrain",
        fix_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.logits_mode = logits_mode
        self.init_pretrained = init_pretrained
        self.token_emb_type = token_emb_type
        self.fix_encoder = fix_encoder

        # Passage encoder (6-layer BERT)
        cfg = BertConfig.from_pretrained(config_name)
        cfg.num_hidden_layers = 6
        self.passage_encoder = BertModel.from_pretrained(config_name, config=cfg)

        config = BertConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = dropout

        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, in_channels)

        # Position embedding
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # LM head
        if logits_mode == 2:
            self.lm_head = nn.Linear(in_channels, vocab_size, bias=True)
        else:
            self.lm_head = nn.Linear(in_channels, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # Input projection
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Layer norm + dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)

        # Cross-attention transformer blocks (6 layers)
        config.num_hidden_layers = 6
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.hidden_size // config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    cross_attention_dim=config.hidden_size,
                    activation_fn="geglu",
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Output projection
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels),
        )

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr: torch.Tensor) -> torch.Tensor:
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(
                self.lm_head.weight, text_emb_t
            )
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1)
            )
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        answer_id: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        src_ids: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb.unsqueeze(1).expand(-1, seq_length, -1)
        )
        hidden_states = self.dropout_layer(self.LayerNorm(emb_inputs))

        if self.fix_encoder:
            with torch.no_grad():
                out = self.passage_encoder(
                    input_ids=src_input_ids, attention_mask=src_attention_mask
                )
                passage_hidden = out.last_hidden_state
        else:
            out = self.passage_encoder(
                input_ids=src_input_ids, attention_mask=src_attention_mask
            )
            passage_hidden = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)

        if answer_id is not None:
            answer_hidden_states = hidden_states.clone()
            answer_out = self.passage_encoder(input_ids=answer_id, attention_mask=answer_mask)
            answer_hidden = answer_out.last_hidden_state + 0 * answer_out.pooler_output.unsqueeze(1)
            for block in self.transformer_blocks:
                answer_hidden_states = block(answer_hidden_states, answer_hidden)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, passage_hidden)

        if answer_id is not None:
            hidden_states = hidden_states + answer_hidden_states

        h = self.output_down_proj(hidden_states)
        h = h.type(x.dtype)
        return h

# ---------------------------------------------------------------------------
# Diffusion noise schedule helper (vendored from GENIE / improved-diffusion)
# ---------------------------------------------------------------------------


def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999):
    """Create a beta schedule from a cumulative alpha_bar function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int):
    """Get a pre-defined beta schedule by name.

    GENIE uses ``"sqrt"`` with 2000 diffusion steps.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


class DiffusionHelper:
    """Diffusion forward and reverse process helper.

    Pre-computes the noise schedule so we can quickly produce noised
    embeddings at any timestep via :meth:`q_sample`, and run the
    reverse denoising chain via :meth:`p_sample`.

    The model is assumed to predict ``x_0`` directly (not noise).

    Args:
        num_timesteps: Total number of diffusion steps (GENIE default: 2000).
        schedule_name: Beta schedule name (GENIE default: ``"sqrt"``).
    """

    def __init__(self, num_timesteps: int = 2000, schedule_name: str = "sqrt") -> None:
        betas = np.array(get_named_beta_schedule(schedule_name, num_timesteps), dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = num_timesteps

        # Forward process
        self.sqrt_alphas_cumprod = torch.from_numpy(
            np.sqrt(alphas_cumprod)
        ).float()
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 - alphas_cumprod)
        ).float()

        # Reverse process — posterior q(x_{t-1} | x_t, x_0)
        # posterior_mean = coeff1 * x_0_pred + coeff2 * x_t
        posterior_mean_coeff1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coeff2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.posterior_mean_coeff1 = torch.from_numpy(posterior_mean_coeff1).float()
        self.posterior_mean_coeff2 = torch.from_numpy(posterior_mean_coeff2).float()
        self.posterior_variance = torch.from_numpy(posterior_variance).float()
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.maximum(posterior_variance, 1e-20))
        ).float()

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Index into a schedule tensor and reshape for broadcasting."""
        schedule = schedule.to(x.device)
        dims_to_add = x.dim() - 1
        shape = (-1,) + (1,) * dims_to_add
        return schedule[t].reshape(shape)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0) — add noise to *x_start* at timestep *t*.

        Args:
            x_start: Clean embeddings, shape ``(batch, seq_len, dim)``.
            t: 1-D integer tensor of timestep indices, shape ``(batch,)``.
            noise: Optional pre-sampled noise (same shape as *x_start*).

        Returns:
            Noised tensor of same shape as *x_start*.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    def q_posterior_mean_variance(
        self,
        x_start_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance q(x_{t-1} | x_t, x_0_pred).

        Args:
            x_start_pred: Model's prediction of x_0, shape ``(batch, seq, dim)``.
            x_t: Current noised latents, shape ``(batch, seq, dim)``.
            t: 1-D integer tensor of timestep indices, shape ``(batch,)``.

        Returns:
            Tuple of (posterior_mean, posterior_variance).
        """
        coeff1 = self._extract(self.posterior_mean_coeff1, t, x_t)
        coeff2 = self._extract(self.posterior_mean_coeff2, t, x_t)
        posterior_mean = coeff1 * x_start_pred + coeff2 * x_t
        posterior_var = self._extract(self.posterior_variance, t, x_t)
        return posterior_mean, posterior_var

    def p_sample(
        self,
        x_start_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse diffusion step: sample x_{t-1} given model output.

        Uses the posterior q(x_{t-1} | x_t, x_0_pred) with the model's
        x_0 prediction. At t=0, returns the mean (no noise added).

        Args:
            x_start_pred: Model's prediction of x_0.
            x_t: Current noised latents.
            t: 1-D integer tensor of timestep indices.

        Returns:
            Denoised latents x_{t-1}.
        """
        mean, var = self.q_posterior_mean_variance(x_start_pred, x_t, t)
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float()
        dims_to_add = x_t.dim() - 1
        nonzero_mask = nonzero_mask.reshape(-1, *([1] * dims_to_add))
        return mean + nonzero_mask * torch.sqrt(var.clamp(min=1e-20)) * noise

