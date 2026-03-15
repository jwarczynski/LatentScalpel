"""Conditional generation samplers for PLAID fine-tuned on XSum.

Two strategies:
1. InpaintingSampler: clamp article prefix at each reverse step
2. GradientGuidanceSampler: gradient-based token guidance on article prefix
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geniesae.plaid_model import (
    DiffusionModel,
    EmbeddingMatrix,
    GammaBounds,
    NoiseSchedule,
)


class InpaintingSampler:
    """Conditional generation via inpainting: clamp article prefix at each step.

    Supports two prefix modes to match the training objective:
    - "renoised" (default): re-noise article embeddings to the current noise
      level at each step. Matches unconditional training mode.
    - "clean": replace article positions with clean (noiseless) embeddings.
      Matches conditional training mode where the model was trained with
      clean article context.
    """

    def __init__(
        self,
        model: DiffusionModel,
        noise_schedule: NoiseSchedule,
        gamma_bounds: GammaBounds,
        embedding_matrix: EmbeddingMatrix,
        sampling_timesteps: int = 256,
        score_temp: float = 0.9,
        prefix_mode: str = "renoised",
    ) -> None:
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma_bounds = gamma_bounds
        self.embedding_matrix_module = embedding_matrix
        self.sampling_timesteps = sampling_timesteps
        self.score_temp = score_temp
        self.prefix_mode = prefix_mode

    def _get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        gamma_0, gamma_1 = self.gamma_bounds()
        gamma_normalized = self.noise_schedule(t).double()
        return gamma_0 + (gamma_1 - gamma_0) * gamma_normalized

    @torch.no_grad()
    def sample(
        self,
        article_token_ids: torch.Tensor,  # (batch, article_len)
        boundary_idx: torch.Tensor,       # (batch,) position of SEP
        seq_len: int,
    ) -> torch.Tensor:
        """Generate sequences with article prefix preserved via inpainting.

        Returns: generated token_ids (batch, seq_len).
        """
        B = article_token_ids.shape[0]
        device = article_token_ids.device
        embedding_matrix = self.embedding_matrix_module()  # (vocab, embed_dim)
        embed_dim = embedding_matrix.shape[1]

        # Get article embeddings
        article_embeds = embedding_matrix[article_token_ids]  # (B, article_len, embed_dim)

        # Initialize z_T from noise
        z = torch.randn(B, seq_len, embed_dim, device=device).double()
        x_selfcond = torch.zeros(B, seq_len, embed_dim, device=device)

        T = self.sampling_timesteps
        for step in range(T):
            t_val = 1.0 - step / T
            s_val = 1.0 - (step + 1) / T
            t = torch.full((B,), t_val, device=device, dtype=torch.float64)
            s = torch.full((B,), s_val, device=device, dtype=torch.float64)

            gamma_t = self._get_gamma(t)
            alpha_sq_t = torch.sigmoid(-gamma_t)
            sigma_sq_t = torch.sigmoid(gamma_t)
            alpha_t = alpha_sq_t.sqrt()
            sigma_t = sigma_sq_t.sqrt()

            # Clamp article prefix
            for b in range(B):
                bi = boundary_idx[b].item()
                if bi > 0:
                    if self.prefix_mode == "clean":
                        # Clean embeddings — matches conditional training mode
                        z[b, :bi] = article_embeds[b, :bi].double()
                    else:
                        # Re-noised — matches unconditional training mode
                        noise_article = torch.randn(bi, embed_dim, device=device, dtype=torch.float64)
                        z[b, :bi] = (
                            alpha_t[b] * article_embeds[b, :bi].double()
                            + sigma_t[b] * noise_article
                        )

            # Forward pass
            _, x_reconst = self.model(
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.0,
                x_selfcond=x_selfcond,
                selfcond_mask=torch.ones(B, device=device),
            )
            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()

            # Score temperature
            epsilon_pred = (z - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
            epsilon_pred /= self.score_temp
            x_reconst = (z - sigma_t[:, None, None] * epsilon_pred) / alpha_t[:, None, None]

            # Reverse step (stochastic)
            if s_val > 0:
                gamma_s = self._get_gamma(s)
                alpha_sq_s = torch.sigmoid(-gamma_s)
                c = -torch.expm1(gamma_s - gamma_t)
                z_new = (
                    (1 - c) * alpha_sq_s.sqrt()[:, None, None]
                    / alpha_sq_t.sqrt()[:, None, None] * z
                )
                z_new += c * alpha_sq_s.sqrt()[:, None, None] * x_reconst
                z_new += (c * (1 - alpha_sq_s)).sqrt()[:, None, None] * torch.randn_like(z)
                z = z_new
            else:
                z = x_reconst

        # Decode: argmax over logits — use gamma_t from last step (gamma at t≈0),
        # NOT literal zero. gamma(0) = gamma_0 ≈ -3, which affects z_variance
        # normalization and bias scaling in the model forward pass.
        z_final = z.float()
        logits, _ = self.model(
            z=z_final,
            gamma=gamma_t.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.0,
            x_selfcond=x_selfcond,
            selfcond_mask=torch.ones(B, device=device),
        )
        token_ids = logits.argmax(dim=-1)  # (B, seq_len)

        # Restore article prefix
        for b in range(B):
            bi = boundary_idx[b].item()
            if bi > 0:
                token_ids[b, :bi] = article_token_ids[b, :bi]

        return token_ids


class TokenGuidanceSampler:
    """Zero-shot conditional generation via token guidance (PLAID paper Section 6.3).

    Uses the pretrained unconditional model with classifier guidance derived
    from the model's own token predictions. At each reverse step, computes
    gradients of log p(prefix tokens | z_t) w.r.t. z_t and adds them to
    x_reconst with the paper's noise-schedule-dependent scaling:
        x_reconst += guidance_grad * sigma^2_t / alpha_t

    This matches the authors' sample.py implementation exactly.
    """

    def __init__(
        self,
        model: DiffusionModel,
        noise_schedule: NoiseSchedule,
        gamma_bounds: GammaBounds,
        embedding_matrix: EmbeddingMatrix,
        sampling_timesteps: int = 1024,
        score_temp: float = 0.9,
        guidance_weight: float = 2.0,
    ) -> None:
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma_bounds = gamma_bounds
        self.embedding_matrix_module = embedding_matrix
        self.sampling_timesteps = sampling_timesteps
        self.score_temp = score_temp
        self.guidance_weight = guidance_weight

    def _get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        gamma_0, gamma_1 = self.gamma_bounds()
        gamma_normalized = self.noise_schedule(t).double()
        return gamma_0 + (gamma_1 - gamma_0) * gamma_normalized

    def sample(
        self,
        prefix_token_ids: list[list[int]],
        seq_len: int,
    ) -> torch.Tensor:
        """Generate sequences conditioned on prefix tokens via token guidance.

        Args:
            prefix_token_ids: List of token ID lists, one per sample in batch.
                Each token is guided at its corresponding position.
            seq_len: Total sequence length to generate.

        Returns: generated token_ids (batch, seq_len).
        """
        device = next(self.model.parameters()).device
        B = len(prefix_token_ids)
        embedding_matrix = self.embedding_matrix_module()
        embed_dim = embedding_matrix.shape[1]

        # Build guidance spec: (token_id, weight, position) per prefix token
        guidance_specs: list[list[tuple[int, float, int]]] = []
        for b in range(B):
            specs = [
                (tid, self.guidance_weight, pos)
                for pos, tid in enumerate(prefix_token_ids[b])
            ]
            guidance_specs.append(specs)

        z = torch.randn((B, seq_len, embed_dim), device=device, dtype=torch.float64)
        x_selfcond = torch.zeros((B, seq_len, embed_dim), device=device, dtype=torch.float32)

        T = self.sampling_timesteps
        for step in range(T):
            t_val = 1.0 - step / T
            s_val = 1.0 - (step + 1) / T
            t = torch.full((B,), t_val, device=device, dtype=torch.float64)
            s = torch.full((B,), s_val, device=device, dtype=torch.float64)

            gamma_t = self._get_gamma(t)
            alpha_sq_t = torch.sigmoid(-gamma_t)
            sigma_sq_t = torch.sigmoid(gamma_t)
            alpha_t = alpha_sq_t.sqrt()
            sigma_t = sigma_sq_t.sqrt()

            # Need gradients for guidance
            z_input = z.float().detach().requires_grad_(True)

            with torch.enable_grad():
                logits, x_reconst = self.model(
                    z=z_input,
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.0,
                    x_selfcond=x_selfcond,
                    selfcond_mask=torch.ones(B, device=device),
                )

                # Compute sum of log-probs for guided tokens
                log_probs = F.log_softmax(logits.float(), dim=2)  # (B, seq_len, vocab)
                sum_logp = torch.zeros(1, device=device, dtype=torch.float32)
                for b in range(B):
                    for token_id, weight, position in guidance_specs[b]:
                        if position < seq_len:
                            sum_logp = sum_logp + weight * log_probs[b, position, token_id]

                guidance_grad = torch.autograd.grad(sum_logp, [z_input])[0]

            z_input.requires_grad = False
            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()

            # Score temperature
            epsilon_pred = (z - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
            epsilon_pred /= self.score_temp
            x_reconst = (z - sigma_t[:, None, None] * epsilon_pred) / alpha_t[:, None, None]

            # Apply guidance with paper's scaling: grad * sigma^2_t / alpha_t
            x_reconst = x_reconst + guidance_grad.double() * sigma_sq_t[:, None, None] / alpha_t[:, None, None]

            # Recompute epsilon after guidance
            epsilon_pred = (z - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]

            # Reverse step (stochastic, VDM Appendix A.4 eqn 33)
            if s_val > 0:
                gamma_s = self._get_gamma(s)
                alpha_sq_s = torch.sigmoid(-gamma_s)
                c = -torch.expm1(gamma_s - gamma_t)
                z_new = (
                    (1 - c) * alpha_sq_s.sqrt()[:, None, None]
                    / alpha_sq_t.sqrt()[:, None, None] * z
                )
                z_new += c * alpha_sq_s.sqrt()[:, None, None] * x_reconst
                z_new += (c * (1 - alpha_sq_s)).sqrt()[:, None, None] * torch.randn_like(z)
                z = z_new.detach()
            else:
                z = x_reconst.detach()

        # Final decode — use gamma_t from last iteration (gamma at t≈0),
        # NOT literal zero. gamma(0) = gamma_0 ≈ -3.
        z_final = z.float()
        with torch.no_grad():
            logits, _ = self.model(
                z=z_final,
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.0,
                x_selfcond=x_selfcond,
                selfcond_mask=torch.ones(B, device=device),
            )
        token_ids = logits.argmax(dim=-1)  # (B, seq_len)
        return token_ids


class GradientGuidanceSampler:
    """Conditional generation via gradient-based token guidance on article prefix."""

    def __init__(
        self,
        model: DiffusionModel,
        noise_schedule: NoiseSchedule,
        gamma_bounds: GammaBounds,
        embedding_matrix: EmbeddingMatrix,
        sampling_timesteps: int = 256,
        score_temp: float = 0.9,
        guidance_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma_bounds = gamma_bounds
        self.embedding_matrix_module = embedding_matrix
        self.sampling_timesteps = sampling_timesteps
        self.score_temp = score_temp
        self.guidance_scale = guidance_scale

    def _get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        gamma_0, gamma_1 = self.gamma_bounds()
        gamma_normalized = self.noise_schedule(t).double()
        return gamma_0 + (gamma_1 - gamma_0) * gamma_normalized

    def sample(
        self,
        article_token_ids: torch.Tensor,  # (batch, article_len)
        boundary_idx: torch.Tensor,       # (batch,) position of SEP
        seq_len: int,
    ) -> torch.Tensor:
        """Generate sequences with gradient-based token guidance.

        At each reverse step:
        1. Enable gradients on z_t
        2. Forward pass → logits
        3. log_softmax at article positions, sum log-probs of ground-truth tokens
        4. Backprop to get grad w.r.t. z_t
        5. Add (guidance_scale * grad) to x_reconst
        6. Reverse step with modified x_reconst

        Returns: generated token_ids (batch, seq_len).
        """
        B = article_token_ids.shape[0]
        device = article_token_ids.device
        embedding_matrix = self.embedding_matrix_module()
        embed_dim = embedding_matrix.shape[1]

        z = torch.randn(B, seq_len, embed_dim, device=device).double()
        x_selfcond = torch.zeros(B, seq_len, embed_dim, device=device)

        T = self.sampling_timesteps
        for step in range(T):
            t_val = 1.0 - step / T
            s_val = 1.0 - (step + 1) / T
            t = torch.full((B,), t_val, device=device, dtype=torch.float64)
            s = torch.full((B,), s_val, device=device, dtype=torch.float64)

            gamma_t = self._get_gamma(t)
            alpha_sq_t = torch.sigmoid(-gamma_t)
            sigma_sq_t = torch.sigmoid(gamma_t)
            alpha_t = alpha_sq_t.sqrt()
            sigma_t = sigma_sq_t.sqrt()

            # Enable gradients on z for guidance
            z_input = z.float().detach().requires_grad_(True)

            # Forward pass
            logits, x_reconst = self.model(
                z=z_input,
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.0,
                x_selfcond=x_selfcond.float(),
                selfcond_mask=torch.ones(B, device=device),
            )

            # Compute guidance: log-prob of article tokens at article positions
            log_probs = F.log_softmax(logits, dim=-1)  # (B, seq_len, vocab)
            guidance_loss = torch.zeros(1, device=device)
            for b in range(B):
                bi = boundary_idx[b].item()
                if bi > 0:
                    article_len = min(bi, article_token_ids.shape[1])
                    target_ids = article_token_ids[b, :article_len]
                    pos_log_probs = log_probs[b, :article_len]
                    guidance_loss = guidance_loss + pos_log_probs[
                        torch.arange(article_len, device=device), target_ids
                    ].sum()

            # Backprop for gradient w.r.t. z
            guidance_loss.backward()
            z_grad = z_input.grad.double()

            # Self-conditioning update
            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()

            # Apply guidance: add scaled gradient to x_reconst
            x_reconst = x_reconst + self.guidance_scale * z_grad

            # Score temperature
            epsilon_pred = (z - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
            epsilon_pred /= self.score_temp
            x_reconst = (z - sigma_t[:, None, None] * epsilon_pred) / alpha_t[:, None, None]

            # Reverse step (stochastic)
            if s_val > 0:
                gamma_s = self._get_gamma(s)
                alpha_sq_s = torch.sigmoid(-gamma_s)
                c = -torch.expm1(gamma_s - gamma_t)
                z_new = (
                    (1 - c) * alpha_sq_s.sqrt()[:, None, None]
                    / alpha_sq_t.sqrt()[:, None, None] * z
                )
                z_new += c * alpha_sq_s.sqrt()[:, None, None] * x_reconst
                z_new += (c * (1 - alpha_sq_s)).sqrt()[:, None, None] * torch.randn_like(z)
                z = z_new.detach()
            else:
                z = x_reconst.detach()

        # Decode: argmax over logits — use gamma_t from last step, not literal zero
        z_final = z.float()
        with torch.no_grad():
            logits, _ = self.model(
                z=z_final,
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.0,
                x_selfcond=x_selfcond.float(),
                selfcond_mask=torch.ones(B, device=device),
            )
        token_ids = logits.argmax(dim=-1)

        return token_ids
