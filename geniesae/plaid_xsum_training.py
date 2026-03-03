"""PyTorch Lightning module for fine-tuning PLAID 1B on XSum.

Implements the VDM variational lower bound loss with low-discrepancy
timestep sampling, self-conditioning, bias warmup, and reconst_bs splitting.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from geniesae.plaid_model import (
    DiffusionModel,
    EmbeddingMatrix,
    GammaBounds,
    NoiseSchedule,
    load_plaid_modules,
)

logger = logging.getLogger(__name__)


def low_discrepancy_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps with low-discrepancy (stratified) sampling.

    Partitions [0, 1] into B equal bins and draws one sample per bin.
    """
    bins = torch.arange(batch_size, device=device, dtype=torch.float64)
    u = torch.rand(batch_size, device=device, dtype=torch.float64)
    t = (bins + u) / batch_size
    return t


def compute_bias_scale(step: int, bias_warmup_steps: int, target: float) -> float:
    """Compute bias scale with linear warmup."""
    if bias_warmup_steps <= 0:
        return target
    return target * min(step / bias_warmup_steps, 1.0)


class PlaidXSumTrainingModule(pl.LightningModule):
    """Lightning module for fine-tuning PLAID on XSum."""

    def __init__(
        self,
        # Model architecture
        dim: int = 2048,
        embed_dim: int = 16,
        n_blocks: int = 24,
        n_heads: int = 32,
        vocab_size: int = 32768,
        gamma_0: float = -3.0,
        gamma_1: float = 6.0,
        # Training hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 1000,
        bias_warmup_steps: int = 5000,
        target_bias_scale: float = 1.0,
        self_cond_prob: float = 0.25,
        reconst_bs: int | None = None,
        clip_quantile: float = 0.95,
        # Layer freezing
        freeze_layers: list[int] | None = None,
        # Gradient checkpointing
        gradient_checkpointing: bool = True,
        # Training mode: "unconditional" or "conditional"
        training_mode: str = "unconditional",
        # Sampling params for epoch-end generation
        sampling_timesteps: int = 256,
        score_temp: float = 0.9,
        guidance_scale: float = 1.0,
        sampler: str = "inpainting",
        num_eval_samples: int = 5,
        log_interval: int = 50,
        noise_schedule_log_interval: int = 500,
        tokenizer_path: str | None = None,
        sample_log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Architecture params
        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.bias_warmup_steps = bias_warmup_steps
        self.target_bias_scale = target_bias_scale
        self.self_cond_prob = self_cond_prob
        self.reconst_bs = reconst_bs
        self.clip_quantile = clip_quantile
        self.freeze_layers = freeze_layers or []
        self.gradient_checkpointing = gradient_checkpointing
        self.training_mode = training_mode
        self.sampling_timesteps = sampling_timesteps
        self.score_temp = score_temp
        self.guidance_scale = guidance_scale
        self.sampler_type = sampler
        self.num_eval_samples = num_eval_samples
        self.log_interval = log_interval
        self.noise_schedule_log_interval = noise_schedule_log_interval
        self.tokenizer_path = tokenizer_path
        self.sample_log_every_n_epochs = sample_log_every_n_epochs
        self._tokenizer = None  # lazy-loaded

        # Build PLAID modules
        self.diffusion_model = DiffusionModel(dim, embed_dim, n_blocks, n_heads, vocab_size).float()
        self.noise_schedule = NoiseSchedule().float()
        self.gamma_bounds = GammaBounds(gamma_0, gamma_1).float()
        self.embedding_matrix = EmbeddingMatrix(vocab_size, embed_dim).float()

        # Current bias scale (updated via warmup)
        self._bias_scale = 0.0
        self._step_time = time.time()

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load four PLAID component weight files via load_plaid_modules."""
        modules = load_plaid_modules(
            weights_path,
            dim=self.dim,
            embed_dim=self.embed_dim,
            n_blocks=self.hparams["n_blocks"],
            n_heads=self.hparams["n_heads"],
            vocab_size=self.vocab_size,
            gamma_0=self.hparams["gamma_0"],
            gamma_1=self.hparams["gamma_1"],
            device="cpu",
        )
        self.diffusion_model.load_state_dict(modules["model"].state_dict())
        self.noise_schedule.load_state_dict(modules["noise_schedule"].state_dict())
        self.gamma_bounds.load_state_dict(modules["gamma_bounds"].state_dict())
        self.embedding_matrix.load_state_dict(modules["embedding_matrix"].state_dict())

        # Apply layer freezing
        self._apply_layer_freezing()

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        logger.info("Loaded pretrained PLAID weights from %s", weights_path)

    def _apply_layer_freezing(self) -> None:
        """Freeze specified transformer blocks."""
        if not self.freeze_layers:
            return
        for idx in self.freeze_layers:
            if idx < 0 or idx >= len(self.diffusion_model.blocks):
                raise IndexError(
                    f"Invalid freeze layer index {idx}. "
                    f"Valid range: [0, {len(self.diffusion_model.blocks) - 1}]"
                )
            for param in self.diffusion_model.blocks[idx].parameters():
                param.requires_grad = False
        logger.info("Froze layers: %s", self.freeze_layers)

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on transformer blocks."""
        for block in self.diffusion_model.blocks:
            block._gradient_checkpointing = True

    def _get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous t to gamma using learned schedule + bounds."""
        gamma_0, gamma_1 = self.gamma_bounds()
        gamma_normalized = self.noise_schedule(t).double()
        return gamma_0 + (gamma_1 - gamma_0) * gamma_normalized

    def _compute_vlb_loss(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        boundary_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the VDM variational lower bound loss.

        Follows the original PLAID implementation (github.com/igul222/plaid):
        - First `reconst_bs` examples get t=0 → CE reconstruction loss only
        - Remaining examples get t>0 → diffusion loss only
        - diffusion_loss = -0.5 * SNR'(t) * MSE  (positive because SNR' < 0)
        - MSE uses mean over seq positions, sum over embed_dim

        Args:
            token_ids: (B, S) token IDs
            attention_mask: (B, S) 1 for real tokens, 0 for padding
            boundary_idx: (B,) position of SEP token per example.

        Returns dict with keys: loss, reconst_loss, diffusion_loss, prior_loss.
        """
        B, S = token_ids.shape
        device = token_ids.device
        embedding_matrix = self.embedding_matrix()  # (vocab_size, embed_dim)

        # Get token embeddings
        x_embed = embedding_matrix[token_ids]  # (B, S, embed_dim)

        # --- Determine reconst_bs (number of examples used for CE loss at t=0) ---
        reconst_bs = self.reconst_bs if self.reconst_bs is not None else max(1, B // 8)
        reconst_bs = min(reconst_bs, B - 1)  # need at least 1 diffusion example

        # --- Sample timesteps following original PLAID ---
        # First reconst_bs examples: t=0 (reconstruction only)
        # Remaining examples: low-discrepancy sampling in (0, 1]
        t = torch.zeros(B, device=device, dtype=torch.float64)
        n_diff = B - reconst_bs
        t[reconst_bs:] = (torch.arange(n_diff, device=device, dtype=torch.float64)
                          + torch.rand(1, device=device, dtype=torch.float64)) / n_diff
        t.requires_grad_(True)

        # Compute gamma — detach for reconst examples (t=0), keep grad for diffusion
        with torch.enable_grad():
            gamma_reconst = self._get_gamma(t[:reconst_bs]).detach()
            gamma_diff = self._get_gamma(t[reconst_bs:])
            gamma_t = torch.cat([gamma_reconst, gamma_diff], dim=0)  # (B,)
            gamma_prime = torch.autograd.grad(
                gamma_t.sum(), t, create_graph=True
            )[0]  # (B,)

        alpha_squared = torch.sigmoid(-gamma_t)
        sigma_squared = torch.sigmoid(gamma_t)
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()

        # SNR'(t) = -exp(-gamma) * gamma'(t)
        # Since gamma increases with t, gamma' > 0, so SNR' < 0
        # and -0.5 * SNR' > 0 → diffusion loss is positive
        snr_prime = -torch.exp(-gamma_t) * gamma_prime  # (B,)

        # gamma_1 quantities for prior loss
        gamma_0_val, gamma_1_val = self.gamma_bounds()
        alpha_1 = torch.sigmoid(-gamma_1_val).sqrt()
        sigma_1 = torch.sigmoid(gamma_1_val).sqrt()

        # Forward diffusion: z_t = alpha * x_embed + sigma * noise
        noise = torch.randn_like(x_embed.double()).float()
        z_t = (alpha[:, None, None] * x_embed.double() + sigma[:, None, None] * noise.double()).float()

        # --- Conditional mode: replace article prefix in z_t ---
        if self.training_mode == "conditional" and boundary_idx is not None:
            for b in range(B):
                bi = boundary_idx[b].item()
                if bi > 0:
                    z_t[b, :bi] = x_embed[b, :bi].float()

        # Build loss mask for conditional mode
        if self.training_mode == "conditional" and boundary_idx is not None:
            loss_mask = torch.zeros(B, S, device=device, dtype=torch.float64)
            for b in range(B):
                bi = boundary_idx[b].item()
                loss_mask[b, bi + 1:] = attention_mask[b, bi + 1:].double()
        else:
            loss_mask = attention_mask.double()

        # Self-conditioning
        x_selfcond = torch.zeros_like(z_t)
        selfcond_mask = torch.zeros(B, device=device)

        sc_mask = torch.rand(B, device=device) < self.self_cond_prob
        if sc_mask.any():
            with torch.no_grad():
                _, x_reconst_sc = self.diffusion_model(
                    z=z_t,
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=self._bias_scale,
                    x_selfcond=x_selfcond,
                    selfcond_mask=torch.zeros(B, device=device),
                )
            x_selfcond = torch.where(
                sc_mask[:, None, None], x_reconst_sc.detach(), x_selfcond
            )
            selfcond_mask = sc_mask.float()

        # Forward pass
        logits, x_reconst = self.diffusion_model(
            z=z_t,
            gamma=gamma_t.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=self._bias_scale,
            x_selfcond=x_selfcond,
            selfcond_mask=selfcond_mask,
        )

        # --- Reconstruction loss (CE) on first reconst_bs examples only ---
        if reconst_bs > 0:
            reconst_logits = logits[:reconst_bs]  # (reconst_bs, S, V)
            reconst_targets = token_ids[:reconst_bs]  # (reconst_bs, S)
            reconst_mask = loss_mask[:reconst_bs]  # (reconst_bs, S)
            # Per-token CE, then mean over valid positions per example, then mean over batch
            ce = F.cross_entropy(
                reconst_logits.reshape(-1, reconst_logits.shape[-1]),
                reconst_targets.reshape(-1),
                reduction="none",
            ).reshape(reconst_bs, S).double()
            # Mean over valid sequence positions per example
            ce_masked = (ce * reconst_mask).sum(dim=1) / reconst_mask.sum(dim=1).clamp(min=1)
            reconst_loss = ce_masked.mean()
        else:
            reconst_loss = torch.tensor(0.0, device=device, dtype=torch.float64)

        # --- Diffusion loss on remaining (B - reconst_bs) examples ---
        if n_diff > 0:
            x_reconst_d = x_reconst[reconst_bs:].double()  # (n_diff, S, E)
            x_embed_d = x_embed[reconst_bs:].double()  # (n_diff, S, E)
            diff_mask = loss_mask[reconst_bs:]  # (n_diff, S)

            # MSE: mean over seq positions (masked), sum over embed_dim
            # Original: (x_embed - x_reconst).pow(2).mean(dim=1).sum(dim=1)
            # With masking: we weight by mask and normalize by valid positions
            mse_per_pos = (x_embed_d - x_reconst_d).pow(2)  # (n_diff, S, E)
            # Apply mask: zero out padding positions
            mse_per_pos = mse_per_pos * diff_mask[:, :, None]
            # Mean over sequence positions (normalized by valid count), sum over embed_dim
            valid_counts = diff_mask.sum(dim=1).clamp(min=1)  # (n_diff,)
            mse = mse_per_pos.sum(dim=1) / valid_counts[:, None]  # (n_diff, E)
            mse = mse.sum(dim=1)  # (n_diff,) — sum over embed_dim

            # diffusion_loss = -0.5 * SNR'(t) * MSE, averaged over diffusion batch
            snr_prime_diff = snr_prime[reconst_bs:]  # (n_diff,)
            per_example_diff_loss = -0.5 * snr_prime_diff * mse  # (n_diff,)
            diffusion_loss = per_example_diff_loss.mean()
        else:
            diffusion_loss = torch.tensor(0.0, device=device, dtype=torch.float64)

        # --- Prior loss: KL(q(z_1|x) || N(0,I)) ---
        # Following original: gaussian_kl(alpha_1 * x_embed, sigma_1, 0, 1)
        # KL = log(1/sigma_1) + (sigma_1^2 + (alpha_1*x_embed)^2) / 2 - 0.5
        # = -log(sigma_1) + sigma_1^2/2 + alpha_1^2 * x_embed^2 / 2 - 0.5
        x_embed_d = x_embed.double()
        prior_kl = (
            -sigma_1.log()
            + 0.5 * sigma_1 ** 2
            + 0.5 * alpha_1 ** 2 * x_embed_d ** 2
            - 0.5
        )  # (B, S, E)
        # Apply loss mask and average: sum over embed_dim, mean over seq
        prior_kl = prior_kl * loss_mask[:, :, None]
        valid_counts_all = loss_mask.sum(dim=1).clamp(min=1)  # (B,)
        prior_loss = (prior_kl.sum(dim=1) / valid_counts_all[:, None]).sum(dim=1).mean()

        total_loss = reconst_loss + diffusion_loss + prior_loss

        return {
            "loss": total_loss,
            "reconst_loss": reconst_loss.detach(),
            "diffusion_loss": diffusion_loss.detach(),
            "prior_loss": prior_loss.detach(),
        }


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute VLB loss with low-discrepancy sampling, self-conditioning, bias warmup."""
        losses = self._compute_vlb_loss(
            batch["token_ids"], batch["attention_mask"], batch.get("boundary_idx")
        )

        # Log at intervals
        if self.global_step % self.log_interval == 0:
            self.log("train/loss", losses["loss"], prog_bar=True, sync_dist=True)
            self.log("train/reconst_loss", losses["reconst_loss"], sync_dist=True)
            self.log("train/diffusion_loss", losses["diffusion_loss"], sync_dist=True)
            self.log("train/prior_loss", losses["prior_loss"], sync_dist=True)
            self.log("train/bias_scale", self._bias_scale, sync_dist=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], sync_dist=True)

            # Throughput
            now = time.time()
            elapsed = now - self._step_time
            if elapsed > 0:
                throughput = batch["token_ids"].shape[0] / elapsed
                self.log("train/throughput", throughput, sync_dist=True)
            self._step_time = now

            # GPU memory
            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / 1e6
                self.log("train/gpu_memory_mb", mem_mb, sync_dist=True)

        return losses["loss"]

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update bias_scale per bias warmup schedule."""
        self._bias_scale = compute_bias_scale(
            self.global_step, self.bias_warmup_steps, self.target_bias_scale
        )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Compute and log validation loss."""
        losses = self._compute_vlb_loss(
            batch["token_ids"], batch["attention_mask"], batch.get("boundary_idx")
        )
        self.log("val/loss", losses["loss"], prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict:
        """AdamW with linear warmup + cosine decay. Excludes frozen layers."""
        # Collect trainable parameters only
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        # Linear warmup + cosine decay
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            # Cosine decay after warmup
            total = self.trainer.estimated_stepping_batches
            progress = (step - self.warmup_steps) / max(total - self.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self) -> None:
        """Generate sample summaries at end of each validation epoch."""
        if not hasattr(self, "trainer") or self.trainer is None:
            return
        if self.trainer.sanity_checking:
            return

        # Log noise schedule curve
        self._log_noise_schedule()

        # Generate and log text samples (only on rank 0, every N epochs)
        if self.global_rank == 0 and (self.current_epoch + 1) % self.sample_log_every_n_epochs == 0:
            try:
                self._generate_and_log_samples()
            except Exception as e:
                logger.warning("Sample generation failed: %s", e)

    def _generate_and_log_samples(self) -> None:
        """Generate text samples using InpaintingSampler and log to wandb + stdout."""
        import tokenizers as _tok

        from geniesae.plaid_samplers import InpaintingSampler

        # Lazy-load tokenizer
        if self._tokenizer is None:
            tok_path = self.tokenizer_path
            if tok_path is None:
                # Try to get from datamodule
                dm = getattr(self.trainer, "datamodule", None)
                if dm is not None and getattr(dm, "tokenizer_path", None):
                    tok_path = dm.tokenizer_path
            if tok_path is None:
                logger.warning("No tokenizer_path available — skipping sample generation")
                return
            self._tokenizer = _tok.Tokenizer.from_file(tok_path)
            logger.info("Loaded tokenizer from %s for sample generation", tok_path)

        tokenizer = self._tokenizer

        # Get validation datamodule for samples
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None or dm.val_dataset is None:
            logger.warning("No validation dataset — skipping sample generation")
            return

        val_dataset = dm.val_dataset
        sep_id = dm.sep_token_id
        seq_len = dm.seq_len
        n_samples = min(self.num_eval_samples, len(val_dataset))

        # Build sampler
        prefix_mode = "clean" if self.training_mode == "conditional" else "renoised"
        sampler = InpaintingSampler(
            model=self.diffusion_model,
            noise_schedule=self.noise_schedule,
            gamma_bounds=self.gamma_bounds,
            embedding_matrix=self.embedding_matrix,
            sampling_timesteps=self.sampling_timesteps,
            score_temp=self.score_temp,
            prefix_mode=prefix_mode,
        )

        # Collect samples
        rows = []
        epoch = self.current_epoch
        logger.info("Generating %d text samples (epoch %d, %d steps, temp=%.2f)...",
                     n_samples, epoch, self.sampling_timesteps, self.score_temp)

        for i in range(n_samples):
            sample = val_dataset[i]
            token_ids = sample["token_ids"].unsqueeze(0).to(self.device)
            boundary_idx = sample["boundary_idx"].unsqueeze(0).to(self.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)

            bi = boundary_idx[0].item()
            all_ids = token_ids[0].tolist()
            real_len = int(attention_mask[0].sum().item())

            # Decode article and reference
            article_ids = all_ids[:bi]
            summary_start = bi + 1
            ref_ids = all_ids[summary_start:real_len]
            article_text = tokenizer.decode(article_ids)
            ref_text = tokenizer.decode(ref_ids) if ref_ids else "(none)"

            # Generate
            with torch.no_grad():
                gen_ids = sampler.sample(
                    article_token_ids=token_ids,
                    boundary_idx=boundary_idx,
                    seq_len=seq_len,
                )

            # Decode generated summary
            gen_all = gen_ids[0].tolist()
            gen_summary_ids = gen_all[summary_start:]
            gen_clean = []
            for tid in gen_summary_ids:
                if tid == sep_id:
                    break
                gen_clean.append(tid)
            gen_text = tokenizer.decode(gen_clean) if gen_clean else "(empty)"

            rows.append({
                "article": article_text[:300],
                "reference": ref_text,
                "generated": gen_text,
            })

            logger.info(
                "--- Sample %d (epoch %d) ---\n"
                "ARTICLE: %.200s...\n"
                "REFERENCE: %s\n"
                "GENERATED: %s",
                i + 1, epoch, article_text, ref_text, gen_text,
            )

        # Log to wandb as a Table
        if self.logger is not None:
            try:
                import wandb

                table = wandb.Table(
                    columns=["epoch", "idx", "article", "reference", "generated"],
                    data=[
                        [epoch, i, r["article"], r["reference"], r["generated"]]
                        for i, r in enumerate(rows)
                    ],
                )
                self.logger.experiment.log(
                    {"val/generated_samples": table},
                    step=self.global_step,
                )
            except Exception as e:
                logger.warning("Failed to log wandb table: %s", e)

    def _log_noise_schedule(self) -> None:
        """Log the learned noise schedule gamma(t) curve."""
        if self.logger is None:
            return
        try:
            import wandb

            t_vals = torch.linspace(0, 1, 100, device=self.device, dtype=torch.float64)
            with torch.no_grad():
                gamma_vals = self._get_gamma(t_vals).cpu().numpy()
            t_np = t_vals.cpu().numpy()

            table = wandb.Table(
                data=[[float(t), float(g)] for t, g in zip(t_np, gamma_vals)],
                columns=["t", "gamma"],
            )
            self.logger.experiment.log(
                {"noise_schedule/gamma_curve": wandb.plot.line(
                    table, "t", "gamma", title="Noise Schedule γ(t)"
                )},
                step=self.global_step,
            )
        except Exception:
            pass  # WandB not available or logging failed
