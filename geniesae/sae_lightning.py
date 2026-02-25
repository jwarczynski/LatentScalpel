"""PyTorch Lightning module wrapping the Top-K SAE for training."""

from __future__ import annotations

import pytorch_lightning as L
import torch
import torch.nn.functional as F

from geniesae.sae import TopKSAE


class SAELightningModule(L.LightningModule):
    """Lightning wrapper for Top-K SAE training.

    Handles:
    - MSE reconstruction loss (training_step)
    - Decoder weight normalization after backward (on_after_backward)
    - K-annealing schedule (on_train_batch_end)
    - Dead feature tracking
    - FVE validation metric (validation_step)

    Args:
        sae: The TopKSAE model instance.
        learning_rate: Learning rate for Adam optimizer.
        k_target: Target k value after annealing.
        k_start: Starting k value (typically 4 * k_target).
        k_anneal_steps: Number of steps to anneal k from k_start to k_target.
        dead_feature_window: Number of tokens for dead feature tracking window.
        resample_dead: Whether to resample dead features.
    """

    def __init__(
        self,
        sae: TopKSAE,
        learning_rate: float = 1e-4,
        k_target: int = 32,
        k_start: int = 128,
        k_anneal_steps: int = 1000,
        dead_feature_window: int = 10000,
        resample_dead: bool = True,
        dead_feature_strategy: str = "none",
        aux_loss_coeff: float = 1e-3,
    ) -> None:
        super().__init__()
        self.sae = sae
        self.learning_rate = learning_rate
        self.k_target = k_target
        self.k_start = k_start
        self.k_anneal_steps = k_anneal_steps
        self.dead_feature_window = dead_feature_window
        self.resample_dead = resample_dead
        self.dead_feature_strategy = dead_feature_strategy
        self.aux_loss_coeff = aux_loss_coeff

        self.save_hyperparameters(ignore=["sae"])

        # Dead feature tracking buffers
        self.register_buffer(
            "feature_activation_count",
            torch.zeros(sae.dictionary_size, dtype=torch.long),
        )
        self.register_buffer(
            "tokens_since_reset",
            torch.tensor(0, dtype=torch.long),
        )

        # Buffer to store high-loss examples for resampling
        self._resample_batch: torch.Tensor | None = None
        self._pending_resample_mask: torch.Tensor | None = None

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat, z = self.sae(x)
        loss = F.mse_loss(x_hat, x)

        # Compute metrics
        l0 = (z != 0).float().sum(dim=-1).mean()

        # Fraction of Variance Explained
        x_mean = x.mean(dim=0, keepdim=True)
        ss_res = (x - x_hat).pow(2).sum()
        ss_tot = (x - x_mean).pow(2).sum()
        fve = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

        self.log("train/mse_loss", loss, prog_bar=True)
        self.log("train/fve", fve, prog_bar=True)
        self.log("train/l0_sparsity", l0)

        # Update dead feature tracking — accumulate first, then report
        # and reset only when the window is full.
        active_features = (z != 0).any(dim=0)
        self.feature_activation_count += active_features.long()
        self.tokens_since_reset += x.shape[0]

        # Aux loss: encourage dead features to produce non-zero pre-activations.
        # Applies a small MSE penalty using only the dead features' decoder columns.
        if self.dead_feature_strategy == "aux_loss":
            dead_mask = self.feature_activation_count == 0
            if dead_mask.any() and self.tokens_since_reset > 0:
                # Reconstruct using only dead features' pre-activations
                pre_act = F.linear(x - self.sae.b_dec, self.sae.W_enc)
                dead_pre_act = pre_act[:, dead_mask]
                # Soft top-k on dead features only to get a sparse signal
                n_dead = dead_mask.sum().item()
                k_aux = min(self.k_target, n_dead)
                if k_aux > 0:
                    topk_vals, topk_idx = torch.topk(dead_pre_act, k_aux, dim=-1)
                    sparse_dead = torch.zeros_like(dead_pre_act)
                    sparse_dead.scatter_(-1, topk_idx, F.relu(topk_vals))
                    # Decode through dead columns only
                    W_dec_dead = self.sae.W_dec[:, dead_mask]
                    x_hat_aux = F.linear(sparse_dead, W_dec_dead)
                    aux_loss = F.mse_loss(x_hat_aux, x - x_hat.detach())
                    loss = loss + self.aux_loss_coeff * aux_loss
                    self.log("train/aux_loss", aux_loss)

        # Store high-loss examples for potential resampling
        if self.dead_feature_strategy == "resample":
            with torch.no_grad():
                per_sample_loss = (x - x_hat).pow(2).mean(dim=-1)
                self._resample_batch = x[per_sample_loss.topk(min(256, x.shape[0])).indices].detach()

        if self.tokens_since_reset >= self.dead_feature_window:
            dead_mask = self.feature_activation_count == 0
            dead_frac = dead_mask.float().mean()
            self.log("train/dead_feature_fraction", dead_frac)

            # Flag for resampling — actual resampling happens in
            # on_train_batch_end (after backward) to avoid in-place
            # modification of tensors still needed by autograd.
            if (self.dead_feature_strategy == "resample"
                    and dead_mask.any()
                    and self._resample_batch is not None):
                self._pending_resample_mask = dead_mask.clone()

            self.tokens_since_reset.zero_()
            self.feature_activation_count.zero_()

        return loss
    @torch.no_grad()
    def _resample_dead_features(self, dead_mask: torch.Tensor) -> None:
        """Re-initialize dead neurons from high-loss examples (Anthropic-style).

        For each dead feature:
        - Set its decoder column to a normalized high-loss example direction
        - Set its encoder row to match (W_enc = W_dec^T for that feature)
        - Reset the optimizer state for those parameters

        Args:
            dead_mask: Boolean tensor of shape (dictionary_size,), True for dead features.
        """
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_dead = dead_indices.shape[0]
        if n_dead == 0 or self._resample_batch is None:
            return

        examples = self._resample_batch  # (N, activation_dim)
        # Sample with replacement from high-loss examples
        chosen = examples[torch.randint(0, examples.shape[0], (n_dead,))]
        # Subtract decoder bias to get centered directions
        directions = chosen - self.sae.b_dec.unsqueeze(0)
        # Normalize to unit vectors
        norms = directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        directions = directions / norms

        # Compute alive feature average norm for scaling
        alive_mask = ~dead_mask
        alive_enc_norms = self.sae.W_enc[alive_mask].norm(dim=-1)
        scale = alive_enc_norms.mean() * 0.2 if alive_mask.any() else 0.1

        # Re-init decoder columns and encoder rows
        self.sae.W_dec[:, dead_indices] = directions.T * scale
        self.sae.W_enc[dead_indices] = directions * scale

        # Reset optimizer state for resampled parameters
        optimizer = self.optimizers()
        if hasattr(optimizer, 'state'):
            for param in [self.sae.W_enc, self.sae.W_dec]:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in state:
                            if param is self.sae.W_enc:
                                state[key][dead_indices] = 0
                            else:
                                state[key][:, dead_indices] = 0

        n_total = dead_mask.shape[0]
        print(f"[Resample] Resampled {n_dead}/{n_total} dead features "
              f"({n_dead/n_total:.1%})", flush=True)

    def on_after_backward(self) -> None:
        """Normalize decoder weights after gradient computation."""
        self.sae.normalize_decoder_()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update k-annealing schedule and run deferred resampling."""
        # Resample dead features (deferred from training_step to avoid
        # in-place weight modification while autograd graph is alive).
        if self._pending_resample_mask is not None:
            self._resample_dead_features(self._pending_resample_mask)
            self._pending_resample_mask = None

        step = self.global_step
        if step < self.k_anneal_steps:
            progress = step / max(self.k_anneal_steps, 1)
            current_k = int(self.k_start - (self.k_start - self.k_target) * progress)
            self.sae.set_k(current_k)
        else:
            self.sae.set_k(self.k_target)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x_hat, z = self.sae(x)
        mse = F.mse_loss(x_hat, x)

        # Fraction of Variance Explained
        x_mean = x.mean(dim=0, keepdim=True)
        ss_res = (x - x_hat).pow(2).sum()
        ss_tot = (x - x_mean).pow(2).sum()
        fve = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

        self.log("val/fve", fve, prog_bar=True)
        self.log("val/mse", mse)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x_hat, z = self.sae(x)
        mse = F.mse_loss(x_hat, x)

        x_mean = x.mean(dim=0, keepdim=True)
        ss_res = (x - x_hat).pow(2).sum()
        ss_tot = (x - x_mean).pow(2).sum()
        fve = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

        l0 = (z != 0).float().sum(dim=-1).mean()

        self.log("test/fve", fve)
        self.log("test/mse", mse)
        self.log("test/l0_sparsity", l0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.sae.parameters(), lr=self.learning_rate)
    @classmethod
    def load_trained(cls, checkpoint_path: str, map_location: str = "cpu") -> "SAELightningModule":
        """Load a trained SAE from a Lightning checkpoint.

        Reconstructs the TopKSAE from saved state dict shapes, then loads
        the full checkpoint. This avoids the ``load_from_checkpoint`` issue
        where ``sae`` is excluded from ``save_hyperparameters``.

        Args:
            checkpoint_path: Path to the ``.ckpt`` file.
            map_location: Device to map tensors to.

        Returns:
            Fully loaded SAELightningModule with trained weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = ckpt["state_dict"]

        # Infer SAE architecture from saved weight shapes
        # sae.W_enc shape: (dictionary_size, activation_dim)
        w_enc = state_dict["sae.W_enc"]
        dictionary_size, activation_dim = w_enc.shape

        hparams = ckpt.get("hyper_parameters", {})
        k_target = hparams.get("k_target", 32)

        sae = TopKSAE(
            activation_dim=activation_dim,
            dictionary_size=dictionary_size,
            k=k_target,
        )

        module = cls(sae=sae, **hparams)
        # Drop legacy b_enc from old checkpoints (removed in TopK SAE fix)
        state_dict = {k: v for k, v in state_dict.items() if k != "sae.b_enc"}
        module.load_state_dict(state_dict)
        return module

