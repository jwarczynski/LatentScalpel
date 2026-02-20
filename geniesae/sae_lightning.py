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
    ) -> None:
        super().__init__()
        self.sae = sae
        self.learning_rate = learning_rate
        self.k_target = k_target
        self.k_start = k_start
        self.k_anneal_steps = k_anneal_steps
        self.dead_feature_window = dead_feature_window
        self.resample_dead = resample_dead

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
        # and reset only when the window is full. This avoids the spike
        # to ~1.0 that happened when we read the counter right after a reset.
        active_features = (z != 0).any(dim=0)
        self.feature_activation_count += active_features.long()
        self.tokens_since_reset += x.shape[0]

        if self.tokens_since_reset >= self.dead_feature_window:
            dead_frac = (self.feature_activation_count == 0).float().mean()
            self.log("train/dead_feature_fraction", dead_frac)
            self.tokens_since_reset.zero_()
            self.feature_activation_count.zero_()

        return loss

    def on_after_backward(self) -> None:
        """Normalize decoder weights after gradient computation."""
        self.sae.normalize_decoder_()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update k-annealing schedule."""
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

