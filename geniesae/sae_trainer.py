"""SAE training loop, evaluation, and reconstruction metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geniesae.config import ExperimentConfig
from geniesae.sae import TopKSAE
from geniesae.activation_collector import ActivationStore

logger = logging.getLogger("geniesae.sae_trainer")


@dataclass
class ReconstructionMetrics:
    """Quantitative measures of SAE reconstruction quality.

    Attributes:
        mse: Mean squared error between original and reconstructed activations.
        explained_variance: Fraction of variance in the original data explained
            by the reconstruction (1.0 = perfect).
        l0_sparsity: Average number of active (non-zero) features per input.
    """

    mse: float
    explained_variance: float
    l0_sparsity: float

    def to_dict(self) -> dict:
        return asdict(self)


class SAETrainer:
    """Trains and evaluates Top-K SAEs on collected activations.

    Args:
        config: Experiment configuration with SAE hyperparameters.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._checkpoint_dir = Path(config.output_dir) / "sae_checkpoints"
        self._results_dir = Path(config.output_dir) / "results"

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, layer_idx: int) -> Path:
        return self._checkpoint_dir / f"layer_{layer_idx:02d}.pt"

    def _save_checkpoint(self, sae: TopKSAE, layer_idx: int) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": sae.state_dict(),
            "activation_dim": sae.activation_dim,
            "dictionary_size": sae.dictionary_size,
            "k": sae.k,
        }
        torch.save(payload, self._checkpoint_path(layer_idx))

    def _load_checkpoint(self, layer_idx: int) -> TopKSAE:
        ckpt = torch.load(
            self._checkpoint_path(layer_idx),
            map_location=self._config.device,
            weights_only=True,
        )
        sae = TopKSAE(
            activation_dim=ckpt["activation_dim"],
            dictionary_size=ckpt["dictionary_size"],
            k=ckpt["k"],
        )
        sae.load_state_dict(ckpt["state_dict"])
        sae.to(self._config.device)
        return sae

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_layer(
        self, layer_idx: int, activation_store: ActivationStore
    ) -> TopKSAE:
        """Train a TopKSAE for one layer.

        If a checkpoint already exists and ``force_retrain`` is False, the
        existing checkpoint is loaded and returned without retraining.
        """
        cfg = self._config
        ckpt_path = self._checkpoint_path(layer_idx)

        # Skip if checkpoint exists and not forcing retrain
        if ckpt_path.exists() and not cfg.force_retrain:
            logger.info("Layer %d: loading existing checkpoint %s", layer_idx, ckpt_path)
            return self._load_checkpoint(layer_idx)

        logger.info("Layer %d: starting training", layer_idx)

        activation_dim = activation_store.activation_dim
        sae = TopKSAE(activation_dim, cfg.sae_dictionary_size, cfg.sae_top_k)
        sae.to(cfg.device)

        dataset = activation_store.get_layer_dataset(layer_idx)
        dataloader = DataLoader(
            dataset, batch_size=cfg.sae_batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.sae_learning_rate)
        loss_fn = nn.MSELoss()

        global_step = 0
        for epoch in range(cfg.sae_training_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                batch = batch.to(cfg.device)
                reconstruction, sparse_code = sae(batch)
                loss = loss_fn(reconstruction, batch)

                if torch.isnan(loss):
                    raise RuntimeError(
                        f"NaN loss at layer {layer_idx}, epoch {epoch}, "
                        f"step {global_step}, lr={cfg.sae_learning_rate}"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    # Compute L0 sparsity for logging
                    l0 = (sparse_code != 0).float().sum(dim=-1).mean().item()
                    logger.info(
                        "Layer %d | epoch %d | step %d | loss %.6f | L0 %.1f",
                        layer_idx, epoch, global_step, loss.item(), l0,
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                "Layer %d | epoch %d complete | avg loss %.6f",
                layer_idx, epoch, avg_loss,
            )

        self._save_checkpoint(sae, layer_idx)
        logger.info("Layer %d: checkpoint saved to %s", layer_idx, ckpt_path)
        return sae

    def train_all_layers(
        self, activation_store: ActivationStore, num_layers: int
    ) -> dict[int, TopKSAE]:
        """Train SAEs for all layers. Returns ``{layer_idx: trained_sae}``."""
        saes: dict[int, TopKSAE] = {}
        for layer_idx in range(num_layers):
            saes[layer_idx] = self.train_layer(layer_idx, activation_store)
        return saes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sae: TopKSAE,
        activation_store: ActivationStore,
        layer_idx: int,
    ) -> ReconstructionMetrics:
        """Compute reconstruction metrics on activations for *layer_idx*."""
        cfg = self._config
        dataset = activation_store.get_layer_dataset(layer_idx)
        dataloader = DataLoader(dataset, batch_size=cfg.sae_batch_size, shuffle=False)

        total_mse = 0.0
        total_l0 = 0.0
        total_samples = 0

        # For explained variance we accumulate sums
        sum_sq_error = 0.0
        sum_sq_total = 0.0
        sum_x = 0.0
        n_elements = 0

        sae.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(cfg.device)
                reconstruction, sparse_code = sae(batch)

                # MSE per sample, then sum
                mse_per_sample = (reconstruction - batch).pow(2).mean(dim=-1)
                total_mse += mse_per_sample.sum().item()

                # L0: count non-zero features per sample
                l0_per_sample = (sparse_code != 0).float().sum(dim=-1)
                total_l0 += l0_per_sample.sum().item()

                total_samples += batch.shape[0]

                # For explained variance: accumulate SS_res and SS_tot
                residual = reconstruction - batch
                sum_sq_error += residual.pow(2).sum().item()
                sum_x += batch.sum().item()
                n_elements += batch.numel()
                sum_sq_total += batch.pow(2).sum().item()

        mse = total_mse / max(total_samples, 1)
        l0_sparsity = total_l0 / max(total_samples, 1)

        # explained_variance = 1 - SS_res / SS_tot
        # SS_tot = sum((x - mean(x))^2) = sum(x^2) - n * mean(x)^2
        mean_x = sum_x / max(n_elements, 1)
        ss_tot = sum_sq_total - n_elements * mean_x * mean_x
        if ss_tot > 0:
            explained_variance = 1.0 - sum_sq_error / ss_tot
            # Clamp to [0, 1] for numerical stability
            explained_variance = max(0.0, min(1.0, explained_variance))
        else:
            # All activations are identical — reconstruction is trivially perfect
            explained_variance = 1.0 if sum_sq_error == 0.0 else 0.0

        return ReconstructionMetrics(
            mse=mse,
            explained_variance=explained_variance,
            l0_sparsity=l0_sparsity,
        )

    def evaluate_all_layers(
        self,
        saes: dict[int, TopKSAE],
        activation_store: ActivationStore,
    ) -> dict[int, ReconstructionMetrics]:
        """Evaluate all SAEs and save metrics JSON to the results directory."""
        metrics: dict[int, ReconstructionMetrics] = {}
        for layer_idx, sae in sorted(saes.items()):
            logger.info("Evaluating layer %d", layer_idx)
            metrics[layer_idx] = self.evaluate(sae, activation_store, layer_idx)
            logger.info(
                "Layer %d metrics: %s", layer_idx, metrics[layer_idx].to_dict()
            )

        # Save to JSON
        self._results_dir.mkdir(parents=True, exist_ok=True)
        json_path = self._results_dir / "reconstruction_metrics.json"
        serializable = {
            f"layer_{idx:02d}": m.to_dict() for idx, m in sorted(metrics.items())
        }
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Reconstruction metrics saved to %s", json_path)

        return metrics
