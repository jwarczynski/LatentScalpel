"""NNsight-based activation patching and loss measurement for GENIE."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geniesae.config import ExperimentConfig
from geniesae.genie_model import DiffusionHelper
from geniesae.sae import TopKSAE

logger = logging.getLogger("geniesae.activation_patcher")


@dataclass
class PatchingResult:
    """Result of a single patching experiment."""

    layer_idx: int | None
    baseline_loss: float
    patched_loss: float
    loss_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


class ActivationPatcher:
    """Replaces decoder-layer activations with SAE reconstructions via NNsight.

    Handles the GENIE-specific forward pass: converts token IDs to embeddings,
    adds diffusion noise, and computes a cross-entropy loss against the target
    tokens using the model's ``get_logits`` method on the denoised output.

    Args:
        model: The GENIE model (unwrapped nn.Module).
        saes: Mapping from layer index to a trained :class:`TopKSAE`.
        config: Experiment configuration.
        layer_accessor: Callable returning NNsight envoy layer proxies.
    """

    def __init__(
        self,
        model: nn.Module,
        saes: dict[int, TopKSAE],
        config: ExperimentConfig,
        layer_accessor: Callable | None = None,
    ) -> None:
        from nnsight import NNsight

        self._model = model
        self._saes = saes
        self._config = config
        self._nnsight_model = NNsight(model)
        self._layer_accessor = layer_accessor
        self._diffusion = DiffusionHelper(
            num_timesteps=config.diffusion_steps,
            schedule_name=config.noise_schedule,
        )

    def _get_layers(self):
        """Return NNsight envoy proxies for the decoder layers."""
        if self._layer_accessor is not None:
            return self._layer_accessor(self._nnsight_model)
        return self._nnsight_model.decoder.layers

    def _prepare_genie_inputs(self, batch, device: str):
        """Convert a dataloader batch into GENIE forward-pass arguments.

        Returns ``(x_noised, t_tensor, input_ids, attention_mask, target_ids)``
        where ``target_ids`` is the same as ``input_ids`` (used for loss).
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
        else:
            input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            attention_mask = None

        batch_size = input_ids.shape[0]

        with torch.no_grad():
            x_start = self._model.get_embeds(input_ids)

        # Use a fixed mid-range timestep for patching evaluation
        timestep = self._config.diffusion_timesteps[len(self._config.diffusion_timesteps) // 2]
        t_tensor = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
        x_noised = self._diffusion.q_sample(x_start, t_tensor)

        return x_noised, t_tensor, input_ids, attention_mask

    def _compute_loss_from_output(
        self, model_output: torch.Tensor, target_ids: torch.Tensor
    ) -> float:
        """Compute cross-entropy loss from GENIE's denoised output.

        GENIE's forward pass returns ``h`` (the denoised prediction in
        embedding space). We convert it to logits via ``get_logits`` and
        compute cross-entropy against the target token IDs.
        """
        with torch.no_grad():
            logits = self._model.get_logits(model_output)
            # logits: (batch, seq_len, vocab_size)
            # target_ids: (batch, seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return loss.item()

    @torch.no_grad()
    def compute_baseline_loss(self, dataloader: DataLoader) -> float:
        """Run unpatched forward pass, return mean cross-entropy loss."""
        self._model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x_noised, t_tensor, input_ids, attention_mask = self._prepare_genie_inputs(
                batch, self._config.device
            )

            saved_output = []
            with self._nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                saved_output.append(self._nnsight_model.output.save())

            output = saved_output[0].value if hasattr(saved_output[0], "value") else saved_output[0]
            if isinstance(output, (tuple, list)):
                output = output[0]

            total_loss += self._compute_loss_from_output(output, input_ids)
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def patch_single_layer(
        self, layer_idx: int, dataloader: DataLoader, baseline_loss: float
    ) -> PatchingResult:
        """Replace activations at *layer_idx* with SAE reconstruction."""
        if layer_idx not in self._saes:
            raise KeyError(f"No trained SAE for layer {layer_idx}")

        self._model.eval()
        sae = self._saes[layer_idx]
        sae.eval()
        layers = self._get_layers()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x_noised, t_tensor, input_ids, attention_mask = self._prepare_genie_inputs(
                batch, self._config.device
            )

            saved_output = []
            with self._nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                layer_output = layers[layer_idx].output
                reconstructed, _ = sae(layer_output)
                layers[layer_idx].output = reconstructed
                saved_output.append(self._nnsight_model.output.save())

            output = saved_output[0].value if hasattr(saved_output[0], "value") else saved_output[0]
            if isinstance(output, (tuple, list)):
                output = output[0]

            total_loss += self._compute_loss_from_output(output, input_ids)
            num_batches += 1

        patched_loss = total_loss / max(num_batches, 1)
        return PatchingResult(
            layer_idx=layer_idx,
            baseline_loss=baseline_loss,
            patched_loss=patched_loss,
            loss_delta=patched_loss - baseline_loss,
        )

    @torch.no_grad()
    def patch_all_layers(
        self, dataloader: DataLoader, baseline_loss: float
    ) -> PatchingResult:
        """Replace activations at ALL layers simultaneously."""
        self._model.eval()
        layers = self._get_layers()

        for sae in self._saes.values():
            sae.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x_noised, t_tensor, input_ids, attention_mask = self._prepare_genie_inputs(
                batch, self._config.device
            )

            saved_output = []
            with self._nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                for li, sae in self._saes.items():
                    layer_output = layers[li].output
                    reconstructed, _ = sae(layer_output)
                    layers[li].output = reconstructed
                saved_output.append(self._nnsight_model.output.save())

            output = saved_output[0].value if hasattr(saved_output[0], "value") else saved_output[0]
            if isinstance(output, (tuple, list)):
                output = output[0]

            total_loss += self._compute_loss_from_output(output, input_ids)
            num_batches += 1

        patched_loss = total_loss / max(num_batches, 1)
        return PatchingResult(
            layer_idx=None,
            baseline_loss=baseline_loss,
            patched_loss=patched_loss,
            loss_delta=patched_loss - baseline_loss,
        )

    def run_full_evaluation(self, dataloader: DataLoader) -> list[PatchingResult]:
        """Run baseline, per-layer patching, and all-layers patching.

        Saves results JSON to ``{output_dir}/results/patching_results.json``.
        """
        logger.info("Computing baseline loss...")
        baseline_loss = self.compute_baseline_loss(dataloader)
        logger.info("Baseline loss: %.6f", baseline_loss)

        results: list[PatchingResult] = []

        for layer_idx in sorted(self._saes.keys()):
            logger.info("Patching layer %d...", layer_idx)
            result = self.patch_single_layer(layer_idx, dataloader, baseline_loss)
            results.append(result)
            logger.info(
                "  Layer %d: patched_loss=%.6f, delta=%.6f",
                layer_idx, result.patched_loss, result.loss_delta,
            )

        logger.info("Patching all layers simultaneously...")
        all_result = self.patch_all_layers(dataloader, baseline_loss)
        results.append(all_result)
        logger.info(
            "  All layers: patched_loss=%.6f, delta=%.6f",
            all_result.patched_loss, all_result.loss_delta,
        )

        # Save results JSON
        results_dir = Path(self._config.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / "patching_results.json"

        per_layer = [
            {"layer_idx": r.layer_idx, "patched_loss": r.patched_loss, "loss_delta": r.loss_delta}
            for r in results if r.layer_idx is not None
        ]
        output_data = {
            "baseline_loss": baseline_loss,
            "per_layer": per_layer,
            "all_layers": {
                "patched_loss": all_result.patched_loss,
                "loss_delta": all_result.loss_delta,
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("Patching results saved to %s", output_path)
        return results
