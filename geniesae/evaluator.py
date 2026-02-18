"""Evaluation logic: loss impact and generation quality of SAE-patched models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geniesae.activation_collector import ActivationStore
from geniesae.activation_patcher import ActivationPatcher
from geniesae.genie_model import DiffusionHelper
from geniesae.sae import TopKSAE

logger = logging.getLogger("geniesae.evaluator")


class Evaluator:
    """Evaluates SAE reconstruction impact on model loss and generation quality.

    Combines:
    - Baseline loss computation
    - Per-layer patching with loss delta measurement
    - All-layers simultaneous patching
    - Generation comparison (original vs patched)
    - Per-layer reconstruction metrics (MSE, FVE, dead %, L0)

    Args:
        model: The GENIE model (unwrapped nn.Module).
        saes: Mapping from layer index to trained TopKSAE.
        config: Any config with diffusion_steps, noise_schedule,
            diffusion_timesteps, device, config_name, and output_dir.
        activation_store: ActivationStore with collected activations.
    """

    def __init__(
        self,
        model: nn.Module,
        saes: dict[int, TopKSAE],
        config: Any,
        activation_store: ActivationStore,
    ) -> None:
        self._model = model
        self._saes = saes
        self._config = config
        self._store = activation_store
        self._patcher = ActivationPatcher(model, saes, config)

    def _compute_reconstruction_metrics(self, layer_idx: int) -> dict:
        """Compute per-layer reconstruction metrics from stored activations."""
        dataset = self._store.get_layer_dataset(layer_idx)
        sae = self._saes[layer_idx]
        sae.eval()

        total_mse = 0.0
        total_ss_res = 0.0
        total_ss_tot = 0.0
        total_l0 = 0.0
        total_dead = 0
        total_features = sae.dictionary_size
        feature_active = torch.zeros(total_features, dtype=torch.bool)
        count = 0

        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        device = next(sae.parameters()).device

        with torch.no_grad():
            all_data = []
            for batch in loader:
                all_data.append(batch)
            all_x = torch.cat(all_data, dim=0)
            x_mean = all_x.mean(dim=0, keepdim=True).to(device)

            for batch in loader:
                x = batch.to(device)
                x_hat, z = sae(x)

                total_mse += F.mse_loss(x_hat, x, reduction="sum").item()
                total_ss_res += (x - x_hat).pow(2).sum().item()
                total_ss_tot += (x - x_mean).pow(2).sum().item()
                total_l0 += (z != 0).float().sum(dim=-1).mean().item()
                feature_active |= (z != 0).any(dim=0).cpu()
                count += 1

        n_samples = len(dataset)
        mse = total_mse / max(n_samples * sae.activation_dim, 1)
        fve = 1.0 - total_ss_res / max(total_ss_tot, 1e-8)
        fve = max(0.0, min(1.0, fve))
        dead_pct = (1.0 - feature_active.float().mean().item()) * 100
        l0 = total_l0 / max(count, 1)

        return {
            "mse": round(mse, 6),
            "fve": round(fve, 4),
            "dead_pct": round(dead_pct, 2),
            "l0": round(l0, 1),
        }

    @torch.no_grad()
    def _generate_samples(
        self, dataloader: DataLoader, num_samples: int
    ) -> list[dict]:
        """Generate text samples with original and patched models."""
        from nnsight import NNsight
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._config.config_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = self._config.device
        diffusion = DiffusionHelper(
            num_timesteps=self._config.diffusion_steps,
            schedule_name=self._config.noise_schedule,
        )
        nnsight_model = NNsight(self._model)

        samples = []
        collected = 0

        for batch in dataloader:
            if collected >= num_samples:
                break

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            bs = input_ids.shape[0]

            # Source text
            source_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Use a mid-range timestep for generation
            timestep = self._config.diffusion_timesteps[len(self._config.diffusion_timesteps) // 2]
            t_tensor = torch.full((bs,), timestep, dtype=torch.long, device=device)

            with torch.no_grad():
                x_start = self._model.get_embeds(input_ids)
            x_noised = diffusion.q_sample(x_start, t_tensor)

            # Original model output
            saved_orig = []
            with nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                saved_orig.append(nnsight_model.output.save())

            orig_output = saved_orig[0].value if hasattr(saved_orig[0], "value") else saved_orig[0]
            if isinstance(orig_output, (tuple, list)):
                orig_output = orig_output[0]
            orig_logits = self._model.get_logits(orig_output)
            orig_tokens = orig_logits.argmax(dim=-1)
            orig_texts = tokenizer.batch_decode(orig_tokens, skip_special_tokens=True)

            # Patched model output (all layers)
            layers = nnsight_model.decoder.layers if hasattr(nnsight_model, "decoder") else []
            saved_patched = []
            try:
                with nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                    for li, sae in self._saes.items():
                        layer_output = layers[li].output
                        reconstructed, _ = sae(layer_output)
                        layers[li].output = reconstructed
                    saved_patched.append(nnsight_model.output.save())

                patched_output = saved_patched[0].value if hasattr(saved_patched[0], "value") else saved_patched[0]
                if isinstance(patched_output, (tuple, list)):
                    patched_output = patched_output[0]
                patched_logits = self._model.get_logits(patched_output)
                patched_tokens = patched_logits.argmax(dim=-1)
                patched_texts = tokenizer.batch_decode(patched_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning("Patched generation failed: %s", e)
                patched_texts = ["[generation failed]"] * bs

            for i in range(min(bs, num_samples - collected)):
                samples.append({
                    "source": source_texts[i][:200],
                    "original": orig_texts[i][:200],
                    "patched": patched_texts[i][:200],
                })
                collected += 1

        return samples

    def run(self, dataloader: DataLoader, num_generation_samples: int = 50) -> dict:
        """Run full evaluation: loss impact + reconstruction metrics + generation comparison."""
        logger.info("Computing baseline loss...")
        baseline_loss = self._patcher.compute_baseline_loss(dataloader)
        logger.info("Baseline loss: %.6f", baseline_loss)

        # Per-layer patching
        per_layer_results = []
        for layer_idx in sorted(self._saes.keys()):
            logger.info("Patching layer %d...", layer_idx)
            result = self._patcher.patch_single_layer(layer_idx, dataloader, baseline_loss)

            # Reconstruction metrics
            recon_metrics = self._compute_reconstruction_metrics(layer_idx)

            per_layer_results.append({
                "layer_idx": layer_idx,
                "patched_loss": round(result.patched_loss, 6),
                "loss_delta": round(result.loss_delta, 6),
                **recon_metrics,
            })
            logger.info(
                "  Layer %d: delta=%.4f, FVE=%.4f, dead=%.1f%%, L0=%.1f",
                layer_idx, result.loss_delta, recon_metrics["fve"],
                recon_metrics["dead_pct"], recon_metrics["l0"],
            )

        # All-layers patching
        logger.info("Patching all layers simultaneously...")
        all_result = self._patcher.patch_all_layers(dataloader, baseline_loss)

        # Generation comparison
        logger.info("Generating %d comparison samples...", num_generation_samples)
        generation_samples = self._generate_samples(dataloader, num_generation_samples)

        results = {
            "baseline_loss": round(baseline_loss, 6),
            "per_layer": per_layer_results,
            "all_layers": {
                "patched_loss": round(all_result.patched_loss, 6),
                "loss_delta": round(all_result.loss_delta, 6),
            },
            "generation_samples": generation_samples,
        }

        logger.info("Evaluation complete.")
        return results
