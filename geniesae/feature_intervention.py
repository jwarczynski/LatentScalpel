"""Feature-level intervention via SAE encode/modify/decode during diffusion.

Implements causal intervention experiments: forcibly activate, suppress, or
shift specific SAE feature activations at chosen NDS values during the full
reverse diffusion chain, then measure the impact on generation quality.

Supports both Genie (discrete 2000-step) and PLAID (continuous VDM) models.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geniesae.genie_model import DiffusionHelper
from geniesae.plaid_model import PlaidDiffusionHelper
from geniesae.sae import TopKSAE

logger = logging.getLogger("geniesae.feature_intervention")


@dataclass
class InterventionSpec:
    """Specification for a feature intervention experiment.

    Attributes:
        feature_indices: SAE feature indices to intervene on.
        target_magnitude: Target activation value for the specified features.
            Use 0.0 to suppress features, positive values to activate.
        intervention_nds_values: NDS values (timestep indices) at which
            to apply the intervention during the reverse diffusion chain.
    """

    feature_indices: list[int]
    target_magnitude: float
    intervention_nds_values: list[int]


@dataclass
class InterventionResult:
    """Result of a feature intervention experiment.

    Attributes:
        baseline_loss: Cross-entropy loss from the unpatched run.
        baseline_perplexity: exp(baseline_loss).
        patched_loss: Cross-entropy loss from the patched run.
        patched_perplexity: exp(patched_loss).
        per_step_losses: Mapping from NDS value to loss at that step.
        baseline_examples: Text examples from the baseline run.
        patched_examples: Text examples from the patched run.
        intervention_spec: Dict representation of the InterventionSpec used.
    """

    baseline_loss: float
    baseline_perplexity: float
    patched_loss: float
    patched_perplexity: float
    per_step_losses: dict[int, float] = field(default_factory=dict)
    baseline_examples: list[str] = field(default_factory=list)
    patched_examples: list[str] = field(default_factory=list)
    intervention_spec: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _intervene_on_sparse_code(
    sparse_code: torch.Tensor,
    feature_indices: list[int],
    target_magnitude: float,
) -> torch.Tensor:
    """Modify specific feature values in a sparse code tensor.

    Args:
        sparse_code: Tensor of shape (..., dictionary_size).
        feature_indices: Indices of features to modify.
        target_magnitude: Value to set for the specified features.

    Returns:
        Modified sparse code tensor (new tensor, does not mutate input).
    """
    modified = sparse_code.clone()
    for idx in feature_indices:
        modified[..., idx] = target_magnitude
    return modified


class FeatureInterventionPatcher:
    """Runs feature-level intervention experiments during reverse diffusion.

    Encodes layer activations through the SAE, modifies specific feature
    values in the sparse code, then decodes back. This operates at the
    feature level rather than full reconstruction.

    Supports both Genie (DiffusionHelper) and PLAID (PlaidDiffusionHelper).

    Args:
        model: The raw nn.Module (Genie Diffusion_LM or PLAID DiffusionModel).
        sae: Trained TopKSAE for the target layer.
        diffusion_helper: DiffusionHelper (Genie) or PlaidDiffusionHelper (PLAID).
        config: Config object with at least: device, num_text_examples,
            loss_log_interval, and model_type attributes.
        layer_accessor: Callable that, given an NNsight model, returns the
            layer module proxies (e.g., model.transformer_blocks).
    """

    def __init__(
        self,
        model: nn.Module,
        sae: TopKSAE,
        diffusion_helper: DiffusionHelper | PlaidDiffusionHelper,
        config: Any,
        layer_accessor: Callable | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        from nnsight import NNsight

        self._model = model
        self._sae = sae
        self._diffusion = diffusion_helper
        self._config = config
        self._nnsight_model = NNsight(model)
        self._layer_accessor = layer_accessor
        self._is_plaid = isinstance(diffusion_helper, PlaidDiffusionHelper)
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layers(self):
        """Return NNsight envoy proxies for the decoder layers."""
        if self._layer_accessor is not None:
            return self._layer_accessor(self._nnsight_model)
        if hasattr(self._nnsight_model, "transformer_blocks"):
            return self._nnsight_model.transformer_blocks
        if hasattr(self._nnsight_model, "blocks"):
            return self._nnsight_model.blocks
        return self._nnsight_model.decoder.layers

    def _compute_loss(
        self, model_output: torch.Tensor, target_ids: torch.Tensor,
    ) -> float:
        """Cross-entropy loss from model output (x_0 prediction) vs targets."""
        with torch.no_grad():
            if self._is_plaid:
                # PLAID forward returns (logits, x_reconst); model_output is logits
                logits = model_output
            else:
                logits = self._model.get_logits(model_output)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return loss.item()

    def _decode_to_text(
        self, model_output: torch.Tensor, tokenizer: Any | None = None,
    ) -> list[str]:
        """Convert model output to text strings via argmax decoding.

        Args:
            model_output: For Genie, the denoised embedding (x_0 pred).
                For PLAID, the logits tensor.
            tokenizer: Optional tokenizer for decoding token IDs to strings.

        Returns:
            List of decoded text strings (one per batch element).
        """
        with torch.no_grad():
            if self._is_plaid:
                logits = model_output
            else:
                logits = self._model.get_logits(model_output)
            token_ids = logits.argmax(dim=-1)  # (batch, seq_len)

        texts = []
        for i in range(token_ids.shape[0]):
            if tokenizer is not None:
                texts.append(tokenizer.decode(token_ids[i], skip_special_tokens=True))
            else:
                texts.append(token_ids[i].tolist())
        return texts


    # ------------------------------------------------------------------
    # NNsight forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_baseline_genie(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unpatched Genie forward pass, returns x_0 prediction."""
        saved = []
        with self._nnsight_model.trace(x_t, t_tensor, input_ids, attention_mask):
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    @torch.no_grad()
    def _forward_patched_genie(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        feature_indices: list[int],
        target_magnitude: float,
        layer_idx: int,
    ) -> torch.Tensor:
        """Patched Genie forward pass with SAE encode/modify/decode."""
        layers = self._get_layers()
        sae = self._sae
        saved = []
        with self._nnsight_model.trace(x_t, t_tensor, input_ids, attention_mask):
            layer_output = layers[layer_idx].output
            if isinstance(layer_output, (tuple, list)):
                act = layer_output[0]
            else:
                act = layer_output
            # Encode → modify → decode
            sparse_code = sae.encode(act)
            modified_code = _intervene_on_sparse_code(
                sparse_code, feature_indices, target_magnitude,
            )
            reconstructed = sae.decode(modified_code)
            if isinstance(layer_output, (tuple, list)):
                layers[layer_idx].output = (reconstructed, *layer_output[1:])
            else:
                layers[layer_idx].output = reconstructed
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    # ------------------------------------------------------------------
    # PLAID forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_baseline_plaid(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
        embedding_matrix: torch.Tensor,
        x_selfcond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpatched PLAID forward pass, returns (logits, x_reconst)."""
        saved = []
        with self._nnsight_model.trace(
            z=z.float(), gamma=gamma.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.0, x_selfcond=x_selfcond,
        ):
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            return output[0], output[1]
        return output, output

    @torch.no_grad()
    def _forward_patched_plaid(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
        embedding_matrix: torch.Tensor,
        x_selfcond: torch.Tensor,
        feature_indices: list[int],
        target_magnitude: float,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patched PLAID forward pass with SAE encode/modify/decode."""
        layers = self._get_layers()
        sae = self._sae
        saved = []
        with self._nnsight_model.trace(
            z=z.float(), gamma=gamma.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.0, x_selfcond=x_selfcond,
        ):
            layer_output = layers[layer_idx].output
            if isinstance(layer_output, (tuple, list)):
                act = layer_output[0]
            else:
                act = layer_output
            sparse_code = sae.encode(act)
            modified_code = _intervene_on_sparse_code(
                sparse_code, feature_indices, target_magnitude,
            )
            reconstructed = sae.decode(modified_code)
            if isinstance(layer_output, (tuple, list)):
                layers[layer_idx].output = (reconstructed, *layer_output[1:])
            else:
                layers[layer_idx].output = reconstructed
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            return output[0], output[1]
        return output, output


    # ------------------------------------------------------------------
    # Genie reverse diffusion chains
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_genie_chain(
        self,
        dataloader: DataLoader,
        spec: InterventionSpec | None = None,
    ) -> tuple[float, dict[int, float], list[str]]:
        """Run full Genie reverse diffusion chain.

        Args:
            dataloader: Yields (input_ids, attention_mask) batches.
            spec: If provided, apply feature intervention at specified NDS
                values. If None, run baseline (no intervention).

        Returns:
            (final_loss, per_step_losses, text_examples)
        """
        device = self._config.device
        layer_idx = self._config.layer
        diffusion = self._diffusion
        T = diffusion.num_timesteps
        log_interval = getattr(self._config, "loss_log_interval", 100)
        num_examples = getattr(self._config, "num_text_examples", 10)
        intervention_steps = set(spec.intervention_nds_values) if spec else set()

        all_losses: dict[int, list[float]] = {}
        text_examples: list[str] = []
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
            else:
                input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
                attention_mask = None

            bs = input_ids.shape[0]

            # Initialize from noise (same shape as embeddings)
            with torch.no_grad():
                x_start_shape = self._model.get_embeds(input_ids)
            x_t = torch.randn_like(x_start_shape)

            last_output = None

            for t in reversed(range(T)):
                t_tensor = torch.full((bs,), t, dtype=torch.long, device=device)

                should_intervene = spec is not None and t in intervention_steps

                if should_intervene:
                    output = self._forward_patched_genie(
                        x_t, t_tensor, input_ids, attention_mask,
                        spec.feature_indices, spec.target_magnitude, layer_idx,
                    )
                else:
                    output = self._forward_baseline_genie(
                        x_t, t_tensor, input_ids, attention_mask,
                    )

                # Update latents via posterior sampling
                x_t = diffusion.p_sample(output, x_t, t_tensor)
                last_output = output

                # Log loss at intervals
                if t % log_interval == 0:
                    loss = self._compute_loss(output, input_ids)
                    all_losses.setdefault(t, []).append(loss)

            # Collect text examples from the final output
            if last_output is not None and len(text_examples) < num_examples:
                examples = self._decode_to_text(last_output, self._tokenizer)
                remaining = num_examples - len(text_examples)
                text_examples.extend(examples[:remaining])

            n_batches += 1
            if (batch_idx + 1) % 4 == 0 or batch_idx == 0:
                logger.info(
                    "Batch %d/%d done", batch_idx + 1, len(dataloader),
                )

        # Aggregate per-step losses
        per_step_losses: dict[int, float] = {}
        for t, losses in all_losses.items():
            per_step_losses[t] = sum(losses) / len(losses)

        # Final loss is at t=0
        final_loss = per_step_losses.get(0, 0.0)

        return final_loss, per_step_losses, text_examples


    # ------------------------------------------------------------------
    # PLAID reverse diffusion chains
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_plaid_chain(
        self,
        dataloader: DataLoader,
        spec: InterventionSpec | None = None,
    ) -> tuple[float, dict[int, float], list[str]]:
        """Run full PLAID reverse diffusion chain.

        Args:
            dataloader: Yields (input_ids, attention_mask) batches.
            spec: If provided, apply feature intervention at specified NDS
                values. If None, run baseline (no intervention).

        Returns:
            (final_loss, per_step_losses, text_examples)
        """
        device = self._config.device
        layer_idx = self._config.layer
        helper = self._diffusion
        T = helper.sampling_timesteps
        log_interval = getattr(self._config, "loss_log_interval", 100)
        num_examples = getattr(self._config, "num_text_examples", 10)
        intervention_steps = set(spec.intervention_nds_values) if spec else set()

        embedding_matrix = helper.embedding_matrix_module()

        all_losses: dict[int, list[float]] = {}
        text_examples: list[str] = []
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
            else:
                input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
                attention_mask = None

            bs = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            # Get embedding dim from the embedding matrix
            embed_dim = embedding_matrix.shape[1]

            # Initialize from noise
            z = torch.randn(bs, seq_len, embed_dim, device=device)
            x_selfcond = torch.zeros_like(z)

            last_logits = None

            for step_idx in range(T):
                t = torch.tensor([1.0 - step_idx / T], device=device)
                s = torch.tensor([1.0 - (step_idx + 1) / T], device=device)

                # Map step index to an NDS-like integer for intervention matching
                nds_value = T - 1 - step_idx
                should_intervene = spec is not None and nds_value in intervention_steps

                gamma_t = helper.get_gamma(t)

                if should_intervene:
                    logits, x_reconst = self._forward_patched_plaid(
                        z, gamma_t, embedding_matrix, x_selfcond,
                        spec.feature_indices, spec.target_magnitude, layer_idx,
                    )
                else:
                    logits, x_reconst = self._forward_baseline_plaid(
                        z, gamma_t, embedding_matrix, x_selfcond,
                    )

                # Update self-conditioning and latents
                x_selfcond = x_reconst.clone().detach()

                # Use the helper's p_sample_step for the actual latent update
                z, _ = helper.p_sample_step(z, t, s, x_selfcond)

                last_logits = logits

                # Log loss at intervals
                if nds_value % log_interval == 0:
                    loss = self._compute_loss(logits, input_ids)
                    all_losses.setdefault(nds_value, []).append(loss)

            # Collect text examples from the final output
            if last_logits is not None and len(text_examples) < num_examples:
                examples = self._decode_to_text(last_logits, self._tokenizer)
                remaining = num_examples - len(text_examples)
                text_examples.extend(examples[:remaining])

            n_batches += 1
            if (batch_idx + 1) % 4 == 0 or batch_idx == 0:
                logger.info(
                    "Batch %d/%d done", batch_idx + 1, len(dataloader),
                )

        # Aggregate per-step losses
        per_step_losses: dict[int, float] = {}
        for t, losses in all_losses.items():
            per_step_losses[t] = sum(losses) / len(losses)

        final_loss = per_step_losses.get(0, 0.0)

        return final_loss, per_step_losses, text_examples


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_baseline(
        self, dataloader: DataLoader,
    ) -> tuple[float, list[str]]:
        """Run full reverse diffusion without intervention.

        Args:
            dataloader: Evaluation data yielding (input_ids, attention_mask).

        Returns:
            Tuple of (loss, text_examples).
        """
        self._model.eval()
        self._sae.eval()

        logger.info("Running baseline (no intervention)...")

        if self._is_plaid:
            loss, _, examples = self._run_plaid_chain(dataloader, spec=None)
        else:
            loss, _, examples = self._run_genie_chain(dataloader, spec=None)

        perplexity = math.exp(min(loss, 100))
        logger.info(
            "Baseline: loss=%.4f, perplexity=%.2f, examples=%d",
            loss, perplexity, len(examples),
        )
        return loss, examples

    @torch.no_grad()
    def run_intervention(
        self, dataloader: DataLoader, spec: InterventionSpec,
    ) -> InterventionResult:
        """Run full reverse diffusion with feature intervention.

        Runs both baseline and patched chains, collecting per-step losses
        and text examples from each.

        Args:
            dataloader: Evaluation data yielding (input_ids, attention_mask).
            spec: Intervention specification (features, magnitude, NDS values).

        Returns:
            InterventionResult with baseline and patched metrics.
        """
        self._model.eval()
        self._sae.eval()

        logger.info(
            "Running intervention: features=%s, magnitude=%.2f, nds_values=%s",
            spec.feature_indices, spec.target_magnitude, spec.intervention_nds_values,
        )

        # Run baseline chain
        logger.info("Running baseline chain...")
        if self._is_plaid:
            bl_loss, bl_per_step, bl_examples = self._run_plaid_chain(
                dataloader, spec=None,
            )
        else:
            bl_loss, bl_per_step, bl_examples = self._run_genie_chain(
                dataloader, spec=None,
            )

        # Run patched chain
        logger.info("Running patched chain...")
        if self._is_plaid:
            p_loss, p_per_step, p_examples = self._run_plaid_chain(
                dataloader, spec=spec,
            )
        else:
            p_loss, p_per_step, p_examples = self._run_genie_chain(
                dataloader, spec=spec,
            )

        bl_ppl = math.exp(min(bl_loss, 100))
        p_ppl = math.exp(min(p_loss, 100))

        logger.info(
            "Results: baseline_loss=%.4f (ppl=%.2f), patched_loss=%.4f (ppl=%.2f)",
            bl_loss, bl_ppl, p_loss, p_ppl,
        )

        return InterventionResult(
            baseline_loss=bl_loss,
            baseline_perplexity=bl_ppl,
            patched_loss=p_loss,
            patched_perplexity=p_ppl,
            per_step_losses=p_per_step,
            baseline_examples=bl_examples,
            patched_examples=p_examples,
            intervention_spec=asdict(spec),
        )
