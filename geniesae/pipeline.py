"""Pipeline Runner orchestrating the full SAE training and evaluation pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from geniesae.config import ExperimentConfig
from geniesae.utils import set_seed, setup_logging, get_device_info
from geniesae.model_loader import load_genie_model, load_xsum_dataset, get_decoder_layers
from geniesae.genie_model import CrossAttention_Diffusion_LM, Diffusion_LM
from geniesae.activation_collector import ActivationCollector
from geniesae.sae_trainer import SAETrainer
from geniesae.activation_patcher import ActivationPatcher

logger = logging.getLogger("geniesae.pipeline")


class PipelineRunner:
    """Top-level orchestrator for the SAE training and evaluation pipeline.

    Sequences: logging/seed setup → model/data loading → activation collection
    → SAE training → reconstruction evaluation → activation patching → summary.

    Args:
        config: Validated experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_stage(stage_name: str, start: float) -> None:
        elapsed = time.time() - start
        logger.info("Stage '%s' completed in %.2f seconds", stage_name, elapsed)

    def _copy_config_to_output(self) -> None:
        """Copy the experiment config YAML into the results directory."""
        results_dir = Path(self._config.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        dest = results_dir / "experiment_config.yaml"
        self._config.to_yaml(str(dest))
        logger.info("Experiment config saved to %s", dest)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline."""
        pipeline_start = time.time()

        # ---- 1. Setup logging and seeds ----
        stage_start = time.time()
        logger.info("Stage 'setup' starting")
        setup_logging(self._config.output_dir)
        set_seed(self._config.random_seed)
        device_info = get_device_info(self._config.device)
        logger.info("Device info: %s", device_info)
        self._copy_config_to_output()
        self._log_stage("setup", stage_start)

        # ---- 2. Load model and dataset ----
        stage_start = time.time()
        logger.info("Stage 'load_model_and_data' starting")
        nnsight_model, tokenizer = load_genie_model(self._config)
        dataloader = load_xsum_dataset(self._config, tokenizer)
        # Unwrap the NNsight wrapper to get the raw nn.Module.
        # ActivationCollector and ActivationPatcher wrap it themselves.
        raw_model = nnsight_model._model

        # Count decoder layers for logging.
        decoder_layers = get_decoder_layers(raw_model)
        num_decoder_layers = len(decoder_layers)
        logger.info("Detected %d decoder layers", num_decoder_layers)

        # Build a layer accessor that navigates the NNsight envoy attribute
        # tree (NOT the raw model) so we get proper proxy objects that
        # support .output.save() inside traces.
        def layer_accessor(nnsight_wrapped):
            """Return NNsight envoy proxies for the GENIE decoder layers."""
            inner = nnsight_wrapped._model
            if isinstance(inner, CrossAttention_Diffusion_LM):
                return nnsight_wrapped.transformer_blocks
            elif isinstance(inner, Diffusion_LM):
                return nnsight_wrapped.input_transformers.layer
            else:
                raise TypeError(
                    f"Cannot determine decoder layers for {type(inner).__name__}"
                )

        self._log_stage("load_model_and_data", stage_start)

        # ---- 3. Collect activations ----
        stage_start = time.time()
        logger.info("Stage 'collect_activations' starting")
        collector = ActivationCollector(raw_model, self._config, layer_accessor=layer_accessor)
        activation_store = collector.collect(dataloader)
        self._log_stage("collect_activations", stage_start)

        # ---- 4. Train SAEs ----
        stage_start = time.time()
        logger.info("Stage 'train_saes' starting")
        trainer = SAETrainer(self._config)
        saes = trainer.train_all_layers(activation_store, activation_store.num_layers)
        self._log_stage("train_saes", stage_start)

        # ---- 5. Evaluate reconstruction quality ----
        stage_start = time.time()
        logger.info("Stage 'evaluate_reconstruction' starting")
        metrics = trainer.evaluate_all_layers(saes, activation_store)
        for layer_idx, m in sorted(metrics.items()):
            logger.info(
                "  Layer %d — MSE: %.6f, Explained Var: %.4f, L0: %.1f",
                layer_idx, m.mse, m.explained_variance, m.l0_sparsity,
            )
        self._log_stage("evaluate_reconstruction", stage_start)

        # ---- 6. Run activation patching ----
        stage_start = time.time()
        logger.info("Stage 'activation_patching' starting")
        patcher = ActivationPatcher(raw_model, saes, self._config, layer_accessor=layer_accessor)
        patching_results = patcher.run_full_evaluation(dataloader)
        self._log_stage("activation_patching", stage_start)

        # ---- 7. Log summary ----
        total_elapsed = time.time() - pipeline_start
        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info("  Total elapsed time: %.2f seconds", total_elapsed)
        logger.info("  Output directory: %s", self._config.output_dir)
        logger.info("  Layers trained: %d", len(saes))
        logger.info("  Patching results: %d entries", len(patching_results))
        logger.info("=" * 60)
