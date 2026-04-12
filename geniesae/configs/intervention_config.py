"""Exca config for feature intervention experiments.

Runs causal intervention experiments: forcibly activate, suppress, or
shift specific SAE feature activations at chosen NDS values during the
full reverse diffusion chain, then measure the impact on generation quality.

Supports both Genie (discrete 2000-step) and PLAID (continuous VDM) models.

Single run (inline)::

    uv run python main.py run-intervention configs/intervention.yaml

Submit to Slurm::

    uv run python main.py run-intervention configs/intervention.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.intervention")


class InterventionConfig(BaseModel):
    """Runs feature-level intervention experiments during reverse diffusion.

    Loads a model and SAE, selects target features (manually or
    automatically from a temporal classification JSON), and runs
    baseline + patched reverse diffusion chains. Results include
    cross-entropy loss, perplexity, per-step losses, and text examples.
    """

    # -- Model ----------------------------------------------------------------
    model_type: str = Field(
        default="genie",
        description='"genie" or "plaid".',
    )
    model_checkpoint_path: str = Field(min_length=1)

    # Genie-specific model params
    model_arch: str = "s2s_CAT"
    in_channel: int = Field(default=128, gt=0)
    model_channels: int = Field(default=128, gt=0)
    out_channel: int = Field(default=128, gt=0)
    vocab_size: int = Field(default=30522, gt=0)
    config_name: str = "bert-base-uncased"
    logits_mode: int = 1
    init_pretrained: bool = False
    token_emb_type: str = "random"
    learn_sigma: bool = False
    fix_encoder: bool = False

    # PLAID-specific model params
    plaid_dim: int = Field(default=2048, gt=0)
    plaid_embed_dim: int = Field(default=16, gt=0)
    plaid_n_blocks: int = Field(default=24, gt=0)
    plaid_n_heads: int = Field(default=32, gt=0)
    plaid_vocab_size: int = Field(default=32768, gt=0)
    plaid_gamma_0: float = -3.0
    plaid_gamma_1: float = 6.0
    plaid_sampling_timesteps: int = Field(default=256, gt=0)
    plaid_score_temp: float = 0.9

    # -- SAE ------------------------------------------------------------------
    sae_checkpoint_dir: str = Field(min_length=1)
    layer: int = Field(ge=0)

    # -- Intervention ---------------------------------------------------------
    feature_selection: str = Field(
        default="manual",
        description='"manual" (explicit feature list) or "auto" (select '
                    'midpoint_exclusive features from classification JSON).',
    )
    feature_indices: list[int] | None = Field(
        default=None,
        description="Explicit feature indices for manual mode.",
    )
    classification_json_path: str | None = Field(
        default=None,
        description="Path to temporal classification JSON for auto mode.",
    )
    target_magnitude: float = Field(
        default=0.0,
        description="Target activation value for intervened features. "
                    "0.0 = suppress, positive = activate.",
    )
    intervention_nds_values: list[int] = Field(
        default_factory=list,
        description="NDS values (timestep indices) at which to apply intervention.",
    )

    # -- Diffusion (Genie) ----------------------------------------------------
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "xsum"
    dataset_split: str = "validation"
    max_samples: int = Field(default=100, gt=0)
    batch_size: int = Field(default=4, gt=0)

    # -- Output ---------------------------------------------------------------
    num_text_examples: int = Field(default=10, gt=0)
    loss_log_interval: int = Field(default=100, gt=0)
    output_path: str = "./experiments/intervention_results.json"

    # -- Device ---------------------------------------------------------------
    device: str = "cuda"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size",
    )

    # ------------------------------------------------------------------
    # Feature selection helpers
    # ------------------------------------------------------------------

    def _resolve_feature_indices(self, sae_dict_size: int) -> list[int]:
        """Resolve feature indices based on selection mode.

        Args:
            sae_dict_size: Dictionary size of the loaded SAE, used for
                validation.

        Returns:
            List of feature indices to intervene on.

        Raises:
            ValueError: If configuration is invalid or no features found.
            FileNotFoundError: If classification JSON is missing in auto mode.
        """
        if self.feature_selection == "manual":
            if not self.feature_indices:
                raise ValueError(
                    "feature_selection='manual' requires non-empty feature_indices."
                )
            # Validate against SAE dictionary size
            invalid = [i for i in self.feature_indices if i < 0 or i >= sae_dict_size]
            if invalid:
                raise ValueError(
                    f"Feature indices out of range [0, {sae_dict_size}): {invalid}"
                )
            return list(self.feature_indices)

        elif self.feature_selection == "auto":
            return self._auto_select_features(sae_dict_size)

        else:
            raise ValueError(
                f"Unknown feature_selection mode: {self.feature_selection!r}. "
                "Expected 'manual' or 'auto'."
            )

    @staticmethod
    def select_midpoint_exclusive_features(
        classification_data: dict,
    ) -> list[int]:
        """Extract midpoint_exclusive feature indices from classification JSON.

        This is a static method so it can be tested independently.

        Args:
            classification_data: Parsed classification JSON dict.

        Returns:
            Sorted list of feature indices with midpoint_exclusive category.
        """
        features = classification_data.get("features", {})
        selected = []
        for feat_key, feat_data in features.items():
            if feat_data.get("category") == "midpoint_exclusive":
                selected.append(int(feat_key))
        return sorted(selected)

    def _auto_select_features(self, sae_dict_size: int) -> list[int]:
        """Auto-select midpoint_exclusive features from classification JSON.

        Args:
            sae_dict_size: Dictionary size of the loaded SAE.

        Returns:
            Sorted list of midpoint_exclusive feature indices.

        Raises:
            ValueError: If no classification path or no features found.
            FileNotFoundError: If classification JSON doesn't exist.
        """
        if not self.classification_json_path:
            raise ValueError(
                "feature_selection='auto' requires classification_json_path."
            )

        cls_path = Path(self.classification_json_path)
        if not cls_path.exists():
            raise FileNotFoundError(
                f"Classification JSON not found: {cls_path}"
            )

        with open(cls_path) as f:
            classification_data = json.load(f)

        selected = self.select_midpoint_exclusive_features(classification_data)

        if not selected:
            raise ValueError(
                "Auto mode found zero midpoint_exclusive features in "
                f"{cls_path}. Consider using manual feature selection."
            )

        # Validate against SAE dictionary size
        invalid = [i for i in selected if i >= sae_dict_size]
        if invalid:
            raise ValueError(
                f"Auto-selected feature indices out of range "
                f"[0, {sae_dict_size}): {invalid}"
            )

        return selected

    def _validate_nds_values(self, max_nds: int) -> None:
        """Validate intervention NDS values against model diffusion steps.

        Args:
            max_nds: Maximum valid NDS value (diffusion_steps - 1 for Genie,
                sampling_timesteps - 1 for PLAID).

        Raises:
            ValueError: If any NDS value is out of range.
        """
        invalid = [v for v in self.intervention_nds_values if v < 0 or v > max_nds]
        if invalid:
            raise ValueError(
                f"Intervention NDS values out of range [0, {max_nds}]: {invalid}"
            )

    def _discover_sae_checkpoint(self) -> str:
        """Find the SAE checkpoint file for the configured layer.

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoint is found.
        """
        import re

        sae_dir = Path(self.sae_checkpoint_dir)
        if not sae_dir.is_dir():
            raise FileNotFoundError(f"SAE checkpoint dir not found: {sae_dir}")

        pattern = re.compile(r"layer_(\d+)(_best)?\.ckpt$")
        candidates: dict[str, str] = {}  # "best" or "last" -> path

        for f in sorted(sae_dir.iterdir()):
            m = pattern.match(f.name)
            if m and int(m.group(1)) == self.layer:
                kind = "best" if m.group(2) else "last"
                candidates[kind] = str(f)

        # Prefer best checkpoint
        if "best" in candidates:
            return candidates["best"]
        if "last" in candidates:
            return candidates["last"]

        raise FileNotFoundError(
            f"No SAE checkpoint found for layer {self.layer} in {sae_dir}"
        )

    @infra.apply
    @notify_on_completion("run-intervention")
    def apply(self) -> str:
        """Run the intervention experiment. Returns the output file path."""
        import math

        import torch
        from datasets import load_dataset
        from torch.utils.data import DataLoader, TensorDataset

        from geniesae.feature_intervention import (
            FeatureInterventionPatcher,
            InterventionSpec,
        )
        from geniesae.sae import TopKSAE
        from geniesae.sae_lightning import SAELightningModule

        torch.set_float32_matmul_precision("high")
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # --- Load model ---
        if self.model_type == "genie":
            from geniesae.genie_model import DiffusionHelper
            from geniesae.model_loader import load_genie_model

            nnsight_model, tokenizer = load_genie_model(self)
            raw_model = (
                nnsight_model._model
                if hasattr(nnsight_model, "_model")
                else nnsight_model
            )
            diffusion_helper = DiffusionHelper(
                num_timesteps=self.diffusion_steps,
                schedule_name=self.noise_schedule,
            )
            max_nds = self.diffusion_steps - 1

        elif self.model_type == "plaid":
            from geniesae.plaid_model import PlaidDiffusionHelper, load_plaid_modules

            modules = load_plaid_modules(
                self.model_checkpoint_path,
                dim=self.plaid_dim,
                embed_dim=self.plaid_embed_dim,
                n_blocks=self.plaid_n_blocks,
                n_heads=self.plaid_n_heads,
                vocab_size=self.plaid_vocab_size,
                gamma_0=self.plaid_gamma_0,
                gamma_1=self.plaid_gamma_1,
                device=str(device),
            )
            raw_model = modules["model"]
            diffusion_helper = PlaidDiffusionHelper(
                modules,
                sampling_timesteps=self.plaid_sampling_timesteps,
                score_temp=self.plaid_score_temp,
            )
            tokenizer = None
            max_nds = self.plaid_sampling_timesteps - 1

            # PLAID uses GPT-2 tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type!r}. "
                "Expected 'genie' or 'plaid'."
            )

        print(
            f"[Intervention] Model type={self.model_type}, layer={self.layer}",
            flush=True,
        )

        # --- Load SAE ---
        ckpt_path = self._discover_sae_checkpoint()
        lightning_module = SAELightningModule.load_trained(
            ckpt_path, map_location=str(device),
        )
        sae = lightning_module.sae.to(device)
        sae.eval()
        print(f"[Intervention] Loaded SAE from {ckpt_path}", flush=True)

        # --- Resolve features and validate ---
        sae_dict_size = sae.dictionary_size
        feature_indices = self._resolve_feature_indices(sae_dict_size)
        self._validate_nds_values(max_nds)

        print(
            f"[Intervention] Features: {feature_indices} "
            f"(selection={self.feature_selection})",
            flush=True,
        )
        print(
            f"[Intervention] Target magnitude={self.target_magnitude}, "
            f"NDS values={self.intervention_nds_values}",
            flush=True,
        )

        # --- Load dataset ---
        if self.model_type == "plaid":
            ds = load_dataset(self.dataset_name, split=self.dataset_split, streaming=True)
            texts: list[str] = []
            for i, row in enumerate(ds):
                if i >= self.max_samples:
                    break
                texts.append(row.get("document", row.get("text", "")))
            seq_len = 256
            encodings = tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=seq_len, return_tensors="pt",
            )
        else:
            ds = load_dataset(self.dataset_name, split=self.dataset_split)
            if self.max_samples < len(ds):
                ds = ds.select(range(self.max_samples))
            texts = list(ds["document"])
            encodings = tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )

        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        print(
            f"[Intervention] Dataset: {self.dataset_name}/{self.dataset_split}, "
            f"{len(dataset)} samples, {len(dataloader)} batches",
            flush=True,
        )

        # --- Create patcher and run ---
        patcher = FeatureInterventionPatcher(
            model=raw_model,
            sae=sae,
            diffusion_helper=diffusion_helper,
            config=self,
            layer_accessor=None,
            tokenizer=tokenizer,
        )

        spec = InterventionSpec(
            feature_indices=feature_indices,
            target_magnitude=self.target_magnitude,
            intervention_nds_values=self.intervention_nds_values,
        )

        result = patcher.run_intervention(dataloader, spec)

        # --- Save results ---
        output = {
            "metadata": {
                "model_type": self.model_type,
                "layer": self.layer,
                "feature_selection": self.feature_selection,
                "num_features_intervened": len(feature_indices),
            },
            "intervention_config": {
                "feature_indices": feature_indices,
                "target_magnitude": self.target_magnitude,
                "intervention_nds_values": self.intervention_nds_values,
            },
            "baseline": {
                "loss": result.baseline_loss,
                "perplexity": result.baseline_perplexity,
                "examples": result.baseline_examples,
            },
            "patched": {
                "loss": result.patched_loss,
                "perplexity": result.patched_perplexity,
                "examples": result.patched_examples,
            },
            "per_step_losses": {
                str(k): v for k, v in result.per_step_losses.items()
            },
        }

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[Intervention] Results saved to {out_path}", flush=True)
        print(
            f"[Intervention] Baseline loss={result.baseline_loss:.4f}, "
            f"Patched loss={result.patched_loss:.4f}",
            flush=True,
        )

        return str(out_path)
