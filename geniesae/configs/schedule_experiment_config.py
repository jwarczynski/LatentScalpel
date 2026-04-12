"""Exca config for denoising schedule modification experiments.

Runs the denoising process with modified schedules (compressed, extended,
reduced, or custom) and compares the resulting temporal profiles against
the standard schedule. Collects trajectory data at relative positions for
direct comparison, computes per-feature correlation scores, identifies
shifted features, and reports generation quality metrics.

Supports both Genie (discrete 2000-step) and PLAID (continuous VDM) models.

Single run (inline)::

    uv run python main.py run-schedule-experiment configs/schedule_experiment.yaml

Submit to Slurm::

    uv run python main.py run-schedule-experiment configs/schedule_experiment.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import math
import typing as tp
from pathlib import Path

import exca
import numpy as np
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.schedule_experiment")


class ScheduleExperimentConfig(BaseModel):
    """Runs schedule modification experiments comparing temporal profiles.

    Loads a model and SAE(s), runs the standard denoising trajectory,
    then runs a modified schedule trajectory, and computes comparison
    metrics (correlation scores, shifted features, quality metrics).
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
    sae_checkpoint_dir: str = Field(
        min_length=1,
        description="Directory containing layer_XX.ckpt SAE checkpoints.",
    )
    layers: list[int] = Field(
        min_length=1,
        description="Layer indices to analyze (must have trained SAEs).",
    )

    # -- Schedule modification ------------------------------------------------
    modification_type: str = Field(
        description=(
            'Schedule modification type: "compress_early", "extend_late", '
            '"reduce_total", or "custom".'
        ),
    )
    modification_params: dict = Field(
        default_factory=dict,
        description=(
            "Modification-specific parameters. "
            'compress_early: {"speedup_steps": N}, '
            'extend_late: {"extra_steps": N}, '
            'reduce_total: {"new_total": N}, '
            'custom: {"timesteps": [int, ...]}.'
        ),
    )

    # -- Trajectory collection ------------------------------------------------
    timestep_subsample: int = Field(
        default=20, gt=0,
        description="Sample every N-th timestep from the full trajectory.",
    )
    top_k_to_record: int = Field(
        default=64, gt=0,
        description="Number of top SAE features to record per timestep.",
    )

    # -- Diffusion (Genie) ----------------------------------------------------
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "xsum"
    dataset_split: str = "validation"
    max_samples: int = Field(default=50, gt=0)
    batch_size: int = Field(default=8, gt=0)

    # -- Output ---------------------------------------------------------------
    output_dir: str = "./experiments/schedule_experiment"

    # -- Device ---------------------------------------------------------------
    device: str = "cuda:0"
    random_seed: int = 42

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size",
    )

    # ------------------------------------------------------------------
    # SAE checkpoint discovery
    # ------------------------------------------------------------------

    def _discover_sae_checkpoints(self) -> dict[int, str]:
        """Find SAE checkpoint files for the configured layers.

        Returns:
            Mapping ``{layer_idx: checkpoint_path}``.

        Raises:
            FileNotFoundError: If checkpoint dir or required layers are missing.
        """
        import re

        sae_dir = Path(self.sae_checkpoint_dir)
        if not sae_dir.is_dir():
            raise FileNotFoundError(f"SAE checkpoint dir not found: {sae_dir}")

        pattern = re.compile(r"layer_(\d+)(_best)?\.ckpt$")
        found: dict[int, dict[str, str]] = {}

        for f in sorted(sae_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                layer_idx = int(m.group(1))
                kind = "best" if m.group(2) else "last"
                found.setdefault(layer_idx, {})[kind] = str(f)

        # Filter to requested layers
        found = {k: v for k, v in found.items() if k in self.layers}

        result: dict[int, str] = {}
        for layer_idx, paths in sorted(found.items()):
            if "best" in paths:
                result[layer_idx] = paths["best"]
            elif "last" in paths:
                result[layer_idx] = paths["last"]

        if not result:
            raise FileNotFoundError(
                f"No SAE checkpoints found in {sae_dir} "
                f"for layers {self.layers}"
            )

        return result

    # ------------------------------------------------------------------
    # Trajectory collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_trajectory_genie(
        raw_model: torch.nn.Module,
        nnsight_model,
        diffusion_helper,
        saes: dict[int, torch.nn.Module],
        layers: list[int],
        input_ids_all: torch.Tensor,
        attention_mask_all: torch.Tensor,
        timesteps_to_record: set[int],
        total_timesteps: int,
        batch_size: int,
        top_k_to_record: int,
        device: torch.device,
    ) -> tuple[dict[int, dict[int, dict[int, float]]], float]:
        """Run Genie reverse diffusion and collect SAE feature activations.

        Returns:
            Tuple of (results_dict, final_loss) where results_dict maps
            layer -> timestep -> {feature_id: mean_activation}.
        """
        import torch.nn.functional as F

        num_samples = input_ids_all.shape[0]
        results: dict[int, dict[int, dict[int, float]]] = {
            li: {} for li in layers
        }
        total_loss = 0.0
        loss_count = 0

        # Determine nnsight layer list
        if hasattr(nnsight_model, "transformer_blocks"):
            nnsight_layer_list = nnsight_model.transformer_blocks
        elif hasattr(nnsight_model, "input_transformers"):
            nnsight_layer_list = nnsight_model.input_transformers.layer
        else:
            raise RuntimeError("Cannot find decoder layers on nnsight model")

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            b_input_ids = input_ids_all[batch_start:batch_end].to(device)
            b_attention_mask = attention_mask_all[batch_start:batch_end].to(device)
            bs = b_input_ids.shape[0]

            with torch.no_grad():
                x_start = raw_model.get_embeds(b_input_ids)
            x_t = torch.randn_like(x_start)

            last_output = None

            for t_val in range(total_timesteps - 1, -1, -1):
                t_tensor = torch.full((bs,), t_val, dtype=torch.long, device=device)
                record = t_val in timesteps_to_record

                with torch.no_grad():
                    if record:
                        saved_outputs: list = []
                        with nnsight_model.trace(x_t, t_tensor, b_input_ids, b_attention_mask):
                            for li in layers:
                                saved_outputs.append(nnsight_layer_list[li].output.save())
                            model_output = nnsight_model.output.save()

                        x_start_pred = model_output.value if hasattr(model_output, "value") else model_output
                        if isinstance(x_start_pred, (tuple, list)):
                            x_start_pred = x_start_pred[0]
                        x_start_pred = x_start_pred.detach()
                    else:
                        x_start_pred = raw_model(x_t, t_tensor, b_input_ids, b_attention_mask)

                    x_t = diffusion_helper.p_sample(x_start_pred, x_t, t_tensor)
                    last_output = x_start_pred

                if record:
                    for idx, li in enumerate(layers):
                        act = saved_outputs[idx].value if hasattr(saved_outputs[idx], "value") else saved_outputs[idx]
                        if isinstance(act, (tuple, list)):
                            act = act[0]
                        act = act.detach().float()

                        if act.dim() == 3:
                            act_flat = act.reshape(-1, act.shape[-1])
                        else:
                            act_flat = act

                        sae = saes[li]
                        with torch.no_grad():
                            sparse_z = sae.encode(act_flat)

                        mean_act = sparse_z.mean(dim=0)
                        topk_vals, topk_ids = torch.topk(
                            mean_act, min(top_k_to_record, mean_act.shape[0])
                        )

                        feat_dict = {
                            int(fid.item()): float(fval.item())
                            for fid, fval in zip(topk_ids.cpu(), topk_vals.cpu())
                            if fval > 0
                        }

                        if t_val not in results[li]:
                            results[li][t_val] = {}
                        existing = results[li][t_val]
                        for fid, fval in feat_dict.items():
                            existing[fid] = existing.get(fid, 0.0) + fval

                    del saved_outputs, model_output
                    torch.cuda.empty_cache()

            # Compute loss from last output
            if last_output is not None:
                with torch.no_grad():
                    logits = raw_model.get_logits(last_output)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        b_input_ids.reshape(-1),
                        reduction="mean",
                    )
                    total_loss += loss.item()
                    loss_count += 1

        # Average across batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        for li in results:
            for t_val in results[li]:
                for fid in results[li][t_val]:
                    results[li][t_val][fid] /= num_batches

        avg_loss = total_loss / max(loss_count, 1)
        return results, avg_loss

    @staticmethod
    def _collect_trajectory_genie_modified(
        raw_model: torch.nn.Module,
        nnsight_model,
        modified_helper,
        modified_timesteps: list[int],
        saes: dict[int, torch.nn.Module],
        layers: list[int],
        input_ids_all: torch.Tensor,
        attention_mask_all: torch.Tensor,
        timesteps_to_record: set[int],
        batch_size: int,
        top_k_to_record: int,
        device: torch.device,
    ) -> tuple[dict[int, dict[int, dict[int, float]]], float]:
        """Run Genie reverse diffusion with a modified schedule.

        Instead of iterating over all timesteps, only visits the timesteps
        in the modified_timesteps list (which may skip or add steps).

        Returns:
            Tuple of (results_dict, final_loss).
        """
        import torch.nn.functional as F

        num_samples = input_ids_all.shape[0]
        results: dict[int, dict[int, dict[int, float]]] = {
            li: {} for li in layers
        }
        total_loss = 0.0
        loss_count = 0

        if hasattr(nnsight_model, "transformer_blocks"):
            nnsight_layer_list = nnsight_model.transformer_blocks
        elif hasattr(nnsight_model, "input_transformers"):
            nnsight_layer_list = nnsight_model.input_transformers.layer
        else:
            raise RuntimeError("Cannot find decoder layers on nnsight model")

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            b_input_ids = input_ids_all[batch_start:batch_end].to(device)
            b_attention_mask = attention_mask_all[batch_start:batch_end].to(device)
            bs = b_input_ids.shape[0]

            with torch.no_grad():
                x_start = raw_model.get_embeds(b_input_ids)
            x_t = torch.randn_like(x_start)

            last_output = None

            for t_val in modified_timesteps:
                t_tensor = torch.full((bs,), t_val, dtype=torch.long, device=device)
                record = t_val in timesteps_to_record

                with torch.no_grad():
                    if record:
                        saved_outputs: list = []
                        with nnsight_model.trace(x_t, t_tensor, b_input_ids, b_attention_mask):
                            for li in layers:
                                saved_outputs.append(nnsight_layer_list[li].output.save())
                            model_output = nnsight_model.output.save()

                        x_start_pred = model_output.value if hasattr(model_output, "value") else model_output
                        if isinstance(x_start_pred, (tuple, list)):
                            x_start_pred = x_start_pred[0]
                        x_start_pred = x_start_pred.detach()
                    else:
                        x_start_pred = raw_model(x_t, t_tensor, b_input_ids, b_attention_mask)

                    x_t = modified_helper.p_sample(x_start_pred, x_t, t_tensor)
                    last_output = x_start_pred

                if record:
                    for idx, li in enumerate(layers):
                        act = saved_outputs[idx].value if hasattr(saved_outputs[idx], "value") else saved_outputs[idx]
                        if isinstance(act, (tuple, list)):
                            act = act[0]
                        act = act.detach().float()

                        if act.dim() == 3:
                            act_flat = act.reshape(-1, act.shape[-1])
                        else:
                            act_flat = act

                        sae = saes[li]
                        with torch.no_grad():
                            sparse_z = sae.encode(act_flat)

                        mean_act = sparse_z.mean(dim=0)
                        topk_vals, topk_ids = torch.topk(
                            mean_act, min(top_k_to_record, mean_act.shape[0])
                        )

                        feat_dict = {
                            int(fid.item()): float(fval.item())
                            for fid, fval in zip(topk_ids.cpu(), topk_vals.cpu())
                            if fval > 0
                        }

                        if t_val not in results[li]:
                            results[li][t_val] = {}
                        existing = results[li][t_val]
                        for fid, fval in feat_dict.items():
                            existing[fid] = existing.get(fid, 0.0) + fval

                    del saved_outputs, model_output
                    torch.cuda.empty_cache()

            # Compute loss from last output
            if last_output is not None:
                with torch.no_grad():
                    logits = raw_model.get_logits(last_output)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        b_input_ids.reshape(-1),
                        reduction="mean",
                    )
                    total_loss += loss.item()
                    loss_count += 1

        # Average across batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        for li in results:
            for t_val in results[li]:
                for fid in results[li][t_val]:
                    results[li][t_val][fid] /= num_batches

        avg_loss = total_loss / max(loss_count, 1)
        return results, avg_loss

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_profiles(
        trajectory_results: dict[int, dict[int, float]],
        sampled_timesteps: list[int],
    ) -> dict[int, list[float]]:
        """Build per-feature activation profiles from trajectory results.

        Args:
            trajectory_results: Mapping timestep -> {feature_id: activation}.
            sampled_timesteps: Sorted list of timesteps that were recorded.

        Returns:
            Mapping feature_id -> [activation_at_t0, activation_at_t1, ...].
        """
        all_features: set[int] = set()
        for feats in trajectory_results.values():
            all_features.update(feats.keys())

        profiles: dict[int, list[float]] = {}
        for fid in sorted(all_features):
            profile = []
            for t in sampled_timesteps:
                feats = trajectory_results.get(t, {})
                profile.append(feats.get(fid, 0.0))
            profiles[fid] = profile

        return profiles

    @staticmethod
    def _compute_correlation(
        profile_a: list[float], profile_b: list[float],
    ) -> float:
        """Compute Pearson correlation between two activation profiles.

        Returns 0.0 if either profile has zero variance.
        """
        a = np.array(profile_a)
        b = np.array(profile_b)

        if len(a) == 0 or len(b) == 0:
            return 0.0

        # Truncate to the shorter length for comparison
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-10 or b_std < 1e-10:
            return 0.0

        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _find_shifted_features(
        original_profiles: dict[int, list[float]],
        modified_profiles: dict[int, list[float]],
        original_timesteps: list[int],
        modified_timesteps: list[int],
        total_steps: int,
        shift_threshold_pct: float = 0.10,
    ) -> list[int]:
        """Find features whose peak NDS shifted by more than threshold.

        Args:
            original_profiles: Feature profiles from standard schedule.
            modified_profiles: Feature profiles from modified schedule.
            original_timesteps: Sorted timesteps for original schedule.
            modified_timesteps: Sorted timesteps for modified schedule.
            total_steps: Total diffusion steps (for NDS normalization).
            shift_threshold_pct: Fraction of total steps for shift threshold.

        Returns:
            Sorted list of feature IDs with significant peak shift.
        """
        from geniesae.temporal_classifier import TemporalClassifier

        shifted = []
        common_features = set(original_profiles.keys()) & set(modified_profiles.keys())

        for fid in common_features:
            orig_profile = original_profiles[fid]
            mod_profile = modified_profiles[fid]

            if not orig_profile or not mod_profile:
                continue

            # Find peak timestep in each
            orig_peak_idx = int(np.argmax(orig_profile))
            mod_peak_idx = int(np.argmax(mod_profile))

            if orig_peak_idx < len(original_timesteps):
                orig_peak_t = original_timesteps[orig_peak_idx]
            else:
                continue
            if mod_peak_idx < len(modified_timesteps):
                mod_peak_t = modified_timesteps[mod_peak_idx]
            else:
                continue

            # Normalize both to [0, 1]
            orig_nds = TemporalClassifier.normalize_nds(orig_peak_t, total_steps)
            mod_nds = TemporalClassifier.normalize_nds(mod_peak_t, total_steps)

            if abs(orig_nds - mod_nds) > shift_threshold_pct:
                shifted.append(fid)

        return sorted(shifted)

    # ------------------------------------------------------------------
    # Main apply
    # ------------------------------------------------------------------

    @infra.apply
    @notify_on_completion("run-schedule-experiment")
    def apply(self) -> str:
        """Run the schedule modification experiment. Returns the output dir path."""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from geniesae.model_loader import load_genie_model, _resolve_device
        from geniesae.sae_lightning import SAELightningModule
        from geniesae.schedule_modifier import ScheduleModifier
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        device = _resolve_device(self.device)

        # -- Load model -------------------------------------------------------
        if self.model_type == "genie":
            print("[ScheduleExperiment] Loading GENIE model...", flush=True)
            nnsight_model, tokenizer = load_genie_model(self)
            raw_model = (
                nnsight_model._model
                if hasattr(nnsight_model, "_model")
                else nnsight_model
            )
            raw_model.eval()

            from geniesae.genie_model import DiffusionHelper

            standard_helper = DiffusionHelper(
                num_timesteps=self.diffusion_steps,
                schedule_name=self.noise_schedule,
            )
            total_steps = self.diffusion_steps

        elif self.model_type == "plaid":
            raise NotImplementedError(
                "PLAID schedule experiments are not yet implemented. "
                "Use model_type='genie'."
            )
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type!r}. "
                "Expected 'genie' or 'plaid'."
            )

        # -- Load SAEs --------------------------------------------------------
        print(f"[ScheduleExperiment] Loading SAEs for layers {self.layers}...", flush=True)
        ckpt_map = self._discover_sae_checkpoints()
        saes: dict[int, torch.nn.Module] = {}
        for layer_idx, ckpt_path in ckpt_map.items():
            module = SAELightningModule.load_trained(ckpt_path, map_location=str(device))
            module.sae.eval()
            module.sae.to(device)
            saes[layer_idx] = module.sae
            print(
                f"  Layer {layer_idx}: dict_size={module.sae.dictionary_size}, "
                f"k={module.sae.k}",
                flush=True,
            )

        # -- Load dataset -----------------------------------------------------
        print(
            f"[ScheduleExperiment] Loading dataset {self.dataset_name} "
            f"({self.dataset_split})...",
            flush=True,
        )
        ds = load_dataset(self.dataset_name, split=self.dataset_split)
        if self.max_samples < len(ds):
            ds = ds.select(range(self.max_samples))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = list(ds["document"])
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        input_ids_all = encodings["input_ids"]
        attention_mask_all = encodings["attention_mask"]

        # -- Standard schedule trajectory -------------------------------------
        print("[ScheduleExperiment] Collecting standard schedule trajectory...", flush=True)
        standard_timesteps = list(range(total_steps - 1, -1, -1))
        sampled_standard = standard_timesteps[::self.timestep_subsample]
        if 0 not in sampled_standard:
            sampled_standard.append(0)
        sampled_standard_set = set(sampled_standard)

        standard_results, standard_loss = self._collect_trajectory_genie(
            raw_model=raw_model,
            nnsight_model=nnsight_model,
            diffusion_helper=standard_helper,
            saes=saes,
            layers=self.layers,
            input_ids_all=input_ids_all,
            attention_mask_all=attention_mask_all,
            timesteps_to_record=sampled_standard_set,
            total_timesteps=total_steps,
            batch_size=self.batch_size,
            top_k_to_record=self.top_k_to_record,
            device=device,
        )
        standard_perplexity = math.exp(min(standard_loss, 100))
        print(
            f"[ScheduleExperiment] Standard: loss={standard_loss:.4f}, "
            f"perplexity={standard_perplexity:.2f}",
            flush=True,
        )

        # -- Modified schedule ------------------------------------------------
        print(
            f"[ScheduleExperiment] Modifying schedule: "
            f"type={self.modification_type}, params={self.modification_params}",
            flush=True,
        )
        modified_helper, modified_timesteps = ScheduleModifier.modify_genie_schedule(
            standard_helper, self.modification_type, self.modification_params,
        )

        # Determine which modified timesteps to record
        sampled_modified = modified_timesteps[::self.timestep_subsample]
        if modified_timesteps and modified_timesteps[-1] not in sampled_modified:
            sampled_modified.append(modified_timesteps[-1])
        sampled_modified_set = set(sampled_modified)

        print(
            f"[ScheduleExperiment] Modified schedule: {len(modified_timesteps)} steps, "
            f"recording {len(sampled_modified)} timesteps",
            flush=True,
        )

        # -- Modified schedule trajectory -------------------------------------
        print("[ScheduleExperiment] Collecting modified schedule trajectory...", flush=True)
        modified_results, modified_loss = self._collect_trajectory_genie_modified(
            raw_model=raw_model,
            nnsight_model=nnsight_model,
            modified_helper=modified_helper,
            modified_timesteps=modified_timesteps,
            saes=saes,
            layers=self.layers,
            input_ids_all=input_ids_all,
            attention_mask_all=attention_mask_all,
            timesteps_to_record=sampled_modified_set,
            batch_size=self.batch_size,
            top_k_to_record=self.top_k_to_record,
            device=device,
        )
        modified_perplexity = math.exp(min(modified_loss, 100))
        print(
            f"[ScheduleExperiment] Modified: loss={modified_loss:.4f}, "
            f"perplexity={modified_perplexity:.2f}",
            flush=True,
        )

        # -- Compute relative positions ---------------------------------------
        relative_positions = ScheduleModifier.compute_relative_positions(
            list(range(total_steps - 1, -1, -1)),
            modified_timesteps,
        )

        # -- Compute comparison metrics per layer -----------------------------
        print("[ScheduleExperiment] Computing comparison metrics...", flush=True)
        layers_output: dict[str, dict] = {}

        for li in self.layers:
            sorted_standard_ts = sorted(standard_results.get(li, {}).keys())
            sorted_modified_ts = sorted(modified_results.get(li, {}).keys())

            original_profiles = self._build_feature_profiles(
                standard_results.get(li, {}), sorted_standard_ts,
            )
            modified_profiles = self._build_feature_profiles(
                modified_results.get(li, {}), sorted_modified_ts,
            )

            # Correlation scores
            correlation_scores: dict[str, float] = {}
            common_features = set(original_profiles.keys()) & set(modified_profiles.keys())
            for fid in sorted(common_features):
                corr = self._compute_correlation(
                    original_profiles[fid], modified_profiles[fid],
                )
                correlation_scores[str(fid)] = round(corr, 4)

            # Shifted features
            shifted = self._find_shifted_features(
                original_profiles, modified_profiles,
                sorted_standard_ts, sorted_modified_ts,
                total_steps,
            )

            layers_output[str(li)] = {
                "original_profiles": {
                    str(fid): profile
                    for fid, profile in original_profiles.items()
                },
                "modified_profiles": {
                    str(fid): profile
                    for fid, profile in modified_profiles.items()
                },
                "correlation_scores": correlation_scores,
                "shifted_features": shifted,
                "quality_metrics": {
                    "original": {
                        "loss": round(standard_loss, 4),
                        "perplexity": round(standard_perplexity, 2),
                    },
                    "modified": {
                        "loss": round(modified_loss, 4),
                        "perplexity": round(modified_perplexity, 2),
                    },
                },
            }

        # -- Save results -----------------------------------------------------
        output = {
            "metadata": {
                "model_type": self.model_type,
                "model_checkpoint": self.model_checkpoint_path,
                "modification_type": self.modification_type,
                "modification_params": self.modification_params,
                "layers": self.layers,
                "diffusion_steps": total_steps,
                "timestep_subsample": self.timestep_subsample,
                "sampled_standard_timesteps": sorted(sampled_standard),
                "sampled_modified_timesteps": sorted(sampled_modified),
                "relative_positions_sample": relative_positions[:10],
                "num_samples": len(texts),
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
            },
            "layers": layers_output,
        }

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "schedule_comparison.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[ScheduleExperiment] Results saved to {out_path}", flush=True)
        return str(out_dir)
