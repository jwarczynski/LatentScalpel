"""Exca config for collecting SAE feature activations along the denoising trajectory.

Runs the full reverse diffusion process (iterative denoising from pure noise),
extracts transformer activations at sampled timesteps, encodes them through
trained SAEs, and records per-feature activation magnitudes.

This produces the data needed to answer RQ1: whether different SAE features
are active at different phases of the denoising process.

Usage:
    uv run python main.py collect-trajectory configs/trajectory.yaml

Submit to Slurm:
    uv run python main.py collect-trajectory configs/trajectory.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.trajectory")


class TrajectoryConfig(BaseModel):
    """Collect SAE feature activations along the full denoising trajectory."""

    # -- Model ----------------------------------------------------------------
    model_checkpoint_path: str = Field(min_length=1)
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

    # -- SAE ------------------------------------------------------------------
    sae_checkpoint_dir: str = Field(
        min_length=1,
        description="Directory containing layer_XX.ckpt SAE checkpoints.",
    )
    layers: list[int] = Field(
        min_length=1,
        description="Layer indices to analyze (must have trained SAEs).",
    )

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "xsum"
    dataset_split: str = "validation"
    # NOTE: Use null (all examples) for reliable results. Small sample sizes
    # (e.g. 50) are insufficient for drawing conclusions from heatmap plots.
    max_samples: int | None = Field(default=None, description="Max examples (None=all)")

    # -- Diffusion ------------------------------------------------------------
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"
    timestep_subsample: int = Field(
        default=20, gt=0,
        description="Sample every N-th timestep from the full trajectory.",
    )

    # -- Collection -----------------------------------------------------------
    batch_size: int = Field(default=8, gt=0)
    top_k_to_record: int = Field(
        default=64, gt=0,
        description="Number of top SAE features to record per timestep.",
    )
    device: str = "cuda:0"
    random_seed: int = 42
    output_path: str = "./experiments/trajectory_features.json"
    checkpoint_interval: int = Field(
        default=100, gt=0,
        description="Save checkpoint every N batches for resume capability.",
    )

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size",
    )

    @infra.apply
    @notify_on_completion("collect-trajectory")
    def apply(self) -> str:
        """Run the full denoising trajectory and collect SAE feature activations."""
        import nnsight
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from geniesae.genie_model import DiffusionHelper
        from geniesae.model_loader import load_genie_model, get_decoder_layers, _resolve_device
        from geniesae.sae_lightning import SAELightningModule
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        device = _resolve_device(self.device)

        # -- Load model -------------------------------------------------------
        print("[Trajectory] Loading GENIE model...", flush=True)
        nnsight_model, tokenizer = load_genie_model(self)
        raw_model = nnsight_model._model if hasattr(nnsight_model, "_model") else nnsight_model
        raw_model.eval()

        diffusion = DiffusionHelper(
            num_timesteps=self.diffusion_steps,
            schedule_name=self.noise_schedule,
        )

        # -- Load SAEs --------------------------------------------------------
        print(f"[Trajectory] Loading SAEs for layers {self.layers}...", flush=True)
        saes: dict[int, torch.nn.Module] = {}
        for layer_idx in self.layers:
            ckpt_path = Path(self.sae_checkpoint_dir) / f"layer_{layer_idx:02d}.ckpt"
            if not ckpt_path.exists():
                # Try best checkpoint
                ckpt_path = Path(self.sae_checkpoint_dir) / f"layer_{layer_idx:02d}_best.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"No SAE checkpoint for layer {layer_idx} in {self.sae_checkpoint_dir}"
                )
            module = SAELightningModule.load_trained(str(ckpt_path), map_location=str(device))
            module.sae.eval()
            module.sae.to(device)
            saes[layer_idx] = module.sae
            print(f"  Layer {layer_idx}: dict_size={module.sae.dictionary_size}, k={module.sae.k}", flush=True)

        # -- Load dataset -----------------------------------------------------
        print(f"[Trajectory] Loading dataset {self.dataset_name} ({self.dataset_split})...", flush=True)
        ds = load_dataset(self.dataset_name, split=self.dataset_split)
        if self.max_samples is not None and self.max_samples < len(ds):
            ds = ds.select(range(self.max_samples))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = list(ds["document"])
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        input_ids_all = encodings["input_ids"]
        attention_mask_all = encodings["attention_mask"]

        # -- Determine timesteps to sample ------------------------------------
        # Full trajectory goes from T-1 down to 0
        all_timesteps = list(range(self.diffusion_steps - 1, -1, -1))
        sampled_timesteps = all_timesteps[::self.timestep_subsample]
        # Always include the last step (t=0) if not already
        if 0 not in sampled_timesteps:
            sampled_timesteps.append(0)
        sampled_set = set(sampled_timesteps)
        print(f"[Trajectory] {len(sampled_timesteps)} timesteps to record "
              f"(subsample every {self.timestep_subsample} from {self.diffusion_steps})", flush=True)

        # -- Get nnsight layer proxies ----------------------------------------
        nnsight_layers = []
        all_decoder_layers = get_decoder_layers(raw_model)
        num_model_layers = len(all_decoder_layers)
        for li in self.layers:
            if li >= num_model_layers:
                raise ValueError(f"layer {li} out of range (model has {num_model_layers})")

        # We need nnsight proxies for the layers we care about
        if hasattr(nnsight_model, "transformer_blocks"):
            nnsight_layer_list = nnsight_model.transformer_blocks
        elif hasattr(nnsight_model, "input_transformers"):
            nnsight_layer_list = nnsight_model.input_transformers.layer
        else:
            raise RuntimeError("Cannot find decoder layers on nnsight model")

        # -- Results structure ------------------------------------------------
        # results[layer_idx][timestep] = {feature_id: mean_activation, ...}
        results: dict[int, dict[int, dict[int, float]]] = {
            li: {} for li in self.layers
        }

        # -- Checkpoint handling ----------------------------------------------
        out_path = Path(self.output_path)
        if not out_path.suffix or out_path.suffix != ".json":
            checkpoint_dir = out_path
        else:
            checkpoint_dir = out_path.parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "_checkpoint.json"
        
        start_batch = 0
        num_batches_processed = 0
        
        # Try to resume from checkpoint
        if checkpoint_file.exists():
            print(f"[Trajectory] Found checkpoint at {checkpoint_file}, attempting resume...", flush=True)
            try:
                with open(checkpoint_file, "r") as f:
                    ckpt_data = json.load(f)
                start_batch = ckpt_data.get("next_batch_start", 0)
                num_batches_processed = ckpt_data.get("num_batches_processed", 0)
                # Restore results
                for li_str, timesteps_data in ckpt_data.get("results", {}).items():
                    li = int(li_str)
                    if li in results:
                        for t_str, feats in timesteps_data.items():
                            results[li][int(t_str)] = {int(k): v for k, v in feats.items()}
                print(f"[Trajectory] Resuming from batch {start_batch // self.batch_size + 1} "
                      f"({num_batches_processed} batches already processed)", flush=True)
            except Exception as e:
                print(f"[Trajectory] Failed to load checkpoint: {e}, starting fresh", flush=True)
                start_batch = 0
                num_batches_processed = 0
                results = {li: {} for li in self.layers}

        # -- Process in batches -----------------------------------------------
        num_samples = len(texts)
        total_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        for batch_start in range(start_batch, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            b_input_ids = input_ids_all[batch_start:batch_end].to(device)
            b_attention_mask = attention_mask_all[batch_start:batch_end].to(device)
            bs = b_input_ids.shape[0]

            print(f"[Trajectory] Batch {batch_start // self.batch_size + 1}"
                  f"/{total_batches}"
                  f" (samples {batch_start}-{batch_end - 1})", flush=True)

            with torch.no_grad():
                x_start = raw_model.get_embeds(b_input_ids)

            # Start from pure noise
            x_t = torch.randn_like(x_start)

            # Iterative denoising
            for t_val in range(self.diffusion_steps - 1, -1, -1):
                t_tensor = torch.full((bs,), t_val, dtype=torch.long, device=device)
                record = t_val in sampled_set

                with torch.no_grad():
                    if record:
                        # Use nnsight to capture layer activations
                        saved_outputs: list = []
                        with nnsight_model.trace(x_t, t_tensor, b_input_ids, b_attention_mask):
                            for li in self.layers:
                                saved_outputs.append(nnsight_layer_list[li].output.save())
                            model_output = nnsight_model.output.save()

                        x_start_pred = model_output.value if hasattr(model_output, "value") else model_output
                        if isinstance(x_start_pred, (tuple, list)):
                            x_start_pred = x_start_pred[0]
                        x_start_pred = x_start_pred.detach()
                    else:
                        # Non-sampled step: plain forward, no hooks
                        x_start_pred = raw_model(x_t, t_tensor, b_input_ids, b_attention_mask)

                    # Reverse diffusion step
                    x_t = diffusion.p_sample(x_start_pred, x_t, t_tensor)

                if t_val % 200 == 0:
                    print(f"    step {self.diffusion_steps - 1 - t_val}/{self.diffusion_steps} (t={t_val})", flush=True)

                # Record SAE features at sampled timesteps
                if record:
                    for idx, li in enumerate(self.layers):
                        act = saved_outputs[idx].value if hasattr(saved_outputs[idx], "value") else saved_outputs[idx]
                        if isinstance(act, (tuple, list)):
                            act = act[0]
                        act = act.detach().float()

                        # act shape: (batch, seq_len, dim) -> flatten to (batch*seq, dim)
                        if act.dim() == 3:
                            act_flat = act.reshape(-1, act.shape[-1])
                        else:
                            act_flat = act

                        # Encode through SAE
                        sae = saes[li]
                        with torch.no_grad():
                            sparse_z = sae.encode(act_flat)

                        # Get top-k features by mean activation across all tokens
                        mean_act = sparse_z.mean(dim=0)  # (dict_size,)
                        topk_vals, topk_ids = torch.topk(
                            mean_act, min(self.top_k_to_record, mean_act.shape[0])
                        )

                        feat_dict = {
                            int(fid.item()): float(fval.item())
                            for fid, fval in zip(topk_ids.cpu(), topk_vals.cpu())
                            if fval > 0
                        }

                        del act, act_flat, sparse_z, mean_act, topk_vals, topk_ids

                        # Accumulate across batches
                        if t_val not in results[li]:
                            results[li][t_val] = {}
                        existing = results[li][t_val]
                        for fid, fval in feat_dict.items():
                            existing[fid] = existing.get(fid, 0.0) + fval

                    # Free nnsight proxies
                    del saved_outputs, model_output
                    torch.cuda.empty_cache()

            num_batches_processed += 1
            print(f"  Batch done.", flush=True)
            
            # Save checkpoint periodically
            if num_batches_processed % self.checkpoint_interval == 0:
                print(f"[Trajectory] Saving checkpoint after {num_batches_processed} batches...", flush=True)
                ckpt_data = {
                    "next_batch_start": batch_start + self.batch_size,
                    "num_batches_processed": num_batches_processed,
                    "results": {
                        str(li): {str(t): feats for t, feats in results[li].items()}
                        for li in results
                    },
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(ckpt_data, f)
                print(f"[Trajectory] Checkpoint saved.", flush=True)

        # Average across batches
        print(f"[Trajectory] All batches complete. Averaging across {num_batches_processed} batches...", flush=True)
        for li in results:
            for t_val in results[li]:
                for fid in results[li][t_val]:
                    results[li][t_val][fid] /= num_batches_processed

        # -- Save results -----------------------------------------------------
        # Remove checkpoint file since we completed successfully
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"[Trajectory] Removed checkpoint file (completed successfully)", flush=True)
        
        out_path = Path(self.output_path)

        # If output_path looks like a directory (no .json suffix) or ends with /,
        # save per-layer files inside it. Otherwise save a single combined file.
        if not out_path.suffix or out_path.suffix != ".json":
            # Per-layer output mode
            out_path.mkdir(parents=True, exist_ok=True)
            common_meta = {
                "model_checkpoint": self.model_checkpoint_path,
                "sae_checkpoint_dir": self.sae_checkpoint_dir,
                "diffusion_steps": self.diffusion_steps,
                "timestep_subsample": self.timestep_subsample,
                "sampled_timesteps": sorted(sampled_timesteps),
                "top_k_to_record": self.top_k_to_record,
                "num_samples": num_samples,
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
            }
            for li in results:
                layer_output = {
                    "metadata": {**common_meta, "layer": li},
                    "timesteps": {
                        str(t): feats
                        for t, feats in sorted(results[li].items())
                    },
                }
                layer_file = out_path / f"layer_{li:02d}_trajectory.json"
                with open(layer_file, "w") as f:
                    json.dump(layer_output, f, indent=2)
                print(f"[Trajectory] Saved layer {li} to {layer_file}", flush=True)
            return str(out_path)
        else:
            # Single-file output mode (legacy)
            output = {
                "metadata": {
                    "model_checkpoint": self.model_checkpoint_path,
                    "sae_checkpoint_dir": self.sae_checkpoint_dir,
                    "layers": self.layers,
                    "diffusion_steps": self.diffusion_steps,
                    "timestep_subsample": self.timestep_subsample,
                    "sampled_timesteps": sorted(sampled_timesteps),
                    "top_k_to_record": self.top_k_to_record,
                    "num_samples": num_samples,
                    "dataset_name": self.dataset_name,
                    "dataset_split": self.dataset_split,
                },
                "layers": {
                    str(li): {
                        str(t): feats
                        for t, feats in sorted(results[li].items())
                    }
                    for li in results
                },
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"[Trajectory] Saved to {out_path}", flush=True)
            return str(out_path)
