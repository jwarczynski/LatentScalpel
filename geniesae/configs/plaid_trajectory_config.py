"""Exca config for collecting SAE feature activations along PLAID's denoising trajectory.

Same as trajectory_config.py but adapted for PLAID's continuous diffusion:
- Uses continuous t in [0,1] instead of discrete timesteps
- Uses PLAID's VDM reverse sampling instead of GENIE's p_sample
- Hooks into model.blocks instead of transformer_blocks

Usage:
    uv run python main.py collect-plaid-trajectory configs/plaid_trajectory.yaml
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

logger = logging.getLogger("geniesae.configs.plaid_trajectory")


class PlaidTrajectoryConfig(BaseModel):
    """Collect SAE feature activations along PLAID's full denoising trajectory."""

    # -- Model ----------------------------------------------------------------
    weights_path: str = Field(min_length=1)
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to Lightning .ckpt (fine-tuned). Overrides weights_path.",
    )
    dim: int = Field(default=2048, gt=0)
    embed_dim: int = Field(default=16, gt=0)
    n_blocks: int = Field(default=24, gt=0)
    n_heads: int = Field(default=32, gt=0)
    vocab_size: int = Field(default=32768, gt=0)
    gamma_0: float = -3.0
    gamma_1: float = 6.0

    # -- SAE ------------------------------------------------------------------
    sae_checkpoint_dir: str = Field(min_length=1)
    layers: list[int] = Field(min_length=1)

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "openwebtext"
    dataset_split: str = "train"
    max_samples: int = Field(default=50, gt=0)
    seq_len: int = Field(default=256, gt=0)
    data_dir: str | None = Field(
        default=None,
        description="Path to XSum .src/.tgt dir. When set, uses XSum instead of HF dataset.",
    )
    tokenizer_path: str | None = Field(
        default=None,
        description="PLAID tokenizer path. Required with data_dir.",
    )

    # -- Diffusion ------------------------------------------------------------
    # NOTE: Standardised to 256 steps to match plaid_evaluation and other
    # PLAID pipelines.  With 256 steps we record every step (subsample=1)
    # giving 256 trajectory data-points — sufficient resolution without
    # the memory/time cost of 4096 steps.
    sampling_timesteps: int = Field(default=256, gt=0)
    score_temp: float = 0.9
    timestep_subsample: int = Field(
        default=1, gt=0,
        description="Sample every N-th step from the full trajectory.",
    )

    # -- Collection -----------------------------------------------------------
    batch_size: int = Field(default=2, gt=0)
    top_k_to_record: int = Field(default=64, gt=0)
    device: str = "cuda:0"
    random_seed: int = 42
    output_path: str = "./experiments/plaid_trajectory_features.json"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device", "batch_size")

    @infra.apply
    @notify_on_completion("collect-plaid-trajectory")
    def apply(self) -> str:
        """Run PLAID's full denoising trajectory and collect SAE feature activations."""
        import nnsight
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from geniesae.plaid_model import load_plaid_modules, PlaidDiffusionHelper
        from geniesae.sae_lightning import SAELightningModule
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # -- Load model -------------------------------------------------------
        print("[PlaidTrajectory] Loading PLAID model...", flush=True)
        if self.checkpoint_path is not None:
            from geniesae.plaid_xsum_training import PlaidXSumTrainingModule
            ckpt_module = PlaidXSumTrainingModule.load_from_checkpoint(
                self.checkpoint_path, map_location=device,
            )
            ckpt_module.eval()
            ckpt_module.to(device)
            modules = {
                "model": ckpt_module.diffusion_model,
                "embedding_matrix": ckpt_module.embedding_matrix,
                "noise_schedule": ckpt_module.noise_schedule,
                "gamma_bounds": ckpt_module.gamma_bounds,
            }
            print(f"[PlaidTrajectory] Loaded fine-tuned checkpoint: {self.checkpoint_path}", flush=True)
        else:
            modules = load_plaid_modules(
                self.weights_path,
                dim=self.dim, embed_dim=self.embed_dim,
                n_blocks=self.n_blocks, n_heads=self.n_heads,
                vocab_size=self.vocab_size,
                gamma_0=self.gamma_0, gamma_1=self.gamma_1,
                device=str(device),
            )
        model = modules["model"]
        diffusion = PlaidDiffusionHelper(modules, self.sampling_timesteps, self.score_temp)
        nnsight_model = nnsight.NNsight(model)

        # -- Load SAEs --------------------------------------------------------
        print(f"[PlaidTrajectory] Loading SAEs for layers {self.layers}...", flush=True)
        saes: dict[int, torch.nn.Module] = {}
        for layer_idx in self.layers:
            ckpt_path = Path(self.sae_checkpoint_dir) / f"layer_{layer_idx:02d}.ckpt"
            if not ckpt_path.exists():
                ckpt_path = Path(self.sae_checkpoint_dir) / f"layer_{layer_idx:02d}_best.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"No SAE checkpoint for layer {layer_idx}")
            module = SAELightningModule.load_trained(str(ckpt_path), map_location=str(device))
            module.sae.eval()
            module.sae.to(device)
            saes[layer_idx] = module.sae
            print(f"  Layer {layer_idx}: dict_size={module.sae.dictionary_size}", flush=True)

        # -- Load dataset -----------------------------------------------------
        print(f"[PlaidTrajectory] Loading dataset...", flush=True)

        if self.data_dir is not None:
            from tokenizers import Tokenizer as HFTokenizer
            tok_path = self.tokenizer_path or "models/plaid/plaid1b_weights/tokenizer.json"
            plaid_tokenizer = HFTokenizer.from_file(tok_path)

            data_path = Path(self.data_dir)
            split_prefix = {"train": "train", "validation": "dev", "test": "test"}
            src_file = data_path / f"{split_prefix.get(self.dataset_split, self.dataset_split)}.src"
            src_lines = src_file.read_text().strip().split("\n")
            if self.max_samples < len(src_lines):
                src_lines = src_lines[:self.max_samples]

            all_ids = []
            for line in src_lines:
                ids = plaid_tokenizer.encode(line).ids[:self.seq_len]
                ids = ids + [0] * (self.seq_len - len(ids))
                all_ids.append(ids)
            input_ids_all = torch.tensor(all_ids, dtype=torch.long)
            texts = src_lines
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            ds = load_dataset(self.dataset_name, split=self.dataset_split, trust_remote_code=True)
            if self.max_samples < len(ds):
                ds = ds.select(range(self.max_samples))

            text_key = "text" if "text" in ds.column_names else ds.column_names[0]
            texts = list(ds[text_key])
            encodings = tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=self.seq_len, return_tensors="pt",
            )
            input_ids_all = encodings["input_ids"]

        # -- Determine timesteps to sample ------------------------------------
        all_steps = list(range(self.sampling_timesteps))
        sampled_steps = all_steps[::self.timestep_subsample]
        if self.sampling_timesteps - 1 not in sampled_steps:
            sampled_steps.append(self.sampling_timesteps - 1)
        sampled_set = set(sampled_steps)
        # Map step index to continuous t value
        step_to_t = {i: 1.0 - i / self.sampling_timesteps for i in range(self.sampling_timesteps)}

        print(f"[PlaidTrajectory] {len(sampled_steps)} steps to record "
              f"(subsample every {self.timestep_subsample} from {self.sampling_timesteps})", flush=True)

        # -- Results structure ------------------------------------------------
        results: dict[int, dict[int, dict[int, float]]] = {li: {} for li in self.layers}

        # -- Process in batches -----------------------------------------------
        embedding_matrix = modules["embedding_matrix"]()
        num_samples = len(texts)

        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            b_ids = input_ids_all[batch_start:batch_end].to(device)
            bs = b_ids.shape[0]

            print(f"[PlaidTrajectory] Batch {batch_start // self.batch_size + 1}"
                  f"/{(num_samples + self.batch_size - 1) // self.batch_size}", flush=True)

            with torch.no_grad():
                # Start from pure noise
                z = torch.randn(bs, self.seq_len, self.embed_dim, device=device).float()
                x_selfcond = torch.zeros_like(z)

                for step_i in range(self.sampling_timesteps):
                    t_val = 1.0 - step_i / self.sampling_timesteps
                    s_val = t_val - 1.0 / self.sampling_timesteps
                    t_tensor = torch.full((bs,), t_val, device=device)
                    s_tensor = torch.full((bs,), max(s_val, 0.0), device=device)

                    record = step_i in sampled_set

                    if record:
                        # Use nnsight to capture layer activations
                        gamma_t = diffusion.get_gamma(t_tensor)
                        emb_mat = embedding_matrix.detach()

                        saved_outputs = []
                        with nnsight_model.trace(
                            z, gamma_t.float(), emb_mat, 1.0, x_selfcond
                        ):
                            for li in self.layers:
                                saved_outputs.append(nnsight_model.blocks[li].output.save())
                            model_output = nnsight_model.output.save()

                        # Extract model output for reverse step
                        out_val = model_output.value if hasattr(model_output, "value") else model_output
                        if isinstance(out_val, (tuple, list)):
                            logits, x_reconst = out_val[0], out_val[1]
                        else:
                            x_reconst = out_val

                        x_selfcond = x_reconst.clone().detach().float()
                        x_reconst_d = x_reconst.double()

                        # Reverse step
                        gamma_t_d = gamma_t.double()
                        gamma_s = diffusion.get_gamma(s_tensor)
                        alpha_sq_t = torch.sigmoid(-gamma_t_d)
                        alpha_sq_s = torch.sigmoid(-gamma_s)
                        sigma_t = torch.sigmoid(gamma_t_d).sqrt()
                        alpha_t = alpha_sq_t.sqrt()

                        eps = (z.double() - alpha_t[:, None, None] * x_reconst_d) / sigma_t[:, None, None]
                        eps /= diffusion.score_temp
                        x_reconst_d = (z.double() - sigma_t[:, None, None] * eps) / alpha_t[:, None, None]

                        if t_val > 1.0 / self.sampling_timesteps:
                            c = -torch.expm1(gamma_s - gamma_t_d)
                            c = c[:, None, None]  # (bs, 1, 1) for broadcasting
                            z_new = (1 - c) * alpha_sq_s.sqrt()[:, None, None] / alpha_sq_t.sqrt()[:, None, None] * z.double()
                            z_new += c * (alpha_sq_s.sqrt()[:, None, None] * x_reconst_d)
                            z_new += (c * (1 - alpha_sq_s[:, None, None])).sqrt() * torch.randn_like(z).double()
                            z = z_new.float()

                        # Record SAE features
                        for idx, li in enumerate(self.layers):
                            act = saved_outputs[idx].value if hasattr(saved_outputs[idx], "value") else saved_outputs[idx]
                            if isinstance(act, (tuple, list)):
                                act = act[0]
                            act = act.detach().float()
                            if act.dim() == 3:
                                act_flat = act.reshape(-1, act.shape[-1])
                            else:
                                act_flat = act

                            sae = saes[li]
                            sparse_z = sae.encode(act_flat)
                            mean_act = sparse_z.mean(dim=0)
                            topk_vals, topk_ids = torch.topk(
                                mean_act, min(self.top_k_to_record, mean_act.shape[0])
                            )
                            feat_dict = {
                                int(fid.item()): float(fval.item())
                                for fid, fval in zip(topk_ids.cpu(), topk_vals.cpu())
                                if fval > 0
                            }
                            del act, act_flat, sparse_z, mean_act

                            if step_i not in results[li]:
                                results[li][step_i] = {}
                            existing = results[li][step_i]
                            for fid, fval in feat_dict.items():
                                existing[fid] = existing.get(fid, 0.0) + fval

                        del saved_outputs, model_output
                        torch.cuda.empty_cache()
                    else:
                        # Non-sampled step: plain forward
                        z, x_selfcond = diffusion.p_sample_step(z, t_tensor, s_tensor, x_selfcond)

                    if step_i % 500 == 0:
                        print(f"    step {step_i}/{self.sampling_timesteps}", flush=True)

            print(f"  Batch done.", flush=True)

        # Average across batches
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for li in results:
            for step_i in results[li]:
                for fid in results[li][step_i]:
                    results[li][step_i][fid] /= num_batches

        # -- Save results -----------------------------------------------------
        output = {
            "metadata": {
                "model": "plaid-1b",
                "weights_path": self.weights_path,
                "sae_checkpoint_dir": self.sae_checkpoint_dir,
                "layers": self.layers,
                "sampling_timesteps": self.sampling_timesteps,
                "timestep_subsample": self.timestep_subsample,
                "sampled_steps": sorted(sampled_steps),
                "top_k_to_record": self.top_k_to_record,
                "num_samples": num_samples,
                "dataset_name": self.dataset_name,
            },
            "layers": {
                str(li): {str(s): feats for s, feats in sorted(results[li].items())}
                for li in results
            },
        }

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[PlaidTrajectory] Saved to {out_path}", flush=True)
        return str(out_path)
