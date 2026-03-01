"""Exca config for collecting activations from the PLAID diffusion model.

PLAID uses a continuous noise schedule (gamma) instead of discrete timesteps.
We sample t values uniformly in [0, 1] and collect transformer block activations
at each noise level.

Usage:
    uv run python main.py collect-plaid-activations configs/plaid_activation_collection.yaml

Submit to Slurm:
    uv run python main.py collect-plaid-activations configs/plaid_activation_collection.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import shutil
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.plaid_collection")


class PlaidCollectionConfig(BaseModel):
    """Collect transformer block activations from PLAID for SAE training."""

    # -- Model ----------------------------------------------------------------
    weights_path: str = Field(min_length=1, description="Directory with PLAID .pt weight files")
    dim: int = Field(default=2048, gt=0)
    embed_dim: int = Field(default=16, gt=0)
    n_blocks: int = Field(default=24, gt=0)
    n_heads: int = Field(default=32, gt=0)
    vocab_size: int = Field(default=32768, gt=0)
    gamma_0: float = -3.0
    gamma_1: float = 6.0

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "openwebtext"
    # NOTE: openwebtext only has a "train" split. For activation collection
    # this is correct (feeds SAE training). For other stages, use skip_samples
    # to carve out a disjoint held-out slice.
    dataset_split: str = "train"
    max_samples: int = Field(default=10000, gt=0)
    skip_samples: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of samples to skip from the beginning of the dataset. "
            "Useful for creating validation splits from datasets that only "
            "have a train split (e.g. openwebtext)."
        ),
    )
    seq_len: int = Field(default=256, gt=0)

    # -- Diffusion sampling ---------------------------------------------------
    # Continuous t values in [0, 1] at which to collect activations
    # These correspond to different noise levels
    diffusion_t_values: list[float] = Field(
        default=[0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95],
        min_length=1,
        description="Continuous t values in [0,1] for noise levels to sample.",
    )

    # -- Collection -----------------------------------------------------------
    batch_size: int = Field(default=8, gt=0)
    output_dir: str = "./experiments/plaid_activations"
    force_overwrite: bool = False
    device: str = "cuda:0"
    random_seed: int = 42
    layers: list[int] | None = Field(
        default=None,
        description="Layer indices to collect. None = all layers.",
    )

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "force_overwrite",
    )

    @infra.apply
    @notify_on_completion("collect-plaid-activations")
    def apply(self) -> str:
        """Collect activations from PLAID transformer blocks.

        Uses plain PyTorch forward hooks instead of nnsight to minimize
        memory overhead (nnsight keeps a full computation graph which
        OOMs on a 1.28B-param model at scale).
        """
        import gc

        from geniesae.plaid_model import load_plaid_modules
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        store_dir = Path(self.output_dir)

        if store_dir.exists() and self.force_overwrite:
            logger.info("Force overwrite: removing %s", store_dir)
            shutil.rmtree(store_dir)

        # -- Load model -------------------------------------------------------
        print("[PlaidCollection] Loading PLAID model...", flush=True)
        modules = load_plaid_modules(
            self.weights_path,
            dim=self.dim, embed_dim=self.embed_dim,
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            gamma_0=self.gamma_0, gamma_1=self.gamma_1,
            device=str(device),
        )
        model = modules["model"]
        embedding_matrix_module = modules["embedding_matrix"]
        noise_schedule = modules["noise_schedule"]
        gamma_bounds = modules["gamma_bounds"]

        # -- Load dataset -----------------------------------------------------
        print(f"[PlaidCollection] Loading dataset {self.dataset_name}...", flush=True)
        from datasets import load_dataset as hf_load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Stream to avoid loading full 8M-row dataset into RAM
        ds = hf_load_dataset(
            self.dataset_name, split=self.dataset_split,
            streaming=True,
        )
        texts: list[str] = []
        target = self.skip_samples + self.max_samples
        for i, row in enumerate(ds):
            if i >= target:
                break
            if i < self.skip_samples:
                continue
            texts.append(row["text"])
        num_texts = len(texts)
        print(f"[PlaidCollection] Loaded {num_texts} samples via streaming", flush=True)

        encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.seq_len, return_tensors="pt",
        )
        input_ids_all = encodings["input_ids"]
        del texts, encodings
        gc.collect()

        # -- Set up forward hooks (lightweight, no computation graph) ----------
        layer_indices = self.layers if self.layers is not None else list(range(self.n_blocks))
        num_layers = len(layer_indices)
        activation_dim = self.dim
        total_samples = 0

        # Dict to capture hook outputs — cleared every batch
        captured: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                captured[layer_idx] = output.detach().cpu().float()
            return hook_fn

        handles = []
        for li in layer_indices:
            h = model.blocks[li].register_forward_hook(_make_hook(li))
            handles.append(h)

        t_to_idx = {t_val: int(t_val * 1000) for t_val in self.diffusion_t_values}

        print(f"[PlaidCollection] Collecting layers {layer_indices} at "
              f"{len(self.diffusion_t_values)} noise levels, "
              f"{num_texts} samples, batch_size={self.batch_size}", flush=True)

        # Write activations directly to disk per-batch to avoid RAM buildup.
        # Each batch file: layer_XX/ts_TTTT_batch_BBBBB.pt
        # After all batches for a timestep, concatenate into one file and
        # delete the per-batch files.
        store_dir.mkdir(parents=True, exist_ok=True)
        for li in layer_indices:
            (store_dir / f"layer_{li:02d}").mkdir(parents=True, exist_ok=True)

        for t_idx, t_val in enumerate(self.diffusion_t_values):
            timestep_idx = t_to_idx[t_val]
            print(f"  Noise level t={t_val:.2f} ({t_idx+1}/{len(self.diffusion_t_values)})...",
                  flush=True)

            batch_counter = 0
            for batch_start in range(0, num_texts, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_texts)
                b_ids = input_ids_all[batch_start:batch_end].to(device)
                bs = b_ids.shape[0]

                captured.clear()

                with torch.no_grad():
                    embedding_matrix = embedding_matrix_module()
                    b_ids_clamped = b_ids.clamp(max=self.vocab_size - 1)
                    x_embed = embedding_matrix[b_ids_clamped]

                    t_tensor = torch.full((bs,), t_val, device=device)
                    gamma_0, gamma_1 = gamma_bounds()
                    gamma_norm = noise_schedule(t_tensor).double()
                    gamma = gamma_0 + (gamma_1 - gamma_0) * gamma_norm

                    alpha_sq = torch.sigmoid(-gamma)[:, None, None]
                    sigma_sq = torch.sigmoid(gamma)[:, None, None]
                    noise = torch.randn_like(x_embed.double())
                    z = alpha_sq.sqrt() * x_embed.double() + sigma_sq.sqrt() * noise
                    z = z.float()

                    x_selfcond = torch.zeros_like(z)
                    model(z, gamma.float(), embedding_matrix, 1.0, x_selfcond)

                # Write each layer's activations directly to disk
                for li in layer_indices:
                    act = captured[li]
                    if act.dim() == 3:
                        act = act.reshape(-1, act.shape[-1])
                    total_samples += act.shape[0]
                    batch_path = store_dir / f"layer_{li:02d}" / f"ts_{timestep_idx:04d}_batch_{batch_counter:05d}.pt"
                    torch.save(act, batch_path)

                captured.clear()
                del b_ids, x_embed, z, x_selfcond, noise
                torch.cuda.empty_cache()
                batch_counter += 1

            # Concatenate per-batch files into one timestep file, then delete batches
            for li in layer_indices:
                layer_dir = store_dir / f"layer_{li:02d}"
                batch_files = sorted(layer_dir.glob(f"ts_{timestep_idx:04d}_batch_*.pt"))
                chunks = [torch.load(bf, map_location="cpu", weights_only=True) for bf in batch_files]
                combined = torch.cat(chunks, dim=0)
                torch.save(combined, layer_dir / f"timestep_{timestep_idx:04d}.pt")
                del chunks, combined
                for bf in batch_files:
                    bf.unlink()

            gc.collect()
            print(f"    Flushed timestep t={t_val:.2f}", flush=True)

        # Remove hooks
        for h in handles:
            h.remove()

        # -- Save metadata ----------------------------------------------------
        metadata = {
            "model": "plaid-1b",
            "num_layers": num_layers,
            "layer_indices": layer_indices,
            "activation_dim": activation_dim,
            "timesteps": [t_to_idx[t] for t in self.diffusion_t_values],
            "t_values": self.diffusion_t_values,
            "num_samples": total_samples,
            "seq_len": self.seq_len,
        }
        import json as _json
        meta_path = store_dir / "metadata.json"
        with open(meta_path, "w") as f:
            _json.dump(metadata, f, indent=2)

        print(f"[PlaidCollection] Done: {num_layers} layers, {total_samples} samples, "
              f"dim={activation_dim}", flush=True)
        return str(store_dir)
