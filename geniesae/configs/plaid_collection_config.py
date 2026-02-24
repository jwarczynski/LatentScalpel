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
    dataset_split: str = "train"
    max_samples: int = Field(default=10000, gt=0)
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

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "force_overwrite",
    )

    @infra.apply
    @notify_on_completion("collect-plaid-activations")
    def apply(self) -> str:
        """Collect activations from PLAID transformer blocks."""
        import nnsight
        from tokenizers import Tokenizer

        from geniesae.activation_collector import ActivationStore
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

        # Wrap model in nnsight for layer hooking
        nnsight_model = nnsight.NNsight(model)

        # -- Load dataset -----------------------------------------------------
        print(f"[PlaidCollection] Loading dataset {self.dataset_name}...", flush=True)
        from datasets import load_dataset as hf_load_dataset

        # Use HF tokenizer for OWT (PLAID uses a custom BPE tokenizer)
        # We'll use GPT-2 tokenizer as a compatible BPE tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = hf_load_dataset(self.dataset_name, split=self.dataset_split, trust_remote_code=True)
        if self.max_samples < len(ds):
            ds = ds.select(range(self.max_samples))

        # Tokenize
        text_key = "text" if "text" in ds.column_names else ds.column_names[0]
        texts = list(ds[text_key])
        encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.seq_len, return_tensors="pt",
        )
        input_ids_all = encodings["input_ids"]

        # -- Collect activations ----------------------------------------------
        store = ActivationStore(str(store_dir))
        num_layers = self.n_blocks
        activation_dim = self.dim  # transformer hidden dim
        total_samples = 0
        batch_counter = 0

        # Map t values to integer "timestep" indices for storage compatibility
        t_to_idx = {t_val: int(t_val * 1000) for t_val in self.diffusion_t_values}

        print(f"[PlaidCollection] Collecting at {len(self.diffusion_t_values)} noise levels, "
              f"{len(texts)} samples, batch_size={self.batch_size}", flush=True)

        for t_idx, t_val in enumerate(self.diffusion_t_values):
            timestep_idx = t_to_idx[t_val]
            print(f"  Noise level t={t_val:.2f} ({t_idx+1}/{len(self.diffusion_t_values)})...", flush=True)

            for batch_start in range(0, len(texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts))
                b_ids = input_ids_all[batch_start:batch_end].to(device)
                bs = b_ids.shape[0]

                with torch.no_grad():
                    # Get embeddings
                    embedding_matrix = embedding_matrix_module()
                    x_embed = embedding_matrix[b_ids]  # (bs, seq_len, embed_dim)

                    # Add noise at time t
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

                    # Trace through model to capture block outputs
                    saved_outputs = []
                    with nnsight_model.trace(
                        z, gamma.float(), embedding_matrix, 1.0, x_selfcond
                    ):
                        for layer in nnsight_model.blocks:
                            saved_outputs.append(layer.output.save())

                    # Save activations per layer
                    for layer_idx, proxy in enumerate(saved_outputs):
                        act = proxy.value if hasattr(proxy, "value") else proxy
                        if isinstance(act, (tuple, list)):
                            act = act[0]
                        act = act.detach().float()

                        if act.dim() == 3:
                            act = act.reshape(-1, act.shape[-1])

                        total_samples += act.shape[0]
                        store.save_activations(layer_idx, timestep_idx, batch_counter, act)

                batch_counter += 1

            store.flush_timestep(timestep_idx)
            print(f"    Flushed timestep t={t_val:.2f}", flush=True)

        # -- Save metadata ----------------------------------------------------
        metadata = {
            "model": "plaid-1b",
            "num_layers": num_layers,
            "activation_dim": activation_dim,
            "timesteps": [t_to_idx[t] for t in self.diffusion_t_values],
            "t_values": self.diffusion_t_values,
            "num_samples": total_samples,
            "seq_len": self.seq_len,
        }
        store.save_metadata(metadata)

        print(f"[PlaidCollection] Done: {num_layers} layers, {total_samples} samples, "
              f"dim={activation_dim}", flush=True)
        return str(store_dir)
