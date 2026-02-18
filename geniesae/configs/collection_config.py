"""Exca config for activation collection stage."""

from __future__ import annotations

import json
import logging
import shutil
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.collection")


class ActivationCollectionConfig(BaseModel):
    """Exca config that collects transformer block activations to disk.

    Supports append mode: when activations already exist and force_overwrite
    is False, new activations are appended (batch file numbering continues).
    """

    # Model
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

    # Dataset
    dataset_name: str = "xsum"
    dataset_split: str = "train"
    max_samples: int = Field(default=10000, gt=0)

    # Diffusion
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"
    diffusion_timesteps: list[int] = Field(default=[100, 200, 300, 400, 500], min_length=1)

    # Collection
    batch_size: int = Field(default=16, gt=0)
    output_dir: str = "./experiments/activations"
    force_overwrite: bool = False
    device: str = "cuda:0"
    random_seed: int = 42

    # Exca infra
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "force_overwrite",
    )

    @infra.apply
    def apply(self) -> str:
        """Collect activations and return the output directory path."""
        import torch
        from datasets import load_dataset
        from torch.utils.data import DataLoader, TensorDataset

        from geniesae.activation_collector import ActivationCollector, ActivationStore
        from geniesae.model_loader import load_genie_model
        from geniesae.genie_model import DiffusionHelper
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        store_dir = Path(self.output_dir)

        # Handle force overwrite vs append
        existing_samples = 0
        batch_offset = 0
        if store_dir.exists():
            meta_path = store_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    existing_meta = json.load(f)
                existing_samples = existing_meta.get("num_samples", 0)
                if self.force_overwrite:
                    logger.info("Force overwrite: removing existing activations at %s", store_dir)
                    shutil.rmtree(store_dir)
                    existing_samples = 0
                else:
                    for layer_dir in sorted(store_dir.glob("layer_*")):
                        n_files = len(list(layer_dir.glob("*.pt")))
                        batch_offset = max(batch_offset, n_files)
                    logger.info(
                        "Append mode: found %d existing samples, batch_offset=%d",
                        existing_samples, batch_offset,
                    )

        # Load model and tokenizer
        nnsight_model, tokenizer = load_genie_model(self)

        # Load dataset
        logger.info("Loading dataset '%s' split='%s' max_samples=%d",
                     self.dataset_name, self.dataset_split, self.max_samples)
        ds = load_dataset(self.dataset_name, split=self.dataset_split)
        if self.max_samples < len(ds):
            ds = ds.select(range(self.max_samples))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts: list[str] = list(ds["document"])
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Collect activations
        raw_model = nnsight_model._model if hasattr(nnsight_model, "_model") else nnsight_model
        collector = ActivationCollector(
            raw_model,
            self,
            layer_accessor=lambda m: m.transformer_blocks,
        )

        store = ActivationStore(str(store_dir))
        layers = collector._get_layers()
        num_layers = len(layers)
        timesteps = self.diffusion_timesteps

        from geniesae.model_loader import _resolve_device
        device = _resolve_device(self.device)
        activation_dim = None
        new_samples = 0
        batch_counter = batch_offset

        raw_model.eval()
        diffusion = DiffusionHelper(
            num_timesteps=self.diffusion_steps,
            schedule_name=self.noise_schedule,
        )

        for ts_idx, timestep in enumerate(timesteps):
            for bi, batch in enumerate(dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                bs = input_ids.shape[0]

                with torch.no_grad():
                    x_start = raw_model.get_embeds(input_ids)

                t_tensor = torch.full((bs,), timestep, dtype=torch.long, device=device)
                x_noised = diffusion.q_sample(x_start, t_tensor)

                saved_outputs = []
                with collector._nnsight_model.trace(x_noised, t_tensor, input_ids, attention_mask):
                    for layer in layers:
                        saved_outputs.append(layer.output.save())

                for layer_idx, proxy in enumerate(saved_outputs):
                    act = proxy.value if hasattr(proxy, "value") else proxy
                    if isinstance(act, (tuple, list)):
                        act = act[0]
                    act = act.detach().float()
                    if act.dim() >= 3:
                        act = act.reshape(-1, act.shape[-1])

                    if activation_dim is None:
                        activation_dim = act.shape[-1]

                    new_samples += act.shape[0]
                    store.save_activations(layer_idx, timestep, batch_counter, act)

                batch_counter += 1

            store.flush_timestep(timestep)
            logger.info("Completed timestep %d (%d/%d)", timestep, ts_idx + 1, len(timesteps))

        metadata = {
            "num_layers": num_layers,
            "activation_dim": activation_dim or 0,
            "timesteps": timesteps,
            "num_samples": existing_samples + new_samples,
        }
        store.save_metadata(metadata)

        logger.info(
            "Activation collection complete: %d layers, %d new samples (total %d), dim=%d",
            num_layers, new_samples, existing_samples + new_samples, activation_dim or 0,
        )
        return str(store_dir)
