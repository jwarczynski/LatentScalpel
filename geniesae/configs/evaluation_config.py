"""Exca config for evaluation stage."""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.evaluation")


class EvaluationConfig(BaseModel):
    """Exca config that evaluates SAE reconstruction impact on model loss and generation quality."""

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

    # Input paths
    sae_checkpoint_dir: str = Field(min_length=1)
    activation_dir: str = Field(min_length=1)

    # Dataset
    dataset_name: str = "xsum"
    dataset_split: str = "train"
    max_samples: int = Field(default=1000, gt=0)

    # Diffusion
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"
    diffusion_timesteps: list[int] = Field(default=[100, 200, 300, 400, 500], min_length=1)

    # Evaluation
    batch_size: int = Field(default=16, gt=0)
    num_generation_samples: int = Field(default=50, gt=0)
    device: str = "cuda:0"

    # Output
    output_dir: str = "./experiments/results"

    # Exca infra
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "num_generation_samples",
    )

    @infra.apply
    def apply(self) -> dict:
        """Run evaluation and return results dict."""
        import json
        import torch
        from datasets import load_dataset
        from torch.utils.data import DataLoader, TensorDataset

        from geniesae.activation_collector import ActivationStore
        from geniesae.evaluator import Evaluator
        from geniesae.model_loader import load_genie_model
        from geniesae.sae import TopKSAE
        from geniesae.sae_lightning import SAELightningModule

        # Load model
        nnsight_model, tokenizer = load_genie_model(self)
        raw_model = nnsight_model._model if hasattr(nnsight_model, "_model") else nnsight_model

        # Load SAE checkpoints
        store = ActivationStore(self.activation_dir)
        meta = store.metadata
        num_layers = meta["num_layers"]

        sae_dir = Path(self.sae_checkpoint_dir)
        saes: dict[int, TopKSAE] = {}
        for layer_idx in range(num_layers):
            ckpt_path = sae_dir / f"layer_{layer_idx:02d}.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
            lightning_module = SAELightningModule.load_from_checkpoint(
                str(ckpt_path), map_location=self.device,
            )
            saes[layer_idx] = lightning_module.sae.to(self.device)

        # Load evaluation dataset
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

        # Run evaluation
        evaluator = Evaluator(
            model=raw_model,
            saes=saes,
            config=self,
            activation_store=store,
        )
        results = evaluator.run(dataloader, num_generation_samples=self.num_generation_samples)

        # Save results
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation results saved to %s", results_path)
        return results
