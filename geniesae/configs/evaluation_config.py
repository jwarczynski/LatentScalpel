"""Exca config for evaluation stage.

Evaluates SAE reconstruction impact by patching activations in the
GENIE model's forward pass and measuring loss / perplexity changes.

Single layer evaluation::

    uv run python main.py evaluate configs/evaluation.yaml

Submit to Slurm::

    uv run python main.py evaluate configs/evaluation.yaml \
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.evaluation")


class EvaluationConfig(BaseModel):
    """Evaluates SAE reconstruction impact on model loss and perplexity.

    For each diffusion timestep, patches decoder layer activations with
    SAE reconstructions and measures cross-entropy loss vs baseline.
    Results are logged to WandB and saved as JSON.

    Only layers with available checkpoints are evaluated — no need to
    train all layers first.
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

    # SAE checkpoints
    sae_checkpoint_dir: str = Field(min_length=1)
    sae_layers: list[int] | None = Field(
        default=None,
        description=(
            "Layer indices to evaluate. null -> auto-detect from "
            "available layer_XX_best.ckpt / layer_XX.ckpt files."
        ),
    )
    prefer_best: bool = Field(
        default=True,
        description="Prefer layer_XX_best.ckpt over layer_XX.ckpt when both exist.",
    )

    # Dataset
    dataset_name: str = "xsum"
    dataset_split: str = "validation"
    max_samples: int = Field(default=500, gt=0)

    # Diffusion
    diffusion_steps: int = Field(default=2000, gt=0)
    noise_schedule: str = "sqrt"
    diffusion_timesteps: list[int] | str = Field(
        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        description=(
            'List of timesteps to evaluate, or "all" for every step '
            "from 1 to diffusion_steps."
        ),
    )
    diffusion_timestep_stride: int | None = Field(
        default=None,
        gt=0,
        description=(
            'When diffusion_timesteps is "all", evaluate every Nth step '
            "instead of all 2000. E.g. stride=100 -> [100, 200, ..., 2000]."
        ),
    )

    def resolve_timesteps(self) -> list[int]:
        """Resolve the timestep list, handling 'all' and stride."""
        if isinstance(self.diffusion_timesteps, str):
            if self.diffusion_timesteps.lower() == "all":
                stride = self.diffusion_timestep_stride or 1
                return list(range(stride, self.diffusion_steps + 1, stride))
            raise ValueError(
                f"diffusion_timesteps must be 'all' or a list of ints, "
                f"got {self.diffusion_timesteps!r}"
            )
        return self.diffusion_timesteps

    # Evaluation
    eval_mode: str = Field(
        default="single_step",
        description=(
            '"single_step": independent per-timestep evaluation. '
            '"iterative": full reverse diffusion chain from t=T-1 to t=0.'
        ),
    )
    batch_size: int = Field(default=16, gt=0)
    device: str = "cuda:0"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "genie-sae"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = Field(
        default=None,
        description="WandB run ID to resume. Continues logging into the same run.",
    )

    # Resume (single_step mode only)
    skip_timesteps: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of timesteps to skip from the beginning of the resolved "
            "timestep list. Use when resuming a crashed single_step run."
        ),
    )

    # Output
    output_dir: str = "./experiments/results"

    # Exca infra
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "use_wandb", "wandb_project", "wandb_run_name",
        "wandb_run_id", "skip_timesteps",
    )

    def _discover_sae_checkpoints(self) -> dict[int, str]:
        """Find available SAE checkpoint files.

        Returns a mapping ``{layer_idx: checkpoint_path}``.
        """
        import re

        sae_dir = Path(self.sae_checkpoint_dir)
        if not sae_dir.is_dir():
            raise FileNotFoundError(f"SAE checkpoint dir not found: {sae_dir}")

        # Scan for layer_XX.ckpt and layer_XX_best.ckpt
        pattern = re.compile(r"layer_(\d+)(_best)?\.ckpt$")
        found: dict[int, dict[str, str]] = {}  # layer -> {"last": path, "best": path}

        for f in sorted(sae_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                layer_idx = int(m.group(1))
                kind = "best" if m.group(2) else "last"
                found.setdefault(layer_idx, {})[kind] = str(f)

        # Filter to requested layers
        if self.sae_layers is not None:
            found = {k: v for k, v in found.items() if k in self.sae_layers}

        # Pick best or last per layer
        result: dict[int, str] = {}
        for layer_idx, paths in sorted(found.items()):
            if self.prefer_best and "best" in paths:
                result[layer_idx] = paths["best"]
            elif "last" in paths:
                result[layer_idx] = paths["last"]
            elif "best" in paths:
                result[layer_idx] = paths["best"]

        if not result:
            raise FileNotFoundError(
                f"No SAE checkpoints found in {sae_dir} "
                f"(layers={self.sae_layers})"
            )

        return result

    @infra.apply
    def apply(self) -> dict:
        """Run evaluation and return results dict."""
        import json
        import math

        import torch
        from datasets import load_dataset
        from torch.utils.data import DataLoader, TensorDataset

        from geniesae.model_loader import load_genie_model
        from geniesae.sae import TopKSAE
        from geniesae.sae_lightning import SAELightningModule

        torch.set_float32_matmul_precision("high")

        # --- Load GENIE model ---
        nnsight_model, tokenizer = load_genie_model(self)
        raw_model = nnsight_model._model if hasattr(nnsight_model, "_model") else nnsight_model

        # --- Discover and load SAE checkpoints ---
        ckpt_map = self._discover_sae_checkpoints()
        print(
            f"[Eval] Found SAE checkpoints for layers: "
            f"{sorted(ckpt_map.keys())}",
            flush=True,
        )

        saes: dict[int, TopKSAE] = {}
        for layer_idx, ckpt_path in ckpt_map.items():
            lightning_module = SAELightningModule.load_trained(
                ckpt_path, map_location=self.device,
            )
            saes[layer_idx] = lightning_module.sae.to(self.device)
            saes[layer_idx].eval()
            print(f"[Eval] Loaded SAE for layer {layer_idx} from {ckpt_path}", flush=True)

        # --- Load evaluation dataset ---
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

        print(
            f"[Eval] Dataset: {self.dataset_name}/{self.dataset_split}, "
            f"{len(dataset)} samples, {len(dataloader)} batches",
            flush=True,
        )

        # --- WandB logger ---
        wandb_run = None
        if self.use_wandb:
            import wandb
            mode_tag = "iter" if self.eval_mode == "iterative" else "ss"
            run_name = self.wandb_run_name or f"{mode_tag}_" + "_".join(
                f"L{l}" for l in sorted(saes.keys())
            )
            wandb_kwargs = dict(
                project=self.wandb_project,
                name=run_name,
                config={
                    "eval_mode": self.eval_mode,
                    "dataset": self.dataset_name,
                    "split": self.dataset_split,
                    "max_samples": self.max_samples,
                    "diffusion_timesteps": self.diffusion_timesteps,
                    "sae_layers": sorted(saes.keys()),
                    "batch_size": self.batch_size,
                },
            )
            if self.wandb_run_id:
                wandb_kwargs["id"] = self.wandb_run_id
                wandb_kwargs["resume"] = "must"
            wandb_run = wandb.init(**wandb_kwargs)

        # --- Run evaluation ---
        from geniesae.evaluator import Evaluator

        # Resolve "all" / stride into concrete timestep list
        resolved_timesteps = self.resolve_timesteps()
        if self.skip_timesteps > 0:
            resolved_timesteps = resolved_timesteps[self.skip_timesteps:]
            print(
                f"[Eval] Skipping first {self.skip_timesteps} timesteps (resume)",
                flush=True,
            )
        print(
            f"[Eval] Evaluating {len(resolved_timesteps)} timesteps "
            f"(first={resolved_timesteps[0]}, last={resolved_timesteps[-1]})",
            flush=True,
        )

        evaluator = Evaluator(
            model=raw_model,
            saes=saes,
            config=self,
            timesteps=resolved_timesteps,
        )
        results = evaluator.run(
            dataloader, wandb_run=wandb_run, eval_mode=self.eval_mode,
            step_offset=self.skip_timesteps,
        )

        # --- Save results ---
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Results saved to {results_path}", flush=True)

        if wandb_run is not None:
            wandb_run.finish()

        return results
