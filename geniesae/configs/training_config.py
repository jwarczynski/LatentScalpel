"""Exca config for SAE training stage.

Trains a Top-K SAE for a single decoder layer.

Single layer (inline)::

    uv run python main.py train-sae configs/train_sae.yaml --layer_idx=1

Resume training from checkpoint::

    uv run python main.py train-sae configs/train_sae.yaml --layer_idx=1 \
        --resume_from=./experiments/sae_checkpoints/layer_01.ckpt

Single layer (slurm via exca)::

    uv run python main.py train-sae configs/train_sae.yaml --layer_idx=1 \
        --submit --infra.cluster=slurm

Multiple layers (slurm job array via exca)::

    uv run python main.py train-sae configs/train_sae.yaml \
        --layers 0 1 2 3 4 5 --submit --infra.cluster=slurm

Resume multiple layers::

    uv run python main.py train-sae configs/train_sae.yaml \
        --layers 0 1 2 3 4 5 --submit --infra.cluster=slurm \
        --resume_from=./experiments/sae_checkpoints
"""

from __future__ import annotations

import json
import logging
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.training")


class SAETrainingConfig(BaseModel):
    """Config that trains a Top-K SAE for a single decoder layer.

    Reads ``metadata.json`` from ``activation_dir`` to discover
    activation dimensionality automatically.

    Separate activation directories can be provided for validation and
    test sets (collected from different dataset splits).  Lightning's
    ``limit_val_batches`` / ``limit_test_batches`` can throttle them.
    """

    activation_dir: str = Field(min_length=1)
    val_activation_dir: str | None = None
    test_activation_dir: str | None = None
    layer_idx: int = Field(ge=0)

    expansion_factor: int = Field(default=16, gt=0)
    k_target: int = Field(default=32, gt=0)
    k_start_multiplier: float = Field(default=4.0, gt=0)
    k_anneal_fraction: float = Field(default=0.1, gt=0)

    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=4096, gt=0)
    max_epochs: int = Field(default=5, gt=0)

    dead_feature_window: int = Field(default=1_000_000, gt=0)
    resample_dead_features: bool = True

    max_samples: int | None = Field(default=None, gt=0)

    # DataLoader performance
    num_workers: int = Field(default=3, ge=0)
    persistent_workers: bool = True
    pin_memory: bool = True
    matmul_precision: str = Field(default="high")

    # Checkpoint resume
    resume_from: str | None = Field(
        default=None,
        description=(
            "Path to a .ckpt file to resume training from. "
            "When used with --layers, can be a directory containing "
            "layer_XX.ckpt files."
        ),
    )

    # Best checkpoint tracking (Lightning ModelCheckpoint)
    save_best: bool = True
    monitor_metric: str = "val/mse"
    monitor_mode: str = "min"

    num_gpus: int = Field(default=1, gt=0)
    strategy: str = "auto"

    wandb_project: str = "genie-sae"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = Field(
        default=None,
        description=(
            "WandB run ID to resume logging into. Set this when resuming "
            "training so metrics continue on the same run instead of "
            "creating a new one. Find the ID in the WandB UI URL or "
            "run overview."
        ),
    )
    use_wandb: bool = True

    output_dir: str = "./experiments/sae_checkpoints"
    random_seed: int = 42

    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "num_gpus", "strategy", "batch_size", "wandb_project", "wandb_run_name",
        "wandb_run_id",
        "num_workers", "persistent_workers", "pin_memory", "matmul_precision",
    )

    def _load_metadata(self) -> dict:
        meta_path = Path(self.activation_dir) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.activation_dir}. "
                "Run activation collection first."
            )
        with open(meta_path) as f:
            return json.load(f)

    def resolve_checkpoint_path(self) -> str | None:
        """Resolve the checkpoint path for the current layer.

        If ``resume_from`` points to a directory, looks for checkpoint
        files matching ``layer_XX*.ckpt``.  When multiple exist (e.g.
        ``layer_00.ckpt`` and ``layer_00_best.ckpt``), the *last*
        checkpoint (most training) is preferred over the best one.
        If it points to a file, uses it directly.
        Returns ``None`` when no checkpoint is found.
        """
        if self.resume_from is None:
            return None
        p = Path(self.resume_from)
        if p.is_file():
            return str(p)
        if p.is_dir():
            # Prefer the "last" checkpoint (most epochs trained)
            last = p / f"layer_{self.layer_idx:02d}.ckpt"
            if last.exists():
                return str(last)
            # Fall back to best checkpoint
            best = p / f"layer_{self.layer_idx:02d}_best.ckpt"
            if best.exists():
                return str(best)
            logger.warning(
                "No checkpoint found for layer %d in %s", self.layer_idx, p,
            )
            return None
        logger.warning("resume_from path does not exist: %s", p)
        return None

    @infra.apply
    @notify_on_completion("train-sae")
    def apply(self) -> str:
        """Train SAE for a single layer and return checkpoint path."""
        import torch
        from torch.utils.data import DataLoader
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import ModelCheckpoint

        from geniesae.activation_collector import ActivationStore
        from geniesae.sae import TopKSAE
        from geniesae.sae_lightning import SAELightningModule
        from geniesae.utils import set_seed

        torch.set_float32_matmul_precision(self.matmul_precision)

        meta = self._load_metadata()
        activation_dim = meta["activation_dim"]
        num_layers = meta["num_layers"]

        if self.layer_idx >= num_layers:
            raise ValueError(
                f"layer_idx={self.layer_idx} out of range "
                f"(model has {num_layers} layers, 0-{num_layers - 1})"
            )

        dictionary_size = activation_dim * self.expansion_factor
        ckpt_path_to_resume = self.resolve_checkpoint_path()

        print(
            f"[SAETraining] Layer {self.layer_idx}: "
            f"dim={activation_dim}, dict={dictionary_size}, "
            f"k={self.k_target}, epochs={self.max_epochs}, bs={self.batch_size}"
            + (f", resuming from {ckpt_path_to_resume}" if ckpt_path_to_resume else ""),
            flush=True,
        )

        set_seed(self.random_seed)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        k_start = int(self.k_target * self.k_start_multiplier)

        # --- DataLoader kwargs ---
        dl_kwargs: dict = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        if self.num_workers > 0:
            dl_kwargs["persistent_workers"] = self.persistent_workers

        # --- Train dataloader ---
        store = ActivationStore(self.activation_dir)
        chunked_ds = store.get_chunked_dataset(
            self.layer_idx, shuffle=True, max_samples=self.max_samples,
        )
        dataloader = DataLoader(chunked_ds, **dl_kwargs)

        # --- Val dataloader (separate activation dir) ---
        val_dataloader = None
        if self.val_activation_dir is not None:
            val_store = ActivationStore(self.val_activation_dir)
            val_ds = val_store.get_chunked_dataset(
                self.layer_idx, shuffle=False,
            )
            val_dataloader = DataLoader(val_ds, **dl_kwargs)

        # --- Test dataloader (separate activation dir) ---
        test_dataloader = None
        if self.test_activation_dir is not None:
            test_store = ActivationStore(self.test_activation_dir)
            test_ds = test_store.get_chunked_dataset(
                self.layer_idx, shuffle=False,
            )
            test_dataloader = DataLoader(test_ds, **dl_kwargs)

        # --- Data mean (from train activations, cached to disk) ---
        print(f"[SAETraining] Computing data mean for layer {self.layer_idx}...", flush=True)
        data_mean = store.compute_layer_mean(self.layer_idx, max_samples=self.max_samples)

        # --- Model ---
        sae = TopKSAE(activation_dim, dictionary_size, self.k_target)
        sae.initialize_from_data(data_mean)

        steps_per_epoch = max(1, chunked_ds.total_samples // self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        k_anneal_steps = int(total_steps * self.k_anneal_fraction)

        lightning_module = SAELightningModule(
            sae=sae,
            learning_rate=self.learning_rate,
            k_target=self.k_target,
            k_start=k_start,
            k_anneal_steps=k_anneal_steps,
            dead_feature_window=self.dead_feature_window,
            resample_dead=self.resample_dead_features,
        )

        # --- Logger ---
        run_name = self.wandb_run_name or f"layer_{self.layer_idx:02d}"
        if self.use_wandb:
            from pytorch_lightning.loggers import WandbLogger
            wandb_kwargs: dict = dict(
                project=self.wandb_project,
                name=run_name,
                save_dir=str(output_dir),
            )
            if self.wandb_run_id:
                wandb_kwargs["id"] = self.wandb_run_id
                wandb_kwargs["resume"] = "must"
            pl_logger = WandbLogger(**wandb_kwargs)
        else:
            from pytorch_lightning.loggers import CSVLogger
            pl_logger = CSVLogger(save_dir=str(output_dir), name=run_name)

        # --- Callbacks ---
        callbacks = []
        if self.save_best and val_dataloader is not None:
            best_ckpt_path = str(output_dir / f"layer_{self.layer_idx:02d}_best.ckpt")
            callbacks.append(ModelCheckpoint(
                dirpath=str(output_dir),
                filename=f"layer_{self.layer_idx:02d}_best",
                monitor=self.monitor_metric,
                mode=self.monitor_mode,
                save_top_k=1,
                verbose=True,
            ))

        # --- Trainer ---
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=self.num_gpus,
            strategy=self.strategy,
            logger=pl_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=bool(callbacks),
            enable_progress_bar=True,
        )

        # --- Fit (with optional resume) ---
        print(f"[SAETraining] Starting training...", flush=True)
        trainer.fit(
            lightning_module,
            dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path_to_resume,
        )

        # --- Optional test ---
        if test_dataloader is not None:
            trainer.test(lightning_module, dataloaders=test_dataloader)

        # --- Save final (last) checkpoint ---
        save_path = output_dir / f"layer_{self.layer_idx:02d}.ckpt"
        trainer.save_checkpoint(str(save_path))
        print(f"[SAETraining] Saved last checkpoint to {save_path}", flush=True)
        return str(save_path)
