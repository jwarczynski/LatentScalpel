"""Exca config for fine-tuning PLAID 1B on XSum summarization.

Drives the full pipeline: data loading, VLB training with Lightning,
conditional generation (inpainting / gradient guidance), and evaluation.

Usage:
    uv run python main.py finetune-plaid configs/plaid_xsum_finetune.yaml

Submit to Slurm:
    uv run python main.py finetune-plaid configs/plaid_xsum_finetune.yaml \
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.plaid_xsum")


class PlaidXSumConfig(BaseModel):
    """Full experiment config for PLAID XSum fine-tuning, generation, and evaluation."""

    # -- Model ----------------------------------------------------------------
    weights_path: str = Field(min_length=1)
    dim: int = Field(default=2048, gt=0)
    embed_dim: int = Field(default=16, gt=0)
    n_blocks: int = Field(default=24, gt=0)
    n_heads: int = Field(default=32, gt=0)
    vocab_size: int = Field(default=32768, gt=0)
    gamma_0: float = -3.0
    gamma_1: float = 6.0

    # -- Dataset --------------------------------------------------------------
    data_dir: str = "datasets/glge-released-dataset/easy/xsum_data/org_data"
    seq_len: int = Field(default=512, gt=0)
    max_summary_len: int = Field(default=64, gt=0)
    tokenizer_path: str | None = None

    # -- Training -------------------------------------------------------------
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    num_epochs: int = Field(default=10, gt=0)
    warmup_steps: int = Field(default=1000, ge=0)
    bias_warmup_steps: int = Field(default=5000, ge=0)
    target_bias_scale: float = 1.0
    self_cond_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    reconst_bs: int | None = None
    clip_quantile: float = Field(default=0.95, gt=0.0, le=1.0)
    freeze_layers: list[int] | None = None
    num_workers: int = Field(default=4, ge=0)

    # -- Training mode --------------------------------------------------------
    training_mode: str = Field(
        default="unconditional",
        description=(
            '"unconditional": VLB loss on full sequence (PLAID paper approach). '
            '"conditional": inpaint article during training, loss on summary only.'
        ),
    )

    # -- Distributed ----------------------------------------------------------
    strategy: str = Field(
        default="ddp",
        description='"ddp" or "fsdp". Use FSDP if model does not fit on a single GPU.',
    )
    num_gpus: int = Field(default=1, ge=1, le=16)
    precision: str = "bf16-mixed"
    gradient_checkpointing: bool = True

    # -- Sampling -------------------------------------------------------------
    sampling_timesteps: int = Field(default=256, gt=0)
    score_temp: float = 0.9
    guidance_scale: float = 1.0
    sampler: str = Field(
        default="inpainting",
        description='"inpainting" or "guidance"',
    )
    num_eval_samples: int = Field(default=5, gt=0)

    # -- Logging --------------------------------------------------------------
    use_wandb: bool = True
    wandb_project: str = "plaid-xsum"
    wandb_run_name: str | None = None
    log_interval: int = Field(default=50, gt=0)
    noise_schedule_log_interval: int = Field(default=500, gt=0)

    # -- Checkpointing --------------------------------------------------------
    output_dir: str = "./experiments/plaid_xsum"
    resume_from: str | None = None

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "batch_size",
        "num_workers",
        "use_wandb",
        "wandb_project",
        "wandb_run_name",
    )

    @infra.apply
    def apply(self) -> dict:
        """Run the full fine-tuning pipeline."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger

        from geniesae.plaid_xsum_training import PlaidXSumTrainingModule
        from geniesae.xsum_data import XSumDataModule

        # --- Data ---
        data_module = XSumDataModule(
            data_dir=self.data_dir,
            seq_len=self.seq_len,
            max_summary_len=self.max_summary_len,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tokenizer_path=self.tokenizer_path,
        )

        # --- Model ---
        training_module = PlaidXSumTrainingModule(
            dim=self.dim,
            embed_dim=self.embed_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            gamma_0=self.gamma_0,
            gamma_1=self.gamma_1,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            warmup_steps=self.warmup_steps,
            bias_warmup_steps=self.bias_warmup_steps,
            target_bias_scale=self.target_bias_scale,
            self_cond_prob=self.self_cond_prob,
            reconst_bs=self.reconst_bs,
            clip_quantile=self.clip_quantile,
            freeze_layers=self.freeze_layers,
            gradient_checkpointing=self.gradient_checkpointing,
            training_mode=self.training_mode,
            # Sampling params for epoch-end generation
            sampling_timesteps=self.sampling_timesteps,
            score_temp=self.score_temp,
            guidance_scale=self.guidance_scale,
            sampler=self.sampler,
            num_eval_samples=self.num_eval_samples,
            log_interval=self.log_interval,
            noise_schedule_log_interval=self.noise_schedule_log_interval,
            tokenizer_path=self.tokenizer_path,
        )
        training_module.load_pretrained_weights(self.weights_path)

        # --- Callbacks ---
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each run gets its own checkpoint subfolder (by run name or timestamp)
        import datetime

        run_tag = self.wandb_run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = output_dir / "checkpoints" / run_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="last",
                save_last=True,
            ),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best-{epoch}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        # --- Logger ---
        wandb_logger = None
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.wandb_run_name,
                save_dir=str(output_dir),
            )

        # --- Strategy ---
        strategy: str | pl.strategies.Strategy
        if self.strategy == "fsdp":
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            from geniesae.plaid_model import TransformerBlock

            strategy = pl.strategies.FSDPStrategy(
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
                mixed_precision=torch.distributed.fsdp.MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                ),
            )
        elif self.num_gpus > 1:
            strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
        else:
            strategy = "auto"

        # --- Trainer ---
        trainer = pl.Trainer(
            max_epochs=self.num_epochs,
            accelerator="gpu",
            devices=self.num_gpus,
            strategy=strategy,
            precision=self.precision,
            callbacks=callbacks,
            logger=wandb_logger,
            default_root_dir=str(output_dir),
            gradient_clip_val=1.0,
        )

        # --- Train ---
        trainer.fit(
            training_module,
            datamodule=data_module,
            ckpt_path=self.resume_from,
        )

        # --- Evaluate ---
        results = {}
        if self.use_wandb and wandb_logger is not None:
            results["wandb_run_id"] = wandb_logger.experiment.id

        logger.info("Fine-tuning complete. Output: %s", self.output_dir)
        return results
