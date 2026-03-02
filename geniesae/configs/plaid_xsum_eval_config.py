"""Exca config for evaluating a fine-tuned PLAID XSum checkpoint.

Loads a checkpoint, generates summaries via inpainting (clean or renoised
prefix mode), computes ROUGE/BLEU/BERTScore, and saves generations + metrics
to JSON.

Usage:
    uv run python main.py eval-plaid-xsum configs/plaid_xsum_eval.yaml

Submit to Slurm:
    uv run python main.py eval-plaid-xsum configs/plaid_xsum_eval.yaml \
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import time
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.plaid_xsum_eval")


class PlaidXSumEvalConfig(BaseModel):
    """Config for PLAID XSum evaluation: generation + metrics."""

    # -- Checkpoint -----------------------------------------------------------
    checkpoint: str = Field(min_length=1, description="Path to .ckpt file")

    # -- Dataset --------------------------------------------------------------
    data_dir: str = "datasets/glge-released-dataset/easy/xsum_data/org_data"
    seq_len: int = Field(default=1024, gt=0)
    max_summary_len: int = Field(default=64, gt=0)
    tokenizer_path: str = "models/plaid/plaid1b_weights/tokenizer.json"

    # -- Sampling -------------------------------------------------------------
    prefix_mode: str = Field(
        default="clean",
        description='"clean": replace prefix with clean embeddings each step. '
        '"renoised": re-noise prefix to match current timestep.',
    )
    sampling_timesteps: int = Field(default=256, gt=0)
    score_temp: float = 0.9
    num_samples: int = Field(default=1000, gt=0)

    # -- Output ---------------------------------------------------------------
    output_dir: str = "./experiments/plaid_xsum/eval_results"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def apply(self) -> dict:
        """Generate summaries and compute metrics."""
        import tokenizers

        from geniesae.plaid_samplers import InpaintingSampler
        from geniesae.plaid_xsum_eval import EvaluationModule
        from geniesae.plaid_xsum_training import PlaidXSumTrainingModule
        from geniesae.xsum_data import XSumDataModule

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        # Load tokenizer
        tokenizer = tokenizers.Tokenizer.from_file(self.tokenizer_path)

        # Load checkpoint
        logger.info("Loading checkpoint: %s", self.checkpoint)
        module = PlaidXSumTrainingModule.load_from_checkpoint(
            self.checkpoint, map_location=device
        )
        module.eval()
        module.to(device)

        # Build sampler
        sampler = InpaintingSampler(
            model=module.diffusion_model,
            noise_schedule=module.noise_schedule,
            gamma_bounds=module.gamma_bounds,
            embedding_matrix=module.embedding_matrix,
            sampling_timesteps=self.sampling_timesteps,
            score_temp=self.score_temp,
            prefix_mode=self.prefix_mode,
        )

        # Load validation data
        dm = XSumDataModule(
            data_dir=self.data_dir,
            seq_len=self.seq_len,
            max_summary_len=self.max_summary_len,
            batch_size=1,
            num_workers=0,
            tokenizer_path=self.tokenizer_path,
        )
        dm.setup("validate")
        val_dataset = dm.val_dataset
        assert val_dataset is not None
        sep_id = dm.sep_token_id

        n = min(self.num_samples, len(val_dataset))
        logger.info(
            "Generating %d summaries (prefix_mode=%s, steps=%d, temp=%.2f)",
            n, self.prefix_mode, self.sampling_timesteps, self.score_temp,
        )

        predictions: list[str] = []
        references: list[str] = []
        generations: list[dict[str, str]] = []

        for i in range(n):
            sample = val_dataset[i]
            token_ids = sample["token_ids"].unsqueeze(0).to(device)
            boundary_idx = sample["boundary_idx"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

            bi = boundary_idx[0].item()
            all_ids = token_ids[0].tolist()
            summary_start = bi + 1
            real_len = attention_mask[0].sum().item()
            ref_ids = all_ids[summary_start:real_len]
            article_text = tokenizer.decode(all_ids[:bi])
            ref_text = tokenizer.decode(ref_ids) if ref_ids else ""

            t0 = time.time()
            with torch.no_grad():
                gen_ids = sampler.sample(
                    article_token_ids=token_ids,
                    boundary_idx=boundary_idx,
                    seq_len=self.seq_len,
                )
            gen_time = time.time() - t0

            gen_all = gen_ids[0].tolist()
            gen_summary_ids = gen_all[summary_start:]
            gen_clean = []
            for tid in gen_summary_ids:
                if tid == sep_id:
                    break
                gen_clean.append(tid)
            gen_text = tokenizer.decode(gen_clean) if gen_clean else ""

            predictions.append(gen_text)
            references.append(ref_text)
            generations.append({
                "idx": i,
                "article": article_text[:500],
                "reference": ref_text,
                "generated": gen_text,
                "gen_time_s": round(gen_time, 1),
            })

            if (i + 1) % 50 == 0:
                logger.info("Generated %d / %d", i + 1, n)

        # Compute metrics
        evaluator = EvaluationModule(output_dir=self.output_dir)
        metrics = evaluator.evaluate(predictions, references)
        logger.info("Metrics: %s", metrics)

        # Save everything
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tag = self.prefix_mode
        results = {
            "checkpoint": self.checkpoint,
            "prefix_mode": self.prefix_mode,
            "num_samples": n,
            "sampling_timesteps": self.sampling_timesteps,
            "score_temp": self.score_temp,
            "metrics": metrics,
            "generations": generations,
        }
        out_path = out_dir / f"eval_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)

        # Also save metrics separately
        evaluator.save_results(metrics, filename=f"metrics_{tag}.json")

        return {"metrics": metrics, "output_file": str(out_path)}
