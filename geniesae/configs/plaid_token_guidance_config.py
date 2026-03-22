"""Exca config for zero-shot conditional generation on XSum using PLAID's token guidance.

Loads pretrained PLAID 1B weights (no fine-tuning), generates summaries
conditioned on article prefixes via the token guidance technique from
Section 6.3 of the paper, computes ROUGE/BLEU/BERTScore, and saves
generations + metrics to JSON.

Usage:
    uv run python main.py token-guidance configs/plaid_token_guidance.yaml

Submit to Slurm:
    uv run python main.py token-guidance configs/plaid_token_guidance.yaml --submit
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

logger = logging.getLogger("geniesae.configs.plaid_token_guidance")


class PlaidTokenGuidanceConfig(BaseModel):
    """Config for zero-shot conditional summarization via PLAID token guidance."""

    # -- Pretrained weights ---------------------------------------------------
    weights_path: str = Field(
        default="models/plaid/plaid1b_weights",
        description="Directory with model.pt, noise_schedule.pt, etc.",
    )
    dim: int = 2048
    embed_dim: int = 16
    n_blocks: int = 24
    n_heads: int = 32
    vocab_size: int = 32768
    gamma_0: float = -3.0
    gamma_1: float = 6.0

    # -- Dataset --------------------------------------------------------------
    data_dir: str = "datasets/glge-released-dataset/easy/xsum_data/org_data"
    tokenizer_path: str = "models/plaid/plaid1b_weights/tokenizer.json"
    eval_split: str = Field(default="dev", description='"dev" or "test"')

    # -- Generation -----------------------------------------------------------
    seq_len: int = Field(default=1024, gt=0)
    max_article_tokens: int = Field(
        default=900, gt=0,
        description="Max tokens for article in the prompt (leaves room for summary).",
    )
    sampling_timesteps: int = Field(default=1024, gt=0)
    score_temp: float = 0.9
    guidance_weight: float = Field(
        default=2.0, description="Token guidance weight (paper default: 2.0).",
    )
    prompt_suffix: str = Field(
        default="\n\nTL;DR:",
        description="Text appended after article to steer generation toward summarization.",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    num_samples: int = Field(default=50, gt=0)

    # -- Output ---------------------------------------------------------------
    output_dir: str = "./experiments/plaid_xsum/token_guidance_results"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("output_dir",)

    @infra.apply
    def apply(self) -> dict:
        """Load pretrained PLAID, generate summaries via token guidance, evaluate."""
        import logging as _logging

        _logging.basicConfig(
            level=_logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        import tokenizers

        from geniesae.plaid_model import load_plaid_modules
        from geniesae.plaid_samplers import TokenGuidanceSampler
        from geniesae.plaid_xsum_eval import EvaluationModule

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}", flush=True)

        # Load tokenizer
        tokenizer = tokenizers.Tokenizer.from_file(self.tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}", flush=True)

        # Load pretrained weights
        print(f"Loading pretrained weights from {self.weights_path}", flush=True)
        t0 = time.time()
        modules = load_plaid_modules(
            self.weights_path,
            dim=self.dim,
            embed_dim=self.embed_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            gamma_0=self.gamma_0,
            gamma_1=self.gamma_1,
            device=str(device),
        )
        print(f"Weights loaded in {time.time() - t0:.1f}s", flush=True)

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # Build sampler
        sampler = TokenGuidanceSampler(
            model=modules["model"],
            noise_schedule=modules["noise_schedule"],
            gamma_bounds=modules["gamma_bounds"],
            embedding_matrix=modules["embedding_matrix"],
            sampling_timesteps=self.sampling_timesteps,
            score_temp=self.score_temp,
            guidance_weight=self.guidance_weight,
        )

        # Load validation data (raw lines, not the XSumDataModule format)
        data_dir = Path(self.data_dir)
        src_path = data_dir / f"{self.eval_split}.src"
        tgt_path = data_dir / f"{self.eval_split}.tgt"

        src_lines = src_path.read_text().strip().split("\n")
        tgt_lines = tgt_path.read_text().strip().split("\n")
        assert len(src_lines) == len(tgt_lines), (
            f"Mismatch: {len(src_lines)} sources vs {len(tgt_lines)} targets"
        )

        n = min(self.num_samples, len(src_lines))

        # Re-seed right before generation loop for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        print(
            f"Generating {n} summaries (steps={self.sampling_timesteps}, "
            f"temp={self.score_temp}, guidance_weight={self.guidance_weight}, "
            f"prompt_suffix={self.prompt_suffix!r})",
            flush=True,
        )

        predictions: list[str] = []
        references: list[str] = []
        generations: list[dict[str, tp.Any]] = []

        for i in range(n):
            article_text = src_lines[i]
            ref_text = tgt_lines[i]

            # Build prompt: article text followed by prompt_suffix
            prompt = article_text + self.prompt_suffix
            prompt_ids = tokenizer.encode(prompt).ids

            # Truncate if needed (leave room for generation)
            if len(prompt_ids) > self.max_article_tokens:
                prompt_ids = prompt_ids[: self.max_article_tokens]

            prefix_len = len(prompt_ids)

            # Generate
            t0 = time.time()
            gen_ids = sampler.sample(
                prefix_token_ids=[prompt_ids],
                seq_len=self.seq_len,
            )
            gen_time = time.time() - t0

            # Decode the generated part (after prefix)
            gen_all = gen_ids[0].tolist()
            gen_suffix_ids = gen_all[prefix_len:]

            # Truncate at newline or end-of-text (token 0 in PLAID tokenizer)
            gen_clean: list[int] = []
            for tid in gen_suffix_ids:
                if tid == 0:
                    break
                gen_clean.append(tid)
            # Also truncate at double newline (paragraph break)
            gen_text = tokenizer.decode(gen_clean) if gen_clean else ""
            # Take first paragraph only
            if "\n\n" in gen_text:
                gen_text = gen_text[: gen_text.index("\n\n")]
            gen_text = gen_text.strip()

            predictions.append(gen_text)
            references.append(ref_text)
            generations.append({
                "idx": i,
                "article": article_text,
                "reference": ref_text,
                "generated": gen_text,
                "prefix_len": prefix_len,
                "gen_time_s": round(gen_time, 1),
            })

            print(f"\n{'='*80}", flush=True)
            print(
                f"[{i + 1}/{n}] ({gen_time:.1f}s) prefix={prefix_len} toks | "
                f"gen={len(gen_clean)} toks",
                flush=True,
            )
            print(f"  ARTICLE: {article_text}", flush=True)
            print(f"  REFERENCE: {ref_text}", flush=True)
            print(f"  GENERATED: {gen_text}", flush=True)

        # Compute metrics
        print("Computing metrics...", flush=True)
        evaluator = EvaluationModule(output_dir=self.output_dir)
        metrics = evaluator.evaluate(predictions, references)
        print(f"Metrics: {metrics}", flush=True)

        # Save results — include hyperparams in filename for easy comparison
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tag = f"gw{self.guidance_weight}_steps{self.sampling_timesteps}_temp{self.score_temp}_n{n}"

        results = {
            "weights_path": self.weights_path,
            "guidance_weight": self.guidance_weight,
            "sampling_timesteps": self.sampling_timesteps,
            "score_temp": self.score_temp,
            "prompt_suffix": self.prompt_suffix,
            "num_samples": n,
            "seq_len": self.seq_len,
            "max_article_tokens": self.max_article_tokens,
            "metrics": metrics,
            "generations": generations,
        }
        out_path = out_dir / f"results_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}", flush=True)

        evaluator.save_results(metrics, filename=f"metrics_{tag}.json")

        return {"metrics": metrics, "output_file": str(out_path)}
