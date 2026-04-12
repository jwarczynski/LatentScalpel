"""Evaluation module for PLAID XSum summarization.

Computes ROUGE, BLEU, and BERTScore on generated vs reference summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationModule:
    """Compute ROUGE, BLEU, BERTScore on generated summaries."""

    def __init__(self, output_dir: str, use_wandb: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute all metrics.

        Returns dict with rouge1, rouge2, rougeL, bleu, bertscore_precision,
        bertscore_recall, bertscore_f1.
        """
        if not predictions or not references:
            raise ValueError("predictions and references must be non-empty")
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have the same length, "
                f"got {len(predictions)} vs {len(references)}"
            )

        results: dict[str, float] = {}

        # ROUGE
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            rouge1_f, rouge2_f, rougeL_f = 0.0, 0.0, 0.0
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_f += scores["rouge1"].fmeasure
                rouge2_f += scores["rouge2"].fmeasure
                rougeL_f += scores["rougeL"].fmeasure
            n = len(predictions)
            results["rouge1"] = rouge1_f / n
            results["rouge2"] = rouge2_f / n
            results["rougeL"] = rougeL_f / n
        except ImportError:
            logger.warning("rouge-score not installed, skipping ROUGE")

        # BLEU
        try:
            import sacrebleu

            bleu = sacrebleu.corpus_bleu(predictions, [references])
            results["bleu"] = bleu.score / 100.0  # normalize to [0, 1]
        except ImportError:
            try:
                from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu

                refs_tok = [[r.split()] for r in references]
                preds_tok = [p.split() for p in predictions]
                results["bleu"] = nltk_corpus_bleu(refs_tok, preds_tok)
            except ImportError:
                logger.warning("Neither sacrebleu nor nltk installed, skipping BLEU")

        # BERTScore
        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions, references, lang="en", verbose=False
            )
            results["bertscore_precision"] = P.mean().item()
            results["bertscore_recall"] = R.mean().item()
            results["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            logger.warning("BERTScore failed: %s. Skipping.", e)

        # Log to WandB
        if self.use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({f"eval/{k}": v for k, v in results.items()})
            except Exception:
                pass

        return results

    def save_results(
        self, results: dict[str, float], filename: str = "metrics.json"
    ) -> Path:
        """Write metrics to JSON file in output_dir."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Metrics saved to %s", path)
        return path
