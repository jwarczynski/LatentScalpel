---
tags: [pipeline, evaluation]
related: [[plaid-finetuned-v3b]]
---

# Stage 7: Evaluation

Compute standard summarization metrics on generated summaries.

## Metrics

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — n-gram and longest-common-subsequence overlap (via `rouge-score`).
- **BLEU** — corpus-level BLEU (via `sacrebleu`; falls back to `nltk.corpus_bleu` if unavailable).
- **BERTScore** — embedding-based F1 (via `bert-score`, lang=en).

## Usage

```python
from geniesae.plaid_xsum_eval import EvaluationModule

ev = EvaluationModule(output_dir="./experiments/results/my_eval")
metrics = ev.evaluate(predictions, references)
# {"rouge1": 0.12, "rouge2": 0.01, "rougeL": 0.09, "bleu": 0.005,
#  "bertscore_precision": 0.33, ...}
ev.save_results(metrics, filename="metrics.json")
```

Class: `EvaluationModule` in `geniesae/plaid_xsum_eval.py`.

## Integration with token guidance

`PlaidTokenGuidanceConfig.apply()` calls `EvaluationModule` automatically after generating predictions. Output includes the generations dict and final metrics JSON.

## Standalone inference evaluation

For the v3b inference run, evaluation was manual because the script used is `scripts/plaid_xsum_inference.py` which doesn't wire up `EvaluationModule`. To compute metrics post-hoc, parse the log file or pipe predictions + references into the module.

## XSum reference availability

- `dev.tgt` is present → dev evaluation is straightforward.
- `test.tgt` is **NOT** present in GLGE XSum (test targets are held secret). Test-set metrics can only be computed via the official XSum leaderboard. For our purposes, we report dev-set metrics.

## Baseline comparisons

Reported numbers from other XSum work (paper values, not reproduced by us):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-1 baseline | 16.3 | 1.6 | 11.8 |
| Pointer-generator | 28.1 | 8.0 | 21.7 |
| BART-large fine-tuned | 45.1 | 22.3 | 37.3 |
| PEGASUS-large | 47.2 | 24.6 | 39.3 |
| Plaid (zero-shot, our setup) | ~5 | ~0 | ~3 |

Our fine-tuned v3b numbers are in a similar ballpark to the Plaid zero-shot paper numbers for XSum — substantially below PEGASUS/BART. This is expected: Plaid is a general-purpose continuous diffusion LM not designed for abstractive summarization, and our fine-tuning hasn't closed that gap.
