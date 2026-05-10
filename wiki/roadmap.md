---
tags: [meta, roadmap]
last_updated: 2026-05-10
---

# Roadmap

Active and upcoming work, rough priority order.

## Active

### Top-examples v3 running (job array 2579629)

Submitted 2026-05-10 18:36. Four jobs on A100 nodes, ~2h timeout.
- Layer 10 (2579629_0), Layer 14 (2579629_1), Layer 20 (2579629_2), Layer 23 (2579629_3).
- Output: `experiments/top_examples/plaid_finetuned_v3b_v3/layer_XX_top_examples.json`.
- Uses v3 `_best-v1` checkpoints (expansion=8, normalize_inputs=true).
- Split: **test** (val was used for SAE early stopping).

## Next up (once top-examples completes)

1. Run `collect-plaid-trajectory` with v3 SAEs (test split) — config ready.
2. Run `classify_temporal_features.py` on new trajectory.
3. Run interpretation per layer via vLLM — config ready, submit once top-examples done.
4. Compare feature quality to [[exp-sae-finetuned-v3b-v1]] (v1 had expansion=32 and val split only; v3 has proper 3-split and normalization).

## Decisions recorded in wiki

- Layers 0 and 4 dropped from downstream pipeline. Val FVE < 0 → features overfit train activations. See [[exp-sae-finetuned-v3b-v3]].
- Layer 20 is the strongest (val FVE 0.63), 10 and 14 also usable (~0.50), 23 marginal (0.28).
- Pipeline continues only with layers **10, 14, 20, 23**.

## Pending issues

- **Interpretation jobs hit 8h Slurm timeout** with no partial-result save. Need to add incremental JSON checkpointing inside `InterpretFeaturesConfig.apply()`. See [[interpretation]].
- **Evaluation metrics not computed for v3b generations**. Either wire up `EvaluationModule` in `plaid_xsum_inference.py` or re-run via `PlaidTokenGuidanceConfig`.
- **Hallucinations in v3b summaries**. Investigate whether specific SAE features correlate with hallucination — intervention target?
- **Revisit layers 0 and 4**. v4 could try much larger expansion (=32) or larger k (=256) to see if they can fit.

## Ideas for future experiments

- **Intervention on Plaid midpoint features**. Plaid layer 14 has 18 midpoint_exclusive features, layer 20 has 17. Analog of [[midpoint-features]] GENIE work.
- **Fine-tune with longer training**. v3b still had decreasing val loss at epoch 100. Try 200 epochs or resume with a fresh small-lr tail.
- **Cross-model SAE features comparison**. What features are shared between pretrained Plaid and fine-tuned v3b? Are new features XSum-specific?
- **Prompt format ablation**. Systematically compare TL;DR / Summary / "In brief" on the fine-tuned model.
- **Larger batch size via gradient accumulation**. Paper uses batch=256; we use 32. Accumulating 4 steps would give 128. Might stabilize training.
- **Input normalization ablation for GENIE**. Does GENIE's pipeline benefit from `normalize_inputs=true` like Plaid did? Retrofitting might boost feature quality there too.

## Deferred

- **Pretraining from scratch**. Out of scope — we're studying the fine-tuned model, not rebuilding Plaid.
- **Switching to Gated or JumpReLU SAEs**. TopK is good enough for now.

## Cleanup

- Remove old v1/v2 SAE checkpoints after v3 is confirmed working.
- Clear stale wandb local caches.
- Update `docs/` references to point at wiki pages (or delete the old docs).
