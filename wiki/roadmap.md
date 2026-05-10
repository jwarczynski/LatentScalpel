---
tags: [meta, roadmap]
last_updated: 2026-05-03
---

# Roadmap

Active and upcoming work, rough priority order.

## Active

### Waiting on SAE v3 training (job 2559832)

Watching wandb for loss curves. Diagnostics for action in [[exp-sae-finetuned-v3b-v3]]:
- If loss decreases smoothly → proceed to next stages.
- If loss is still flat → try larger k (128 or 256) or higher lr (1e-3).

## Next up (once v3 SAE completes)

1. Re-run `find-top-examples` on **test** split activations (held-out).
2. Re-run `collect-plaid-trajectory` with new SAEs.
3. Run `classify_temporal_features.py` on new trajectory.
4. Re-run interpretation — this time with temporal context (`trajectory_data_path` set in config) for midpoint_exclusive features.
5. Compare feature quality to v1 interpretations (we expect: more diverse explanations, higher interpretability scores, more obvious topic/syntax features).

## Pending issues

- **Interpretation jobs hit 8h Slurm timeout** with no partial-result save. Need to add incremental JSON checkpointing inside `InterpretFeaturesConfig.apply()`. See [[interpretation]].
- **Evaluation metrics not computed for v3b generations**. Either wire up `EvaluationModule` in `plaid_xsum_inference.py` or re-run via `PlaidTokenGuidanceConfig`.
- **Hallucinations in v3b summaries**. Investigate whether specific SAE features correlate with hallucination — intervention target?

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
