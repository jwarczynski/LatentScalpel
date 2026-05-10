---
tags: [pipeline, overview]
---

# Pipeline overview

End-to-end SAE analysis. Each stage takes inputs from the previous stage or raw sources.

```
collect-activations ──► train-sae ──► find-top-examples ──► interpret-features
                                  ├─► collect-trajectory ──► classify-temporal ──► plot
                                  ├─► evaluate (SAE reconstruction)
                                  └─► run-intervention
```

## Stages

| Stage | Command | Input | Output | Page |
|---|---|---|---|---|
| 1 | `collect-plaid-activations` | model + dataset | per-layer/timestep `.pt` files | [[activation-collection]] |
| 2 | `train-sae` | activation dirs | per-layer SAE checkpoints | [[sae-training]] |
| 3a | `find-top-examples` | SAE + activations | per-feature top-k dataset examples (JSON) | [[top-examples]] |
| 3b | `collect-plaid-trajectory` | model + SAEs + dataset | feature activations across denoising steps (JSON) | [[trajectory-collection]] |
| 4a | `interpret-features` | top-examples + LLM | per-feature explanation + score (JSON) | [[interpretation]] |
| 4b | `scripts/classify_temporal_features.py` | trajectory JSON | per-feature temporal category (JSON) | [[temporal-classification]] |
| 5 | `scripts/plot_trajectory_organized.py` | trajectory JSON | PNG plots (heatmaps, category profiles) | [[trajectory-collection]] |
| 6 | intervention experiments | model + SAEs + chosen features | patched generations | [[intervention]] |
| 7 | `evaluate` | predictions + references | ROUGE/BLEU/BERTScore | [[evaluation]] |

## Data splits per stage (our convention)

Matches GENIE. See [[decision-split-policy]].

- **Train split** activations → SAE **training**.
- **Validation split** activations → SAE **validation** (val loss, monitoring).
- **Test split** activations → SAE **test evaluation** + **top-examples** (held-out for feature analysis).
- Trajectory: **validation** split (or any held-out subset).

Rationale: top-examples and trajectory feed into feature interpretation, which should use held-out data to avoid overfitting explanations to training samples. Using test for top-examples matches this principle while keeping validation free for SAE hyperparameter tuning.

## Running the pipeline for a new model

Rough order of operations:

1. Collect activations on all three splits. Jobs are independent → can run in parallel.
2. Train SAEs on train activations (usually job array, one layer per Slurm task).
3. Collect trajectory on validation split (hooks all layers into one denoising run).
4. Find top-examples on test activations (one job per layer).
5. Classify temporal features from the trajectory JSON (fast, CPU-only).
6. Generate plots from the trajectory JSON.
7. Interpret features with vLLM (GPU, ~hours per layer for ~17k+ features).
8. Intervention / evaluation on demand.

All stages are exca-based so they respect `--submit --infra.cluster=slurm` and cache results by UID.

## Wandb projects

- `genie-sae` — GENIE SAEs
- `plaid-sae` — pretrained Plaid SAEs (OpenWebText)
- `plaid-sae-finetuned` — fine-tuned Plaid v3b SAEs
- `plaid-xsum` — Plaid XSum fine-tuning runs

See [[wandb-projects]].
