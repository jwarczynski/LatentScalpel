---
tags: [experiment, sae, plaid, finetuned]
status: completed-but-flawed
date: 2026-04-17
slurm_job: 2527994
related: [[plaid-finetuned-v3b]] [[decision-split-policy]]
---

# Exp: SAEs on fine-tuned v3b — v1 (validation-only split)

First SAE training run on v3b activations. Used **only** validation-split activations, which turned out to be a methodological mistake.

## Setup

| Stage | Split | Data |
|---|---|---|
| Activation collection | validation | 3k XSum dev samples |
| SAE train | validation activations | (same) |
| SAE val | none | (no validation loop!) |
| Top-examples | validation activations | (same) |

`configs/train_sae_plaid_finetuned_v3b.yaml` at infra.version=1:

```yaml
activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/validation"
val_activation_dir: null        # ← no validation
test_activation_dir: null
layer_idx: 0
expansion_factor: 16
k_target: 64
k_start_multiplier: 4.0
max_samples: 5000000
learning_rate: 3e-4
```

## Slurm

Job array 2527994 (6 tasks, one per layer). All completed in ~38 min each.

## Output

- 6 checkpoints at `experiments/sae_checkpoints/plaid_finetuned_v3b/layer_XX.ckpt` (v1, later overwritten by v3).
- 6 top-examples JSONs at `experiments/top_examples/plaid_finetuned_v3b/layer_XX_top_examples.json` (388 MB total).
- 6 interpretation JSONs (4 of them cut off by Slurm timeout).

## Outcome

- **Training curves**: no validation charts because `val_activation_dir` was null.
- **Top examples**: ran successfully.
- **Interpretation**: Layer 0 got 2567 features interpreted successfully. Inspection showed interpretations were near-identical ("direct quotes from named individuals") across most features. This was the trigger to rethink the pipeline.

## Problems identified

1. **No held-out data for SAE training**. Both SAE train and top-examples used the same 3k validation samples → features overfit to those specific articles.
2. **XSum news bias**. Every article has quotes → LLM anchors on "quotes" as the common property of top examples.
3. **Interpretation timeouts**. 4 of 6 jobs hit Slurm 8h limit without saving partial progress (layers 10, 14, 20, 23).

## Superseded by

- [[exp-sae-finetuned-v3b-v2]] — proper 3-split setup.
- [[exp-sae-finetuned-v3b-v3]] — v2 + normalization + tighter config.

## Lessons

- Always have a held-out evaluation set when training SAEs — even if the val loss doesn't tell you much, it catches pipeline errors.
- Interpretation on a dataset-biased corpus (news) needs additional distinguishing signal beyond top examples — maybe show the LLM non-activating examples too with explicit contrast instructions.
