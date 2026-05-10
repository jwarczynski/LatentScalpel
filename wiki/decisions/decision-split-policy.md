---
tags: [decision, sae, splits]
status: adopted
date: 2026-04-26
related: [[exp-sae-finetuned-v3b-v1]] [[genie]]
---

# Decision: 3-split policy for fine-tuned Plaid SAE

Adopted the same split policy for fine-tuned Plaid SAEs that was used in the GENIE SAE pipeline.

## Policy

For each fine-tuned model:

| Stage | XSum split | Samples |
|---|---|---|
| Activation collection — train | train | 10,000 |
| Activation collection — validation | dev | 3,000 |
| Activation collection — test | test | 3,000 |
| SAE training | train activations | |
| SAE validation (during training) | val activations | |
| SAE evaluation | test activations | |
| Top-examples for interpretation | **test** activations | |
| Trajectory collection | val activations | 50 samples full-chain |

## Why not use validation for both training and top-examples

Our initial run ([[exp-sae-finetuned-v3b-v1]]) used the same 3k validation samples for both SAE training and top-examples. The result: near-identical feature interpretations ("direct quotes from named individuals" repeated hundreds of times). The features overfit to the specific articles in the 3k slice — top-activating examples were trivial to predict because they were the SAE's training data.

## Why this is the right setup

- **SAE train on train activations**: more data, different from what we use downstream.
- **SAE val on val activations**: monitors generalization during training.
- **SAE test + top-examples on test activations**: held-out evaluation. Feature explanations based on test examples are more likely to generalize because the SAE didn't optimize on them.

This matches the methodology in the GENIE SAE work. Confirming this is the right call for Plaid: the SAEs now see ~150M train activations vs ~46M in the single-split setup, and top-examples come from a properly held-out set.

## Overlap with model fine-tuning data

The SAE train split and the model fine-tune train split are the **same XSum train data**. The model has "seen" these activations during fine-tuning. This is identical to how GENIE did it — the SAEs train on the model's in-distribution data, which is what we want for finding features the model actually uses.

## Activation collection configs

- `configs/plaid_finetuned_collection_train.yaml` — 10k train samples.
- `configs/plaid_finetuned_collection_val.yaml` — 3k dev samples.
- `configs/plaid_finetuned_collection_test.yaml` — 3k test samples.

## SAE training config

`configs/train_sae_plaid_finetuned_v3b.yaml`:

```yaml
activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/train"
val_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/validation"
test_activation_dir: "./experiments/activations/plaid_finetuned_v3b/xsum/test"
save_best: true
monitor_metric: "val/mse"
```

## Alternatives considered

1. **Collect more val activations, skip test** — would save ~1.5h of collection time. Rejected: gives weaker held-out guarantees for top-examples.
2. **Use the original GENIE 3k/3k/3k** — rejected because Plaid has 16× larger activation_dim so needs proportionally more training data.
3. **Use all 204k train samples** — rejected: ~24 TB of activations. We'd need to sparsify much more aggressively first.

## Related

- [[exp-sae-finetuned-v3b-v2]] — first run with this setup, still had other bugs.
- [[exp-sae-finetuned-v3b-v3]] — current best attempt combining this with normalization fixes.
