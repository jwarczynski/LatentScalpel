---
tags: [experiment, finetune, plaid, template]
status: completed-but-superseded
date: 2026-03-28
---

# Exp: template v2

Template-mode counterpart of [[exp-conditional-v2]]. Same hyperparameters, same bugs still outstanding, different data format.

## Config

`configs/plaid_xsum_template_8gpu_v2.yaml`

| Field | Value |
|---|---|
| lr | 1e-5 |
| weight_decay | 4e-5 |
| schedule | linear |
| num_epochs | 30 |
| batch_size | 4 |
| seq_len | 1024 |
| training_mode | template |

Data format: `"ARTICLE: <article>\n\nSUMMARY: <summary>"`. See [[conditional-vs-template-training]].

## Slurm

- Job 2500093 (after OOM/dtype fixes).

## Outcome

Similar to [[exp-conditional-v2]]: readable but unreliable generations, shaky loss curves. Superseded by v3c with better hyperparams.
