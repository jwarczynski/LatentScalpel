---
tags: [experiment, finetune, plaid, template]
status: completed
date: 2026-04-12
---

# Exp: template v3c (seq_len=256)

Template-mode counterpart of v3b. Same hyperparams, different format.

## Config

`configs/plaid_xsum_template_8gpu_v3c.yaml`

| Field | Value |
|---|---|
| training_mode | template |
| lr | 1e-6 |
| schedule | cosine |
| epochs | 100 |
| seq_len | 256 |

Data format: `"ARTICLE: <article>\n\nSUMMARY: <summary>"`.

## Slurm

Job 2517799, submitted 2026-04-12 alongside v3a and v3b.

## Wandb

Run name: `template-256-lr1e6-cosine-100ep`

## Outcome

Completed 100 epochs. Generations were less reliable than v3b conditional — the lack of strong fixed-prefix conditioning made summaries more loosely tied to the article.

## Conclusion

Conditional training + short seq_len is the current winning combination. Template mode is kept as a reference / alternative format but not the primary focus.

## Related

- [[conditional-vs-template-training]]
- [[exp-conditional-v3b]] — the winner.
