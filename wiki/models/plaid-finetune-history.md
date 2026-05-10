---
tags: [model, plaid, finetune, history]
last_updated: 2026-05-03
related: [[plaid-finetuned-v3b]] [[bugs-summary]]
---

# Plaid fine-tune history

All XSum fine-tuning runs in chronological order.

## v1 — first attempt (broken, pre-bugfix)

Ran before the [[bug-mup-scaling]] and [[bug-rotary-values]] fixes were in. Model was generating gibberish. Loss looked plausible (because the bugs produced mathematically valid but useless gradients). Unrecoverable — weights permanently contaminated.

- Artifacts discarded.

## v2 — first post-bugfix run (shaky)

Commit range: after ``d06f5dd``, before ``759231c``.

- Config: `configs/plaid_xsum_conditional_8gpu_v2.yaml`
- Params: lr=1e-5, linear schedule, weight_decay=4e-5, beta2=0.99, batch_size=4, 30 epochs
- Wandb: `5gepdez9` belongs to v3b not v2; v2 run IDs need to be recovered from wandb project.
- Outcome: Loss oscillated 2-4 after initial drop. Summaries readable but unreliable.

Problem root-caused to [[bug-selfcond-detachment]] — noise schedule was being corrupted during training.

Companion template-mode run: [[exp-template-v2]].

### Resume attempt: v2 + lr=1e-7 linear tail (cancelled)

- Slurm job 2505405 / 2506663
- Config: `configs/plaid_xsum_conditional_8gpu_resume.yaml`
- Intended: 10 extra epochs resuming from `last-v1.ckpt`, with `resume_lr=1e-7` to skip warmup and decay from a small non-zero lr.
- Added `resume_lr` field to `PlaidXSumTrainingModule` in commit ``0c67b10``.
- Cancelled because attention shifted to v3 runs with better hyperparams.

## v3a / v3b / v3c — current generation

Three parallel runs, all with lr=1e-6, cosine schedule, 100 epochs, 8× A100. All post-[[bug-selfcond-detachment]].

| Config | seq_len | training_mode | Wandb run name | Status |
|---|---|---|---|---|
| v3a | 1024 | conditional | `cond-1024-lr1e6-cosine-100ep` | Completed; lower quality than v3b |
| v3b | 256 | conditional | `cond-256-lr1e6-cosine-100ep` | **Completed; best** |
| v3c | 256 | template | `template-256-lr1e6-cosine-100ep` | Completed |

Submitted 2026-04-12, jobs 2517797/2517798/2517799. All started instantly because the cluster had many fully-free 8-GPU nodes.

**Key finding**: shorter seq_len (256) worked substantially better than 1024 for summarization. See [[decision-seq-len-256]].

Best checkpoint overall: **v3b epoch 85** (`best-epoch85.ckpt`).

## Hyperparameter lineage

| Version | lr | wd | betas | schedule | batch | seq_len | epochs |
|---|---|---|---|---|---|---|---|
| v1 | 1e-4 | 0.01 | (0.9, 0.999) | cosine | 4 | 1024 | 30 |
| v2 | 1e-5 | 4e-5 | (0.9, 0.99) | linear | 4 | 1024 | 30 |
| v3a | 1e-6 | 4e-5 | (0.9, 0.99) | cosine | 4 | 1024 | 100 |
| v3b | 1e-6 | 4e-5 | (0.9, 0.99) | cosine | 4 | 256 | 100 |
| v3c | 1e-6 | 4e-5 | (0.9, 0.99) | cosine | 4 | 256 | 100 (template) |

The move from 0.01 to 4e-5 weight_decay matches the original Plaid `train.py`. See [[decision-lr-1e-6]].

## Original paper comparison

Original Plaid pretraining used:
- lr=1.4e-3 (with MuAdam — mup-aware)
- beta2=0.99
- weight_decay=4e-5
- linear lr decay to zero
- batch_size=256
- warmup_steps=2500

We use AdamW instead of MuAdam (see [[decision-no-mup-adam]]) and a much smaller effective batch (32 vs 256) because we're fine-tuning on a smaller dataset. The lower lr (1e-6 vs 1.4e-3) reflects both the scale difference and the standard fine-tuning heuristic.
