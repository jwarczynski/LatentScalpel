---
tags: [experiment, finetune, plaid]
status: completed-but-superseded
date: 2026-03-28
result: shaky-training
related: [[bug-selfcond-detachment]] [[exp-conditional-v3b]]
---

# Exp: conditional v2

First run after [[bug-mup-scaling]] and [[bug-rotary-values]] were fixed, but **before** [[bug-selfcond-detachment]] was found.

## Config

`configs/plaid_xsum_conditional_8gpu_v2.yaml`

| Field | Value |
|---|---|
| lr | 1e-5 |
| weight_decay | 4e-5 |
| betas | (0.9, 0.99) |
| lr_schedule | linear |
| num_epochs | 30 |
| warmup_steps | 1000 |
| seq_len | 1024 |
| batch_size | 4 |
| training_mode | conditional |
| num_gpus | 8 |

Same lr/schedule as the original Plaid paper for fine-tuning but with batch=32 effective vs paper's 256.

## Slurm

- Initial job: 2500094 (after OOM fix bumping from batch=8 to batch=4).
- First failed submissions: 2493108 (dtype mismatch), 2496851 (OOM).

## Wandb

Runs in `plaid-xsum` project. Specific IDs not recorded here; accessible via wandb UI filtering for `conditional-8gpu-v2-lr1e5-linear`.

## Training behavior

- Loss dropped enormously in first ~500 steps.
- Then oscillated between 2 and 4 without clear decrease.
- Validation loss shaky too.

Subjectively the generations were readable English with accurate topic words, but many hallucinations. Classic "model learned something but not the task."

## Root cause

Found afterward: [[bug-selfcond-detachment]]. Without detaching $\gamma$, $\gamma'$, $x_{\text{embed}}$, $\alpha_1$/$\sigma_1$ for selfcond examples, the noise schedule was being pulled in opposite directions by selfcond vs non-selfcond examples. Training was unstable as a result.

## Resume attempt

- Config: `configs/plaid_xsum_conditional_8gpu_resume.yaml` with `resume_lr=1e-7`, schedule continues linear decay, 10 more epochs.
- Same wandb run (resumed via `wandb_run_id: 5gepdez9`).
- Eventually cancelled — attention shifted to v3 runs with fundamentally different hyperparams (lr=1e-6, cosine).

## Sibling run

[[exp-template-v2]] — same hyperparams but with template-mode format.

## Fix + next

Fixed selfcond detachment in ``759231c`` + ``0042bcc``, plus lerp dtype ``34c4f6e`` + ``16a3477``. Then submitted v3 series with lr=1e-6 + cosine + 100 epochs.
