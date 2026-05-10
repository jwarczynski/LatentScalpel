---
tags: [experiment, finetune, resume]
status: cancelled
date: 2026-04-03
---

# Exp: resume v2 with lr=1e-7 linear tail

Attempt to extend [[exp-conditional-v2]] training after seeing that its val loss was still decreasing at epoch 30. Resume from the last checkpoint, continue for 10 more epochs, but with a tiny fresh lr to avoid the "oscillation at lr=1e-5" problem.

## Config

`configs/plaid_xsum_conditional_8gpu_resume.yaml`

```yaml
# Key resume fields
resume_from: "./experiments/plaid_xsum_conditional_v2/checkpoints/conditional-8gpu-v2-lr1e5-linear/last-v1.ckpt"
wandb_run_id: "5gepdez9"            # same wandb run
wandb_run_name: "conditional-8gpu-v2-lr1e5-linear"
num_epochs: 40                       # was 30, extending by 10
lr_schedule: "linear"
resume_lr: 1.0e-7                    # new field — see below
```

## `resume_lr` mechanism

Added to `PlaidXSumTrainingModule` in commit ``0c67b10``. When set, the optimizer is re-initialized with `lr = resume_lr` and the scheduler **skips warmup** — decays linearly from this value to zero over the remaining epochs.

```python
def lr_lambda(step):
    if self.resume_lr is not None:
        # Skip warmup, just decay from resume_lr
        total = self.trainer.estimated_stepping_batches
        progress = step / max(total, 1)
        return (1.0 - progress) if self.lr_schedule == "linear" else ...
    # Normal path: warmup then decay
    ...
```

## Slurm jobs

- 2505405 — cancelled (wrong wandb run ID; see below).
- 2506663 — cancelled after attention shifted to v3 runs.

## Why cancelled

Between submission and runtime, we decided that v3 hyperparams (lr=1e-6 cosine, 100 epochs from scratch) would give cleaner results than patching v2. Rather than split compute between two approaches, we cancelled the resume and committed to v3.

## Status of `resume_lr` feature

Still available in the codebase. Usable for future resume-with-fresh-lr scenarios.

## Takeaway

The resume-with-small-lr approach is sound for extending a run that's close to converged. But if a run has fundamental issues (like v2's hyperparams being off), it's usually better to start fresh with better settings than to patch.
