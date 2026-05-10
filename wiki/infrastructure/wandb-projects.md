---
tags: [infra, wandb]
---

# Wandb projects

Organization: [`jedrasowicz`](https://wandb.ai/jedrasowicz) on wandb.ai.

## Projects

| Project | Purpose |
|---|---|
| [`plaid-xsum`](https://wandb.ai/jedrasowicz/plaid-xsum) | Plaid XSum fine-tuning runs |
| [`plaid-sae`](https://wandb.ai/jedrasowicz/plaid-sae) | Pretrained Plaid SAE training (OpenWebText) |
| [`plaid-sae-finetuned`](https://wandb.ai/jedrasowicz/plaid-sae-finetuned) | Fine-tuned Plaid SAE training |
| [`genie-sae`](https://wandb.ai/jedrasowicz/genie-sae) | GENIE SAE training |

## Key runs

### plaid-xsum

- `5gepdez9` — `cond-256-lr1e6-cosine-100ep` — **current best** fine-tune ([[exp-conditional-v3b]]).
- `cond-1024-lr1e6-cosine-100ep` — seq_len=1024 sibling ([[exp-conditional-v3a]]).
- `template-256-lr1e6-cosine-100ep` — template mode sibling ([[exp-template-v3c]]).
- `conditional-8gpu-v2-lr1e5-linear` — older v2 runs ([[exp-conditional-v2]]).
- `conditional-8gpu-bugfix-v3` — early v2 debug iteration.

### plaid-sae-finetuned

- New runs from `exp-sae-finetuned-v3b-v3` (job 2559832) submitted 2026-05-03.
- Prior v1/v2 runs also here but superseded.

## Authentication

Uses `~/.netrc` on the remote for wandb API keys. Already set up; no action needed for new submissions.

## Resuming a wandb run

In config:

```yaml
wandb_run_id: "5gepdez9"    # existing run to resume
wandb_run_name: "cond-256-lr1e6-cosine-100ep"  # must match
```

The Lightning WandbLogger will use `resume="must"` internally — fails if the run doesn't exist.

## Logged metrics conventions

Training steps log:
- `train/loss`, `train/reconst_loss`, `train/diffusion_loss`, `train/prior_loss` — for Plaid fine-tuning.
- `train/bias_scale`, `train/lr` — monitoring.
- `train/mse_loss`, `train/fve`, `train/l0_sparsity`, `train/active_feat_frac` — for SAE training.

Validation / test similar with `val/` or `test/` prefix.

## Tables

Generated samples logged as a single accumulated table: `val/generated_samples` with columns `[epoch, idx, article, reference, generated]`. Updated on every validation epoch by appending rows (matches ShortcutFM pattern). See [[related-shortcutfm]].

Older runs used per-epoch tables (`val/generated_samples_epoch_0`, `_epoch_1`, ...) — cluttered the wandb UI. Fixed in commit ``5b182e9``.

## Removed metrics

- `train/throughput` — never useful; removed.
- `train/gpu_memory_mb` — also not useful enough; removed.
- `noise_schedule/gamma_curve` — used to log the learned gamma(t) curve every val epoch. Confusing visualization. Removed.
