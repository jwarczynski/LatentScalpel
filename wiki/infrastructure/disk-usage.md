---
tags: [infra, disk]
---

# Disk usage

Remote scratch: `/net/tscratch/people/plgjentker/GenieSAE/` (165 TB quota, ~174 TB free on the filesystem as of 2026-05-03).

## Current footprint (approximate)

| Path | Size | Notes |
|---|---|---|
| `experiments/activations/plaid_finetuned_v3b/xsum/train/` | ~1.2 TB | 10k samples × 6 layers × 10 ts |
| `experiments/activations/plaid_finetuned_v3b/xsum/validation/` | ~354 GB | 3k samples |
| `experiments/activations/plaid_finetuned_v3b/xsum/test/` | ~354 GB | 3k samples |
| `experiments/activations/plaid/openwebtext/` | ~400 GB (est) | Pretrained SAE data |
| `experiments/activations/genie/xsum/` | ~10 GB | GENIE dim is smaller |
| `experiments/plaid_xsum_v3b/checkpoints/` | ~46 GB | 3 × 15 GB full-model Lightning ckpts |
| `experiments/plaid_xsum_v3b/wandb/` | ~15 GB | local wandb cache |
| `experiments/sae_checkpoints/plaid_finetuned_v3b/` | ~3 GB | 6 × ~500 MB SAE ckpts |
| `experiments/top_examples/plaid_finetuned_v3b/` | ~400 MB | 6 layer JSONs |
| `experiments/results/plaid_finetuned_v3b/` | ~100 MB | trajectory + interpretations |
| `.venv/` | ~20 GB | uv environment incl. torch + vllm |
| **Total** | **~3.4 TB** | |

## Disk math per collection run

Plaid activations, bf16→float32:
- 1 sample × 256 tokens × 2048 dim × 4 bytes = 2 MB per layer per timestep
- 1 sample × 6 layers × 10 timesteps = **120 MB** per sample
- 3000 samples = **360 GB** per split
- 10000 samples = **1.2 TB** per split

## Large files to be careful about

- `models/plaid/plaid1b_weights/model.pt` — 5.2 GB raw model weights.
- `experiments/plaid_xsum_v3b/checkpoints/*.ckpt` — 15 GB each (contains full 1.3B-param model + optimizer state).
  - `best-epoch85.ckpt` — keep.
  - `last.ckpt`, `last-v1.ckpt` — can delete if short on space.
- `experiments/activations/*/timestep_*.pt` — 5.9 GB each (3k sample slab). 60 of these per split per experiment.

## Cleanup candidates

- Wandb local cache under `experiments/plaid_xsum_v3b/wandb/` — syncs to wandb cloud, safe to delete.
- Older SAE checkpoints from v1/v2 runs (overwritten but old files might linger).
- Interpretation JSON from timed-out v1 runs (already lost, nothing to clean).

## Monitoring commands

```bash
du -sh /net/tscratch/people/plgjentker/GenieSAE/experiments/
du -sh /net/tscratch/people/plgjentker/GenieSAE/experiments/*/
df -h /net/tscratch
```

## Quota policy

Scratch is purged periodically (details vary by site — check CYFRONET policy). Don't store anything irreplaceable here. For permanent artifacts use your home dir (slower) or transfer to external storage.
