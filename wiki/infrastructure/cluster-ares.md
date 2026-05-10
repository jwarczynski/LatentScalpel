---
tags: [infra, cluster]
---

# Cluster: Ares/Athena (CYFRONET)

Polish academic HPC. Slurm-managed A100 partition we use for all GPU work.

## Access

- SSH alias: `athena` (resolves to `athena.cyfronet.pl`).
- Remote user: `plgjentker`.
- Home: `/net/people/plgrid/plgjentker/`.
- Scratch (project data lives here): `/net/tscratch/people/plgjentker/`.

SSH config:

```
Host athena ares helios 
    User plgjentker
    IdentityFile ~/.ssh/id_ed25519_plgrid
    IdentitiesOnly yes

Host athena
    HostName athena.cyfronet.pl

Host ares
    HostName ares.cyfronet.pl
```

## GPU partition

`plgrid-gpu-a100`:

- Nodes: t0001–t0048 (not contiguous; some drained/reserved).
- Per node: 8× A100 40GB + 128 CPUs + 1 TB RAM.
- Max time: 2 days.
- Default mem per CPU: 8000 MB.

Queue can be fast if you find a free node:

```bash
# Nodes with 8 GPUs fully free
sinfo -p plgrid-gpu-a100 -N -O NodeHost,StateLong,GresUsed \
    | grep "gpu:a100:0"
```

## Our QOS

`normal` QOS, `plgnarnlg-gpu-a100` account. No documented per-user GPU limits — we've used 8× A100 full nodes without issues.

## Disk

`/net/tscratch` is shared across nodes, 665 TB total, ~174 TB free as of 2026-05-03. Project uses ~3.4 TB currently.

Disk breakdown:
- `experiments/activations/plaid_finetuned_v3b/` — ~1.9 TB (train + val + test splits).
- `experiments/sae_checkpoints/plaid_finetuned_v3b/` — ~0.5 GB per checkpoint × 6 = ~3 GB.
- `experiments/top_examples/plaid_finetuned_v3b/` — ~400 MB.
- `experiments/results/plaid_finetuned_v3b/` — trajectory + interpretations (~100 MB).
- `experiments/plaid_xsum_v3b/checkpoints/` — ~46 GB (Plaid 1B model × 3 ckpts).

## Useful Slurm commands

```bash
# Queue status
squeue -u $USER --format="%.10i %.30j %.8T %.10M %.20R"

# Job history
sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode

# Cluster-wide free GPUs
sinfo -p plgrid-gpu-a100 -N -O NodeHost:15,StateLong:12,GresUsed:25 | grep -v drain

# Peek at a running job's stdout
find experiments/cache -name "*${JOB_ID}*log.out" -exec tail -40 {} \;

# Cancel a job array
scancel <JOB_ID>

# Test a submission without queuing (reserve GPUs)
sbatch --test-only --partition=plgrid-gpu-a100 --gpus=8 --time=2-00:00:00 --wrap="echo test"
```

## Connecting via MCP

MCP server config supports running remote operations via:
- `mcp_remote_gpu_run_command` — execute shell on `athena` login node.
- Syncs via `rsync -avz <local> athena:<remote>` (or just `git pull` on remote now).

## Gotchas

- **Login nodes have no GPU**. `python -c "import torch; torch.cuda.init()"` fails there.
- **Nodes have no internet access**. Must download datasets/models on login node first.
- **Slurm `--test-only` doesn't actually run** — just validates the submission.
- **`uv sync` on login node is slow** — the NFS-backed home dir is slow for lots of small file writes. Prefer running it on a compute node if possible, or be patient.
- **Job arrays**: use `--submit --infra.cluster=slurm` via exca's `infra.job_array()`. One Slurm job ID with multiple task indices: `2559832_[0-5]`.
