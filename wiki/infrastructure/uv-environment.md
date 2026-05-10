---
tags: [infra, uv, python]
---

# uv environment

Project uses [`uv`](https://docs.astral.sh/uv/) as the Python package manager.

## Setup

- `pyproject.toml` — dependencies (project + transitive pinned in `uv.lock`).
- `uv.lock` — cross-platform lockfile, committed to git.
- `.venv/` — created by `uv sync`, not committed.

## Locking strategy

The lockfile is **Linux-canonical**:

1. Add/change deps on remote (Linux) with `uv add <pkg>` or `uv lock`.
2. Commit the resulting `uv.lock` from the remote.
3. Pull locally on macOS and run `uv sync` — it'll install macOS wheels using the same resolution.

This avoids the lockfile flip-flopping between macOS and Linux resolutions.

## Common commands

```bash
# Install everything
uv sync

# Add a dep
uv add torch

# Add a dev dep
uv add --dev pytest

# Run a command in the env
uv run python main.py ...

# Run pytest
uv run pytest

# Upgrade a single package
uv lock --upgrade-package torch
```

## Python version

`.python-version` pins 3.12. `uv sync` creates a 3.12 venv.

## Platform notes

- On Ares (Linux, CUDA 11.8), `uv sync` installs `torch` with CUDA support automatically via the lockfile's wheel selectors.
- On macOS, `torch` installs CPU-only (no GPU — doesn't matter since we don't run GPU work locally).

## If `uv sync` hangs on login node

NFS can be slow for the 30k+ small file writes of installing PyTorch wheels. Workarounds:

1. Run `uv sync` inside a Slurm job on a compute node (faster local disk).
2. Use `uv sync --frozen` to skip lockfile resolution.
3. Be patient — sometimes it's just network latency.

## Dependencies worth noting

- `torch` 2.9.1+cu128 (Linux) — supports the hardware on Ares.
- `exca` — our orchestration layer for Slurm + caching.
- `nnsight` — used in trajectory collection and interventions.
- `pytorch_lightning` — wrapper for the XSum fine-tuning and SAE training.
- `mup==1.0.0` — used at training time for proper muP initialization.
- `tokenizers` — Plaid's BPE.
- `vllm` — for interpretation stage.
- `wandb` — experiment tracking.
- `flash-attn` / `apex` — NOT in our lockfile. We replaced them with pure-PyTorch equivalents. See [[plaid-1b]] "Reimplementation".
