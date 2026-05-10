---
tags: [infra, git]
---

# Git workflow

## Repo

`git@github.com:jwarczynski/LatentScalpel.git` (private).

## Branches

- `main` — the only long-lived branch.
- Feature branches only when doing large-enough work to merit a PR. For ongoing iteration we just commit to main.

## Commit conventions

- Prefer small, focused commits with a clear subject line.
- For bug fixes, include the symptom and root cause in the commit body.
- Reference wiki pages or commit SHAs when relevant.

Example:

```
Fix muP output_mult: should be 1.0 not 0.125

Our _apply_mup_shapes set output_mult = base_dim/dim = 0.125.
Combined with width_mult = 8.0, effective scaling was 0.015625 —
8× too small. Logits crushed → gibberish generation.

Original mup.MuReadout uses output_mult=1.0 (default).

See wiki/bugs/bug-mup-scaling.md.
```

## Author identity

Both local (macOS) and remote (Ares login node) use:
- Name: `Jędrzej Warczyński`
- Email: `jwarczynski@users.noreply.github.com`

Remote had auto-detected identity (`plgjentker@login01.athena.cyfronet.pl`) at first, which made GitHub show anonymous avatars. Fixed by running `git config user.name/email` on the remote.

History was rewritten with `git filter-branch` to normalize author across all commits. If you push a new commit, use the above identity or the anonymous avatar returns for that commit.

## Local ↔ remote sync

Early in the project we used rsync between macOS and Ares. That's now retired in favor of git:

1. Edit locally.
2. `git add -A && git commit -m "..." && git push origin main`.
3. On remote: `ssh athena "cd /net/tscratch/people/plgjentker/GenieSAE && git pull --rebase origin main"`.

The MCP workflow runs `cd /net/tscratch/people/plgjentker/GenieSAE && git pull origin main` directly.

## Things NOT tracked

- `.venv/` — Python environment.
- `experiments/` — training outputs, checkpoints, activations.
- `datasets/` — raw data.
- `wandb/` — wandb local cache.
- `models/` — model weights.
- `logs/` — slurm logs.
- Large binary files generally.

Keep these out of git; they live on scratch and get cleaned periodically.

## `uv.lock`

Cross-platform lockfile committed to the repo. Canonical version resolved on Linux (Ares). Locally on macOS, `uv sync` will install the platform-specific wheels using the resolution markers in the lockfile, so the same lockfile works both places.

When adding a dependency:
1. Run `uv add <pkg>` on remote (Linux).
2. Commit and push the resulting `uv.lock`.
3. Pull locally; `uv sync` installs the macOS wheels.

This avoids flip-flopping the lockfile between platforms.

## `.gitignore` highlights

```
__pycache__/
*.py[oc]
.venv
.pytest_cache
experiments
docs
.hypothesis
```

(Yes, `docs/` is gitignored but actually contains checked-in files — historical oversight. Wiki is tracked because it's under `wiki/`.)
