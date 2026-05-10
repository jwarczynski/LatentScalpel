---
tags: [related, tooling]
---

# SAEScope

Interactive dashboard for exploring SAE feature interpretation results. Sibling project in the workspace (not a Python dependency of GenieSAE, but useful for inspecting our output).

## Path

`/Users/warczynj/Projects/GenieSAE/saescope/` (also a git repo of its own).

## Purpose

Loads the JSON outputs from our pipeline (interpretation results, top examples, trajectory data) and provides a web UI for:
- Browsing SAE features by layer.
- Reading explanations and interpretability scores.
- Viewing top-activating examples.
- Cross-referencing temporal profiles.

## Usage

```bash
cd saescope/
uv run saescope serve <path_to_interpretation_results.json>
```

Frontend under `saescope/frontend/`, backend under `saescope/saescope/`.

## Relation to GenieSAE

Consumes outputs from:
- `find-top-examples` (top activating examples JSON).
- `interpret-features` (explanations JSON).
- `collect-trajectory` (trajectory JSON).

If the JSON schemas change on the GenieSAE side, saescope needs a corresponding update.

## Status

Works. Used for ad-hoc feature exploration after each analysis run. Not part of automated pipeline.
