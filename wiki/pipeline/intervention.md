---
tags: [pipeline, sae, intervention]
related: [[midpoint-features]] [[temporal-classification]]
---

# Stage 6: Feature Intervention

Causally test what a feature does by artificially modifying its activation during generation and observing the effect.

## What it does

During the reverse diffusion chain, intercept the model's intermediate activations at a target layer via NNsight, encode them through the trained SAE, modify specific feature magnitudes in the sparse code, decode back, and replace the layer's output. Then continue the chain.

Produces:
- Baseline generation (no modification).
- Patched generation.
- Per-step losses.
- Text outputs.

## Modes

- **Suppress** — zero out target features at the timestep where they naturally fire.
- **Enforce** — force target features to a fixed magnitude (e.g. 5.0) at a timestep where they don't naturally fire.
- **Enforce early / Enforce late** — force features at timestep numerically before or after their natural peak.

## Configs

- `configs/intervention.yaml` — general intervention runner
- `configs/intervention_enforce_early_layer0.yaml` — force layer 0 features early
- `configs/intervention_suppress_midpoint_layer0.yaml` — suppress layer 0 features at midpoint
- Similar configs exist per layer: `intervention_enforce_early.yaml`, `intervention_suppress_midpoint.yaml`

Config class: `InterventionConfig` in `geniesae/configs/intervention_config.py`. Implementation: `geniesae/feature_intervention.py`.

## Key findings (GENIE, carried over)

See [[midpoint-features]] for detail. Summary:

- GENIE layer 5 has 56 features classified as `midpoint_exclusive`.
- Suppressing them at their natural peak timestep (NDS≈0.5) degrades generation.
- Forcing them **earlier** than natural (NDS≈0.25) — where the text is already partially resolved — also degrades generation.
- These features are a kind of "mid-denoising transition" signal that the model relies on at a specific point of the chain.

Analogous features exist in Plaid (layer 14: 18 midpoint_exclusive features, layer 20: 17). Intervention experiments on Plaid have not yet been run.

## Planned next

- Apply the same enforce/suppress protocol to Plaid v3 SAEs.
- Compare: are Plaid's midpoint features analogous to GENIE's?
- Look for topic-coherent features (via interpretation) and test if suppressing them removes topic coherence from generated summaries.
