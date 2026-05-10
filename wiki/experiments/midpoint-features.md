---
tags: [experiment, genie, intervention, legacy]
status: completed
related: [[temporal-classification]] [[intervention]]
---

# Midpoint features — GENIE intervention experiments

Legacy intervention work on GENIE's layer-5 midpoint-exclusive features. Kept here for reference; Plaid analog not yet run.

Source: `docs/midpoint_features_experiments.md` (pre-wiki).

## Setup

- Model: GENIE XSum checkpoint.
- SAE: layer 5, trained previously (see [[exp-sae-pretrained-plaid]]-adjacent GENIE runs).
- Target features: all 56 classified as `midpoint_exclusive` in `classification_layer5.json`. These peak at NDS≈0.5 (t=1000 of GENIE's 2000 reverse steps).

## Procedure

Two runs per experiment — baseline and patched — over 50 XSum validation samples:

1. **Baseline**: standard 2000-step reverse diffusion.
2. **Patched**: at the specified intervention timestep, use NNsight to intercept layer-5 output, encode via SAE, modify the 56 features' activations, decode, replace.

Metrics recorded every 100 steps: cross-entropy loss vs ground truth, argmax decoded text.

## Variants

- **Suppress**: `intervention_nds_values=[1000]`, `target_magnitude=0.0` — zero out features at natural peak.
- **Enforce-early**: `intervention_nds_values=[500]`, `target_magnitude=5.0` — force features earlier than natural.
  - "Early" here means at a more-denoised stage (NDS≈0.25 is closer to clean text than NDS=0.5).

## Findings

(Summary from docs/midpoint_features_experiments.md — interpret carefully, names were confusing.)

- Suppressing at natural peak: degraded generation quality at that step but text recovered by end.
- Enforcing at NDS=0.5 with magnitude 5: pushed model toward patterns consistent with those features.
- Enforcing at NDS=0.25 ("early"): also degraded generation — features are a transition signal the model expected later.

## Plaid analog

Plaid has 18 midpoint_exclusive features in layer 14 and 17 in layer 20 (see [[temporal-classification]]). Not yet intervened on. If we do, use the same enforce/suppress protocol but with Plaid's 256 sampling steps instead of GENIE's 2000.

## Terminology warning

"Early" / "late" are ambiguous — could mean:
- Early in the **denoising trajectory** (high-noise side, t=1000 in 2000 steps).
- Early in the **timestep number** (t=0 is the END of denoising, t=T is the START).

The original script uses "early" to mean "earlier in denoising than natural peak" → **lower timestep number** = **more denoised**. This is confusing and should be clarified in the config filename if we re-run on Plaid.
