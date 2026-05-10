---
tags: [model, genie, baseline]
---

# GENIE

Diffusion language model on which our SAE analysis pipeline was first built. GENIE uses a different formulation from Plaid (discrete timestep schedule, different architecture) and is mostly of interest as the baseline our Plaid work mirrors.

## Architecture snapshot

| Property | Value |
|---|---|
| architecture | `s2s_CAT` (encoder-decoder BERT-based) |
| config_name | `bert-base-uncased` |
| in_channel / model_channels | 128 |
| vocab_size | 30522 |
| diffusion_steps | 2000 |
| noise_schedule | sqrt |
| diffusion_timesteps sampled | [100, 200, …, 1000] (10 values) |

## Paths

- Weights (on cluster): `models/genie/GENIE_ckpt-XSum`
- Configs:
  - `configs/activation_collection.yaml`
  - `configs/activation_collection_val.yaml`
  - `configs/activation_collection_test.yaml`
  - `configs/train_sae.yaml`
  - `configs/trajectory.yaml`
  - `configs/find_top_examples.yaml`
  - `configs/interpret_features.yaml`

## SAE pipeline for GENIE

Trained SAEs on layers [0, 1, 4, 5] (trajectory), [0, 5] (schedule experiments). Top-examples collected for layers 0, 1, 4, 5. Feature interpretations via LLM-as-judge. Midpoint-exclusive features on layer 5 were studied via intervention — see [[midpoint-features]].

## Split policy (the one we're mirroring for Plaid)

- **Train split activations** → SAE training
- **Validation split activations** → SAE validation + top-examples (wait, actually GENIE uses validation for top-examples too?)

See [[decision-split-policy]] for the clarification: GENIE used train activations for SAE training, validation for SAE validation and top-examples, test for final SAE evaluation. This is what we're now applying to Plaid v3b SAE training.

## Relation to Plaid

Mostly unrelated architecturally — GENIE is encoder-decoder BERT-based, Plaid is decoder-only. But the downstream SAE analysis pipeline is shared: same activation collection framework, same SAE training code, same trajectory recording, same LLM-as-judge interpretation. See [[pipeline-overview]].
