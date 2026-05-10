---
tags: [experiment, sae, plaid, pretrained]
status: completed
date: 2026-02-??
---

# Exp: SAEs on pretrained Plaid (OpenWebText)

SAE training on the original pretrained Plaid 1B model's activations collected from OpenWebText. This was our first SAE pipeline run, predating all the Plaid fine-tuning work.

## Setup

- Activations: 10k OpenWebText train samples + 2k disjoint validation samples.
- Configs: `configs/plaid_activation_collection.yaml`, `configs/plaid_activation_collection_val.yaml`, `configs/train_sae_plaid.yaml`.
- Layers: [0, 23] (first + last only — for quick exploration).
- Checkpoints: `experiments/sae_checkpoints/plaid/layer_00.ckpt` + `layer_23.ckpt`.

## Wandb

Project: `plaid-sae`.

## Outcome

Baseline SAEs that worked reasonably on the pretrained model on its in-distribution data. Feature interpretations via LLM-as-judge produced mostly coherent explanations about OpenWebText topics (news, politics, sports, etc.).

## Relation to fine-tuned runs

These SAEs are NOT used for analyzing the fine-tuned model — fine-tuning changes which features exist. For the v3b model we trained new SAEs from scratch. See [[exp-sae-finetuned-v3b-v1]], [[exp-sae-finetuned-v3b-v2]], [[exp-sae-finetuned-v3b-v3]].

Kept around as a reference for what "healthy" SAE training curves look like on Plaid activations.
