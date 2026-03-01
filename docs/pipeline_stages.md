# Pipeline Stages

This document describes the main pipeline stages, what data they produce, and how they relate to each other.

## Overview

```
collect-activations ──► train-sae ──► collect-trajectory ──► plot (heatmap, profiles)
                                  └──► find-top-examples ──► interpret-features (LLM judge)
                                  └──► run-intervention
                                  └──► evaluate
```

## Stage 1: Activation Collection

Runs the model on a dataset and saves intermediate decoder activations to disk.

- **Commands**: `collect-activations` (Genie), `collect-plaid-activations` (PLAID), `collect-t5-activations` (T5)
- **Input**: Model weights + dataset
- **Output**: `experiments/activations/<model>/<dataset>/<split>/layer_XX/timestep_XXXX.pt`

Each `.pt` file contains a tensor of shape `(num_examples × seq_len, activation_dim)` — all token positions from all examples flattened into rows. A `metadata.json` records layer indices, timesteps, dimensions, and seq_len.

For diffusion models (Genie, PLAID), activations are collected at multiple noise levels (timesteps). For T5, there's a single timestep (0) since it's not a diffusion model.

## Stage 2: SAE Training

Trains a Top-K Sparse Autoencoder on the collected activations.

- **Command**: `train-sae`
- **Input**: Activation directory (train + optionally val)
- **Output**: `experiments/sae_checkpoints/<model>/layer_XX.ckpt` (and `_best.ckpt` if validation is enabled)

## Stage 3a: Trajectory Collection

Runs the full denoising process end-to-end, encoding each intermediate state through the trained SAE, and records which features fire at each timestep.

- **Commands**: `collect-trajectory` (Genie), `collect-plaid-trajectory` (PLAID)
- **Input**: Model weights + SAE checkpoints + dataset
- **Output**: `experiments/results/<model>/trajectory_features.json`

### What trajectory_features.json contains

For each layer and each denoising timestep, it stores a dict of `{feature_id: mean_activation}` — the mean activation of each feature across all examples at that timestep. Only the `top_k_to_record` (default: 64) most active features per token position are tracked during collection.

This data is used for:
- **Heatmap plots**: Normalized activation over time, sorted by peak timestep
- **Category profile plots**: Top features per temporal category (early_only, late_only, etc.)
- **Active feature count / total mass plots**
- **Feature classification**: Assigning each feature a temporal category

### How to generate plots

```bash
uv run python scripts/plot_trajectory_organized.py \
    experiments/results/<model>/trajectory_features.json \
    --model <model_name> --save-json
```

Output goes to `experiments/plots/trajectory_analysis/<model>/layer_XX/`.

## Stage 3b: Find Top Examples

Runs SAE inference over stored activations to find, for each feature, the dataset examples that maximally activate it. This is the data the LLM-as-judge protocol needs.

- **Command**: `find-top-examples`
- **Input**: SAE checkpoint + activation directory
- **Output**: `experiments/top_examples/<model>/layer_XX_top_examples.json`

### How find-top-examples works

The goal is to answer: "For feature F, which specific dataset examples cause the highest activation?"

The algorithm:

1. **Load SAE** and the stored activation tensors for the target layer.
2. **For each timestep file** (`timestep_XXXX.pt`):
   - Load the tensor of shape `(num_examples × seq_len, activation_dim)`.
   - Process in batches through `sae.encode()` to get sparse codes of shape `(batch_size, dictionary_size)`.
   - For each row in the sparse codes, identify the non-zero features (there are exactly K per row, where K is the SAE's top-k).
   - For each active feature, compute the `(example_id, token_position)` from the row index: `example_id = row // seq_len`, `token_pos = row % seq_len`.
   - Maintain a per-feature min-heap of size `top_k` (default: 30). Each heap entry is `(activation_value, example_id, timestep, token_position)`. If a new activation exceeds the current minimum in the heap, it replaces it.
   - With `unique_examples=True` (default), each example_id appears at most once per feature — only the highest activation across all timesteps/tokens is kept.
3. **After all timesteps**, sort each feature's heap descending by activation and write the result.

The key difference from trajectory collection: trajectory collection averages activations across examples to get a temporal profile per feature. Find-top-examples keeps track of individual examples to identify which specific inputs maximally activate each feature.

### Output format

```json
{
  "metadata": {
    "dataset_name": "xsum",
    "layer_idx": 0,
    "sae_checkpoint": "...",
    "top_k": 30,
    "num_features": 32768,
    "timesteps_used": [50, 150, ...],
    "seq_len": 256
  },
  "features": {
    "0": [
      {"example_id": 42, "activation": 3.14, "timestep": 500, "token_position": 17},
      ...
    ],
    ...
  }
}
```

## Stage 3c: Interpret Features (LLM-as-Judge)

Uses an LLM to generate natural-language explanations for each SAE feature, then scores them via a discrimination task.

- **Command**: `interpret-features`
- **Input**: Top-examples JSON from stage 3b
- **Output**: `experiments/results/<model>/interpretation_results_layer_XX.json`

The protocol (based on DLM-Scope):
1. For each feature, retrieve its top-activating examples from the top-examples JSON.
2. Present these examples to the LLM and ask it to generate a natural-language explanation of what the feature detects.
3. Score the explanation via a discrimination task: given a mix of high-activation and random examples, can the LLM (using only the explanation) correctly identify which examples would activate the feature?
4. The discrimination accuracy is the "interpretability score" for that feature.

## Stage 3d: Intervention Experiments

Patches SAE feature activations during the denoising process to measure causal impact.

- **Command**: `run-intervention`
- **Input**: Model + SAE + classification JSON (to select which features to intervene on)
- **Output**: `experiments/results/<model>/intervention_*.json`

## Stage 3e: Evaluation

Measures SAE reconstruction quality by patching activations during denoising and comparing loss.

- **Commands**: `evaluate` (Genie), `evaluate-plaid` (PLAID)
- **Input**: Model + SAE checkpoints
- **Output**: `experiments/results/<model>/evaluation_results.json`
