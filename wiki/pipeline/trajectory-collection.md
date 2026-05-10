---
tags: [pipeline, sae, trajectory]
related: [[temporal-classification]] [[sae-training]]
---

# Stage 3b: Trajectory Collection

Runs the full denoising chain end-to-end and records SAE feature activations at every step. This is what lets us classify features by when (at what noise level) they fire.

## Command

```bash
uv run python main.py collect-plaid-trajectory \
    configs/plaid_trajectory_finetuned_v3b.yaml --submit
```

Config class: `PlaidTrajectoryConfig` in `geniesae/configs/plaid_trajectory_config.py`.

## What it does

1. Loads the model + SAEs for the specified layers.
2. Starts denoising from pure noise.
3. At every sampling step:
   - Uses `nnsight` to capture the block output (we hook block outputs, not residual stream).
   - Encodes through the SAE for each selected layer.
   - For each layer, computes `mean over batch × seq_len` of SAE feature activations.
   - Keeps the top-64 features per step (by mean activation) and stores `{feature_id: mean_activation}`.
4. Completes one full denoising run per batch.
5. Averages across batches.
6. Writes a single JSON with the full trajectory.

## Config

| Field | Purpose |
|---|---|
| `weights_path` / `checkpoint_path` | Same as activation collection — supports both raw and fine-tuned. |
| `sae_checkpoint_dir` | Directory with `layer_XX.ckpt` files. |
| `layers` | List of layer indices to hook (must have SAE checkpoints). |
| `dataset_name` / `dataset_split` / `data_dir` / `tokenizer_path` | Source data. |
| `max_samples` | Number of denoising runs (each is one sample). Small because full chain is expensive. |
| `seq_len` | Sequence length. |
| `sampling_timesteps` | Total denoising steps (256 for our Plaid work). |
| `score_temp` | Sampling temperature. |
| `timestep_subsample` | Record every N-th step. `1` = all steps. |
| `batch_size` | Keep small — full trajectory uses lots of GPU memory. |
| `top_k_to_record` | Top-K feature count stored per step. |
| `output_path` | JSON output. |

## Output JSON

```json
{
  "metadata": {
    "model": "plaid-1b",
    "checkpoint_path": "...best-epoch85.ckpt",
    "sae_checkpoint_dir": "./experiments/sae_checkpoints/plaid_finetuned_v3b",
    "layers": [0, 4, 10, 14, 20, 23],
    "sampling_timesteps": 256,
    "timestep_subsample": 1,
    "sampled_steps": [0, 1, 2, ..., 255],
    "top_k_to_record": 64,
    "num_samples": 50,
    "dataset_name": "xsum"
  },
  "layers": {
    "0": {
      "0": {"feature_id_a": 1.23, "feature_id_b": 0.89, ...},
      "1": {...},
      ...
      "255": {...}
    },
    "4": {...},
    ...
  }
}
```

Step numbers in `layers.<layer_idx>` are the step indices in the denoising chain (0 = start, at high noise; 255 = end, at low noise).

Size: ~10 MB for 6 layers × 256 steps × top-64 features.

## Gotchas

- Uses `nnsight` so must have the model loaded as an `NNsight` wrapper. That's fine for inference-only work but adds a layer of indirection.
- The original code looks at `model.blocks` (the `ModuleList`); if the layer structure changes the accessor breaks.
- **Timestep numbering convention**: step 0 is the first reverse step (still at near-max noise); step `T-1` is the last reverse step (at near-zero noise). `sampled_steps` is a list of which step indices we actually recorded.
- NDS (Normalized Diffusion Step) maps step → [0, 1]. For step counting: `nds = step / (total_steps - 1)`. Convention: nds=0 means clean/low-noise side, nds=1 means noise/high-noise side. **Confusingly, for GENIE the original convention was the reverse** — verify before interpreting NDS values.

## Visualization

After collection, generate plots:

```bash
uv run python scripts/plot_trajectory_organized.py \
    experiments/results/plaid_finetuned_v3b/trajectory_features.json \
    --model plaid_finetuned_v3b
```

Produces per-layer directories under `experiments/plots/trajectory_analysis/plaid_finetuned_v3b/layer_XX/` with:

- `heatmap.png` — features × timesteps heatmap.
- `counts.png` — number of active features per step.
- `early_only.png` / `late_only.png` / `midpoint_transition.png` / `starting_spike.png` / `finishing_spike.png` — per-category feature profiles.

## Temporal classification feeds off this

See [[temporal-classification]] for the separate standalone script `scripts/classify_temporal_features.py` that reads the trajectory JSON and emits per-feature category labels as JSON.
