---
tags: [pipeline, plaid, genie]
related: [[pipeline-overview]] [[sae-training]]
---

# Stage 1: Activation Collection

Runs the model on a dataset and records transformer block outputs at multiple noise levels.

## Commands

| Model | Command | Config class |
|---|---|---|
| Plaid | `collect-plaid-activations` | `PlaidCollectionConfig` |
| GENIE | `collect-activations` | `CollectionConfig` |
| T5 | `collect-t5-activations` | `T5CollectionConfig` |

## Input

- Model weights (pretrained `.pt` files for Plaid, Lightning `.ckpt` for fine-tuned).
- Dataset (HuggingFace stream or local `.src`/`.tgt` files).

## Output directory layout

```
experiments/activations/<model>/<dataset>/<split>/
├── metadata.json
├── layer_00/
│   ├── mean.pt
│   ├── std.pt
│   ├── mean_50000k.pt           # computed for max_samples=50M
│   ├── std_50000k.pt
│   ├── timestep_0050.pt
│   ├── timestep_0150.pt
│   └── ...
├── layer_04/
│   ├── timestep_0050.pt
│   └── ...
└── ...
```

Each `timestep_TTTT.pt` is a flat tensor of shape `(num_samples × seq_len, activation_dim)`. `metadata.json` records layer indices, timesteps, activation_dim, seq_len, num_samples.

## Key config fields (Plaid)

| Field | Purpose |
|---|---|
| `weights_path` | Directory with raw Plaid `.pt` files. |
| `checkpoint_path` | *Optional*. Lightning `.ckpt` for fine-tuned model. Overrides `weights_path`. |
| `dataset_name`, `dataset_split` | HuggingFace dataset identifier. |
| `data_dir` | *Optional*. Local XSum `.src`/`.tgt` directory. Overrides HF. |
| `tokenizer_path` | Required with `data_dir`. Path to Plaid `tokenizer.json`. |
| `max_samples`, `skip_samples` | Sample count and offset (for carving out splits). |
| `seq_len` | Tokenizer truncation length. |
| `diffusion_t_values` | Continuous `t ∈ [0,1]` values to sample. 10 evenly spaced for GENIE parity. |
| `batch_size` | Batch size during collection. Lower for large models. |
| `layers` | Block indices to hook. `None` = all blocks. |
| `force_overwrite` | Delete existing output dir before starting. |

## Layer choices for Plaid (24 blocks)

**Our policy**: 2 early, 2 middle, 2 late → `[0, 4, 10, 14, 20, 23]`.

Reasoning:
- Block 0 captures very low-level patterns right out of embeddings.
- Block 4 is still in "surface features" territory.
- Block 10 is early-middle.
- Block 14 is late-middle — typically where semantic composition peaks.
- Block 20 is late.
- Block 23 is the output layer pre-LayerNorm.

For quick exploration you can use just `[0, 23]` (first and last) — these already surface most of the phenomena.

## Noise levels (timesteps)

Plaid's diffusion is continuous but we sample 10 noise levels for parity with GENIE's 10 discrete timesteps:

```yaml
diffusion_t_values: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
```

`t=0` is clean, `t=1` is maximum noise. Note: trajectory collection (stage 3b) records all 256 sampling steps, not just these 10.

## Disk sizes (Plaid, dim=2048, bf16→float32 saved)

Per 1000 samples × 256 seq_len × 6 layers × 10 timesteps ≈ ~120 GB.

| Split | Samples | Disk | Job time |
|---|---|---|---|
| train | 10,000 | ~1.2 TB | ~2h50m |
| validation | 3,000 | ~354 GB | ~55 min |
| test | 3,000 | ~354 GB | ~1h8m |

See [[disk-usage]].

## Implementation notes

- Uses plain PyTorch forward hooks, NOT nnsight — nnsight keeps the computation graph alive and OOMs on 1.3B-param model.
- Activations are written to per-batch files during the pass (to avoid RAM buildup), then concatenated into a single `timestep_TTTT.pt` at the end of each timestep.
- `store.compute_layer_mean()` and `compute_layer_std()` cache their results to `mean.pt` / `std.pt` (or `_NNNk.pt` variants for sampled subsets).

## Fine-tuned model gotchas

When loading from a Lightning checkpoint (`checkpoint_path` set):

```python
from geniesae.plaid_xsum_training import PlaidXSumTrainingModule
ckpt_module = PlaidXSumTrainingModule.load_from_checkpoint(
    self.checkpoint_path, map_location=device,
)
model = ckpt_module.diffusion_model
embedding_matrix_module = ckpt_module.embedding_matrix
noise_schedule = ckpt_module.noise_schedule
gamma_bounds = ckpt_module.gamma_bounds
```

This bypasses `load_plaid_modules` which expects raw `.pt` files.

## XSum data gotcha

The GLGE XSum test split has `.src` but no `.tgt` file — references are unavailable. For top-examples collection this doesn't matter; for interpretation it means test-split explanations can't be scored against references (we use test for top-examples precisely because it's held out from training and validation).
