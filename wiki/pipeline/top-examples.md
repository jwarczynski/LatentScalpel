---
tags: [pipeline, sae, interpretation]
related: [[sae-training]] [[interpretation]]
---

# Stage 3a: Find Top Examples

For each SAE feature, find the dataset examples that maximally activate it.

## Command

```bash
uv run python main.py find-top-examples \
    configs/find_top_examples_plaid_finetuned_v3b.yaml \
    --layer_idx=0 \
    --sae_checkpoint_path=./experiments/sae_checkpoints/plaid_finetuned_v3b/layer_00.ckpt \
    --submit --infra.cluster=slurm
```

One job per layer. CLI overrides are how we substitute the checkpoint path.

Config class: `TopExamplesConfig` in `geniesae/configs/top_examples_config.py`.

## Algorithm

1. Load the trained SAE for the layer.
2. For each timestep file `timestep_TTTT.pt`:
   - Process in batches through `sae.encode()` → sparse codes `(batch, dictionary_size)`.
   - For each row, identify the non-zero features (there are exactly K per row).
   - For each non-zero feature, push `(activation, example_id, timestep, token_pos)` onto a per-feature min-heap of size `top_k`.
3. Output the sorted top-k list per feature as JSON.

example_id and token_pos are inferred from the row index given `seq_len` from metadata.

## Config

| Field | Purpose |
|---|---|
| `sae_checkpoint_path` | Trained SAE for this layer. |
| `activation_dir` | Should be the **test** or **held-out** split (see [[decision-split-policy]]). |
| `layer_idx` | Layer to process. |
| `dataset_name`, `dataset_split` | Propagated to metadata for later interpretation. |
| `top_k` | Number of top examples per feature (default 20, we use 30). |
| `unique_examples` | If True, same example can't appear twice per feature (keeps the max activation across timesteps/tokens). |
| `features` | Subset of feature indices to process. `None` = all. |
| `timesteps` | Subset of timesteps. `None` = all. |
| `output_dir` | Output directory. |

## Output format

`<output_dir>/layer_XX_top_examples.json`:

```json
{
  "metadata": {
    "dataset_name": "xsum",
    "dataset_split": "validation",
    "layer_idx": 0,
    "sae_checkpoint": "./experiments/sae_checkpoints/plaid_finetuned_v3b/layer_00.ckpt",
    "top_k": 30,
    "num_features": 16384,
    "activation_dim": 2048,
    "timesteps_used": [50, 150, 250, 350, 450, 550, 650, 750, 850, 950],
    "seq_len": 256,
    "unique_examples": true
  },
  "features": {
    "0": [
      { "example_id": 1337, "activation": 3.245, "timestep": 450, "token_position": 87 },
      { "example_id": 42, "activation": 2.891, "timestep": 650, "token_position": 12 },
      ...
    ],
    "1": [...],
    ...
  }
}
```

## Disk sizes

Per-layer JSON files for Plaid v3b (v1 run, 32k dictionary, top_k=30):

| Layer | Size |
|---|---|
| 0 | 9.6 MB |
| 4 | 38 MB |
| 10 | 56 MB |
| 14 | 65 MB |
| 20 | 106 MB |
| 23 | 116 MB |

Size scales with number of *live* features (dead features have empty lists). Total ~388 MB for all 6 layers.

## Gotchas

- `dataset_split` must match the split used when collecting activations. `example_id` values are indices into the raw dataset split ordering.
- When using local GLGE XSum data, remember the `test.tgt` file doesn't exist — test-split tops can be generated but interpretation-by-reference cannot use the targets.
- Examples with extremely high activation values can be embedding outliers (very short documents or tokenization artifacts). Inspect the top of top-of-feature-0 manually before trusting interpretations.

## Sanity check via CLI

After running, quickly verify activations are reasonable:

```python
import json
d = json.load(open("layer_00_top_examples.json"))
# Pick a few features
for feat in ["0", "100", "1000"]:
    if feat in d["features"] and d["features"][feat]:
        top = d["features"][feat][0]
        print(f"F{feat}: act={top['activation']:.2f} ex={top['example_id']} ts={top['timestep']}")
```

Healthy: max activations typically 2–10. If all features show max activation < 0.5 → something is wrong (possibly normalization mismatch or dead features dominate).
