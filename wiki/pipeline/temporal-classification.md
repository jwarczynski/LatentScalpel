---
tags: [pipeline, sae, temporal]
related: [[trajectory-collection]] [[interpretation]] [[midpoint-features]]
---

# Stage 4b: Temporal Classification

Classifies each SAE feature by its temporal activation pattern along the denoising trajectory. Categories: early_only, late_only, midpoint_transition, starting_spike, finishing_spike, stable, variable, midpoint_exclusive, other.

## Command

```bash
PYTHONPATH=. uv run python scripts/classify_temporal_features.py \
    experiments/results/plaid_finetuned_v3b/trajectory_features.json \
    --output_dir experiments/results/plaid_finetuned_v3b/temporal
```

CPU-only, fast. One output JSON per layer.

## Categories

Defined in `geniesae/temporal_classifier.py::TemporalClassifier`:

| Category | Condition |
|---|---|
| `midpoint_exclusive` | Peak within ±midpoint_window_pct of midpoint AND midpoint/outside ratio > midpoint_ratio_threshold (default 10) |
| `stable` | Coefficient of variation < 0.3 |
| `early_only` | first_half_mean / second_half_mean > 3 |
| `late_only` | first_half_mean / second_half_mean < 0.33 |
| `finishing_spike` | last_10%_mean / overall_mean > 2.5 |
| `starting_spike` | first_10%_mean / overall_mean > 2.5 |
| `midpoint_transition` | `abs(log(half_ratio)) > 0.5` (clear transition between halves) |
| `variable` | none of the above (catch-all for moderately-variable features) |

Classification happens in order — `midpoint_exclusive` wins over others.

## Output

```json
{
  "metadata": {
    "layer": 0,
    "total_features_classified": 253,
    "midpoint_ratio_threshold": 10.0,
    "midpoint_window_pct": 0.10,
    "diffusion_steps": 256
  },
  "features": {
    "42": {
      "category": "early_only",
      "mean_activation": 0.134,
      "peak_nds": 0.88,
      "peak_nds_raw": 225,
      "coefficient_of_variation": 1.42,
      "first_half_mean": 0.21,
      "second_half_mean": 0.05,
      "midpoint_activation": 0.10,
      "midpoint_to_outside_ratio": 0.7
    },
    ...
  }
}
```

## Plaid v3b actual counts (per layer, from 2026-04-21 run)

| Layer | early_only | late_only | midpoint_trans. | midpoint_excl. | variable | stable | starting | finishing |
|---|---|---|---|---|---|---|---|---|
| 0 | 87 | 132 | 19 | 1 | 11 | 3 | - | - |
| 4 | 183 | 63 | 11 | 1 | 9 | 13 | - | - |
| 10 | 624 | 103 | 43 | 3 | 26 | - | - | 2 |
| 14 | 541 | 198 | 61 | 18 | 53 | 3 | - | 3 |
| 20 | 332 | 155 | 62 | 17 | 53 | 1 | - | - |
| 23 | 399 | 98 | 56 | 5 | 36 | 2 | 1 | - |

**Observations:**

- Layer 0 is the exception — more `late_only` than `early_only`. Often block 0 is still doing linear-ish surface work at low noise.
- Deeper layers (10, 14) are dominated by `early_only` features — classic pattern for diffusion LMs where deeper layers do heavy structural work at high noise then quiet down.
- `midpoint_exclusive` features cluster in layers 14 and 20 — these are the ones intervention experiments target (see [[midpoint-features]] for the GENIE analog).

## Bridging metadata

GENIE trajectory JSON has `diffusion_steps` in metadata. Plaid uses `sampling_timesteps`. The standalone script `scripts/classify_temporal_features.py` bridges this:

```python
if "diffusion_steps" not in meta and "sampling_timesteps" in meta:
    meta["diffusion_steps"] = meta["sampling_timesteps"]
```

## NDS convention

Normalized diffusion step = step / (total_steps - 1). Convention used here: **0 is clean (low noise), 1 is maximum noise**. A feature that peaks at NDS=0.5 fires most strongly at mid-diffusion.

Be careful reading GENIE code — the original GENIE classifier may have used the reverse convention. Our classifier follows the Plaid-natural direction.

## Use in interpretation

The LLM interpretation stage ([[interpretation]]) can optionally take a `trajectory_data_path` to include temporal info in prompts. When a feature is `midpoint_exclusive`, the prompt includes that category and the LLM can produce explanations like "this feature fires specifically during the midpoint of denoising."
