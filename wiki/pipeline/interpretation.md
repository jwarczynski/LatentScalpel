---
tags: [pipeline, sae, llm]
related: [[top-examples]] [[temporal-classification]]
---

# Stage 4a: Feature Interpretation (LLM-as-judge)

Uses a large LLM (via vLLM) to generate natural-language explanations for SAE features based on their top-activating examples.

## Command

```bash
uv run python main.py interpret-features \
    configs/interpret_features_plaid_finetuned_v3b.yaml \
    --top_examples_path=./experiments/top_examples/plaid_finetuned_v3b/layer_00_top_examples.json \
    --output_path=./experiments/results/plaid_finetuned_v3b/interpretation_layer_00.json \
    --submit --infra.cluster=slurm
```

One job per layer. Use a shell loop to submit all 6:

```bash
for L in 0 4 10 14 20 23; do
  LSTR=$(printf "%02d" $L)
  uv run python main.py interpret-features \
    configs/interpret_features_plaid_finetuned_v3b.yaml \
    --top_examples_path=./experiments/top_examples/plaid_finetuned_v3b/layer_${LSTR}_top_examples.json \
    --output_path=./experiments/results/plaid_finetuned_v3b/interpretation_layer_${LSTR}.json \
    --submit --infra.cluster=slurm
done
```

Config class: `InterpretFeaturesConfig` in `geniesae/configs/interpret_config.py`.

## Protocol (DLM-Scope style)

Per feature:

1. **Explanation prompt** — show the LLM the top-activating documents and ask for a one-sentence explanation of what the feature detects.
2. **Scoring prompt** — give the LLM its own explanation, then present a mix of activating + non-activating examples. Ask it to identify which ones the feature would activate on. Compare predictions to ground truth → interpretability score.

## Inputs

| Field | Purpose |
|---|---|
| `top_examples_path` | JSON from find-top-examples. |
| `llm_model` | HF model ID. Default: `Qwen/Qwen2.5-32B-Instruct-AWQ`. |
| `vllm_kwargs` | Passed to `vllm.LLM()`. AWQ quantization, eager mode, 32k context. |
| `num_scoring_examples` | How many examples in scoring prompt (half activating, half not). Default 10. |
| `max_doc_chars` | Truncate each document in prompts. Default 500. |
| `features` | Subset to interpret. `None` = all. |
| `trajectory_data_path` | Optional. Enables temporal-aware prompts. |
| `data_dir` | Optional. Local XSum `.src` instead of HF dataset. |

## Output

```json
{
  "metadata": {
    "llm_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "top_examples_path": "...",
    "num_scoring_examples": 10,
    "num_features_interpreted": 2567
  },
  "features": {
    "42": {
      "explanation": "The neuron is activating on sentences containing direct quotes from named individuals.",
      "interpretability_score": 0.9,
      "predicted_indices": [1, 4, 9],
      "ground_truth_indices": [4, 9],
      "temporal_category": "early_only",            // if trajectory_data_path was provided
      "is_timing_feature": false
    },
    ...
  }
}
```

## Dataset handling

Supports two modes:
- HF dataset (default): `load_dataset(dataset_name, split=dataset_split)`, accessing `dataset[eid]["document"]`.
- Local files (`data_dir` set): reads `<dev|test|train>.src` from the directory and wraps lines as `[{"document": line}, ...]` — list-of-dicts that duck-types as HF for this access pattern.

## Runtime

**Long**. For Plaid v3b layer 23 (~31k features, 2 LLM calls per feature), expected runtime on 1× A100 with Qwen 32B: ~13h. Layer 4 (~17k features): ~5h. This blew past the 8h Slurm limit on 4 of 6 layers in our first run.

Observed times from 2026-04-21 run (layer-wise):

| Layer | Features | Wall time | Status |
|---|---|---|---|
| 0 | 2567 | 1.8h | ✅ completed |
| 4 | 17789 | 5.5h | ✅ completed |
| 10 | 27208 | 8h | ⏱ timeout (95% done) |
| 14 | 30029 | 8h | ⏱ timeout (80% done) |
| 20 | 32108 | 8h | ⏱ timeout (58% done) |
| 23 | 31269 | 8h | ⏱ timeout (57% done) |

Critical issue: partial progress is **not saved incrementally**. If the job is killed mid-run, all interpretations are lost. Needs a TODO: periodic checkpointing so partial results survive Slurm timeouts.

## Feature quality (first run observation)

First run on layer 0 produced interpretations that looked near-identical across features — "direct quotes from named individuals" appeared dozens of times. Root causes discussed in [[bug-sae-stuck-loss]]:

- Validation-only split used for both SAE training AND top-examples → overfitting + weak features.
- XSum's news-only distribution → quotes are everywhere → LLM anchors on them.
- Layer 0 is early → features are mostly low-level.

Expected to improve substantially after:
- v3 SAE retraining with normalization + proper splits (see [[exp-sae-finetuned-v3b-v3]]).
- Using test-split top-examples (see [[decision-split-policy]]).

## Prompt design

Prompts live in `geniesae/prompts.py`:

- `build_explanation_prompt(documents)` — Given top-K docs, one-sentence neuron description.
- `build_scoring_prompt(explanation, texts)` — Given explanation and shuffled mix, pick which texts activate the neuron.
- `build_temporal_explanation_prompt(documents, temporal_category, temporal_summary, nds_values, activating_tokens)` — Temporal-aware version for features with trajectory data.
- `parse_scoring_response(response, total)` — Parse LLM output into predicted indices.
- `compute_interpretability_score(predicted, gt, total)` — Accuracy.

## Gotchas

- vLLM must be installed with CUDA 11.8-compatible build on our cluster.
- AWQ quantization halves VRAM; without it 32B doesn't fit on 1 A100 40GB.
- The LLM tends to describe common properties of top examples rather than distinguishing what makes them different from non-activating ones. Possible prompt improvement: emphasize contrastive thinking.
