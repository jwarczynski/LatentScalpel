# T5 SAE Feature Interpretation Results

## Setup

- Model: T5-Large fine-tuned on XSum (`sysresearch101/t5-large-finetuned-xsum`)
- SAE: Top-K SAE (expansion_factor=16, k=32) trained on decoder residual stream activations
- Layers: 0 (first), 11 (middle), 23 (last)
- LLM judge: Qwen/Qwen2.5-32B-Instruct-AWQ
- Protocol: DLM-Scope auto-interpretation (generate explanation, then score via discrimination task with 20 examples)
- Dataset: XSum train split (10k examples for activation collection)

## Results

| Layer | Active Features | Avg Interpretability | Median | Features > 0.7 |
|-------|----------------|---------------------|--------|----------------|
| 0     | 5,514          | 0.750               | 0.750  | 3,091 (56%)    |
| 11    | 8,852          | 0.754               | 0.750  | 4,852 (55%)    |
| 23    | 15,837         | 0.691               | 0.700  | 5,962 (38%)    |

## Observations

- The middle layer (11) has the highest average interpretability, consistent with the hypothesis that middle layers encode more semantically meaningful features.
- Layer 23 has the most active features (15,837 out of 16,384 dictionary size) but the lowest average interpretability score. Deeper layers likely encode more distributed, abstract representations that are harder for the LLM to describe concisely.
- Layer 0 has the fewest active features (5,514) — many dictionary entries are dead at the first layer, suggesting the input representation is relatively low-dimensional.
- All layers show reasonable interpretability (avg > 0.69), indicating the SAE successfully decomposes T5 decoder activations into meaningful features.

## Output Files

- `experiments/results/t5/interpretation_results_layer_00.json`
- `experiments/results/t5/interpretation_results_layer_11.json`
- `experiments/results/t5/interpretation_results_layer_23.json`
- `experiments/top_examples/t5/layer_{00,11,23}_top_examples.json`
