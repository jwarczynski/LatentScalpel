# T5 SAE Feature Interpretation Results

## Setup

- Model: T5-Large fine-tuned on XSum (`sysresearch101/t5-large-finetuned-xsum`)
- SAE: Top-K SAE (expansion_factor=16, k=32) trained on decoder residual stream activations
- Layers: 0 (first), 11 (middle), 23 (last)
- LLM judge: Qwen/Qwen2.5-32B-Instruct-AWQ
- Protocol: DLM-Scope auto-interpretation (arXiv:2602.05859, Appendix D) — two-stage LLM-as-judge pipeline described below
- Dataset: XSum train split (10k examples for activation collection)

### Interpretation Protocol

The DLM-Scope protocol has two stages:

1. **Explanation generation**: The judge LLM receives top-activating documents (with activating tokens marked by `<< >>`) ordered by activation strength, and produces a one-sentence explanation of what the feature activates on.

2. **Discrimination scoring**: The judge LLM receives the explanation and a shuffled mix of activating and non-activating examples (without markers). It predicts which examples the feature would activate on. The interpretability score = (TP + TN) / total examples.

System prompt (explanation stage):
> "We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words. Pay attention to capitalization and punctuation, since they might matter."

System prompt (scoring stage):
> "We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown several example sequences in random order. You must return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like '2, 9, 10, 12'. Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with 'None'. You should include nothing else in your response other than comma-separated numbers or the word 'None' - this is important."

20 scoring examples were used per feature (10 activating, 10 non-activating).

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
