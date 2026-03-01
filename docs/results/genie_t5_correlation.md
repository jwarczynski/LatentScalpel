# Cross-Model Feature Correlation: Genie L0 vs T5 L0

## Setup

- Genie SAE: layer 0, 12,288 features (expansion_factor=16, activation_dim=128 from Genie decoder)
- T5 SAE: layer 0, 16,384 features (expansion_factor=16, activation_dim=1024 from T5 decoder)
- Method: For 100,000 activation rows from the XSum train split, encode each through both SAEs to get binary co-activation vectors (feature active or not), then compute pairwise Pearson correlation across all feature pairs.
- Both models were trained/fine-tuned on XSum, so they share the same data domain.

## LLM-as-Judge Interpretation Protocol

Feature explanations and interpretability scores were produced using the DLM-Scope auto-interpretation protocol (arXiv:2602.05859, Appendix D). The protocol has two stages:

**Stage 1 — Explanation generation.** The LLM receives a system message framing the task as studying neurons in a neural network, where each neuron activates on particular words/substrings/concepts. The top-activating documents (with activating tokens marked by `<< >>`) are listed from most to least strongly activating. The LLM is asked to summarize in a single sentence what the neuron activates on.

System prompt (explanation):
> "We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words. Pay attention to capitalization and punctuation, since they might matter."

**Stage 2 — Discrimination scoring.** The LLM receives the explanation from Stage 1 and a shuffled mix of activating and non-activating examples (without markers). It must predict which examples the neuron would activate on. The interpretability score is computed as accuracy: (TP + TN) / total.

System prompt (scoring):
> "We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown several example sequences in random order. You must return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like '2, 9, 10, 12'. Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with 'None'. You should include nothing else in your response other than comma-separated numbers or the word 'None' - this is important."

Both Genie and T5 features were interpreted independently using the same protocol with `Qwen/Qwen2.5-32B-Instruct-AWQ` as the judge LLM and 20 scoring examples per feature.

## Statistics

| Metric | Value |
|--------|-------|
| Mean correlation | ~0.0 |
| Median correlation | 0.0 |
| Max correlation | 1.0 |
| Std deviation | 0.0011 |
| Pairs with r > 0.3 | 69 |
| Pairs with r > 0.5 | 9 |
| Pairs with r > 0.7 | 3 |

The vast majority of cross-model feature pairs are uncorrelated (mean ~0), which is expected — the two models have different architectures (diffusion LM vs encoder-decoder), different activation dimensions (128 vs 1024), and different internal representations. The few highly correlated pairs are notable because they suggest shared semantic concepts despite these architectural differences.

## Top Correlated Feature Pairs (r > 0.5)

### T5 F1700 <-> Genie F7352 (r=1.00)

- T5 explanation (score 0.85): "Notable events, figures, or changes in various fields such as sports, music, politics, economics, and environmental issues, often including names of specific people, places, or organizations."
- Genie explanation (score 0.50): "Reports or descriptions of events, incidents, or news stories, particularly those involving crime, accidents, natural phenomena, or noteworthy occurrences."
- Both capture general news event reporting. The perfect correlation suggests these features fire on the exact same XSum examples.

### T5 F5559 <-> Genie F7352 (r=1.00)

- T5 explanation (score 0.80): "Reports or news articles discussing various events, often involving unexpected occurrences, controversies, or significant issues related to health, environment, politics, and social matters."
- Same Genie feature as above. Two distinct T5 features both perfectly correlate with one Genie feature, suggesting Genie's representation is more compressed at layer 0.

### T5 F95 <-> Genie F10239 (r=0.71)

- T5 explanation (score 0.65): "Words and phrases related to the place 'Orkney' or names that sound similar."
- Genie explanation (score 0.75): "Texts that discuss legal or disciplinary actions, often in the context of sports, politics, or journalism."
- Weaker semantic overlap — the correlation may be driven by specific XSum articles about Orkney that happen to involve legal/disciplinary contexts.

### T5 F13162 <-> Genie F8832 (r=0.58)

- T5 explanation (score 0.90): "Player's age, as indicated by phrases like 'the 21-year-old joined Liverpool from Manchester United'."
- Genie explanation (score 0.90): "Sentences that describe the transfer or loan of a player, including details such as the player's age, previous teams, and performance statistics."
- Strong semantic match — both features detect sports transfer news with player age details. Both have high interpretability (0.90).

### T5 F777 <-> Genie F10570 (r=0.58)

- T5 explanation (score 0.60): "Statements or announcements made by official entities, particularly in contexts involving legal, political, or organizational matters."
- Genie explanation (score 0.80): "Crime, legal proceedings, and reports of incidents, often including details about suspects, victims, charges, and court appearances."
- Overlapping domain: official/legal statements and crime reporting.

### T5 F11309 <-> Genie F10570 (r=0.52)

- T5 explanation (score 0.80): "Instances and descriptions of abductions, kidnappings, and their aftermath."
- Genie explanation (score 0.80): "Crime, legal proceedings, and reports of incidents."
- Clear semantic overlap — kidnapping is a subset of crime reporting.

### T5 F14308 <-> Genie F10570 (r=0.51)

- T5 explanation (score 0.75): "Texts that discuss violence, conflict, and militant activities, often involving specific groups like Boko Haram, Abu Sayyaf, or ISIS."
- Genie explanation (score 0.80): "Crime, legal proceedings, and reports of incidents."
- Again, violence/conflict maps to the broader crime/incident Genie feature.

## Key Observations

1. Genie feature 10570 (crime/legal) correlates with multiple T5 features covering different crime subtypes (official statements, kidnappings, militant violence). This suggests Genie L0 has a broader "crime" feature while T5 L0 splits this into finer-grained categories.

2. Genie feature 7352 (news events) correlates perfectly with two T5 features. At layer 0, Genie may use a single general "news event" detector where T5 uses multiple.

3. The strongest semantic match is T5 F13162 <-> Genie F8832 (sports transfers with player age) — both models independently learned a specific feature for this common XSum pattern, and both achieve 0.90 interpretability.

4. Only 69 out of ~200 million possible pairs have r > 0.3, confirming that cross-model feature alignment is sparse but meaningful where it exists.

## Output Files

- `experiments/correlation/genie_l0_vs_t5_l0/correlation_matrix.pt` (16384 x 12288 tensor)
- `experiments/correlation/genie_l0_vs_t5_l0/correlation_summary.json`
