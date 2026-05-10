---
tags: [decision, data, xsum]
status: adopted
---

# Decision: local GLGE XSum files over HuggingFace xsum

For XSum data, use the local `.src`/`.tgt` files from GLGE (`datasets/glge-released-dataset/easy/xsum_data/org_data/`) instead of the HuggingFace `xsum` dataset.

## Why

1. **Pre-tokenized and cleaned**. GLGE provides line-aligned `.src` (articles) and `.tgt` (summaries) already cleaned. HF xsum is raw and requires custom parsing for some edge cases.
2. **Offline**. Ares compute nodes have no internet. HF's `load_dataset("xsum")` with streaming works on the login node only. GLGE files are on scratch.
3. **Consistent splits**. GLGE uses the "easy" variant with a fixed train/dev/test split. HF xsum has had multiple versions and the split indices differ.

## Code paths

Collection configs (`PlaidCollectionConfig`, `PlaidTrajectoryConfig`) support either:

- **HF**: set `dataset_name="xsum"` and `dataset_split="validation"`.
- **Local**: set `data_dir` to the GLGE path and `tokenizer_path` to the Plaid tokenizer.

When both are set, `data_dir` wins. Example:

```yaml
dataset_name: "xsum"
dataset_split: "validation"
data_dir: "datasets/glge-released-dataset/easy/xsum_data/org_data"
tokenizer_path: "models/plaid/plaid1b_weights/tokenizer.json"
```

## Split name mapping

GLGE uses `dev.src`/`dev.tgt` for validation, not `validation.src`:

```python
split_prefix = {"train": "train", "validation": "dev", "test": "test"}
src_file = Path(self.data_dir) / f"{split_prefix.get(self.dataset_split, self.dataset_split)}.src"
```

## Gotcha: test set has no targets

GLGE XSum test set has `.src` but no `.tgt` — test labels are held secret for leaderboard purposes. Affects:

- [[evaluation]] — can't compute metrics on test.
- [[top-examples]] — can still mine test for top-activating examples, which is exactly why we use test for this stage.

## For `InterpretFeaturesConfig`

Same pattern: `data_dir` override. When set, reads `.src` lines and wraps them as `[{"document": line}, ...]` — list-of-dicts that duck-types as HF for the `dataset[eid]["document"]` access pattern used internally.

## Not using for pretrained Plaid SAEs

For the pretrained SAE runs we used HF `openwebtext` instead — XSum is too small a distribution for pretrained-model analysis. GLGE XSum is only appropriate for the fine-tuned model.
