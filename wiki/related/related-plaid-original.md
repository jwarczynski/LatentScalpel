---
tags: [related, reference]
---

# Plaid original repo

The canonical `igul222/plaid` source. Our reimplementation is in `geniesae/plaid_model.py`; the standalone `scripts/plaid_original_xsum_generate.py` is a near-verbatim copy of the upstream code with flash-attn / apex replaced by pure PyTorch.

## Links

- Repo: https://github.com/igul222/plaid
- Paper: https://arxiv.org/abs/2305.18619
- Weights: GitHub releases `v1.0.0`.

## Remote path

Cloned at `$SCRATCH/plaid` (on Athena) for cross-validation use. This is where the standalone `plaid_original_xsum_generate.py` was derived from.

## Key files

- `sample.py` — authoritative reference for generation (guidance modes, reverse sampling).
- `train.py` — authoritative reference for the VDM loss and selfcond handling.
- `lib/models.py` — `DiffusionModel` class.
- `lib/rotary.py` — rotary embedding implementation we had to match.
- `misc/owt2_tokenizer.json` — the Plaid tokenizer (we also have this at `models/plaid/plaid1b_weights/tokenizer.json`).

## Used for

- **Bug hunting**: when our reimplementation produced gibberish, we diff'd intermediate values against this. See [[exp-original-code-comparison]].
- **Inference reference**: when in doubt about a sampling detail, read the upstream `sample.py`.
- **Loss reference**: when in doubt about training loss terms, read `train.py`.

## Don't modify

Treat the clone as read-only. If a behavior diverges, fix **our** code to match, not the reference.

## Relation to `plaid-finetuned-v3b`

v3b's model code is `geniesae/plaid_model.py`, which is our reimplementation. It should behave identically to `$SCRATCH/plaid/lib/models.py` on a forward pass (modulo the three bugs we found — see [[bugs-summary]]).

For a side-by-side forward-pass comparison, run `scripts/compare_forward_pass.py` and `scripts/compare_sampling.py`.
