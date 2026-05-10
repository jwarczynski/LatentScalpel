---
tags: [experiment, inference, plaid, evaluation]
status: completed
date: 2026-04-21
slurm_job: 2526183
related: [[plaid-finetuned-v3b]] [[evaluation]]
---

# Exp: v3b inference on XSum dev+test

100 sample generations from the v3b best checkpoint — 50 from XSum dev, 50 from XSum test.

## Setup

Script: `scripts/plaid_xsum_inference.py`.

```bash
# In scripts/submit_eval_v3b.sh (run as sbatch on 1× A100)
CKPT="experiments/plaid_xsum_v3b/checkpoints/cond-256-lr1e6-cosine-100ep/best-epoch85.ckpt"

# Dev split (50 samples)
python scripts/plaid_xsum_inference.py \
    --checkpoint "$CKPT" \
    --data_dir "datasets/glge-released-dataset/easy/xsum_data/org_data" \
    --tokenizer_path "models/plaid/plaid1b_weights/tokenizer.json" \
    --num_samples 50 \
    --seq_len 256 \
    --max_summary_len 64 \
    --sampling_timesteps 256 \
    --score_temp 0.9 \
    --prefix_mode clean \
    --split validate

# Test split (50 samples) — same flags, --split test
```

## Sampler

`InpaintingSampler(prefix_mode="clean")`:
- Article tokens are held as clean embeddings throughout the reverse chain.
- Summary positions are denoised from pure noise.
- 256 sampling steps with `score_temp=0.9`.

## Output

`logs/eval_v3b_2526183.out` (134 KB). Structured as:

```
========== DEV SPLIT (50 samples) ==========
--- Example 1 (generated in 5.7s) ---
ARTICLE: <full article text>
REFERENCE: <reference summary from dev.tgt>
GENERATED: <our generation>
--- Example 2 ...
...
========== TEST SPLIT (50 samples) ==========
--- Example 1 ...
ARTICLE: <...>
REFERENCE: (no reference)        # test.tgt doesn't exist in GLGE
GENERATED: <...>
```

## Sample quality observations

**Good:**
- `Burberry`: "Burberry, the UK's biggest luxury fashion brand, has reported a 7% rise in profits over last year..." — accurate, well-formed.
- `Kermit`: "The first drawing of Kermit the Frog made 60 years ago has been donated to the Smithsonian in Washington." — very close to reference.
- `Magistrates' courts`: "Thirty-eight magistrates' courts in towns in England and Wales are set to close." — concise.

**Issues:**
- Hallucinated specifics: wrong names (`Nathan Pipe` instead of `David Pipe`), wrong numbers, wrong places.
- Some outputs empty (4 tokens before EOT).
- Longer articles suffer from seq_len=256 truncation — article eats most of the budget.

## Metrics

Not computed for this run — the inference script doesn't invoke `EvaluationModule`. For future runs, use `PlaidTokenGuidanceConfig` which wires up evaluation automatically.

## Reference file issue

GLGE XSum test set has no `.tgt` file (targets held secret). All test-split references show `(no reference)` in the log. Metrics computable only for dev.

## Reproducibility

Script is deterministic given `--seed` (not currently exposed in CLI, defaults to `torch.manual_seed(0)` implicitly via `random.seed`).
