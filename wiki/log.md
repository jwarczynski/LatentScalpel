---
tags: [meta, log]
---

# Log

Append-only chronological record. Newest entries at the top. Each entry uses `## [YYYY-MM-DD] <type> | <short title>` for easy grepping:

```bash
grep "^## \[" wiki/log.md | head -20
grep "ingest\|experiment" wiki/log.md
```

Types: `ingest`, `experiment`, `bug`, `decision`, `lint`, `meta`, `note`.

---

## [2026-05-03] meta | Wiki initialized

Bootstrapped the persistent wiki under `GenieSAE/wiki/`. Populated from conversation history, git log, existing `docs/`, configs, and code. See [[index]] for the catalog.

## [2026-05-03] experiment | SAE training v3 submitted (job 2559832)

Cancelled v2 SAE training (job 2559785) because loss was flat from step 0 (see [[bug-sae-stuck-loss]]). Resubmitted with:
- `normalize_inputs: true` (per-dim mean/std from train activations)
- `expansion_factor: 8` (dictionary=16384 instead of 32768)
- `k_start_multiplier: 1.0` (no k-warmup)
- `max_samples: 50_000_000` (10× more data)
- Added `train/active_feat_frac` metric

Config bumped to `infra.version=3`. All 6 layers [0, 4, 10, 14, 20, 23] submitted. See [[exp-sae-finetuned-v3b-v3]].

## [2026-05-03] experiment | Activation collection on train+test splits

Collected activations on XSum `train` (10K samples, job 2547972, ~2h50min) and `test` (3K samples, job 2547973, ~1h8min) for v3b checkpoint. Matches GENIE's 3-split policy. See [[decision-split-policy]].

## [2026-04-26] decision | Adopt 3-split policy for fine-tuned Plaid SAE

Previous SAE training used validation activations for BOTH training AND top-examples, leading to generic/repetitive feature explanations. Switching to GENIE-style: SAE train on XSum train, val on dev, top-examples on test. See [[decision-split-policy]].

## [2026-04-21] experiment | First v3b SAE analysis pipeline complete

Trained SAEs on 6 layers (v1: validation-only split, 32768 dict, k=64). Top-examples collected. Trajectory recorded. Interpretation via Qwen 2.5 32B vLLM. Layer 0 interpretations look near-identical ("direct quotes from named individuals") — symptom of poor features + XSum news bias. See [[exp-sae-finetuned-v3b-v1]].

## [2026-04-14] experiment | v3b fine-tune complete (epoch 85 is best)

Best run so far: wandb `cond-256-lr1e6-cosine-100ep` (run `5gepdez9`). Validation loss still decreasing at epoch 100 but lr reached near-zero. See [[exp-conditional-v3b]] and [[decision-seq-len-256]].

## [2026-04-12] experiment | v3 runs submitted (3 parallel)

Submitted v3a (seq_len=1024), v3b (seq_len=256 conditional), v3c (seq_len=256 template) with lr=1e-6, cosine schedule, 100 epochs. All 8-GPU on A100s. GPU cluster had many fully-free nodes so all started immediately.

## [2026-03-29] decision | Lower lr and cosine schedule

Oscillating loss at lr=1e-5 prompted switch to 1e-6 + cosine (easier to resume). See [[decision-lr-1e-6]].

## [2026-03-28] experiment | v2 fine-tune 30 epochs completed

Two runs completed (conditional + template) with lr=1e-5, linear decay, weight_decay=4e-5, beta2=0.99. Loss dropped ~enormously in first 500 steps then oscillated 2-4. Generated summaries readable but many hallucinations and inconsistent quality.

## [2026-03-24] bug | Selfcond detachment missing in VLB loss

Found critical training bug: `gamma`, `gamma_prime`, `x_embed`, `alpha_1`/`sigma_1` must be detached for self-conditioned examples (original train.py does this). Our code skipped it → noise schedule received conflicting gradients. See [[bug-selfcond-detachment]].

## [2026-03-24] bug | Lerp dtype mismatch

`torch.lerp` requires weight arg dtype to match input. `gamma_t`/`alpha_1` are float64, `selfcond_mask` was float32 → RuntimeError during validation. Fixed by casting to `.double()` for float64 inputs and keeping `.float()` for `x_embed`. See [[bug-lerp-dtype]].

## [2026-03-22] bug | Sampler final decode used gamma=0

All three samplers (Inpainting, TokenGuidance, GradientGuidance) passed `gamma=torch.zeros(B)` to the final decode forward pass. Original uses `gamma_t` from last iteration (= gamma(0) = gamma_0 ≈ -3). Affects z_variance normalization and bias scaling. See [[bug-sampler-gamma-zero]]. Fixed in commit ``205c301``.

## [2026-03-22] bug | Rotary rotated value vectors

Our `Rotary.forward` produced cos/sin with shape `(1, seq, 1, dim)` — missing the qkv dimension. Original produces `(1, seq, 3, 1, dim)` with v's slot set to cos=1, sin=0. Our version rotated v too, corrupting attention. Fixed in ``d06f5dd``. See [[bug-rotary-values]].

## [2026-03-22] bug | muP output_mult = 0.125 instead of 1.0

`_apply_mup_shapes` set `output_mult = base_dim / dim = 0.125`. Original mup.MuReadout defaults to `output_mult = 1.0`. Combined with `width_mult = 8.0`, our effective scaling was `0.015625` (8× too small) → crushed logits → gibberish generation. Fixed in ``d06f5dd``. See [[bug-mup-scaling]].

## [2026-03-14] ingest | Cloned original Plaid repo for validation

Cloned `igul222/plaid` to `$SCRATCH/plaid` to cross-check our reimplementation. Built a standalone script `scripts/plaid_original_xsum_generate.py` that runs the original model code with pure-PyTorch replacements for flash-attn and apex. Used to compare outputs. See [[related-plaid-original]] and [[exp-original-code-comparison]].
