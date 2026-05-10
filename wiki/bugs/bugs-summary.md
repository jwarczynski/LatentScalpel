---
tags: [bug, meta]
last_updated: 2026-05-03
---

# Bugs summary

One-line list of every bug found, with severity and fix commit. For details see individual pages.

## Plaid model (inference + training)

| Bug | Sev | Fix commit | Page |
|---|---|---|---|
| muP output_mult = 0.125 instead of 1.0 | critical | ``d06f5dd`` | [[bug-mup-scaling]] |
| Rotary rotated value vectors | critical | ``d06f5dd`` | [[bug-rotary-values]] |
| Samplers used gamma=0 for final decode | high | ``205c301`` | [[bug-sampler-gamma-zero]] |

## Plaid XSum training

| Bug | Sev | Fix commit | Page |
|---|---|---|---|
| Selfcond detachment missing | critical | ``759231c`` | [[bug-selfcond-detachment]] |
| torch.lerp dtype mismatch | low | ``34c4f6e`` | [[bug-lerp-dtype]] |
| Prior loss reduction mismatch | medium | ``0042bcc`` | [[bug-prior-loss-reduction]] |

## SAE training

| Bug | Sev | Fix commit | Page |
|---|---|---|---|
| SAE loss flat from step 0 | high | ``92a7752`` | [[bug-sae-stuck-loss]] |

## Pattern of discovery

Most of these were found by cross-validating against the original Plaid source code (`igul222/plaid` at `$SCRATCH/plaid`). The standalone script `scripts/plaid_original_xsum_generate.py` uses pure-PyTorch replacements for flash-attn and apex but otherwise matches the original exactly. Whenever our code diverged in quality, we compared intermediate tensor values between the two implementations to isolate the difference.

See [[exp-original-code-comparison]] for the methodology.

## Lessons

1. **Cross-validate by running a reference implementation** — 90% of our bug-finding came from this.
2. **Smoke-test on 1 GPU before multi-GPU jobs** — caught the lerp dtype bug in 2 minutes.
3. **When all samples look the same, the model is broken, not the data** — generic explanations and identical output patterns are symptoms of crushed logits or rotated values.
4. **Noise schedule modifications are a red flag** — if `gamma_0`/`gamma_1` params drift >1% during fine-tuning, a training bug exists.
5. **Compare against loss curves from published work** — our v1 losses "looked okay" in isolation but were off vs plaid's published values.
