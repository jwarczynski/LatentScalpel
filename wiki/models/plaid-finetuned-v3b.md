---
tags: [model, plaid, finetune]
date: 2026-04-14
status: current-best
related: [[plaid-1b]] [[plaid-finetune-history]] [[decision-seq-len-256]] [[exp-conditional-v3b]]
---

# Plaid fine-tuned v3b (current best)

Our current best fine-tuned Plaid checkpoint on XSum.

## Identifiers

- Wandb: [`5gepdez9`](https://wandb.ai/jedrasowicz/plaid-xsum/runs/5gepdez9) — `cond-256-lr1e6-cosine-100ep`
- Slurm jobs: 2517798 (initial), 2506663 (resume; cancelled)
- Best checkpoint: `experiments/plaid_xsum_v3b/checkpoints/cond-256-lr1e6-cosine-100ep/best-epoch85.ckpt`
- Last checkpoint (epoch 100): `last-v1.ckpt`
- Config: [`configs/plaid_xsum_conditional_8gpu_v3b.yaml`](../../configs/plaid_xsum_conditional_8gpu_v3b.yaml)

## Training setup

| Property | Value |
|---|---|
| starting weights | [[plaid-1b]] pretrained |
| dataset | XSum (easy / GLGE) train split |
| seq_len | 256 |
| max_summary_len | 64 |
| training mode | conditional (see [[conditional-vs-template-training]]) |
| batch_size | 4 per GPU × 8 GPUs = 32 effective |
| learning_rate | 1e-6 |
| weight_decay | 4e-5 |
| betas | (0.9, 0.99) |
| lr_schedule | cosine |
| warmup_steps | 500 |
| bias_warmup_steps | 5000 |
| num_epochs | 100 |
| self_cond_prob | 0.25 |
| precision | bf16-mixed |
| gradient_checkpointing | true |
| hardware | 8× A100 40GB |

See [[decision-lr-1e-6]] for why we chose this lr, [[decision-seq-len-256]] for why 256.

## Data format

Each training example is formatted as `[article | SEP | summary]` where SEP is token 0 (the `<|endoftext_R9VQqF0Ag7|>` token). `boundary_idx` marks the SEP position.

In **conditional** mode (this run), during training:
1. `z_t` is constructed by noising the full sequence.
2. The article prefix part (`z_t[:bi]`) is then replaced with clean embeddings (no noise).
3. Loss is computed only on positions `[bi+1 .. real_len]` (the summary).

Loss mask construction in `_compute_vlb_loss`:

```python
if self.training_mode == "conditional" and boundary_idx is not None:
    loss_mask = torch.zeros(B, S, device=device, dtype=torch.float64)
    for b in range(B):
        bi = boundary_idx[b].item()
        loss_mask[b, bi:] = attention_mask[b, bi:].double()
```

## Training outcome

- Train loss dropped sharply in first ~500 steps then continued decreasing smoothly.
- Validation loss still decreasing at epoch 100 — run ended because LR reached ~0 from cosine schedule.
- Attempted resume run (job 2506663) with lr=1e-7 linear tail to extend training, but it was cancelled early on.

Training curves are on wandb (link above).

## Evaluation (inference on XSum)

100 generations (50 dev + 50 test) via `scripts/plaid_xsum_inference.py` with `InpaintingSampler(prefix_mode="clean")`, 256 sampling steps. See [[exp-eval-v3b-inference]] and `logs/eval_v3b_2526183.out`.

**Sample quality (dev split, 50 examples):**
- Many well-formed summaries with accurate factual content (Burberry, Romanian stowaway, Kermit donation, etc.).
- Hallucinated specifics in a notable fraction (wrong names, wrong numbers).
- Some outputs empty or truncated.
- Test split had no reference file (`test.tgt` missing) — but generations are similarly variable in quality.

**Known issues:**
- Some samples produce only 0-4 tokens before hitting EOT — model emits EOT too aggressively.
- Longer articles truncated heavily (seq_len=256 includes both article and summary budget).

## Bug fixes incorporated

This run used the first code version that had all known bugs fixed:
- [[bug-mup-scaling]] ✅
- [[bug-rotary-values]] ✅
- [[bug-sampler-gamma-zero]] ✅ (inference only)
- [[bug-selfcond-detachment]] ✅
- [[bug-lerp-dtype]] ✅

## What we used the checkpoint for

- [[exp-eval-v3b-inference]] — 100 XSum generations
- [[exp-sae-finetuned-v3b-v1]] — initial SAE training (validation split only, poor features)
- [[exp-sae-finetuned-v3b-v2]] — SAE with proper 3-split policy
- [[exp-sae-finetuned-v3b-v3]] — SAE with normalization (current)

## Related

- [[plaid-finetune-history]] — all fine-tune attempts
- [[exp-conditional-v3a]] — sibling seq_len=1024 run
- [[exp-template-v3c]] — sibling template-mode run
