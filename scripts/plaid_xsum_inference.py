"""Quick inference script: load a fine-tuned PLAID checkpoint and generate
summaries for a few validation examples.

Usage (on Slurm via exca, or directly on a GPU node):
    uv run python scripts/plaid_xsum_inference.py \
        --checkpoint experiments/plaid_xsum/checkpoints/last.ckpt \
        --num_samples 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import tokenizers

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geniesae.plaid_model import (
    DiffusionModel,
    EmbeddingMatrix,
    GammaBounds,
    NoiseSchedule,
)
from geniesae.plaid_samplers import InpaintingSampler
from geniesae.plaid_xsum_training import PlaidXSumTrainingModule
from geniesae.xsum_data import XSumDataModule


def main():
    parser = argparse.ArgumentParser(description="PLAID XSum inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="datasets/glge-released-dataset/easy/xsum_data/org_data")
    parser.add_argument("--tokenizer_path", type=str,
                        default="models/plaid/plaid1b_weights/tokenizer.json")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_summary_len", type=int, default=64)
    parser.add_argument("--sampling_timesteps", type=int, default=256)
    parser.add_argument("--score_temp", type=float, default=0.9)
    parser.add_argument("--prefix_mode", type=str, default="renoised",
                        choices=["clean", "renoised"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    t0 = time.time()
    module = PlaidXSumTrainingModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    module.eval()
    module.to(device)
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s")

    # Build sampler
    sampler = InpaintingSampler(
        model=module.diffusion_model,
        noise_schedule=module.noise_schedule,
        gamma_bounds=module.gamma_bounds,
        embedding_matrix=module.embedding_matrix,
        sampling_timesteps=args.sampling_timesteps,
        score_temp=args.score_temp,
        prefix_mode=args.prefix_mode,
    )

    # Load validation data
    dm = XSumDataModule(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        max_summary_len=args.max_summary_len,
        batch_size=1,
        num_workers=0,
        tokenizer_path=args.tokenizer_path,
    )
    dm.setup("validate")
    val_dataset = dm.val_dataset
    assert val_dataset is not None, "No validation dataset found"
    print(f"Validation set: {len(val_dataset)} examples")

    # Resolve SEP token ID
    sep_id = dm.sep_token_id
    print(f"SEP token ID: {sep_id}")

    # Generate summaries
    print(f"\n{'='*80}")
    print(f"Generating {args.num_samples} summaries ({args.sampling_timesteps} steps, "
          f"temp={args.score_temp}, prefix_mode={args.prefix_mode})")
    print(f"{'='*80}\n")

    for i in range(min(args.num_samples, len(val_dataset))):
        sample = val_dataset[i]
        token_ids = sample["token_ids"].unsqueeze(0).to(device)      # (1, seq_len)
        boundary_idx = sample["boundary_idx"].unsqueeze(0).to(device) # (1,)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        bi = boundary_idx[0].item()

        # Decode original article and reference summary
        all_ids = token_ids[0].tolist()
        article_ids = all_ids[:bi]
        # Summary starts after SEP
        summary_start = bi + 1
        # Find where padding starts
        real_len = attention_mask[0].sum().item()
        ref_summary_ids = all_ids[summary_start:real_len]

        article_text = tokenizer.decode(article_ids)
        ref_summary_text = tokenizer.decode(ref_summary_ids) if ref_summary_ids else "(no reference)"

        # Generate
        t0 = time.time()
        with torch.no_grad():
            generated_ids = sampler.sample(
                article_token_ids=token_ids,
                boundary_idx=boundary_idx,
                seq_len=args.seq_len,
            )
        gen_time = time.time() - t0

        # Decode generated summary (after boundary_idx + 1)
        gen_all = generated_ids[0].tolist()
        gen_summary_ids = gen_all[summary_start:]
        # Truncate at first SEP/PAD (token 0)
        gen_summary_clean = []
        for tid in gen_summary_ids:
            if tid == sep_id:
                break
            gen_summary_clean.append(tid)
        gen_summary_text = tokenizer.decode(gen_summary_clean) if gen_summary_clean else "(empty)"

        print(f"--- Example {i+1} (generated in {gen_time:.1f}s) ---")
        print(f"ARTICLE (first 200 chars): {article_text[:200]}...")
        print(f"REFERENCE: {ref_summary_text}")
        print(f"GENERATED: {gen_summary_text}")
        print()


if __name__ == "__main__":
    main()
