#!/bin/bash
#SBATCH --job-name=plaid-orig-xsum
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/plaid_orig_xsum_%j.out
#SBATCH --error=logs/plaid_orig_xsum_%j.err

set -euo pipefail

SCRATCH="/net/tscratch/people/plgjentker"
WORKDIR="$SCRATCH/GenieSAE"
cd "$WORKDIR"

# Activate venv
source .venv/bin/activate

mkdir -p logs
mkdir -p experiments/plaid_original_xsum

python scripts/plaid_original_xsum_generate.py \
    --weights_path "$SCRATCH/GenieSAE/models/plaid/plaid1b_weights" \
    --tokenizer_path "$SCRATCH/plaid/misc/owt2_tokenizer.json" \
    --xsum_src_path "$SCRATCH/GenieSAE/datasets/glge-released-dataset/easy/xsum_data/org_data/dev.src" \
    --xsum_tgt_path "$SCRATCH/GenieSAE/datasets/glge-released-dataset/easy/xsum_data/org_data/dev.tgt" \
    --output_path "experiments/plaid_original_xsum/dev_results.jsonl" \
    --num_samples 50 \
    --n_samples_per_article 1 \
    --max_article_tokens 200 \
    --max_summary_tokens 80 \
    --sampling_timesteps 256 \
    --guidance_weight 2.0 \
    --score_temp 0.9
