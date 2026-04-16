#!/bin/bash
#SBATCH --job-name=eval-v3b
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_v3b_%j.out
#SBATCH --error=logs/eval_v3b_%j.err

set -euo pipefail
cd /net/tscratch/people/plgjentker/GenieSAE
source .venv/bin/activate
mkdir -p logs

CKPT="experiments/plaid_xsum_v3b/checkpoints/cond-256-lr1e6-cosine-100ep/best-epoch85.ckpt"
DATA_DIR="datasets/glge-released-dataset/easy/xsum_data/org_data"
TOK="models/plaid/plaid1b_weights/tokenizer.json"

echo "========== DEV SPLIT (50 samples) =========="
python scripts/plaid_xsum_inference.py \
    --checkpoint "$CKPT" \
    --data_dir "$DATA_DIR" \
    --tokenizer_path "$TOK" \
    --num_samples 50 \
    --seq_len 256 \
    --max_summary_len 64 \
    --sampling_timesteps 256 \
    --score_temp 0.9 \
    --prefix_mode clean \
    --split validate

echo ""
echo "========== TEST SPLIT (50 samples) =========="
python scripts/plaid_xsum_inference.py \
    --checkpoint "$CKPT" \
    --data_dir "$DATA_DIR" \
    --tokenizer_path "$TOK" \
    --num_samples 50 \
    --seq_len 256 \
    --max_summary_len 64 \
    --sampling_timesteps 256 \
    --score_temp 0.9 \
    --prefix_mode clean \
    --split test
