#!/bin/bash
#SBATCH --job-name=plaid-compare
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/plaid_compare_%j.out
#SBATCH --error=logs/plaid_compare_%j.err

set -euo pipefail
cd /net/tscratch/people/plgjentker/GenieSAE
source .venv/bin/activate
mkdir -p logs
python scripts/compare_sampling.py
