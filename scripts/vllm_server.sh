#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/vllm_server_%j.log

# Launch a vLLM OpenAI-compatible server for LLM-as-judge interpretation.
#
# Usage:
#   sbatch scripts/vllm_server.sh
#
# Then point interpret-features at it:
#   uv run python main.py interpret-features configs/interpret_features.yaml \
#       --vllm_base_url=http://$(hostname):8000/v1
#
# To check which node it's running on:
#   squeue -u $USER --name=vllm-server -o "%N"

set -euo pipefail

cd /net/tscratch/people/plgjentker/GenieSAE

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"

echo "Starting vLLM server on $(hostname) at $(date)"
echo "Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --quantization awq \
    --enforce-eager \
    --dtype auto
