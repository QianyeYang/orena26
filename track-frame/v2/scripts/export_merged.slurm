#!/bin/bash
#SBATCH --job-name=frame-v2-export
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=4:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/export-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/export-%j.log

# Merge a LoRA into the base and write a deployable model directory.
#
# Runs entirely on CPU (the merge is an addition and torchao's int8 weight-only
# conversion needs no GPU), but 52 GiB of bf16 weights plus the merged copy have
# to be resident, hence --mem=150G. A GPU is requested only because the h200
# partition allocates whole nodes by GPU; nothing here touches it.
#
# Usage: sbatch export_merged.slurm <adapter-ckpt> <name>
#   sbatch export_merged.slurm \
#     track-frame/v2/logs/Qwen3.6-27B-frame-v1prompt/checkpoint-23136 ep24
set -euo pipefail
REPO=/datasets/engs2732/orena
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

ADAPTER="${1:?adapter checkpoint is required}"
NAME="${2:?short name (e.g. ep24) is required}"

python track-frame/v2/src/export_merged.py \
    --base os-models/Qwen3.6-27B \
    --adapter "$ADAPTER" \
    --out-merged "os-models/Qwen3.6-27B-frame-$NAME-merged" \
    --out-quant  "os-models/Qwen3.6-27B-frame-$NAME-int8mlp"

echo "done -> os-models/Qwen3.6-27B-frame-$NAME-{merged,int8mlp}"
