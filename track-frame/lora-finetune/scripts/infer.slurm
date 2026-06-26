#!/bin/bash
#SBATCH --job-name=frame-lora-infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/infer-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/infer-%j.log

# Usage: sbatch infer.slurm <adapter_dir> [split] [extra runner args...]
#   sbatch infer.slurm logs/Qwen2.5-VL-7B-Instruct/checkpoint-250 test
#   sbatch infer.slurm logs/Qwen2.5-VL-7B-Instruct/final test --limit 16   # smoke
#
# max-pixels MUST match the training config (602112) for train/inference parity.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ADAPTER="${1:?usage: sbatch infer.slurm <adapter_dir> [split] [extra args]}"
[[ "$ADAPTER" = /* ]] || ADAPTER="$REPO/$ADAPTER"
SPLIT="${2:-test}"
MODEL_SRC="$REPO/os-models/Qwen2.5-VL-7B-Instruct"
STAGE="/dev/shm/$USER/$(basename "$MODEL_SRC")"
trap 'rm -rf "$STAGE"' EXIT

echo "infer: adapter=$ADAPTER split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging base -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$MODEL_SRC"/. "$STAGE"/

TAG="$(basename "$ADAPTER")"
OUT="$REPO/track-frame/lora-finetune/logs/resp_${SPLIT}_${TAG}"
python track-frame/lora-finetune/src/runner.py --split "$SPLIT" --base "$STAGE" \
    --adapter "$ADAPTER" --out "$OUT" --max-pixels 602112 "${@:3}"
echo "done. -> $OUT"
