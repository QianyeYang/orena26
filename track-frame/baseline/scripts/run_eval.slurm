#!/bin/bash
#SBATCH --job-name=frame-base-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/baseline/logs/eval-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/baseline/logs/eval-%j.log

# Usage: sbatch run_eval.slurm [split]   (needs os-models/Qwen3.5-4B for the judge)
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SPLIT="${1:-test}"
OUT="$REPO/track-frame/baseline/logs/resp_${SPLIT}"

# Stage judge to node-local /dev/shm (same reason as run_inference.slurm: Lustre
# mmap random-reads stall; sequential copy + load-from-RAM avoids it).
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
STAGE="/dev/shm/$USER/$(basename "$JUDGE_SRC")"
trap 'rm -rf "$STAGE"' EXIT
echo "eval: split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging judge -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$JUDGE_SRC"/. "$STAGE"/

python track-frame/baseline/src/evaluate.py --split "$SPLIT" \
    --responses "$OUT/responses.json" --out "$OUT/eval" --judge-model "$STAGE"
echo "done."
