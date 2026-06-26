#!/bin/bash
#SBATCH --job-name=frame-lora-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/eval-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/eval-%j.log

# Usage: sbatch eval.slurm <resp_dir> [split]   (resp_dir holds responses.json)
#   sbatch eval.slurm logs/resp_test_final test
# Reuses the baseline evaluator (focus.Evaluator, judge = Qwen3.5-4B).
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RESP="${1:?usage: sbatch eval.slurm <resp_dir> [split]}"
[[ "$RESP" = /* ]] || RESP="$REPO/$RESP"
SPLIT="${2:-test}"

# Stage judge to node-local /dev/shm (same Lustre-mmap reason as inference).
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
STAGE="/dev/shm/$USER/$(basename "$JUDGE_SRC")"
trap 'rm -rf "$STAGE"' EXIT
echo "eval: resp=$RESP split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging judge -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$JUDGE_SRC"/. "$STAGE"/

python track-frame/baseline/src/evaluate.py --split "$SPLIT" \
    --responses "$RESP/responses.json" --out "$RESP/eval" --judge-model "$STAGE"
echo "done. -> $RESP/eval"
