#!/bin/bash
#SBATCH --job-name=seg-base-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/eval-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/eval-%j.log

# Usage: sbatch run_eval.slurm <model-local-name> [split]
# Scores responses.json with the focus Evaluator (Qwen3.5-4B judge), reusing the
# track-parameterised frame-track evaluate.py unchanged.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"

NAME="${1:?usage: sbatch run_eval.slurm <model-local-name> [split]}"
SPLIT="${2:-test}"
OUT="$REPO/track-segment/baseline/logs/resp_${SPLIT}/$NAME"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
STAGE="/dev/shm/$USER/Qwen3.5-4B"
trap 'rm -rf "$STAGE" "$TMPDIR"' EXIT

echo "eval: track=segment model=$NAME split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging judge -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$JUDGE_SRC"/. "$STAGE"/

python track-frame/baseline/src/evaluate.py --track segment --split "$SPLIT" \
    --responses "$OUT/responses.json" --out "$OUT/eval" --judge-model "$STAGE"
