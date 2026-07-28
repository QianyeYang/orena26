#!/bin/bash
#SBATCH --job-name=seg-lora-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/lora-finetune/logs/eval-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/lora-finetune/logs/eval-%j.log

# Usage: sbatch eval.slurm <resp_dir> [split]   (resp_dir holds responses.json)
#   sbatch eval.slurm track-segment/lora-finetune/logs/resp_test_checkpoint-3000 test
# Reuses the shared evaluator (focus.Evaluator, judge = Qwen3.5-4B).
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"

RESP="${1:?usage: sbatch eval.slurm <resp_dir> [split]}"
[[ "$RESP" = /* ]] || RESP="$REPO/$RESP"
SPLIT="${2:-test}"

JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
STAGE="/dev/shm/$USER/$(basename "$JUDGE_SRC")"
trap 'rm -rf "$STAGE" "$TMPDIR"' EXIT
echo "eval: track=segment resp=$RESP split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging judge -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$JUDGE_SRC"/. "$STAGE"/

python track-frame/baseline/src/evaluate.py --track segment --split "$SPLIT" \
    --responses "$RESP/responses.json" --out "$RESP/eval" --judge-model "$STAGE"
echo "done. -> $RESP/eval"
