#!/bin/bash
#SBATCH --job-name=proc-lora-infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-procedure/lora-finetune/logs/infer-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-procedure/lora-finetune/logs/infer-%j.log

# Usage: sbatch infer.slurm <adapter_dir> [split] [extra runner args...]
#   sbatch infer.slurm track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct/checkpoint-1000 test
# Reuses the BASELINE runner with --adapter, at the TRAINING sampling settings
# (96 frames @ max_pixels 131072 — mirrors src/configs/qwen3_vl_4b.yaml, which
# is reduced vs the baseline defaults to fit training memory).
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"

ADAPTER="${1:?usage: sbatch infer.slurm <adapter_dir> [split] [extra args]}"
[[ "$ADAPTER" = /* ]] || ADAPTER="$REPO/$ADAPTER"
SPLIT="${2:-test}"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
STAGE="/dev/shm/$USER/$(basename "$MODEL_SRC")"
trap 'rm -rf "$STAGE" "$TMPDIR"' EXIT

echo "infer: track=procedure adapter=$ADAPTER split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging base -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$MODEL_SRC"/. "$STAGE"/

TAG="$(basename "$ADAPTER")"
OUT="$REPO/track-procedure/lora-finetune/logs/resp_${SPLIT}_${TAG}"
python track-procedure/baseline/src/runner.py --split "$SPLIT" --model "$STAGE" \
    --model-name "Qwen3-VL-4B-Instruct-lora-$TAG" --adapter "$ADAPTER" --out "$OUT" \
    --max-frames 96 --max-pixels 131072 "${@:3}"
echo "done. -> $OUT"
