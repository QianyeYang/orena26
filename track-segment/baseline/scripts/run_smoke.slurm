#!/bin/bash
#SBATCH --job-name=seg-base-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/smoke-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/smoke-%j.log

# Usage: sbatch run_smoke.slurm <model-local-name> [N=12] [extra runner args...]
# Tiny --limit run straight from os-models/ (no staging) to validate the pipeline.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

NAME="${1:?usage: sbatch run_smoke.slurm <model-local-name> [N] [extra args]}"
N="${2:-12}"
echo "smoke: model=$NAME N=$N host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

OUT="$REPO/track-segment/baseline/logs/smoke/$NAME"
rm -rf "$OUT"   # smokes are always fresh (never resumed)
python track-segment/baseline/src/runner.py \
    --split test --model "$REPO/os-models/$NAME" --model-name "$NAME" \
    --out "$OUT" --limit "$N" "${@:3}"
echo "=== smoke output dir: $OUT ==="
ls -la "$OUT"
