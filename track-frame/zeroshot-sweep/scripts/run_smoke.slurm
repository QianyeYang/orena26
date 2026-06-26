#!/bin/bash
#SBATCH --job-name=frame-sweep-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/zeroshot-sweep/logs/smoke-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/zeroshot-sweep/logs/smoke-%j.log

# Usage: sbatch run_smoke.slurm <model-local-name> [N=8] [extra runner args...]
# Tiny --limit run straight from os-models/ (no staging) to validate a new VLM.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Pin caches + TMPDIR to /dev/shm; some nodes have an unwritable /local/user/<uid> $TMPDIR
# that breaks torch's import-time cache_dir(). Cheap on tmpfs.
export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

NAME="${1:?usage: sbatch run_smoke.slurm <model-local-name> [N] [extra args]}"
N="${2:-8}"
echo "smoke: model=$NAME N=$N host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

OUT="$REPO/track-frame/zeroshot-sweep/logs/smoke/$NAME"
python track-frame/zeroshot-sweep/src/runner.py \
    --split test --model "$REPO/os-models/$NAME" --model-name "$NAME" \
    --out "$OUT" --limit "$N" "${@:3}"
echo "=== smoke output dir: $OUT ==="
ls -la "$OUT"
