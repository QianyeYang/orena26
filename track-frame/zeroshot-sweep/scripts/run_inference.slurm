#!/bin/bash
#SBATCH --job-name=frame-sweep-infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=02:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/zeroshot-sweep/logs/infer-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/zeroshot-sweep/logs/infer-%j.log

# Usage: sbatch run_inference.slurm <model-local-name> [split] [extra runner args...]
# Stages the model to node-local /dev/shm (avoids Lustre mmap contention).
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Some compute nodes have an unwritable node-local $TMPDIR (/local/user/<uid>), which
# breaks torch's import-time inductor/triton cache_dir(). Pin caches + TMPDIR to /dev/shm
# (tmpfs, reliably writable; tiny footprint vs --mem). STAGED holds the temp model copy
# (if any); a single trap cleans both. Never put $MODEL_SRC in STAGED.
export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"
STAGED=""
cleanup() { rm -rf "$TMPDIR" ${STAGED:+"$STAGED"}; }
trap cleanup EXIT

NAME="${1:?usage: sbatch run_inference.slurm <model-local-name> [split] [extra args]}"
SPLIT="${2:-test}"
MODEL_SRC="$REPO/os-models/$NAME"

# /dev/shm is tmpfs (RAM, counts against the cgroup): staging is great for small
# models (avoids Lustre mmap contention) but a 60GB+ model staged there + the
# load working set blows --mem. Set STAGE_MODEL=0 for big models to load direct.
STAGE_MODEL="${STAGE_MODEL:-1}"
if [ "$STAGE_MODEL" = "1" ]; then
    MODEL_PATH="/dev/shm/$USER/$NAME"
    STAGED="$MODEL_PATH"
    echo "infer: model=$NAME split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
    echo "staging model -> $MODEL_PATH"
    mkdir -p "$MODEL_PATH"
    time cp -ru "$MODEL_SRC"/. "$MODEL_PATH"/
else
    MODEL_PATH="$MODEL_SRC"
    echo "infer: model=$NAME split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} (no staging, direct load)"
fi

OUT="$REPO/track-frame/zeroshot-sweep/logs/resp_${SPLIT}/$NAME"
python track-frame/zeroshot-sweep/src/runner.py \
    --split "$SPLIT" --model "$MODEL_PATH" --model-name "$NAME" --out "$OUT" "${@:3}"
