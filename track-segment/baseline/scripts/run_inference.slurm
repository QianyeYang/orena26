#!/bin/bash
#SBATCH --job-name=seg-base-infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/infer-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-segment/baseline/logs/infer-%j.log

# Usage: sbatch run_inference.slurm <model-local-name> [split] [extra runner args...]
# Stages the model to node-local /dev/shm (avoids Lustre mmap contention).
# Requeue-safe: the runner resumes from predictions.jsonl in the same --out.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Pin caches + TMPDIR to /dev/shm; some nodes have an unwritable /local/user/<uid>
# $TMPDIR that breaks torch's import-time cache_dir(). STAGED holds the temp model
# copy (if any); one trap cleans both. Never put $MODEL_SRC in STAGED.
export TMPDIR="/dev/shm/$USER/tmp-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton" XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR"
STAGED=""
cleanup() { rm -rf "$TMPDIR" ${STAGED:+"$STAGED"}; }
trap cleanup EXIT

NAME="${1:?usage: sbatch run_inference.slurm <model-local-name> [split] [extra args]}"
SPLIT="${2:-test}"
MODEL_SRC="$REPO/os-models/$NAME"

# Set STAGE_MODEL=0 for >=30B models (a big copy in tmpfs blows --mem).
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

OUT="$REPO/track-segment/baseline/logs/resp_${SPLIT}/$NAME"
python track-segment/baseline/src/runner.py \
    --split "$SPLIT" --model "$MODEL_PATH" --model-name "$NAME" --out "$OUT" "${@:3}"
