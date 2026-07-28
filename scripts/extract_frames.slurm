#!/bin/bash
#SBATCH --job-name=orena-extract
#SBATCH --partition=allgpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/logs/extract-%x-%j.log
#SBATCH --error=/datasets/engs2732/orena/logs/extract-%x-%j.log

# Official-compatible sparse extraction for one dataset/track/sampling plan.
# Usage:
#   sbatch scripts/extract_frames.slurm DATASET TRACK STRIDE MAX_FRAMES [WORKERS]
# Examples:
#   sbatch scripts/extract_frames.slurm heico frame 1 1
#   sbatch scripts/extract_frames.slurm heico segment 25 64
#   sbatch scripts/extract_frames.slurm lapchole procedure 300 96

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/datasets/engs2732/orena}"
if [[ ! -f "$REPO/src/frame_track_extraction.py" ]]; then
    echo "invalid repository root: $REPO" >&2
    exit 2
fi
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The conda-forge Decord build is linked to FFmpeg 8 and cannot initialise its
# pixel-format filter. Prefer the self-contained PyPI wheel pulled by the
# official `pip install orena-focus` installation path.
DECORD_PYPI="$REPO/data/vendor/decord-pypi"
if [[ -f "$DECORD_PYPI/decord/__init__.py" ]]; then
    export PYTHONPATH="$DECORD_PYPI${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch scripts/extract_frames.slurm DATASET TRACK STRIDE MAX_FRAMES [WORKERS]" >&2
    exit 2
fi
DATASET="$1"
TRACK="${2:-frame}"
STRIDE="${3:-1}"
MAX_FRAMES="${4:-1}"
WORKERS="${5:-${SLURM_CPUS_PER_TASK:-32}}"

if [[ "$DATASET" != "heico" && "$DATASET" != "lapchole" ]]; then
    echo "dataset must be 'heico' or 'lapchole' (got: $DATASET)" >&2
    exit 2
fi
if [[ "$TRACK" != "frame" && "$TRACK" != "segment" && "$TRACK" != "procedure" ]]; then
    echo "track must be frame, segment, or procedure (got: $TRACK)" >&2
    exit 2
fi
if [[ ! "$STRIDE" =~ ^[1-9][0-9]*$ || ! "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
    echo "stride and max_frames must be positive integers" >&2
    exit 2
fi

echo "dataset=$DATASET track=$TRACK stride=$STRIDE max_frames=$MAX_FRAMES workers=$WORKERS host=$(hostname) job=${SLURM_JOB_ID:-none}"
python src/frame_track_extraction.py \
    --dataset "$DATASET" \
    --track "$TRACK" \
    --stride "$STRIDE" \
    --max-frames "$MAX_FRAMES" \
    --workers "$WORKERS"
echo "done dataset=$DATASET track=$TRACK"
