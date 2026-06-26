#!/bin/bash
#SBATCH --job-name=orena-extract
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/logs/extract-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/logs/extract-%j.log

# Frame extraction for the FRAME track (train+test). CPU-only (PyAV).
# One worker process per video; each pinned to a single decode/math thread.
# Override with: sbatch extract_frames.slurm <track> "<splits>" <workers>

set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

TRACK="${1:-frame}"
SPLITS="${2:-train test}"
WORKERS="${3:-30}"

echo "extract: track=$TRACK splits=$SPLITS workers=$WORKERS  host=$(hostname)"
python -m src.frames --track "$TRACK" --splits $SPLITS --workers "$WORKERS"
echo "done."
