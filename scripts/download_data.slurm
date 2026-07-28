#!/bin/bash
#SBATCH --job-name=orena-download
#SBATCH --partition=allgpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download-data-%j.log
#SBATCH --error=logs/download-data-%j.log

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p logs data/.cache/huggingface

# No credential is written to the Slurm log. The Python downloader prefers the
# HF_KEY assignment in .bashrc and uses this gitignored file as its fallback.
set -a
if [[ -f "$REPO/.env" ]]; then
    source "$REPO/.env"
fi
set +a

# Keep caches off the 50 GB home filesystem.
export HF_HOME="$REPO/data/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate orena

if (( $# > 0 )); then
    download_args=("$@")
else
    download_args=(--datasets lapchole)
fi

python -u "$REPO/scripts/download_data.py" "${download_args[@]}"
