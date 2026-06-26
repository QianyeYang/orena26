#!/bin/bash
#SBATCH --job-name=orena-dl-models
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/logs/download-models-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/logs/download-models-%j.log

# Download VLMs/LLMs into os-models/ (compute node: has internet + memory headroom;
# login node OOM-kills large parallel downloads). Args forwarded to the script.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

echo "downloading on $(hostname)"
python scripts/download_models.py "$@"
echo "done."
