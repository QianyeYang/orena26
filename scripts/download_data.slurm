#!/bin/bash
#SBATCH --job-name=orena-download
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/data/download-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/data/download-%j.log

REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena

# HF_TOKEN is loaded from the gitignored .env (never hardcode secrets here).
# See .env.example. Create .env with: HF_TOKEN=hf_xxx  (chmod 600 .env)
set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set — create $REPO/.env from .env.example}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena

python /lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/scripts/download_data.py
