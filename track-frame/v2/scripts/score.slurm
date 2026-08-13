#!/bin/bash
#SBATCH --job-name=frame-v2-score
#SBATCH --partition=h200
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=100G --time=4:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/score-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/score-%j.log
set -euo pipefail
source /users/engs2732/miniconda3/etc/profile.d/conda.sh; conda activate orena
cd /datasets/engs2732/orena
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python track-frame/v2/src/score.py --run "$1"
