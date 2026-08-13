#!/bin/bash
#SBATCH --job-name=frame-count-head
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/counting-head/logs/head-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/counting-head/logs/head-%j.log

# Extract per-question hidden states from a frozen VLM, then fit the count head
# on them. Both stages in one job so the features and the head that consumes
# them can never drift apart.
#
# Usage: sbatch counting_head.slurm <model-dir> <label> [--adapter DIR] [extra features.py args]
#   sbatch counting_head.slurm os-models/Qwen3-VL-4B-Instruct 4b-ep30 \
#          --adapter track-frame/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/final
#   sbatch counting_head.slurm os-models/Qwen3-VL-4B-Instruct smoke --limit 60
set -euo pipefail
REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-frame/counting-head"
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

MODEL_REL="${1:?model dir (repo-relative or absolute) is required}"
LABEL="${2:?run label is required}"
shift 2

MODEL="$MODEL_REL"
[[ "$MODEL" != /* ]] && MODEL="$REPO/$MODEL"
[[ -d "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }

OUT="$TRACKDIR/logs/$LABEL"
mkdir -p "$OUT"

echo "count-head: model=$MODEL label=$LABEL host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python "$TRACKDIR/src/features.py" \
    --model "$MODEL" --model-name "$LABEL" --out "$OUT/features" "$@"

python "$TRACKDIR/src/train_head.py" \
    --features "$OUT/features" --out "$OUT/head"

echo "done -> $OUT/head/results.json"
