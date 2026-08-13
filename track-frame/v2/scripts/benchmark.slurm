#!/bin/bash
#SBATCH --job-name=frame-v2-bench
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/bench-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/bench-%j.log

# Stage 1: run one model over the full FRAME test set, then score it with the
# leaderboard-shaped four-bucket metric.
#
# Usage: sbatch benchmark.slurm <model-dir> <label> [extra benchmark.py args...]
#   sbatch benchmark.slurm os-models/Qwen3.6-27B qwen3.6-27b-zeroshot-v2
#   sbatch benchmark.slurm os-models/Qwen3-VL-4B-Instruct qwen3vl-4b-lora-v1 \
#          --adapter track-frame/lora-finetune/logs/.../final --prompt-strategy direct
#   sbatch benchmark.slurm os-models/Qwen3.6-27B smoke --limit 40
#
# The h200 partition is eight single-GPU nodes (141 GB each), so a 27B runs in
# bf16 without sharding and without touching the a100 box.
set -euo pipefail
REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-frame/v2"
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
# Overridable: the finegrained-fp8 kernel is fetched from the Hub, and offline
# resolution of it is still unsolved (a submission-image problem, not a
# benchmarking one). Pass HF_HUB_OFFLINE=0 to let an FP8 run reach the Hub.
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

echo "bench: model=$MODEL label=$LABEL host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python "$TRACKDIR/src/benchmark.py" \
    --model "$MODEL" --model-name "$LABEL" --out "$OUT" "$@"

python "$TRACKDIR/src/score.py" --run "$OUT"
echo "done -> $OUT/eval/leaderboard.md"
