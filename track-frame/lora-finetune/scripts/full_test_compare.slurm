#!/bin/bash
#SBATCH --job-name=frame-full-test
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/lora-finetune/logs/full-test-comparison/slurm-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/lora-finetune/logs/full-test-comparison/slurm-%j.log

# Run one adapter over every official FRAME test question in both datasets,
# then score each dataset with the same local Qwen3.5-4B FOCUS judge.
#
# Usage:
#   sbatch full_test_compare.slurm <label> <adapter-dir> [limit]
#
# Submit the old and new adapters as separate jobs so they run concurrently on
# identical B200 hardware.  `limit` is only for an end-to-end smoke test.
set -euo pipefail

REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-frame/lora-finetune"
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

LABEL="${1:?usage: sbatch full_test_compare.slurm <label> <adapter-dir> [limit]}"
[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "label may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}

ADAPTER="${2:?usage: sbatch full_test_compare.slurm <label> <adapter-dir> [limit]}"
[[ "$ADAPTER" = /* ]] || ADAPTER="$REPO/$ADAPTER"
[[ -d "$ADAPTER" ]] || { echo "adapter not found: $ADAPTER" >&2; exit 2; }

LIMIT="${3:-}"
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "limit must be a positive integer, got: $LIMIT" >&2
  exit 2
fi

JOB_TAG="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
JOB_SCRATCH="/dev/shm/$USER/orena-frame-full-$JOB_TAG"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"
OUT_ROOT="$TRACKDIR/logs/full-test-comparison/$LABEL"

trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$JOB_SCRATCH/tmp" "$MODEL_STAGE" "$JUDGE_STAGE" "$OUT_ROOT"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "label=$LABEL adapter=$ADAPTER host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "staging base  -> $MODEL_STAGE"
time cp -ru "$MODEL_SRC"/. "$MODEL_STAGE"/
echo "staging judge -> $JUDGE_STAGE"
time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/

for DATASET in heico lapchole; do
  OUT="$OUT_ROOT/$DATASET"
  RUNNER_ARGS=(
    --track frame
    --split test
    --dataset "$DATASET"
    --base "$MODEL_STAGE"
    --adapter "$ADAPTER"
    --model-name "$LABEL"
    --out "$OUT"
    --max-new-tokens 64
    --max-pixels 602112
  )
  EVAL_ARGS=(
    --track frame
    --split test
    --dataset "$DATASET"
    --responses "$OUT/responses.json"
    --out "$OUT/eval"
    --judge-model "$JUDGE_STAGE"
  )
  if [[ -n "$LIMIT" ]]; then
    RUNNER_ARGS+=(--limit "$LIMIT")
    EVAL_ARGS+=(--limit "$LIMIT")
  fi

  echo "[$DATASET] inference start"
  python "$TRACKDIR/src/runner.py" "${RUNNER_ARGS[@]}"
  echo "[$DATASET] evaluation start"
  python "$REPO/track-frame/baseline/src/evaluate.py" "${EVAL_ARGS[@]}"
done

touch "$OUT_ROOT/COMPLETE"
echo "complete -> $OUT_ROOT"
