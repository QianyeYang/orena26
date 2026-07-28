#!/bin/bash
#SBATCH --job-name=frame-bbox-count
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/counting-bbox-prompt/logs/slurm-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/counting-bbox-prompt/logs/slurm-%j.log

# Re-run only FRAME counting rows with the one-shot bounding-box JSON prompt,
# merge them into the completed epoch-30 direct-prompt run, and (for a no-limit
# run) evaluate each full test set.
#
# Usage:
#   sbatch run_experiment.slurm <label> <adapter-dir> [counting-limit]
set -euo pipefail

REPO=/datasets/engs2732/orena
METHOD_DIR="$REPO/track-frame/counting-bbox-prompt"
LORA_DIR="$REPO/track-frame/lora-finetune"
BASELINE_ROOT="$LORA_DIR/logs/full-test-comparison/new-epoch30"

source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

LABEL="${1:?usage: sbatch run_experiment.slurm <label> <adapter-dir> [counting-limit]}"
[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "label may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}

ADAPTER="${2:?usage: sbatch run_experiment.slurm <label> <adapter-dir> [counting-limit]}"
[[ "$ADAPTER" = /* ]] || ADAPTER="$REPO/$ADAPTER"
[[ -d "$ADAPTER" ]] || { echo "adapter not found: $ADAPTER" >&2; exit 2; }

LIMIT="${3:-}"
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "counting-limit must be a positive integer, got: $LIMIT" >&2
  exit 2
fi

OUT_ROOT="$METHOD_DIR/logs/$LABEL"
if [[ -e "$OUT_ROOT" ]]; then
  echo "refusing to overwrite existing experiment directory: $OUT_ROOT" >&2
  exit 2
fi

JOB_TAG="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
JOB_SCRATCH="/dev/shm/$USER/orena-frame-bbox-$JOB_TAG"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"

trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$METHOD_DIR/logs" "$OUT_ROOT" "$JOB_SCRATCH/tmp" "$MODEL_STAGE"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "label=$LABEL adapter=$ADAPTER limit=${LIMIT:-full} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "staging base -> $MODEL_STAGE"
time cp -ru "$MODEL_SRC"/. "$MODEL_STAGE"/

if [[ -z "$LIMIT" ]]; then
  mkdir -p "$JUDGE_STAGE"
  echo "staging judge -> $JUDGE_STAGE"
  time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/
fi

for DATASET in heico lapchole; do
  DATASET_OUT="$OUT_ROOT/$DATASET"
  COUNTING_OUT="$DATASET_OUT/counting-only"
  RUNNER_ARGS=(
    --track frame
    --split test
    --dataset "$DATASET"
    --base "$MODEL_STAGE"
    --adapter "$ADAPTER"
    --model-name "$LABEL-counting-only"
    --prompt-strategy bbox-json
    --question-filter counting
    --warmup
    --out "$COUNTING_OUT"
    --max-new-tokens 512
    --max-pixels 602112
  )
  if [[ -n "$LIMIT" ]]; then
    RUNNER_ARGS+=(--limit "$LIMIT")
  fi

  echo "[$DATASET] structured counting inference start"
  python "$LORA_DIR/src/runner.py" "${RUNNER_ARGS[@]}"

  EXPECTED_ARGS=()
  if [[ -z "$LIMIT" ]]; then
    if [[ "$DATASET" == heico ]]; then
      EXPECTED_ARGS=(--expected-replacements 1326)
    else
      EXPECTED_ARGS=(--expected-replacements 768)
    fi
  fi

  python "$METHOD_DIR/src/merge_outputs.py" \
    --baseline "$BASELINE_ROOT/$DATASET" \
    --counting "$COUNTING_OUT" \
    --out "$DATASET_OUT" \
    --model-name "$LABEL" \
    "${EXPECTED_ARGS[@]}"

  if [[ -z "$LIMIT" ]]; then
    echo "[$DATASET] full merged evaluation start"
    python "$REPO/track-frame/baseline/src/evaluate.py" \
      --track frame \
      --split test \
      --dataset "$DATASET" \
      --responses "$DATASET_OUT/responses.json" \
      --out "$DATASET_OUT/eval" \
      --judge-model "$JUDGE_STAGE"
  fi
done

if [[ -z "$LIMIT" ]]; then
  touch "$OUT_ROOT/COMPLETE"
  echo "complete -> $OUT_ROOT"
else
  touch "$OUT_ROOT/SMOKE_COMPLETE"
  echo "limited inference complete; full evaluation intentionally skipped -> $OUT_ROOT"
fi
