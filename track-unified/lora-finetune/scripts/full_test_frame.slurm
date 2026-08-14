#!/bin/bash
# Full official FRAME test for three epochs of the unified 3-track LoRA.
#
#   sbatch track-unified/lora-finetune/scripts/full_test_frame.slurm
#
# Array task i runs epoch EPOCHS[i] over both datasets.  FRAME is single-frame
# and takes ~16 min for all 6,252 questions, so splitting the datasets across
# tasks buys nothing.
#
# Pinned to b200 because the FRAME specialist baseline this is compared against
# was measured on b200: greedy bf16 decoding is not bit-identical across GPU
# families, and the epoch deltas under test are ~0.01.
#SBATCH --job-name=uni-frame-full
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/track-unified/lora-finetune/logs/full-test/frame-%A_%a.log
#SBATCH --error=/datasets/engs2732/orena/track-unified/lora-finetune/logs/full-test/frame-%A_%a.log

set -euo pipefail

REPO=/datasets/engs2732/orena
RUN="$REPO/track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"

EPOCHS=(4 8 12)
CKPTS=(8592 17184 25776)
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK < 0 || TASK >= ${#EPOCHS[@]} )); then
    echo "invalid array task index: $TASK" >&2
    exit 2
fi
EPOCH="${EPOCHS[$TASK]}"
CKPT="${CKPTS[$TASK]}"
ADAPTER="$RUN/checkpoint-$CKPT"
OUT_ROOT="$RUN/full_test/epoch_$EPOCH/frame"
MODEL_NAME="Qwen3-VL-4B-Instruct-lora-unified-epoch$EPOCH-frame-full-test"

source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "adapter not found: $ADAPTER" >&2; exit 2; }
[[ -d "$MODEL_SRC" ]] || { echo "base model not found: $MODEL_SRC" >&2; exit 2; }
[[ -d "$JUDGE_SRC" ]] || { echo "judge model not found: $JUDGE_SRC" >&2; exit 2; }
ADAPTER_SHA="$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')"

JOB_TAG="${SLURM_ARRAY_JOB_ID}_${TASK}"
JOB_SCRATCH="/dev/shm/$USER/orena-uni-frame-$JOB_TAG"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$JOB_SCRATCH/tmp" "$MODEL_STAGE" "$JUDGE_STAGE" "$OUT_ROOT"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "track=frame epoch=$EPOCH checkpoint=$CKPT adapter_sha256=$ADAPTER_SHA"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} out=$OUT_ROOT"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "staging base  -> $MODEL_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$MODEL_SRC"/. "$MODEL_STAGE"/
echo "staging judge -> $JUDGE_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$JUDGE_SRC"/. "$JUDGE_STAGE"/

DATASETS=(heico lapchole)
EXPECTED_ROWS=(4000 2252)
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    ROWS="${EXPECTED_ROWS[$i]}"
    OUT="$OUT_ROOT/$DATASET"
    mkdir -p "$OUT"

    echo "[epoch $EPOCH][$DATASET] inference start (expected_rows=$ROWS)"
    python "$REPO/track-frame/lora-finetune/src/runner.py" \
        --track frame \
        --split test \
        --dataset "$DATASET" \
        --base "$MODEL_STAGE" \
        --adapter "$ADAPTER" \
        --model-name "$MODEL_NAME" \
        --out "$OUT" \
        --max-new-tokens 64 \
        --max-pixels 602112 \
        --dtype bfloat16

    echo "[epoch $EPOCH][$DATASET] judge evaluation start"
    python "$REPO/track-frame/baseline/src/evaluate.py" \
        --track frame \
        --split test \
        --dataset "$DATASET" \
        --responses "$OUT/responses.json" \
        --out "$OUT/eval" \
        --judge-model "$JUDGE_STAGE"

    python "$REPO/track-unified/lora-finetune/src/validate_full_test.py" \
        --track frame --dataset "$DATASET" --out "$OUT" --expected-rows "$ROWS"

    touch "$OUT/COMPLETE"
done

echo "full FRAME test complete for epoch $EPOCH -> $OUT_ROOT"
