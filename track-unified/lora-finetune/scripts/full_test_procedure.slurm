#!/bin/bash
# Full official PROCEDURE test for three epochs of the unified 3-track LoRA.
#
#   sbatch track-unified/lora-finetune/scripts/full_test_procedure.slurm
#
# Six array tasks = 3 epochs x 2 datasets, one GPU each, so all six can run
# concurrently.  PROCEDURE feeds 96 frames per question; the specialist took
# ~3h36m for both datasets back to back, so ~2h20m / ~1h20m split out.
#
# Pinned to a100 because the PROCEDURE specialist baseline was measured on A100:
# greedy bf16 decoding is not bit-identical across GPU families, and the epoch
# deltas under test are ~0.01.
#SBATCH --job-name=uni-procedure-full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/track-unified/lora-finetune/logs/full-test/procedure-%A_%a.log
#SBATCH --error=/datasets/engs2732/orena/track-unified/lora-finetune/logs/full-test/procedure-%A_%a.log

set -euo pipefail

REPO=/datasets/engs2732/orena
RUN="$REPO/track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"

EPOCHS=(4 8 12)
CKPTS=(8592 17184 25776)
DATASETS=(heico lapchole)
EXPECTED_ROWS=(2000 1127)
STRIDES=(250 300)

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK < 0 || TASK >= ${#EPOCHS[@]} * ${#DATASETS[@]} )); then
    echo "invalid array task index: $TASK" >&2
    exit 2
fi
EPOCH_IDX=$(( TASK / ${#DATASETS[@]} ))
DATA_IDX=$(( TASK % ${#DATASETS[@]} ))
EPOCH="${EPOCHS[$EPOCH_IDX]}"
CKPT="${CKPTS[$EPOCH_IDX]}"
DATASET="${DATASETS[$DATA_IDX]}"
ROWS="${EXPECTED_ROWS[$DATA_IDX]}"
STRIDE="${STRIDES[$DATA_IDX]}"
ADAPTER="$RUN/checkpoint-$CKPT"
OUT="$RUN/full_test/epoch_$EPOCH/procedure/$DATASET"
MODEL_NAME="Qwen3-VL-4B-Instruct-lora-unified-epoch$EPOCH-procedure-full-test"

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
JOB_SCRATCH="/dev/shm/$USER/orena-uni-procedure-$JOB_TAG"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$JOB_SCRATCH/tmp" "$MODEL_STAGE" "$JUDGE_STAGE" "$OUT"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "track=procedure epoch=$EPOCH checkpoint=$CKPT dataset=$DATASET rows=$ROWS stride=$STRIDE"
echo "adapter_sha256=$ADAPTER_SHA"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} out=$OUT"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "staging base  -> $MODEL_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$MODEL_SRC"/. "$MODEL_STAGE"/
echo "staging judge -> $JUDGE_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$JUDGE_SRC"/. "$JUDGE_STAGE"/

echo "[epoch $EPOCH][$DATASET] full inference start"
python "$REPO/track-procedure/baseline/src/runner.py" \
    --dataset "$DATASET" \
    --track procedure \
    --split test \
    --model "$MODEL_STAGE" \
    --model-name "$MODEL_NAME" \
    --adapter "$ADAPTER" \
    --out "$OUT" \
    --stride "$STRIDE" \
    --max-frames 96 \
    --max-pixels 131072 \
    --max-new-tokens 64 \
    --dtype bfloat16

echo "[epoch $EPOCH][$DATASET] full judge evaluation start"
python "$REPO/track-frame/baseline/src/evaluate.py" \
    --track procedure \
    --split test \
    --dataset "$DATASET" \
    --responses "$OUT/responses.json" \
    --out "$OUT/eval" \
    --judge-model "$JUDGE_STAGE"

python "$REPO/track-unified/lora-finetune/src/validate_full_test.py" \
    --track procedure --dataset "$DATASET" --out "$OUT" --expected-rows "$ROWS"

touch "$OUT/COMPLETE"
echo "full PROCEDURE $DATASET test complete for epoch $EPOCH: $ROWS/$ROWS -> $OUT"
