#!/bin/bash
#SBATCH --job-name=frame-res-diag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/lora-finetune/logs/resolution-diagnostic-20260726/slurm-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/lora-finetune/logs/resolution-diagnostic-20260726/slurm-%j.log

# Compare the current 602,112-pixel cap with a 1,003,520-pixel native-input
# cap on the complete affected number/FO-class test subset plus controls.
set -euo pipefail

REPO=/datasets/engs2732/orena
TRACK_DIR="$REPO/track-frame/lora-finetune"
OUTPUT="$TRACK_DIR/logs/resolution-diagnostic-20260726"
BASE_SOURCE="$REPO/os-models/Qwen3-VL-4B-Instruct"
ADAPTER_SOURCE="$TRACK_DIR/logs/Qwen3-VL-4B-Instruct-both-official/final"
PREVIOUS_ROOT="$TRACK_DIR/logs/full-test-comparison/new-epoch30"
DEFINITIONS="$REPO/submissions/frame/qwen3-vl-4b-lora-both-official-epoch30-20260725/test/input/interface_1/FO_definitions.json"

source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export ORENA_BASE_SOURCE="$BASE_SOURCE"
export ORENA_ADAPTER_SOURCE="$ADAPTER_SOURCE"

mkdir -p "$OUTPUT"
JOB_SCRATCH="$(mktemp -d "/dev/shm/orena-frame-res-${USER}-${SLURM_JOB_ID}-XXXXXX")"
BASE_STAGE="$JOB_SCRATCH/base-model"
ADAPTER_STAGE="$JOB_SCRATCH/adapter"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

mkdir -p "$BASE_STAGE" "$ADAPTER_STAGE" "$JOB_SCRATCH/tmp"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/torchinductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

echo "job=$SLURM_JOB_ID host=$(hostname) visible_gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
scontrol show job "$SLURM_JOB_ID" > "$OUTPUT/slurm-job.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$OUTPUT/gpu-start.csv"

echo "staging base model -> $BASE_STAGE"
time cp -ru "$BASE_SOURCE"/. "$BASE_STAGE"/
echo "staging adapter -> $ADAPTER_STAGE"
time cp -ru "$ADAPTER_SOURCE"/. "$ADAPTER_STAGE"/

python "$TRACK_DIR/src/resolution_diagnostic.py" \
  --base "$BASE_STAGE" \
  --adapter "$ADAPTER_STAGE" \
  --definitions "$DEFINITIONS" \
  --previous-root "$PREVIOUS_ROOT" \
  --output "$OUTPUT" \
  --baseline-max-pixels 602112 \
  --high-max-pixels 1003520 \
  --control-per-stratum 50 \
  --batch-size 8 \
  --max-new-tokens 64 \
  --dtype bfloat16 \
  --seed 2732

nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$OUTPUT/gpu-end.csv"
echo "complete -> $OUTPUT"
