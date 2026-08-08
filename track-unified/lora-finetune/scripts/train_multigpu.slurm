#!/bin/bash
#SBATCH --job-name=unified-qwen4b-ddp
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/datasets/engs2732/orena/track-unified/lora-finetune/logs/train-ddp-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-unified/lora-finetune/logs/train-ddp-%j.log

# Usage:
#   sbatch train_multigpu.slurm [config] [extra train.py args...]
#
# Default: 4x A100, pdbs 1 x grad_accum 4 (from the config) = effective 16 —
# the geometry the segment both-official production run validated on this
# node (64 frames @ 262144 px at pdbs 1 fits an 80 GB A100; procedure items
# are strictly smaller). Keep effective batch 16 when changing NUM_GPUS:
#   2 GPUs: --grad-accum 8    8 GPUs: --grad-accum 2
#
#   RUN_NAME_OVERRIDE=_smoke_a100_unified NUM_GPUS=4 \
#     sbatch train_multigpu.slurm src/configs/qwen3_vl_4b_unified.yaml \
#       --limit-per-track 8 --epochs 1 --eval-limit 6 --save-steps 2   # smoke
set -euo pipefail
REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-unified/lora-finetune"
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_GPUS="${NUM_GPUS:-4}"
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
}

JOB_TAG="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
JOB_SCRATCH="/dev/shm/$USER/orena-$JOB_TAG"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME/torch/kernels"

CFG="${1:-$TRACKDIR/src/configs/qwen3_vl_4b_unified.yaml}"
if [[ "$CFG" != /* ]]; then
  if [[ -f "$REPO/$CFG" ]]; then
    CFG="$REPO/$CFG"
  elif [[ -f "$TRACKDIR/$CFG" ]]; then
    CFG="$TRACKDIR/$CFG"
  fi
fi
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 2; }

BASE_REL=$(python -c \
  'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["base_path"])' "$CFG")
MODEL_SRC="$REPO/$BASE_REL"
STAGE="$JOB_SCRATCH/model"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

echo "train-ddp: track=unified cfg=$CFG base=$MODEL_SRC host=$(hostname) nproc=$NUM_GPUS visible=${CUDA_VISIBLE_DEVICES:-?}"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "staging base  -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$MODEL_SRC"/. "$STAGE"/
echo "staging judge -> $JUDGE_STAGE"
mkdir -p "$JUDGE_STAGE"
time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/

RUN_NAME="${RUN_NAME_OVERRIDE:-$(python -c \
  'import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get("run_name", pathlib.Path(c["base_path"]).name))' \
  "$CFG")}"
OUT="$TRACKDIR/logs/$RUN_NAME"
RESUME=()
if compgen -G "$OUT/checkpoint-*" >/dev/null 2>&1; then
  RESUME=(--resume)
  echo "found checkpoints in $OUT -> resuming"
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  "$TRACKDIR/src/train.py" --config "$CFG" --base "$STAGE" \
  --out "$OUT" --judge-model "$JUDGE_STAGE" "${RESUME[@]}" "${@:2}"
echo "done. adapters + per-epoch per-track evals under: $OUT"
