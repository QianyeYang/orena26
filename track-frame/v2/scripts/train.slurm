#!/bin/bash
#SBATCH --job-name=frame-v2-train
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=155G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/train-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/train-%j.log

# Usage: sbatch train.slurm [config] [extra train.py args...]
#   sbatch train.slurm                                        # 27B, 8 epochs
#   sbatch train.slurm src/configs/qwen3_6_27b.yaml --limit 32 --epochs 1   # smoke
#
# One H200 NVL (141 GB) holds a 27B in bf16 with gradient checkpointing, so no
# sharding and no multi-node rendezvous — the h200 partition is eight
# single-GPU nodes. Evaluation is NOT run here: adapters are saved per epoch and
# scored afterwards by benchmark.slurm under the leaderboard-shaped metric.
#
# Resumable: re-submitting the same config continues from the latest epoch
# checkpoint (optimizer/scheduler/RNG restored).
set -euo pipefail
REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-frame/v2"
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

JOB_TAG="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
JOB_SCRATCH="/dev/shm/$USER/orena-$JOB_TAG"
export TMPDIR="$JOB_SCRATCH/tmp"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

CFG="${1:-$TRACKDIR/src/configs/qwen3_6_27b.yaml}"
if [[ "$CFG" != /* ]]; then
  if   [[ -f "$REPO/$CFG"     ]]; then CFG="$REPO/$CFG"
  elif [[ -f "$TRACKDIR/$CFG" ]]; then CFG="$TRACKDIR/$CFG"; fi
fi
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 2; }
shift || true

BASE_REL=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["base_path"])' "$CFG")
RUN_NAME=$(python -c 'import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get("run_name", pathlib.Path(c["base_path"]).name))' "$CFG")
OUT="$TRACKDIR/logs/$RUN_NAME"
mkdir -p "$OUT"

echo "train: cfg=$CFG base=$BASE_REL host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'

# 54 GB of weights is too much to stage into /dev/shm alongside the run, so the
# base is read from the shared filesystem; only caches are node-local.
RESUME=""
if compgen -G "$OUT/checkpoint-*" >/dev/null 2>&1; then
  RESUME="--resume"; echo "found checkpoints in $OUT -> resuming"
fi

python "$TRACKDIR/src/train.py" --config "$CFG" --out "$OUT" $RESUME "$@"
echo "done. per-epoch adapters under: $OUT"
