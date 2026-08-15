#!/bin/bash
#SBATCH --job-name=frame-v2-ddp
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/train-ddp-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/train-ddp-%j.log

# Multi-GPU sibling of train.slurm: same trainer, run under torchrun DDP.
#
# Usage: sbatch train_ddp.slurm [config] [extra train.py args...]
#   sbatch train_ddp.slurm                                          # 3.8-27B, 30 epochs
#   sbatch train_ddp.slurm src/configs/qwen3_8_27b_v1prompt.yaml --limit 32 --epochs 1
#
# DDP is plain data parallelism — each rank holds the whole 52 GB bf16 model, so
# this needs GPUs that fit the model on their own (a B200 is 183 GB). It buys
# throughput, not capacity. The config's grad_accum must be divided by the GPU
# count to hold the effective batch fixed; qwen3_8_27b_v1prompt.yaml already is.
#
# Evaluation is NOT run here: adapters are saved per epoch and scored afterwards
# by benchmark.slurm under the leaderboard-shaped metric.
#
# Resumable: re-submitting the same config continues from the latest epoch
# checkpoint (optimizer/scheduler/RNG restored), which is also what --requeue
# relies on if the 2-day wall is hit.
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

CFG="${1:-$TRACKDIR/src/configs/qwen3_8_27b_v1prompt.yaml}"
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

NGPU=$(nvidia-smi --list-gpus | wc -l)
[[ "$NGPU" -ge 1 ]] || { echo "no GPUs visible" >&2; exit 2; }

echo "train-ddp: cfg=$CFG base=$BASE_REL host=$(hostname) ngpu=$NGPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'

# Effective batch must stay at 16 to match the single-GPU Qwen3.6 run; fail loudly
# rather than silently training a different recipe.
python - "$CFG" "$NGPU" <<'PY'
import sys, yaml
cfg, ngpu = yaml.safe_load(open(sys.argv[1]))["train"], int(sys.argv[2])
eff = cfg["per_device_batch_size"] * cfg["grad_accum"] * ngpu
print(f"effective batch = {cfg['per_device_batch_size']} x {cfg['grad_accum']} x {ngpu} = {eff}")
if eff != 16:
    raise SystemExit(f"effective batch {eff} != 16; adjust grad_accum for {ngpu} GPUs")
PY

# 52 GB of weights is too much to stage into /dev/shm alongside the run, so the
# base is read from the shared filesystem; only caches are node-local.
RESUME=""
if compgen -G "$OUT/checkpoint-*" >/dev/null 2>&1; then
  RESUME="--resume"; echo "found checkpoints in $OUT -> resuming"
fi

# The b200 node is shared, so pick a per-job rendezvous port instead of
# torchrun's fixed default, which would collide with a concurrent job.
RDZV_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

torchrun \
  --nnodes=1 --nproc-per-node="$NGPU" \
  --rdzv-backend=c10d --rdzv-endpoint="localhost:$RDZV_PORT" \
  "$TRACKDIR/src/train.py" --config "$CFG" --out "$OUT" $RESUME "$@"
echo "done. per-epoch adapters under: $OUT"
