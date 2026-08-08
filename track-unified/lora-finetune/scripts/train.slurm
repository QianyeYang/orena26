#!/bin/bash
#SBATCH --job-name=unified-qwen4b-both
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/datasets/engs2732/orena/track-unified/lora-finetune/logs/train-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-unified/lora-finetune/logs/train-%j.log

# Single-GPU FALLBACK (b200) — the primary launch is train_multigpu.slurm on
# 4x A100. The config carries the DDP grad_accum (4), so single-GPU runs MUST
# pass --grad-accum 16 to keep effective batch 16.
# Usage: sbatch train.slurm [config] [extra train.py args...]
#   sbatch train.slurm src/configs/qwen3_vl_4b_unified.yaml --grad-accum 16
# Per-epoch per-track seeded-subset inference + judge scoring run INSIDE this
# job. Resumable: re-submitting the same config auto-continues from the latest
# checkpoint (optimizer/scheduler/RNG restored), so the 12-epoch target can
# span multiple 2-day jobs. Stages base + judge to an isolated /dev/shm path.
set -euo pipefail
REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-unified/lora-finetune"
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
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

CFG="${1:-$TRACKDIR/src/configs/qwen3_vl_4b_unified.yaml}"
# accept absolute, REPO-relative, or lora-finetune-relative config paths
if [[ "$CFG" != /* ]]; then
  if [[ -f "$REPO/$CFG" ]]; then CFG="$REPO/$CFG"
  elif [[ -f "$TRACKDIR/$CFG" ]]; then CFG="$TRACKDIR/$CFG"; fi
fi
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 2; }
# base_path is read from the config so the script stays model-agnostic.
BASE_REL=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["base_path"])' "$CFG")
MODEL_SRC="$REPO/$BASE_REL"
STAGE="$JOB_SCRATCH/model"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

echo "train: track=unified cfg=$CFG base=$MODEL_SRC host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "staging base  -> $STAGE";       mkdir -p "$STAGE";       time cp -ru "$MODEL_SRC"/. "$STAGE"/
echo "staging judge -> $JUDGE_STAGE"; mkdir -p "$JUDGE_STAGE"; time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/

RUN_NAME="${RUN_NAME_OVERRIDE:-$(python -c 'import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get("run_name", pathlib.Path(c["base_path"]).name))' "$CFG")}"
OUT="$TRACKDIR/logs/$RUN_NAME"
# auto-resume if checkpoints already exist (e.g. a previous 2-day job walled out)
RESUME=""
if compgen -G "$OUT/checkpoint-*" >/dev/null 2>&1; then
  RESUME="--resume"; echo "found checkpoints in $OUT -> resuming"
fi

python "$TRACKDIR/src/train.py" --config "$CFG" --base "$STAGE" \
    --out "$OUT" --judge-model "$JUDGE_STAGE" $RESUME "${@:2}"
echo "done. adapters + per-epoch per-track evals under: $OUT"
