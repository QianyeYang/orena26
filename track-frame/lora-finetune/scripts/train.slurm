#!/bin/bash
#SBATCH --job-name=frame-lora-train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/train-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/lora-finetune/logs/train-%j.log

# Usage: sbatch train.slurm [config] [extra train.py args...]
#   sbatch train.slurm track-frame/lora-finetune/src/configs/qwen3_vl_8b.yaml         # full 30-epoch run
#   sbatch train.slurm <config> --limit 16 --epochs 1 --eval-limit 16                 # smoke
# Per-epoch test inference + judge scoring run INSIDE this job (no separate eval task).
# Resumable: re-submitting the same config auto-continues from the latest epoch
# checkpoint (optimizer/scheduler/RNG restored), so a 30-epoch target can span
# multiple 24h jobs. Stages base + judge to node-local /dev/shm (avoids Lustre
# mmap stalls).
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CFG="${1:-track-frame/lora-finetune/src/configs/qwen2_5_vl_7b.yaml}"
# accept absolute, REPO-relative, or lora-finetune-relative config paths
if [[ "$CFG" != /* ]]; then
  if [[ -f "$REPO/$CFG" ]]; then CFG="$REPO/$CFG"
  elif [[ -f "$REPO/track-frame/lora-finetune/$CFG" ]]; then CFG="$REPO/track-frame/lora-finetune/$CFG"; fi
fi
# base_path is read from the config so the script stays model-agnostic.
BASE_REL=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['base_path'])")
MODEL_SRC="$REPO/$BASE_REL"
STAGE="/dev/shm/$USER/$(basename "$MODEL_SRC")"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
JUDGE_STAGE="/dev/shm/$USER/$(basename "$JUDGE_SRC")"
trap 'rm -rf "$STAGE" "$JUDGE_STAGE"' EXIT

echo "train: cfg=$CFG base=$MODEL_SRC host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging base  -> $STAGE";       mkdir -p "$STAGE";       time cp -ru "$MODEL_SRC"/. "$STAGE"/
echo "staging judge -> $JUDGE_STAGE"; mkdir -p "$JUDGE_STAGE"; time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/

OUT="$REPO/track-frame/lora-finetune/logs/$(basename "$MODEL_SRC")"
# auto-resume if epoch checkpoints already exist (e.g. a previous 24h job walled out)
RESUME=""
if compgen -G "$OUT/checkpoint-*" >/dev/null 2>&1; then
  RESUME="--resume"; echo "found checkpoints in $OUT -> resuming"
fi

python track-frame/lora-finetune/src/train.py --config "$CFG" --base "$STAGE" \
    --out "$OUT" --judge-model "$JUDGE_STAGE" $RESUME "${@:2}"
echo "done. adapters + per-epoch evals under: $OUT"
