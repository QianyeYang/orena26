#!/bin/bash
#SBATCH --job-name=frame-base-infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/baseline/logs/infer-%j.log
#SBATCH --error=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena/track-frame/baseline/logs/infer-%j.log

# Usage: sbatch run_inference.slurm [split] [extra runner args...]
#   sbatch run_inference.slurm test --limit 50      # smoke test
#   sbatch run_inference.slurm test                 # full public test
#
# Stages the model to node-local /dev/shm first: safetensors loads via mmap
# (729 random reads) which stalls badly on contended Lustre; a single sequential
# copy + load-from-RAM avoids that.
set -euo pipefail
REPO=/lus/lfs1aip2/projects/u6ix/yipei/Q/project/orena
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SPLIT="${1:-test}"
MODEL_SRC="$REPO/os-models/Qwen2.5-VL-7B-Instruct"
STAGE="/dev/shm/$USER/$(basename "$MODEL_SRC")"
trap 'rm -rf "$STAGE"' EXIT

echo "infer: split=$SPLIT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "staging model -> $STAGE"
mkdir -p "$STAGE"
time cp -ru "$MODEL_SRC"/. "$STAGE"/

OUT="$REPO/track-frame/baseline/logs/resp_${SPLIT}"
python track-frame/baseline/src/runner.py --split "$SPLIT" --model "$STAGE" --out "$OUT" "${@:2}"
echo "done."
