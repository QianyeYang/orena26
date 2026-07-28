#!/usr/bin/env bash

#SBATCH --job-name=focus-frame-smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=submissions/frame/qwen3-vl-4b-lora-both-official-epoch30-20260725/test/apptainer-smoke-%j.log

set -euo pipefail

BUNDLE_NAME="qwen3-vl-4b-lora-both-official-epoch30-20260725"
if [[ -f "${SLURM_SUBMIT_DIR}/inference.py" && -f "${SLURM_SUBMIT_DIR}/apptainer.def" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -d "${SLURM_SUBMIT_DIR}/submissions/frame/${BUNDLE_NAME}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/submissions/frame/${BUNDLE_NAME}"
else
    echo "ERROR: submit from the repository root or the bundle directory" >&2
    exit 2
fi
SCRATCH_ROOT="/scratch/shared/engs2732/focus-submission-build/qwen3-vl-4b-frame-epoch30"
IMAGE_PATH="${SCRATCH_ROOT}/focus-frame-qwen3-vl-4b-epoch30.sif"

mkdir -p "${SCRATCH_ROOT}"
export FOCUS_APPTAINER_TMP_BASE="/dev/shm"
export KEEP_APPTAINER_IMAGE=0

echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer version
df -h /dev/shm "${SCRATCH_ROOT}"

"${SCRIPT_DIR}/do_test_apptainer.sh" "${IMAGE_PATH}"

echo "finished=$(date --iso-8601=seconds)"
if [[ -e "${IMAGE_PATH}" ]]; then
    echo "ERROR: passing smoke-test SIF was not removed" >&2
    exit 1
fi
echo "PASS: smoke test passed and SIF was removed"
