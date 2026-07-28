#!/usr/bin/env bash

#SBATCH --job-name=focus-procedure-smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=submissions/procedure/qwen3-vl-4b-lora-both-official-epoch8-20260727/test/apptainer-smoke-%j.log

set -euo pipefail

BUNDLE_NAME="qwen3-vl-4b-lora-both-official-epoch8-20260727"
if [[ -f "${SLURM_SUBMIT_DIR}/inference.py" \
    && -f "${SLURM_SUBMIT_DIR}/apptainer.def" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -d "${SLURM_SUBMIT_DIR}/submissions/procedure/${BUNDLE_NAME}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/submissions/procedure/${BUNDLE_NAME}"
else
    echo "ERROR: submit from the repository root or the bundle directory" >&2
    exit 2
fi
SCRATCH_ROOT="/scratch/shared/engs2732/focus-submission-build/qwen3-vl-4b-procedure-epoch8"
IMAGE_PATH="${SCRATCH_ROOT}/focus-procedure-qwen3-vl-4b-epoch8.sif"

mkdir -p "${SCRATCH_ROOT}"
export FOCUS_APPTAINER_TMP_BASE="/dev/shm"
export FOCUS_TEST_MODEL_DTYPE="bfloat16"
export FOCUS_TEST_MAX_FRAMES="96"
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
