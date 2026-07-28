#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_ROOT="${REPO_ROOT}/tmp/submission-build/qwen3-vl-4b-procedure-epoch8"
IMAGE_PATH="${1:-${BUILD_ROOT}/focus-procedure-qwen3-vl-4b-epoch8.sif}"
INPUT_DIR="${SCRIPT_DIR}/test/input/interface_1"
OUTPUT_DIR="${SCRIPT_DIR}/test/output/interface_1"
KEEP_IMAGE="${KEEP_APPTAINER_IMAGE:-0}"
TEST_MODEL_DTYPE="${FOCUS_TEST_MODEL_DTYPE:-bfloat16}"
TEST_MAX_FRAMES="${FOCUS_TEST_MAX_FRAMES:-96}"
RUNTIME_TMP_BASE="${FOCUS_APPTAINER_TMP_BASE:-/tmp}"
RUNTIME_TMP_DIR=$(mktemp -d "${RUNTIME_TMP_BASE%/}/orena-procedure-runtime.XXXXXX")
passed=0

cleanup() {
    rm -rf -- "${RUNTIME_TMP_DIR}"
    if [[ "${passed}" == "1" && "${KEEP_IMAGE}" != "1" ]]; then
        rm -f -- "${IMAGE_PATH}"
        rm -rf -- "${BUILD_ROOT}/cache" "${BUILD_ROOT}/tmp"
        rmdir --ignore-fail-on-non-empty "${BUILD_ROOT}" 2>/dev/null || true
        echo "Removed passing smoke-test SIF and disposable Apptainer cache."
    fi
}
trap cleanup EXIT

case "${TEST_MODEL_DTYPE}" in
    bfloat16|float16) ;;
    *)
        echo "ERROR: FOCUS_TEST_MODEL_DTYPE must be bfloat16 or float16." >&2
        exit 1
        ;;
esac
if ! [[ "${TEST_MAX_FRAMES}" =~ ^[1-9][0-9]*$ ]] \
    || ((TEST_MAX_FRAMES > 96)); then
    echo "ERROR: FOCUS_TEST_MAX_FRAMES must be an integer from 1 to 96." >&2
    exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
    "${SCRIPT_DIR}/do_build_apptainer.sh" "${IMAGE_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
rm -f -- "${OUTPUT_DIR}/answer.json"
echo "Apptainer smoke: dtype=${TEST_MODEL_DTYPE} max_frames=${TEST_MAX_FRAMES}"

apptainer run \
    --nv \
    --containall \
    --no-home \
    --cleanenv \
    --net \
    --network=none \
    --pwd /opt/app \
    --env "FOCUS_MODEL_DTYPE=${TEST_MODEL_DTYPE}" \
    --env "FOCUS_MAX_FRAMES=${TEST_MAX_FRAMES}" \
    --bind "${INPUT_DIR}":/input:ro \
    --bind "${OUTPUT_DIR}":/output \
    --bind "${RUNTIME_TMP_DIR}":/tmp \
    "${IMAGE_PATH}"

python3 "${SCRIPT_DIR}/validate_output.py" \
    --requests "${INPUT_DIR}/request.json" \
    --answers "${OUTPUT_DIR}/answer.json"

passed=1
