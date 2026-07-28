#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE_TAG="focus-frame-qwen3-vl-4b-epoch30"
INPUT_DIR="${SCRIPT_DIR}/test/input/interface_1"
OUTPUT_DIR="${SCRIPT_DIR}/test/output/interface_1"
TMP_VOLUME="${DOCKER_IMAGE_TAG}-tmp"
TEST_MODEL_DTYPE="${FOCUS_TEST_MODEL_DTYPE:-bfloat16}"
TEST_BATCH_SIZE="${FOCUS_TEST_BATCH_SIZE:-8}"

case "${TEST_MODEL_DTYPE}" in
    bfloat16|float16) ;;
    *)
        echo "ERROR: FOCUS_TEST_MODEL_DTYPE must be bfloat16 or float16." >&2
        exit 1
        ;;
esac
if ! [[ "${TEST_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FOCUS_TEST_BATCH_SIZE must be a positive integer." >&2
    exit 1
fi

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "ERROR: NVIDIA Container Toolkit is required for this model." >&2
    exit 1
fi

"${SCRIPT_DIR}/do_build.sh"
mkdir -p "${OUTPUT_DIR}"
rm -f -- "${OUTPUT_DIR}/answer.json"
docker volume create "${TMP_VOLUME}" >/dev/null

cleanup() {
    docker volume rm "${TMP_VOLUME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Smoke-test runtime override: dtype=${TEST_MODEL_DTYPE} batch=${TEST_BATCH_SIZE}"
if [[ "${TEST_MODEL_DTYPE}" == "float16" ]]; then
    echo "FP16 is test-only; the Docker image default remains BF16."
fi

docker run --rm \
    --platform=linux/amd64 \
    --network=none \
    --gpus=all \
    --shm-size=2g \
    --env "FOCUS_MODEL_DTYPE=${TEST_MODEL_DTYPE}" \
    --env "FOCUS_BATCH_SIZE=${TEST_BATCH_SIZE}" \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output \
    --volume "${TMP_VOLUME}":/tmp \
    "${DOCKER_IMAGE_TAG}"

python3 "${SCRIPT_DIR}/validate_output.py" \
    --requests "${INPUT_DIR}/request.json" \
    --answers "${OUTPUT_DIR}/answer.json"
