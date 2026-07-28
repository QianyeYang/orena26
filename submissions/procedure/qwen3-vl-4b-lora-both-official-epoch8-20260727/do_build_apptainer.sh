#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_ROOT="${REPO_ROOT}/tmp/submission-build/qwen3-vl-4b-procedure-epoch8"
IMAGE_PATH="${1:-${BUILD_ROOT}/focus-procedure-qwen3-vl-4b-epoch8.sif}"
CACHE_DIR="${BUILD_ROOT}/cache"
LOCAL_TMP_BASE="${FOCUS_APPTAINER_TMP_BASE:-/tmp}"
LOCAL_TMP_DIR=$(mktemp -d "${LOCAL_TMP_BASE%/}/orena-procedure-apptainer.XXXXXX")

cleanup() {
    rm -rf -- "${LOCAL_TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "$(dirname -- "${IMAGE_PATH}")" "${CACHE_DIR}"
cd "${SCRIPT_DIR}"

APPTAINER_CACHEDIR="${CACHE_DIR}" \
APPTAINER_TMPDIR="${LOCAL_TMP_DIR}" \
apptainer build --fakeroot --ignore-fakeroot-command "${IMAGE_PATH}" apptainer.def

echo "${IMAGE_PATH}"
