#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_ROOT="${REPO_ROOT}/tmp/submission-build/qwen3-vl-4b-procedure-epoch8-v2"
IMAGE_PATH="${1:-${BUILD_ROOT}/focus-procedure-qwen3-vl-4b-epoch8-v2.sif}"
CACHE_DIR="${BUILD_ROOT}/cache"
LOCAL_TMP_BASE="${FOCUS_APPTAINER_TMP_BASE:-/tmp}"
LOCAL_TMP_DIR=""

cleanup() {
    if [[ -n "${LOCAL_TMP_DIR}" ]]; then
        rm -rf -- "${LOCAL_TMP_DIR}"
    fi
}
trap cleanup EXIT

mkdir -p "$(dirname -- "${IMAGE_PATH}")" "${CACHE_DIR}"
cd "${SCRIPT_DIR}"

for attempt in 1 2; do
    LOCAL_TMP_DIR=$(mktemp -d \
        "${LOCAL_TMP_BASE%/}/orena-procedure-apptainer.XXXXXX")
    if APPTAINER_CACHEDIR="${CACHE_DIR}" \
        APPTAINER_TMPDIR="${LOCAL_TMP_DIR}" \
        apptainer build --fakeroot --ignore-fakeroot-command \
            "${IMAGE_PATH}" apptainer.def; then
        rm -rf -- "${LOCAL_TMP_DIR}"
        LOCAL_TMP_DIR=""
        echo "${IMAGE_PATH}"
        exit 0
    else
        build_status=$?
    fi
    rm -rf -- "${LOCAL_TMP_DIR}"
    LOCAL_TMP_DIR=""
    rm -f -- "${IMAGE_PATH}"
    if ((attempt == 2)); then
        exit "${build_status}"
    fi
    echo "WARNING: Apptainer build failed; retrying once with a fresh tmpfs directory." >&2
done

exit 1
