#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE_TAG="focus-frame-qwen3-vl-4b-epoch30"

"${SCRIPT_DIR}/do_build.sh"
if ! docker image inspect "${DOCKER_IMAGE_TAG}" \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -Fxq 'FOCUS_MODEL_DTYPE=bfloat16'; then
    echo "ERROR: refusing to save an image whose default dtype is not BF16." >&2
    exit 1
fi
created=$(docker inspect --format='{{.Created}}' "${DOCKER_IMAGE_TAG}")
stamp=$(date --date="${created}" --utc +%Y-%m-%d_%H-%M-%S)
archive="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${stamp}.tar.gz"

if command -v pigz >/dev/null 2>&1; then
    compressor=(pigz -c)
else
    compressor=(gzip -c)
fi

echo "Saving ${DOCKER_IMAGE_TAG} to ${archive}"
docker save "${DOCKER_IMAGE_TAG}" | "${compressor[@]}" >"${archive}"
gzip --test "${archive}"
echo "Saved and verified ${archive}"
