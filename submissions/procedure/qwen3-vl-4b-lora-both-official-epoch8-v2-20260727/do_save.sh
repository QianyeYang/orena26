#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE_TAG="focus-procedure-qwen3-vl-4b-epoch8-v2"

"${SCRIPT_DIR}/do_build.sh"
environment=$(docker image inspect "${DOCKER_IMAGE_TAG}" \
    --format '{{range .Config.Env}}{{println .}}{{end}}')
if ! grep -Fxq 'FOCUS_MODEL_DTYPE=bfloat16' <<<"${environment}"; then
    echo "ERROR: refusing to save an image whose default dtype is not BF16." >&2
    exit 1
fi
if ! grep -Fxq 'FOCUS_MAX_FRAMES=96' <<<"${environment}"; then
    echo "ERROR: refusing to save an image whose default frame cap is not 96." >&2
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
