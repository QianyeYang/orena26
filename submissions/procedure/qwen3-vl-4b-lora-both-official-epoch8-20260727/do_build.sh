#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE_TAG="focus-procedure-qwen3-vl-4b-epoch8"

docker build \
    --platform=linux/amd64 \
    --tag "${DOCKER_IMAGE_TAG}" \
    "${SCRIPT_DIR}"
