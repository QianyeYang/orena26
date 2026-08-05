#!/usr/bin/env bash
# Convenience launcher: open the unified visualiser on the Segment module.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FOCUS_VIS_PORT="${SEGMENT_VIS_PORT:-8767}"
export FOCUS_VIS_INITIAL_TRACK="segment"
exec "${REPO_ROOT}/vis-tool.sh" "$@"
