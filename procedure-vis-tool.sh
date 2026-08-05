#!/usr/bin/env bash
# Compatibility launcher: open the unified visualiser on the Procedure module.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FOCUS_VIS_PORT="${PROCEDURE_VIS_PORT:-8766}"
export FOCUS_VIS_INITIAL_TRACK="procedure"
exec "${REPO_ROOT}/vis-tool.sh" "$@"
