#!/usr/bin/env bash
# Launch the local, SSH-friendly FOCUS Frame bounding-box annotation tool.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOL_ROOT="${REPO_ROOT}/track-frame/frame-annotation-tool"
TOOL_PORT="${FRAME_ANNOTATION_PORT:-8770}"
TOOL_HOST="${FRAME_ANNOTATION_HOST:-127.0.0.1}"

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "orena" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate orena
  else
    echo "error: conda not found; run 'conda activate orena' first." >&2
    exit 1
  fi
fi

# The login node has a conservative process limit; one thread is ample for this
# local metadata/web workload and avoids accidental BLAS worker fan-out.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${TOOL_ROOT}/src/build_pool.py" --ensure

echo
echo "Frame Annotation Tool will listen on ${TOOL_HOST}:${TOOL_PORT}"
echo "From your own computer, open a second terminal and run:"
echo "  ssh -N -L ${TOOL_PORT}:127.0.0.1:${TOOL_PORT} <user>@<cluster-host>"
echo "Then open: http://localhost:${TOOL_PORT}"
echo

exec python "${TOOL_ROOT}/src/server.py" \
  --host "${TOOL_HOST}" \
  --port "${TOOL_PORT}" \
  "$@"
