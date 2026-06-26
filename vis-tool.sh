#!/usr/bin/env bash
# Launch the ORena FOCUS prediction visualiser.
# Then in VSCode: Cmd/Ctrl-Shift-P -> "Simple Browser: Show" -> http://localhost:8765
set -euo pipefail

cd "$(dirname "$0")"

# Activate the orena conda env (unless it is already the active one).
if [[ "${CONDA_DEFAULT_ENV:-}" != "orena" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate orena
  else
    echo "error: conda not found on PATH; run 'conda activate orena' first." >&2
    exit 1
  fi
fi

# Keep BLAS/OMP single-threaded (login node caps process count -> OpenBLAS crashes).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec python -m src.visualisation.server --port 8765 "$@"
