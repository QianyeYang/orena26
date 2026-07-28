#!/usr/bin/env bash
# Launch the latest Frame-track prediction visualiser.
# Then in VSCode: Cmd/Ctrl-Shift-P -> "Simple Browser: Show" -> http://localhost:8765
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LATEST_RESULTS="${REPO_ROOT}/track-frame/lora-finetune/logs/full-test-comparison/new-epoch30"
cd "${REPO_ROOT}"

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

for dataset in heico lapchole; do
  if [[ ! -f "${LATEST_RESULTS}/${dataset}/predictions.parquet" ]]; then
    echo "error: latest Frame results are incomplete: ${LATEST_RESULTS}/${dataset}" >&2
    exit 1
  fi
done

exec python -m src.visualisation.server \
  --port 8765 \
  --run-root "${LATEST_RESULTS}" \
  "$@"
