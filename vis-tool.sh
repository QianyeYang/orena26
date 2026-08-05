#!/usr/bin/env bash
# Launch the unified Frame / Segment / Procedure prediction visualiser.
# Then in VSCode: Cmd/Ctrl-Shift-P -> "Simple Browser: Show" -> http://localhost:8765
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

FRAME_DEFAULT="${REPO_ROOT}/track-frame/lora-finetune/logs/full-test-comparison/new-epoch30"
SEGMENT_TRAIN_RUN="${REPO_ROOT}/track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official"
if [[ -f "${SEGMENT_TRAIN_RUN}/full_test_epoch_6/COMPLETE" ]]; then
  SEGMENT_DEFAULT="${SEGMENT_TRAIN_RUN}/full_test_epoch_6"
else
  SEGMENT_DEFAULT="${SEGMENT_TRAIN_RUN}/eval_epoch_6"
fi
PROCEDURE_TRAIN_RUN="${REPO_ROOT}/track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official"
if [[ -f "${PROCEDURE_TRAIN_RUN}/full_test_epoch_8/COMPLETE" ]]; then
  PROCEDURE_DEFAULT="${PROCEDURE_TRAIN_RUN}/full_test_epoch_8"
else
  PROCEDURE_DEFAULT="${PROCEDURE_TRAIN_RUN}/eval_epoch_8"
fi

FRAME_RUN="${FRAME_RESULTS:-${FRAME_DEFAULT}}"
SEGMENT_RUN="${SEGMENT_RESULTS:-${SEGMENT_DEFAULT}}"
PROCEDURE_RUN="${PROCEDURE_RESULTS:-${PROCEDURE_DEFAULT}}"
PORT="${FOCUS_VIS_PORT:-8765}"
HOST="${FOCUS_VIS_HOST:-127.0.0.1}"
INITIAL_TRACK="${FOCUS_VIS_INITIAL_TRACK:-frame}"

cd "${REPO_ROOT}"

# Activate the repository environment unless it is already active.
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

# Keep the metadata server lightweight on process-constrained login nodes.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for specification in \
  "Frame|${FRAME_RUN}" \
  "Segment|${SEGMENT_RUN}" \
  "Procedure|${PROCEDURE_RUN}"
do
  track_name=${specification%%|*}
  result_root=${specification#*|}
  for dataset in heico lapchole; do
    if [[ ! -f "${result_root}/${dataset}/predictions.parquet" ]]; then
      echo "error: ${track_name} results are incomplete: ${result_root}/${dataset}" >&2
      exit 1
    fi
  done
done

exec python -m src.visualisation.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --auto-port \
  --initial-track "${INITIAL_TRACK}" \
  --run-root "${FRAME_RUN}" \
  --run-root "${SEGMENT_RUN}" \
  --run-root "${PROCEDURE_RUN}" \
  "$@"
