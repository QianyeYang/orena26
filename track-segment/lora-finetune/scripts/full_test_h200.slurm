#!/bin/bash
# Run the selected epoch-6 Segment adapter over every official test question,
# then judge every response. Array task 0 handles HeiCo and task 1 LapChole.
# The inference JSONL makes re-submission resumable.
#
# Usage:
#   sbatch track-segment/lora-finetune/scripts/full_test_h200.slurm
#
# The typed GRES guarantees that each array task receives one NVIDIA H200 NVL
# rather than another GPU family.
#SBATCH --job-name=segment-full-ep6
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/track-segment/lora-finetune/logs/full-test-h200-%A_%a.log
#SBATCH --error=/datasets/engs2732/orena/track-segment/lora-finetune/logs/full-test-h200-%A_%a.log

set -euo pipefail

REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-segment/lora-finetune"
TRAIN_RUN="$TRACKDIR/logs/Qwen3-VL-4B-Instruct-both-official"
ADAPTER="$TRAIN_RUN/checkpoint-5160"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
OUT_ROOT="$TRAIN_RUN/full_test_epoch_6"
MODEL_NAME="Qwen3-VL-4B-Instruct-lora-both-official-epoch6-full-test"
EXPECTED_ADAPTER_SHA256=92556248ad9e8a82b7f1d295bac3c32f0984e7e247ef1a74499e7134032b73cb

DATASETS=(heico lapchole)
EXPECTED_ROWS=(4000 2254)
STRIDES=(25 30)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= ${#DATASETS[@]} )); then
    echo "invalid array task index: $TASK_ID" >&2
    exit 2
fi
DATASET="${DATASETS[$TASK_ID]}"
EXPECTED_COUNT="${EXPECTED_ROWS[$TASK_ID]}"
STRIDE="${STRIDES[$TASK_ID]}"
OUT="$OUT_ROOT/$DATASET"

source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

[[ -f "$ADAPTER/adapter_model.safetensors" ]] || {
    echo "epoch-6 adapter not found: $ADAPTER" >&2
    exit 2
}
[[ -d "$MODEL_SRC" ]] || { echo "base model not found: $MODEL_SRC" >&2; exit 2; }
[[ -d "$JUDGE_SRC" ]] || { echo "judge model not found: $JUDGE_SRC" >&2; exit 2; }

actual_adapter_sha256="$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
if [[ "$actual_adapter_sha256" != "$EXPECTED_ADAPTER_SHA256" ]]; then
    echo "epoch-6 adapter hash mismatch: $actual_adapter_sha256" >&2
    exit 2
fi
RUNNER_SHA256="$(sha256sum "$REPO/track-segment/baseline/src/runner.py" | awk '{print $1}')"
PROMPTS_SHA256="$(sha256sum "$REPO/src/prompts.py" | awk '{print $1}')"
ADAPTER_CODE_SHA256="$(sha256sum "$REPO/src/adapter.py" | awk '{print $1}')"

JOB_TAG="${SLURM_ARRAY_JOB_ID:?}_${TASK_ID}"
JOB_SCRATCH="/dev/shm/$USER/orena-segment-full-$JOB_TAG"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$JOB_SCRATCH/tmp" "$MODEL_STAGE" "$JUDGE_STAGE" "$OUT"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "track=segment dataset=$DATASET split=test checkpoint=5160 epoch=6"
echo "adapter_sha256=$actual_adapter_sha256 expected_rows=$EXPECTED_COUNT stride=$STRIDE"
echo "runner_sha256=$RUNNER_SHA256 prompts_sha256=$PROMPTS_SHA256 adapter_code_sha256=$ADAPTER_CODE_SHA256"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} output=$OUT"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - "$DATASET" "$EXPECTED_COUNT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dataset, expected_text = sys.argv[1:]
expected = int(expected_text)
path = (
    Path("data/parquet/segment/test/0000.parquet")
    if dataset == "heico"
    else Path("data/parquet/lapchole/segment/test/0000.parquet")
)
rows = len(pd.read_parquet(path, columns=["id"]))
if rows != expected:
    raise SystemExit(f"{dataset}: expected-row constant is stale: {expected} != {rows}")
print(f"{dataset}: annotation preflight passed ({rows} rows)")
PY

echo "staging base  -> $MODEL_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$MODEL_SRC"/. "$MODEL_STAGE"/
echo "staging judge -> $JUDGE_STAGE"
time cp -r --no-preserve=mode,ownership,timestamps "$JUDGE_SRC"/. "$JUDGE_STAGE"/

echo "[$DATASET] full inference start"
python "$REPO/track-segment/baseline/src/runner.py" \
    --dataset "$DATASET" \
    --track segment \
    --split test \
    --model "$MODEL_STAGE" \
    --model-name "$MODEL_NAME" \
    --adapter "$ADAPTER" \
    --out "$OUT" \
    --stride "$STRIDE" \
    --max-frames 64 \
    --max-pixels 262144 \
    --max-new-tokens 64 \
    --dtype bfloat16

python - "$DATASET" "$OUT/predictions.parquet" "$EXPECTED_COUNT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dataset, prediction_path, expected_text = sys.argv[1:]
expected = int(expected_text)
ground_truth_path = (
    Path("data/parquet/segment/test/0000.parquet")
    if dataset == "heico"
    else Path("data/parquet/lapchole/segment/test/0000.parquet")
)
expected_ids = pd.read_parquet(ground_truth_path, columns=["id"])["id"].astype(str)
predictions = pd.read_parquet(prediction_path)
actual_ids = predictions["sample_id"].astype(str)
if len(predictions) != expected:
    raise SystemExit(f"{dataset}: incomplete predictions: {len(predictions)} != {expected}")
if actual_ids.nunique() != expected:
    raise SystemExit(f"{dataset}: prediction IDs are not unique")
if set(actual_ids) != set(expected_ids):
    missing = sorted(set(expected_ids) - set(actual_ids))[:5]
    extra = sorted(set(actual_ids) - set(expected_ids))[:5]
    raise SystemExit(f"{dataset}: prediction ID mismatch; missing={missing} extra={extra}")
errors = int(predictions["error"].fillna("").astype(str).str.len().gt(0).sum())
if errors:
    raise SystemExit(f"{dataset}: {errors} inference rows contain errors")
empty = int(predictions["raw_model_output"].fillna("").astype(str).str.strip().eq("").sum())
if empty:
    raise SystemExit(f"{dataset}: {empty} inference rows have empty outputs")
print(f"{dataset}: inference validation passed ({expected} unique rows, no errors)")
PY

echo "[$DATASET] full judge evaluation start"
python "$REPO/track-frame/baseline/src/evaluate.py" \
    --track segment \
    --split test \
    --dataset "$DATASET" \
    --responses "$OUT/responses.json" \
    --out "$OUT/eval" \
    --judge-model "$JUDGE_STAGE"

python - "$DATASET" "$OUT/eval/results.csv" "$EXPECTED_COUNT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dataset, results_path, expected_text = sys.argv[1:]
expected = int(expected_text)
ground_truth_path = (
    Path("data/parquet/segment/test/0000.parquet")
    if dataset == "heico"
    else Path("data/parquet/lapchole/segment/test/0000.parquet")
)
expected_ids = set(pd.read_parquet(ground_truth_path, columns=["id"])["id"].astype(str))
results = pd.read_csv(results_path, dtype={"qID": str})
actual_ids = set(results["qID"])
if len(results) != expected:
    raise SystemExit(f"{dataset}: incomplete evaluation: {len(results)} != {expected}")
if len(actual_ids) != expected:
    raise SystemExit(f"{dataset}: evaluation IDs are not unique")
if actual_ids != expected_ids:
    missing = sorted(expected_ids - actual_ids)[:5]
    extra = sorted(actual_ids - expected_ids)[:5]
    raise SystemExit(f"{dataset}: evaluation ID mismatch; missing={missing} extra={extra}")
print(f"{dataset}: evaluation validation passed ({expected} unique rows)")
PY

python - "$OUT/run_manifest.json" "$DATASET" "$EXPECTED_COUNT" "$STRIDE" \
    "$actual_adapter_sha256" "$RUNNER_SHA256" "$PROMPTS_SHA256" \
    "$ADAPTER_CODE_SHA256" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path,
    dataset,
    expected_text,
    stride_text,
    adapter_sha256,
    runner_sha256,
    prompts_sha256,
    adapter_code_sha256,
) = sys.argv[1:]
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
payload = {
    "track": "segment",
    "split": "test",
    "dataset": dataset,
    "rows": int(expected_text),
    "checkpoint": "checkpoint-5160",
    "epoch": 6,
    "adapter_sha256": adapter_sha256,
    "code_sha256": {
        "track-segment/baseline/src/runner.py": runner_sha256,
        "src/prompts.py": prompts_sha256,
        "src/adapter.py": adapter_code_sha256,
    },
    "sampling": {"stride": int(stride_text), "max_frames": 64, "max_pixels": 262144},
    "slurm": {
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "gpu": gpu_name,
    },
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

touch "$OUT/COMPLETE"
echo "full Segment $DATASET test complete: $EXPECTED_COUNT/$EXPECTED_COUNT -> $OUT"
