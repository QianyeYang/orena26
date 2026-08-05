#!/bin/bash
# Run the exact epoch-8 Procedure adapter over every official test question,
# then judge every response. The inference JSONL makes re-submission resumable.
#
# Usage:
#   sbatch track-procedure/lora-finetune/scripts/full_test_a100.slurm
#SBATCH --job-name=procedure-full-ep8
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/track-procedure/lora-finetune/logs/full-test-a100-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-procedure/lora-finetune/logs/full-test-a100-%j.log

set -euo pipefail

REPO=/datasets/engs2732/orena
TRACKDIR="$REPO/track-procedure/lora-finetune"
TRAIN_RUN="$TRACKDIR/logs/Qwen3-VL-4B-Instruct-both-official"
ADAPTER="$TRAIN_RUN/checkpoint-3440"
MODEL_SRC="$REPO/os-models/Qwen3-VL-4B-Instruct"
JUDGE_SRC="$REPO/os-models/Qwen3.5-4B"
OUT_ROOT="$TRAIN_RUN/full_test_epoch_8"
MODEL_NAME="Qwen3-VL-4B-Instruct-lora-both-official-epoch8-full-test"

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
    echo "epoch-8 adapter not found: $ADAPTER" >&2
    exit 2
}
[[ -d "$MODEL_SRC" ]] || { echo "base model not found: $MODEL_SRC" >&2; exit 2; }
[[ -d "$JUDGE_SRC" ]] || { echo "judge model not found: $JUDGE_SRC" >&2; exit 2; }

# Pin the exact evaluated epoch-8 artifact. The final adapter is currently
# byte-identical, but checkpoint-3440 makes the requested provenance explicit.
EXPECTED_ADAPTER_SHA256=7927db9c55ecbaf2fad4f23c4686b9507a7bbb7e2d8537507c5086cbe6ae7c6f
actual_adapter_sha256="$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
if [[ "$actual_adapter_sha256" != "$EXPECTED_ADAPTER_SHA256" ]]; then
    echo "epoch-8 adapter hash mismatch: $actual_adapter_sha256" >&2
    exit 2
fi

JOB_TAG="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
JOB_SCRATCH="/dev/shm/$USER/orena-procedure-full-$JOB_TAG"
MODEL_STAGE="$JOB_SCRATCH/model"
JUDGE_STAGE="$JOB_SCRATCH/judge"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
mkdir -p "$JOB_SCRATCH/tmp" "$MODEL_STAGE" "$JUDGE_STAGE" "$OUT_ROOT"
export TMPDIR="$JOB_SCRATCH/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/inductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels"

echo "track=procedure split=test checkpoint=3440 adapter_sha256=$actual_adapter_sha256"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} output=$OUT_ROOT"
python -c 'import accelerate,peft,torch,transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "staging base  -> $MODEL_STAGE"
time cp -ru "$MODEL_SRC"/. "$MODEL_STAGE"/
echo "staging judge -> $JUDGE_STAGE"
time cp -ru "$JUDGE_SRC"/. "$JUDGE_STAGE"/

for DATASET in heico lapchole; do
    if [[ "$DATASET" == "heico" ]]; then
        STRIDE=250
        EXPECTED_ROWS=2000
    else
        STRIDE=300
        EXPECTED_ROWS=1127
    fi
    OUT="$OUT_ROOT/$DATASET"

    echo "[$DATASET] full inference start: expected_rows=$EXPECTED_ROWS stride=$STRIDE"
    python "$REPO/track-procedure/baseline/src/runner.py" \
        --dataset "$DATASET" \
        --track procedure \
        --split test \
        --model "$MODEL_STAGE" \
        --model-name "$MODEL_NAME" \
        --adapter "$ADAPTER" \
        --out "$OUT" \
        --stride "$STRIDE" \
        --max-frames 96 \
        --max-pixels 131072 \
        --max-new-tokens 64 \
        --dtype bfloat16

    python - "$DATASET" "$OUT/predictions.parquet" "$EXPECTED_ROWS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dataset, prediction_path, expected_rows_text = sys.argv[1:]
expected_rows = int(expected_rows_text)
ground_truth_path = (
    Path("data/parquet/procedure/test/0000.parquet")
    if dataset == "heico"
    else Path("data/parquet/lapchole/procedure/test/0000.parquet")
)
ground_truth = pd.read_parquet(ground_truth_path, columns=["id"])
predictions = pd.read_parquet(prediction_path)
expected_ids = ground_truth["id"].astype(str).tolist()
actual_ids = predictions["sample_id"].astype(str).tolist()
if len(ground_truth) != expected_rows:
    raise SystemExit(
        f"{dataset}: expected-row constant is stale: {expected_rows} != {len(ground_truth)}"
    )
if len(predictions) != expected_rows:
    raise SystemExit(
        f"{dataset}: incomplete predictions: {len(predictions)} != {expected_rows}"
    )
if len(set(actual_ids)) != expected_rows:
    raise SystemExit(f"{dataset}: prediction IDs are not unique")
if set(actual_ids) != set(expected_ids):
    missing = sorted(set(expected_ids) - set(actual_ids))[:5]
    extra = sorted(set(actual_ids) - set(expected_ids))[:5]
    raise SystemExit(f"{dataset}: prediction ID mismatch; missing={missing} extra={extra}")
errors = int(predictions["error"].fillna("").astype(str).str.len().gt(0).sum())
if errors:
    raise SystemExit(f"{dataset}: {errors} inference rows contain errors")
print(f"{dataset}: inference validation passed ({expected_rows} unique rows, no errors)")
PY

    echo "[$DATASET] full judge evaluation start"
    python "$REPO/track-frame/baseline/src/evaluate.py" \
        --track procedure \
        --split test \
        --dataset "$DATASET" \
        --responses "$OUT/responses.json" \
        --out "$OUT/eval" \
        --judge-model "$JUDGE_STAGE"

    python - "$DATASET" "$OUT/eval/results.csv" "$EXPECTED_ROWS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dataset, results_path, expected_rows_text = sys.argv[1:]
expected_rows = int(expected_rows_text)
results = pd.read_csv(results_path)
ground_truth_path = (
    Path("data/parquet/procedure/test/0000.parquet")
    if dataset == "heico"
    else Path("data/parquet/lapchole/procedure/test/0000.parquet")
)
expected_ids = set(
    pd.read_parquet(ground_truth_path, columns=["id"])["id"].astype(str)
)
actual_ids = set(results["qID"].astype(str))
if len(results) != expected_rows:
    raise SystemExit(f"{dataset}: incomplete evaluation: {len(results)} != {expected_rows}")
if len(actual_ids) != expected_rows:
    raise SystemExit(f"{dataset}: evaluation IDs are not unique")
if actual_ids != expected_ids:
    missing = sorted(expected_ids - actual_ids)[:5]
    extra = sorted(actual_ids - expected_ids)[:5]
    raise SystemExit(f"{dataset}: evaluation ID mismatch; missing={missing} extra={extra}")
print(f"{dataset}: evaluation validation passed ({expected_rows} unique rows)")
PY
done

python - "$OUT_ROOT/run_manifest.json" "$actual_adapter_sha256" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, adapter_sha256 = sys.argv[1:]
payload = {
    "track": "procedure",
    "split": "test",
    "datasets": {"heico": 2000, "lapchole": 1127},
    "checkpoint": "checkpoint-3440",
    "epoch": 8,
    "adapter_sha256": adapter_sha256,
    "sampling": {
        "stride": {"heico": 250, "lapchole": 300},
        "max_frames": 96,
        "max_pixels": 131072,
    },
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

touch "$OUT_ROOT/COMPLETE"
echo "full Procedure test complete: 3127/3127 -> $OUT_ROOT"
