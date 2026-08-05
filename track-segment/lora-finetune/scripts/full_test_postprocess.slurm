#!/bin/bash
# Validate the completed H200 array and publish the full Segment result report.
# Submit with: sbatch --dependency=afterok:<inference-array-id> <this-file>
#SBATCH --job-name=segment-full-report
#SBATCH --partition=allgpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/track-segment/lora-finetune/logs/full-test-report-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-segment/lora-finetune/logs/full-test-report-%j.log

set -euo pipefail

REPO=/datasets/engs2732/orena
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/summarize_segment_full_test.py

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

repo = Path("/datasets/engs2732/orena")
run = (
    repo
    / "track-segment/lora-finetune/logs/"
    / "Qwen3-VL-4B-Instruct-both-official/full_test_epoch_6"
)
summary = (
    repo
    / "result-summary/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-full-test"
)
expected_rows = {"heico": 4000, "lapchole": 2254}
expected_files = {
    "answer-format.csv",
    "capability-answer-format.csv",
    "capability-dataset-gap.csv",
    "capability-group.csv",
    "capability-leaf.csv",
    "cardinality-audit.csv",
    "categorical-diagnostics.csv",
    "checkpoint-epoch6-vs-epoch8-paired.csv",
    "checkpoint-selection.csv",
    "dataset-overall.csv",
    "fo-class-cardinality.csv",
    "fo-class-label.csv",
    "numeric-diagnostics.csv",
    "provenance.json",
    "runtime.csv",
    "secondary-capability.csv",
    "slice-performance.csv",
    "subset-vs-full.csv",
    "time-diagnostics.csv",
    "video-performance.csv",
}
actual_files = {path.name for path in summary.iterdir() if path.is_file()}
if actual_files != expected_files:
    raise SystemExit(
        f"summary artifact mismatch; missing={sorted(expected_files - actual_files)} "
        f"extra={sorted(actual_files - expected_files)}"
    )

overall = pd.read_csv(summary / "dataset-overall.csv").set_index("dataset")
if overall["rows"].astype(int).to_dict() != expected_rows:
    raise SystemExit(f"wrong report row counts: {overall['rows'].to_dict()}")
if not overall["evaluated_fraction"].eq(1.0).all():
    raise SystemExit("full-test report contains incomplete dataset coverage")
if int(overall["inference_error_rows"].sum()) != 0:
    raise SystemExit("full-test report contains inference errors")
if int(overall["timed_out_rows"].sum()) != 0:
    raise SystemExit("full-test report contains timeouts")

provenance = json.loads((summary / "provenance.json").read_text())
if provenance["coverage"]["rows"] != sum(expected_rows.values()):
    raise SystemExit("provenance does not cover all full-test rows")
if provenance["coverage"]["coverage_fraction"] != 1.0:
    raise SystemExit("provenance coverage fraction is not one")
if not (run / "COMPLETE").is_file():
    raise SystemExit("aggregate run COMPLETE marker is absent")

report_text = (
    repo
    / "result-summary/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-full-test.md"
).read_text()
if "Authoritative local full-test evaluation" not in report_text:
    raise SystemExit("full-test Markdown status is missing")
if "weakest ability" not in report_text:
    raise SystemExit("full-test Markdown does not identify the weakest ability")
index_text = (repo / "result-summary/segment/README.md").read_text()
if "Authoritative local full-test report" not in index_text:
    raise SystemExit("Segment result index was not promoted to full-test status")

print("full Segment post-processing validation passed")
print(overall.to_string())
PY

echo "full Segment report published and validated"
