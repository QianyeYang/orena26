"""Validate one (track, dataset) full-test output directory.

Replaces the heredoc checks the per-track full-test scripts each inlined, so
the three unified full-test scripts assert identical invariants:

* the expected-row constant still matches the annotation parquet,
* inference covered every official question exactly once, with no errors and
  no empty generations,
* the judge scored every one of those questions.

    python validate_full_test.py --track segment --dataset heico \
        --out <run>/full_test/epoch_8/segment/heico --expected-rows 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.paths import DATASETS, TRACKS, parquet_path  # noqa: E402


def _fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def check_inference(out: Path, expected_ids: set[str], label: str) -> None:
    predictions = pd.read_parquet(out / "predictions.parquet")
    actual_ids = predictions["sample_id"].astype(str)
    if len(predictions) != len(expected_ids):
        _fail(f"{label}: incomplete predictions: {len(predictions)} != {len(expected_ids)}")
    if actual_ids.nunique() != len(expected_ids):
        _fail(f"{label}: prediction IDs are not unique")
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))[:5]
        extra = sorted(set(actual_ids) - expected_ids)[:5]
        _fail(f"{label}: prediction ID mismatch; missing={missing} extra={extra}")
    errors = int(predictions["error"].fillna("").astype(str).str.len().gt(0).sum())
    if errors:
        _fail(f"{label}: {errors} inference rows contain errors")
    empty = int(predictions["raw_model_output"].fillna("").astype(str).str.strip().eq("").sum())
    if empty:
        _fail(f"{label}: {empty} inference rows have empty outputs")
    print(f"{label}: inference validation passed ({len(expected_ids)} unique rows, no errors)")


def check_evaluation(out: Path, expected_ids: set[str], label: str) -> None:
    results = pd.read_csv(out / "eval" / "results.csv", dtype={"qID": str})
    actual_ids = set(results["qID"])
    if len(results) != len(expected_ids):
        _fail(f"{label}: incomplete evaluation: {len(results)} != {len(expected_ids)}")
    if len(actual_ids) != len(expected_ids):
        _fail(f"{label}: evaluation IDs are not unique")
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        extra = sorted(actual_ids - expected_ids)[:5]
        _fail(f"{label}: evaluation ID mismatch; missing={missing} extra={extra}")
    print(f"{label}: evaluation validation passed ({len(expected_ids)} unique rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=TRACKS)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-rows", required=True, type=int)
    args = ap.parse_args()

    label = f"{args.track}/{args.dataset}"
    ground_truth = pd.read_parquet(
        parquet_path(args.track, args.split, args.dataset), columns=["id"]
    )
    if len(ground_truth) != args.expected_rows:
        _fail(
            f"{label}: expected-row constant is stale: "
            f"{args.expected_rows} != {len(ground_truth)}"
        )
    expected_ids = set(ground_truth["id"].astype(str))

    check_inference(args.out, expected_ids, label)
    check_evaluation(args.out, expected_ids, label)


if __name__ == "__main__":
    main()
