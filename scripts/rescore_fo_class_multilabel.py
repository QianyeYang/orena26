#!/usr/bin/env python
"""Rescore fo_class rows of a saved run through the CURRENT multi-label adapter.

The 2026-07-24 FRAME full-test eval predates the multi-label ``fo_class`` fix
(first-match reduction scored multi-object answers as single labels), so rows
whose raw output was already a correct label set were marked wrong. This
recomputes ``normalized_prediction`` for fo_class rows from the SAVED
``raw_model_output`` (no inference) via ``src.adapter.build_response`` and
rescores them with the official format's set-equality read; every other row
keeps its stored ``eval/results.csv`` correctness. CPU-only, judge-free —
fo_class scoring is deterministic.

Usage (training cluster, conda env orena, repo root):
    python scripts/rescore_fo_class_multilabel.py \
        track-frame/lora-finetune/logs/full-test-comparison/new-epoch30 \
        --datasets heico lapchole

Writes ``eval/rescore-results.csv`` + ``eval/rescore-summary.csv`` per dataset
(originals untouched) and prints old vs new bucket accuracies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from focus import get_format_class

from src.adapter import build_response, is_multi_select

FO_FORMAT = get_format_class("fo_class")()


def fo_class_correct(prediction: str, answer: str) -> bool:
    """Official fo_class correctness: parsed label-set equality."""
    try:
        return FO_FORMAT.read(str(prediction)) == FO_FORMAT.read(str(answer))
    except Exception:  # noqa: BLE001 — unparseable prediction scores wrong
        return False


def rescore_dataset(run_dir: Path, dataset: str) -> dict:
    ddir = run_dir / dataset
    preds = pd.read_parquet(ddir / "predictions.parquet")
    results = pd.read_csv(ddir / "eval" / "results.csv")
    results["qID"] = results["qID"].astype(str)
    preds["sample_id"] = preds["sample_id"].astype(str)

    merged = results.merge(
        preds[["sample_id", "question", "raw_model_output", "answer",
               "normalized_prediction"]],
        left_on="qID", right_on="sample_id", how="left", validate="one_to_one",
    )
    if merged["raw_model_output"].isna().any():
        missing = int(merged["raw_model_output"].isna().sum())
        raise RuntimeError(f"{dataset}: {missing} results rows have no prediction row")

    fo = merged["answer_format"] == "fo_class"
    new_norm, new_correct = [], []
    for row in merged[fo].itertuples():
        resp = build_response(
            row.qID, str(row.raw_model_output), "fo_class",
            multi=is_multi_select(str(row.question)),
        )
        new_norm.append(resp.content)
        new_correct.append(fo_class_correct(resp.content, row.answer))

    merged["correctness_new"] = merged["correctness"]
    merged.loc[fo, "rescored_prediction"] = new_norm
    merged.loc[fo, "correctness_new"] = new_correct

    changed = merged[fo & (merged["correctness"] != merged["correctness_new"])]
    out_results = merged.drop(columns=["sample_id"])
    out_results.to_csv(ddir / "eval" / "rescore-results.csv", index=False)

    def bucket_table(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        rows = []
        for name, sub in frame.groupby(col):
            rows.append({
                "level": col, "name": name, "count": len(sub),
                "acc_old": sub["correctness"].mean(),
                "acc_new": sub["correctness_new"].mean(),
            })
        return pd.DataFrame(rows)

    tables = [
        pd.DataFrame([{
            "level": "overall", "name": "MEAN", "count": len(merged),
            "acc_old": merged["correctness"].mean(),
            "acc_new": merged["correctness_new"].mean(),
        }]),
        bucket_table(merged, "answer_format"),
        bucket_table(merged, "primary"),
        bucket_table(merged, "ood"),
        bucket_table(merged, "clinical"),
    ]
    summary = pd.concat(tables, ignore_index=True)
    summary["delta"] = summary["acc_new"] - summary["acc_old"]
    summary.to_csv(ddir / "eval" / "rescore-summary.csv", index=False)

    n_fo = int(fo.sum())
    flipped_up = int((changed["correctness_new"] & ~changed["correctness"]).sum())
    flipped_down = int((~changed["correctness_new"] & changed["correctness"]).sum())
    print(f"\n== {dataset}: {len(merged)} rows, {n_fo} fo_class, "
          f"{flipped_up} flipped wrong->right, {flipped_down} right->wrong")
    show = summary[summary.level.isin(["overall", "answer_format"])]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return {
        "dataset": dataset, "rows": len(merged), "fo_rows": n_fo,
        "flipped_up": flipped_up, "flipped_down": flipped_down,
        "overall_old": float(merged["correctness"].mean()),
        "overall_new": float(merged["correctness_new"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path,
                    help="run dir containing <dataset>/{predictions.parquet,eval/results.csv}")
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"])
    a = ap.parse_args()

    stats = [rescore_dataset(a.run_dir, dataset) for dataset in a.datasets]
    print("\n== rescore complete ==")
    for s in stats:
        print(f"{s['dataset']}: overall {s['overall_old']:.4f} -> {s['overall_new']:.4f} "
              f"(+{s['flipped_up']}/-{s['flipped_down']} of {s['fo_rows']} fo_class rows)")


if __name__ == "__main__":
    main()
