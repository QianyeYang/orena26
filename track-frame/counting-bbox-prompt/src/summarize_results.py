#!/usr/bin/env python
"""Create reproducible distribution tables for the paired counting experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from focus.taxonomy import Capability


DATASETS = ("heico", "lapchole")


def _paired_video_ci(
    frame: pd.DataFrame,
    *,
    direct_col: str,
    experiment_col: str,
    seed: int = 42,
    samples: int = 10_000,
) -> tuple[float, float]:
    per_video = frame.groupby("video", sort=False)[[direct_col, experiment_col]].mean()
    differences = (
        per_video[experiment_col].to_numpy(dtype=float)
        - per_video[direct_col].to_numpy(dtype=float)
    )
    if len(differences) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True)
    boot = draws.mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _macro_accuracy(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("video", sort=False)[column].mean().mean())


def _subset_rows(
    frame: pd.DataFrame,
    dimensions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = dimensions[0] if len(dimensions) == 1 else dimensions
    for keys, part in frame.groupby(group_key, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        low, high = _paired_video_ci(
            part,
            direct_col="direct_correct",
            experiment_col="experiment_correct",
        )
        row = {name: value for name, value in zip(dimensions, keys)}
        direct_macro = _macro_accuracy(part, "direct_correct")
        experiment_macro = _macro_accuracy(part, "experiment_correct")
        row.update(
            {
                "count": len(part),
                "videos": part["video"].nunique(),
                "direct_accuracy": direct_macro,
                "bbox_accuracy": experiment_macro,
                "delta": experiment_macro - direct_macro,
                "paired_delta_ci_low": low,
                "paired_delta_ci_high": high,
                "direct_micro_accuracy": float(part["direct_correct"].mean()),
                "bbox_micro_accuracy": float(part["experiment_correct"].mean()),
            }
        )
        rows.append(row)
    return rows


def _answer_bucket(value: Any) -> str:
    number = int(value)
    return str(number) if number <= 5 else "6+"


def _capability_group(value: str) -> str:
    capability = Capability.from_any(value)
    if capability is None:
        raise ValueError(f"unrecognized capability: {value}")
    group = capability.group
    return str(group.value if hasattr(group, "value") else group)


def _read_dataset(
    baseline_root: Path,
    experiment_root: Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline_dir = baseline_root / dataset
    experiment_dir = experiment_root / dataset

    baseline_predictions = pd.read_parquet(baseline_dir / "predictions.parquet")
    experiment_predictions = pd.read_parquet(experiment_dir / "predictions.parquet")
    baseline_results = pd.read_csv(baseline_dir / "eval" / "results.csv")
    experiment_results = pd.read_csv(experiment_dir / "eval" / "results.csv")

    baseline_predictions["sample_id"] = baseline_predictions["sample_id"].astype(str)
    experiment_predictions["sample_id"] = experiment_predictions["sample_id"].astype(str)
    baseline_results["qID"] = baseline_results["qID"].astype(str)
    experiment_results["qID"] = experiment_results["qID"].astype(str)

    if len(baseline_predictions) != len(experiment_predictions):
        raise ValueError(f"{dataset}: baseline/experiment prediction counts differ")
    if set(baseline_predictions["sample_id"]) != set(experiment_predictions["sample_id"]):
        raise ValueError(f"{dataset}: baseline/experiment prediction IDs differ")
    if set(baseline_results["qID"]) != set(experiment_results["qID"]):
        raise ValueError(f"{dataset}: baseline/experiment result IDs differ")

    direct_correct = baseline_results.set_index("qID")["correctness"].astype(bool)
    evaluator_correct = experiment_results.set_index("qID")["correctness"].astype(bool)
    changed = experiment_predictions["changed_by_prompt"].fillna(False).astype(bool)
    changed_ids = set(experiment_predictions.loc[changed, "sample_id"])
    unchanged_ids = set(experiment_predictions.loc[~changed, "sample_id"])
    isolated_correct = direct_correct.copy()
    isolated_correct.loc[list(changed_ids)] = evaluator_correct.loc[list(changed_ids)]

    unchanged_direct = direct_correct.loc[list(unchanged_ids)]
    unchanged_rerun = evaluator_correct.loc[list(unchanged_ids)]
    judge_mismatch_mask = unchanged_direct.ne(unchanged_rerun)
    judge_audit = {
        "dataset": dataset,
        "unchanged_rows": len(unchanged_ids),
        "judge_mismatches": int(judge_mismatch_mask.sum()),
        "direct_only_correct": int((unchanged_direct & ~unchanged_rerun).sum()),
        "rerun_only_correct": int((~unchanged_direct & unchanged_rerun).sum()),
    }

    counting = experiment_predictions.loc[changed].copy()
    counting["direct_correct"] = (
        counting["sample_id"].map(direct_correct).astype(bool)
    )
    counting["experiment_correct"] = (
        counting["sample_id"].map(isolated_correct).astype(bool)
    )
    counting["dataset"] = dataset
    counting["answer_count_bucket"] = counting["answer"].map(_answer_bucket)
    counting["question_type"] = counting["counting_mode"].astype(str)
    target_mask = counting["counting_mode"].eq("target")
    counting.loc[target_mask, "question_type"] = (
        "target:" + counting.loc[target_mask, "counting_target"].astype(str)
    )

    baseline_summary = pd.read_csv(baseline_dir / "eval" / "summary.csv")
    experiment_summary = pd.read_csv(experiment_dir / "eval" / "summary.csv")
    keys = ["level", "name"]
    baseline_summary = baseline_summary[
        baseline_summary["level"].isin(("leaf", "group", "answer_format", "overall"))
    ]
    experiment_summary = experiment_summary[
        experiment_summary["level"].isin(("leaf", "group", "answer_format", "overall"))
    ]
    paired_summary = baseline_summary.merge(
        experiment_summary,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_direct", "_bbox_evaluator"),
    )
    paired_summary.insert(0, "dataset", dataset)
    paired_summary["delta_evaluator"] = (
        paired_summary["accuracy_bbox_evaluator"]
        - paired_summary["accuracy_direct"]
    )

    full = baseline_results[
        ["qID", "video", "primary", "answer_format", "correctness"]
    ].rename(
        columns={"correctness": "direct_correct"}
    )
    full = full.merge(
        experiment_results[["qID", "correctness"]].rename(
            columns={"correctness": "experiment_correct_evaluator"}
        ),
        on="qID",
        validate="one_to_one",
    )
    full["experiment_correct"] = full["qID"].map(isolated_correct).astype(bool)
    full["changed_by_prompt"] = full["qID"].isin(changed_ids)
    full["group"] = full["primary"].map(_capability_group)
    full["dataset"] = dataset

    isolated_accuracies: list[float] = []
    for row in paired_summary.itertuples(index=False):
        if row.level == "overall":
            part = full
        elif row.level == "answer_format":
            part = full[full["answer_format"].eq(row.name)]
        elif row.level == "leaf":
            part = full[full["primary"].eq(row.name)]
        elif row.level == "group":
            part = full[full["group"].eq(row.name)]
        else:
            raise ValueError(f"unsupported summary level: {row.level}")
        if part.empty:
            raise ValueError(f"{dataset}: empty isolated summary for {row.level}/{row.name}")
        isolated_accuracies.append(_macro_accuracy(part, "experiment_correct"))
    paired_summary["accuracy_bbox_isolated"] = isolated_accuracies
    paired_summary["delta_isolated"] = (
        paired_summary["accuracy_bbox_isolated"]
        - paired_summary["accuracy_direct"]
    )
    return counting, paired_summary, full, judge_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    counting_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    full_parts: list[pd.DataFrame] = []
    judge_audits: list[dict[str, Any]] = []
    for dataset in DATASETS:
        counting, paired_summary, full, judge_audit = _read_dataset(
            baseline_root, experiment_root, dataset
        )
        counting_parts.append(counting)
        summary_parts.append(paired_summary)
        full_parts.append(full)
        judge_audits.append(judge_audit)

    counting = pd.concat(counting_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    full = pd.concat(full_parts, ignore_index=True)
    counting["gold_count"] = pd.to_numeric(counting["answer"], errors="raise")
    counting["direct_count"] = pd.to_numeric(
        counting["baseline_prediction"], errors="raise"
    )
    counting["bbox_count"] = pd.to_numeric(counting["prediction"], errors="raise")

    summary[summary["level"].eq("overall")].to_csv(
        out / "dataset-overall.csv", index=False
    )
    summary[summary["level"].isin(("group", "leaf"))].to_csv(
        out / "capability.csv", index=False
    )
    summary[summary["level"].eq("answer_format")].to_csv(
        out / "answer-format.csv", index=False
    )
    pd.DataFrame(judge_audits).to_csv(out / "judge-rerun-audit.csv", index=False)

    pd.DataFrame(_subset_rows(counting, ["dataset"])).to_csv(
        out / "counting-overall.csv", index=False
    )
    pd.DataFrame(_subset_rows(counting, ["dataset", "counting_mode"])).to_csv(
        out / "counting-mode.csv", index=False
    )
    pd.DataFrame(_subset_rows(counting, ["dataset", "question_type"])).to_csv(
        out / "question-type.csv", index=False
    )
    pd.DataFrame(
        _subset_rows(counting, ["dataset", "answer_count_bucket"])
    ).to_csv(out / "answer-count-bucket.csv", index=False)

    bias_rows: list[dict[str, Any]] = []
    for (dataset, mode), part in counting.groupby(
        ["dataset", "counting_mode"], sort=False
    ):
        direct_error = part["direct_count"] - part["gold_count"]
        bbox_error = part["bbox_count"] - part["gold_count"]
        bias_rows.append(
            {
                "dataset": dataset,
                "counting_mode": mode,
                "count": len(part),
                "gold_mean": float(part["gold_count"].mean()),
                "direct_prediction_mean": float(part["direct_count"].mean()),
                "bbox_prediction_mean": float(part["bbox_count"].mean()),
                "direct_mae": float(direct_error.abs().mean()),
                "bbox_mae": float(bbox_error.abs().mean()),
                "direct_undercount_rate": float(direct_error.lt(0).mean()),
                "bbox_undercount_rate": float(bbox_error.lt(0).mean()),
                "direct_overcount_rate": float(direct_error.gt(0).mean()),
                "bbox_overcount_rate": float(bbox_error.gt(0).mean()),
            }
        )
    pd.DataFrame(bias_rows).to_csv(out / "count-bias.csv", index=False)

    paired_outcomes: list[dict[str, Any]] = []
    for dataset, part in counting.groupby("dataset", sort=False):
        direct = part["direct_correct"]
        experiment = part["experiment_correct"]
        paired_outcomes.append(
            {
                "dataset": dataset,
                "count": len(part),
                "both_correct": int((direct & experiment).sum()),
                "bbox_only_correct": int((~direct & experiment).sum()),
                "direct_only_correct": int((direct & ~experiment).sum()),
                "both_wrong": int((~direct & ~experiment).sum()),
            }
        )
    pd.DataFrame(paired_outcomes).to_csv(out / "paired-outcomes.csv", index=False)

    status = (
        counting.groupby(
            ["dataset", "structured_output_status"],
            dropna=False,
            sort=False,
        )
        .agg(
            count=("sample_id", "size"),
            schema_valid=("structured_output_valid", "sum"),
            bbox_correct=("experiment_correct", "sum"),
        )
        .reset_index()
    )
    status["rate"] = status["count"] / status.groupby("dataset")["count"].transform("sum")
    status["bbox_micro_accuracy"] = status["bbox_correct"] / status["count"]
    status.to_csv(out / "structured-output-status.csv", index=False)

    runtime_rows: list[dict[str, Any]] = []
    for dataset, part in counting.groupby("dataset", sort=False):
        latency = part["latency_sec"].astype(float)
        runtime_rows.append(
            {
                "dataset": dataset,
                "count": len(part),
                "schema_valid_count": int(
                    part["structured_output_valid"].fillna(False).sum()
                ),
                "schema_valid_rate": float(
                    part["structured_output_valid"].fillna(False).mean()
                ),
                "errors": int(part["error"].fillna("").astype(str).ne("").sum()),
                "latency_median_sec": float(latency.median()),
                "latency_p95_sec": float(latency.quantile(0.95)),
                "latency_max_sec": float(latency.max()),
                "latency_over_5_sec": int(latency.gt(5.0).sum()),
            }
        )
    pd.DataFrame(runtime_rows).to_csv(out / "runtime.csv", index=False)

    overall_rows: list[dict[str, Any]] = []
    for dataset, part in full.groupby("dataset", sort=False):
        direct_macro = _macro_accuracy(part, "direct_correct")
        bbox_macro = _macro_accuracy(part, "experiment_correct")
        low, high = _paired_video_ci(
            part,
            direct_col="direct_correct",
            experiment_col="experiment_correct",
        )
        overall_rows.append(
            {
                "dataset": dataset,
                "count": len(part),
                "videos": part["video"].nunique(),
                "direct_accuracy": direct_macro,
                "bbox_accuracy": bbox_macro,
                "delta": bbox_macro - direct_macro,
                "paired_delta_ci_low": low,
                "paired_delta_ci_high": high,
                "direct_micro_accuracy": float(part["direct_correct"].mean()),
                "bbox_micro_accuracy": float(part["experiment_correct"].mean()),
            }
        )
    pd.DataFrame(overall_rows).to_csv(out / "paired-overall.csv", index=False)

    provenance = {
        "baseline_root": str(baseline_root),
        "experiment_root": str(experiment_root),
        "datasets": list(DATASETS),
        "counting_rows": len(counting),
        "bootstrap": {
            "unit": "video",
            "samples": 10_000,
            "seed": 42,
            "interval": "percentile 95%",
        },
    }
    with (out / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(f"wrote comparison tables for {len(counting)} counting rows -> {out}")


if __name__ == "__main__":
    main()
