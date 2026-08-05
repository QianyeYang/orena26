#!/usr/bin/env python
"""Build the epoch-8 Procedure full-test result-summary tables.

The official evaluator summaries remain authoritative for primary capability,
answer-format, and overall scores. Additional tables are descriptive slices of
the same evaluated rows and use the same mean-of-per-video-means point metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    REPO
    / "track-procedure/lora-finetune/logs/"
    / "Qwen3-VL-4B-Instruct-both-official/full_test_epoch_8"
)
OUT_ROOT = (
    REPO
    / "result-summary/procedure/"
    / "qwen3-vl-4b-lora-both-official-epoch8-full-test"
)

DATASETS = ("heico", "lapchole")

GROUPS = (
    ("1", "object_recognition"),
    ("2", "temporal_grounding"),
    ("3", "aggregation"),
    ("4", "event_understanding"),
    ("5", "complex_reasoning"),
)

LEAVES = (
    ("1a", "object_identification", "object_recognition"),
    ("1b", "instance_matching", "object_recognition"),
    ("1c", "object_attributes", "object_recognition"),
    ("1d", "spatial_localization_camera", "object_recognition"),
    ("1e", "spatial_localization_situs", "object_recognition"),
    ("2a", "temporal_localization", "temporal_grounding"),
    ("2b", "duration_estimation", "temporal_grounding"),
    ("3a", "object_aggregation", "aggregation"),
    ("3b", "event_aggregation", "aggregation"),
    ("4a", "fo_interaction_recognition", "event_understanding"),
    ("4b", "fo_usage_purpose", "event_understanding"),
    ("4c", "temporal_ordering", "event_understanding"),
    ("5a", "functional_reasoning", "complex_reasoning"),
    ("5b", "causal_consequence_reasoning", "complex_reasoning"),
    ("5c", "multi_step_reasoning", "complex_reasoning"),
)

CODE_TO_LEAF = {code: name for code, name, _ in LEAVES}
LEAF_TO_CODE = {name: code for code, name, _ in LEAVES}
LEAF_TO_GROUP = {name: group for _, name, group in LEAVES}
GROUP_TO_CODE = {name: code for code, name in GROUPS}

TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]?\d):([0-5]?\d)(?!\d)")
MULTI_SELECT_RE = re.compile(
    r"select (?:none, )?one or multiple answers?", re.IGNORECASE
)


def annotation_path(dataset: str) -> Path:
    if dataset == "heico":
        return REPO / "data/parquet/procedure/test/0000.parquet"
    return REPO / "data/parquet/lapchole/procedure/test/0000.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(part).strip() for part in value if str(part).strip()]
    if pd.isna(value):
        return []
    return [str(value).strip()]


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_parquet(annotation_path(dataset)).copy()
    annotations["qID"] = annotations["id"].astype(str)
    annotations["secondary_capabilities"] = annotations[
        "secondary_capabilities"
    ].apply(as_list)

    results_path = RUN_ROOT / dataset / "eval/results.csv"
    results = pd.read_csv(results_path, dtype={"qID": str})
    results = results.rename(
        columns={
            "video": "evaluated_video",
            "ood": "evaluated_ood",
            "clinical": "evaluated_clinical",
            "primary": "evaluated_primary",
            "answer_format": "evaluated_answer_format",
            "latency": "evaluated_latency_sec",
        }
    )

    prediction_columns = [
        "sample_id",
        "raw_model_output",
        "normalized_prediction",
        "prediction",
        "n_frames",
        "latency_sec",
        "error",
    ]
    predictions = pd.read_parquet(
        RUN_ROOT / dataset / "predictions.parquet", columns=prediction_columns
    ).copy()
    predictions["qID"] = predictions["sample_id"].astype(str)
    predictions = predictions.drop(columns=["sample_id"])

    merged = annotations.merge(results, on="qID", validate="one_to_one")
    merged = merged.merge(predictions, on="qID", validate="one_to_one")
    merged["dataset"] = dataset
    merged["primary"] = merged["primary_capability"].map(CODE_TO_LEAF)
    merged["group"] = merged["primary"].map(LEAF_TO_GROUP)

    if len(merged) != len(annotations) or len(merged) != len(results):
        raise RuntimeError(f"{dataset}: incomplete annotation/evaluation join")
    if merged["qID"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate qIDs after join")
    if not merged["primary"].eq(merged["evaluated_primary"]).all():
        raise RuntimeError(f"{dataset}: primary capability mismatch")
    if not merged["answer_format"].eq(merged["evaluated_answer_format"]).all():
        raise RuntimeError(f"{dataset}: answer-format mismatch")
    if not merged["video"].eq(merged["evaluated_video"]).all():
        raise RuntimeError(f"{dataset}: video mismatch")
    if not merged["ood"].eq(merged["evaluated_ood"]).all():
        raise RuntimeError(f"{dataset}: OOD flag mismatch")
    if not merged["clinical_relevance"].eq(merged["evaluated_clinical"]).all():
        raise RuntimeError(f"{dataset}: clinical flag mismatch")
    if merged["error"].fillna("").astype(str).str.len().gt(0).any():
        raise RuntimeError(f"{dataset}: inference errors are present")

    summary = pd.read_csv(RUN_ROOT / dataset / "eval/summary.csv")
    return merged, summary


def point_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "videos": 0,
            "correct": 0,
            "micro_accuracy": np.nan,
            "video_macro_accuracy": np.nan,
        }
    return {
        "rows": len(frame),
        "videos": frame["video"].nunique(),
        "correct": int(frame["correctness"].sum()),
        "micro_accuracy": float(frame["correctness"].mean()),
        "video_macro_accuracy": float(
            frame.groupby("video", sort=False)["correctness"].mean().mean()
        ),
    }


def official_row(
    summary: pd.DataFrame, level: str, name: str
) -> dict[str, float] | None:
    match = summary[(summary["level"] == level) & (summary["name"] == name)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "official_video_macro_accuracy": float(row["accuracy"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
    }


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(
        OUT_ROOT / name,
        index=False,
        float_format="%.9f",
        na_rep="",
    )


def parse_timestamps(value: Any) -> list[int]:
    output: list[int] = []
    for hour, minute, second in TIME_RE.findall(str(value)):
        output.append(int(hour) * 3600 + int(minute) * 60 + int(second))
    return sorted(output)


def parse_numeric(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_fo_set(value: Any) -> frozenset[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1 and parts[0].lower() == "none":
        return frozenset()
    return frozenset(part.lower() for part in parts)


def set_precision(gold: frozenset[str], pred: frozenset[str]) -> float:
    if not pred:
        return float(not gold)
    return len(gold & pred) / len(pred)


def set_recall(gold: frozenset[str], pred: frozenset[str]) -> float:
    if not gold:
        return float(not pred)
    return len(gold & pred) / len(gold)


def build_dataset_overall(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        metrics = point_metrics(frame)
        official = official_row(summaries[dataset], "overall", "MEAN")
        pre = summaries[dataset].loc[
            summaries[dataset]["level"] == "pre_evaluation", "accuracy"
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                **metrics,
                **(official or {}),
                "pre_evaluation_score": float(pre),
                "clinical_rows": int(frame["clinical_relevance"].sum()),
                "ood_rows": int(frame["ood"].sum()),
                "timed_out_rows": int(frame["timed_out"].sum()),
                "inference_error_rows": int(
                    frame["error"].fillna("").astype(str).str.len().gt(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_capability_groups(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for code, group in GROUPS:
            subset = frame[frame["group"] == group]
            metrics = point_metrics(subset)
            official = official_row(summaries[dataset], "group", group)
            rows.append(
                {
                    "dataset": dataset,
                    "group_code": code,
                    "capability_group": group,
                    "row_share": len(subset) / len(frame),
                    **metrics,
                    **(official or {}),
                }
            )
    return pd.DataFrame(rows)


def build_capability_leaves(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for code, leaf, group in LEAVES:
            subset = frame[frame["primary"] == leaf]
            metrics = point_metrics(subset)
            official = official_row(summaries[dataset], "leaf", leaf)
            rows.append(
                {
                    "dataset": dataset,
                    "capability_code": code,
                    "capability_group": group,
                    "primary_capability": leaf,
                    "row_share": len(subset) / len(frame),
                    **metrics,
                    **(official or {}),
                }
            )
    return pd.DataFrame(rows)


def build_capability_gap(
    groups: pd.DataFrame, leaves: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for level, table, name_column in (
        ("group", groups, "capability_group"),
        ("leaf", leaves, "primary_capability"),
    ):
        for name, slice_frame in table.groupby(name_column, sort=False):
            by_dataset = slice_frame.set_index("dataset")
            heico = by_dataset.loc["heico"]
            lapchole = by_dataset.loc["lapchole"]
            if not heico["rows"] or not lapchole["rows"]:
                delta = np.nan
            else:
                delta = (
                    lapchole["official_video_macro_accuracy"]
                    - heico["official_video_macro_accuracy"]
                )
            rows.append(
                {
                    "level": level,
                    "capability": name,
                    "heico_rows": int(heico["rows"]),
                    "lapchole_rows": int(lapchole["rows"]),
                    "heico_official_video_macro_accuracy": heico.get(
                        "official_video_macro_accuracy", np.nan
                    ),
                    "lapchole_official_video_macro_accuracy": lapchole.get(
                        "official_video_macro_accuracy", np.nan
                    ),
                    "lapchole_minus_heico": delta,
                }
            )
    return pd.DataFrame(rows)


def build_answer_formats(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    formats = sorted(set().union(*(set(f["answer_format"]) for f in frames.values())))
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for answer_format in formats:
            subset = frame[frame["answer_format"] == answer_format]
            metrics = point_metrics(subset)
            official = official_row(
                summaries[dataset], "answer_format", answer_format
            )
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": answer_format,
                    "row_share": len(subset) / len(frame),
                    **metrics,
                    **(official or {}),
                }
            )
    return pd.DataFrame(rows)


def build_capability_answer_format(
    frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for (leaf, answer_format), subset in frame.groupby(
            ["primary", "answer_format"], sort=False
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "capability_code": LEAF_TO_CODE[leaf],
                    "capability_group": LEAF_TO_GROUP[leaf],
                    "primary_capability": leaf,
                    "answer_format": answer_format,
                    **point_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def build_secondary_capabilities(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        exploded = frame.explode("secondary_capabilities")
        for code, leaf, group in LEAVES:
            subset = exploded[exploded["secondary_capabilities"] == code]
            rows.append(
                {
                    "dataset": dataset,
                    "capability_code": code,
                    "capability_group": group,
                    "secondary_capability": leaf,
                    **point_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def build_gap_decomposition(
    groups: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    indexed = groups.set_index(["dataset", "capability_group"])
    rows = []
    for _, group in GROUPS:
        heico = indexed.loc[("heico", group)]
        lapchole = indexed.loc[("lapchole", group)]
        p_h = heico["row_share"]
        p_l = lapchole["row_share"]
        a_h = heico["micro_accuracy"]
        a_l = lapchole["micro_accuracy"]
        rows.append(
            {
                "capability_group": group,
                "heico_row_share": p_h,
                "lapchole_row_share": p_l,
                "heico_micro_accuracy": a_h,
                "lapchole_micro_accuracy": a_l,
                "composition_contribution": (p_l - p_h) * (a_l + a_h) / 2,
                "within_group_performance_contribution": (a_l - a_h)
                * (p_l + p_h)
                / 2,
            }
        )
    output = pd.DataFrame(rows)
    total = {
        "capability_group": "TOTAL",
        "heico_row_share": output["heico_row_share"].sum(),
        "lapchole_row_share": output["lapchole_row_share"].sum(),
        "heico_micro_accuracy": frames["heico"]["correctness"].mean(),
        "lapchole_micro_accuracy": frames["lapchole"]["correctness"].mean(),
        "composition_contribution": output["composition_contribution"].sum(),
        "within_group_performance_contribution": output[
            "within_group_performance_contribution"
        ].sum(),
    }
    return pd.concat([output, pd.DataFrame([total])], ignore_index=True)


def build_slices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for dimension, column in (
            ("generation", "generation"),
            ("clinical_relevance", "clinical_relevance"),
            ("ood", "ood"),
        ):
            for value, subset in frame.groupby(column, dropna=False, sort=True):
                rows.append(
                    {
                        "dataset": dataset,
                        "dimension": dimension,
                        "value": str(value).lower()
                        if isinstance(value, (bool, np.bool_))
                        else str(value),
                        **point_metrics(subset),
                    }
                )
    return pd.DataFrame(rows)


def build_video_performance(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for video, subset in frames[dataset].groupby("video", sort=True):
            rows.append({"dataset": dataset, "video": video, **point_metrics(subset)})
    return pd.DataFrame(rows)


def latency_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    latency = subset["evaluated_latency_sec"].astype(float)
    return {
        "rows": len(subset),
        "mean_latency_sec": latency.mean(),
        "median_latency_sec": latency.median(),
        "p95_latency_sec": latency.quantile(0.95),
        "max_latency_sec": latency.max(),
        "over_5_sec_rows": int(latency.gt(5).sum()),
        "over_30_sec_rows": int(latency.gt(30).sum()),
        "timed_out_rows": int(subset["timed_out"].sum()),
    }


def build_runtime(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        rows.append(
            {
                "dataset": dataset,
                "scope": "dataset",
                "name": "all",
                **latency_metrics(frame),
            }
        )
        for group, subset in frame.groupby("group", sort=False):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "group",
                    "name": group,
                    **latency_metrics(subset),
                }
            )
        for leaf, subset in frame.groupby("primary", sort=False):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "leaf",
                    "name": leaf,
                    **latency_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def time_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_timestamps)
    normalized = subset["prediction"].apply(parse_timestamps)
    raw = subset["raw_model_output"].apply(parse_timestamps)
    gold_count = gold.str.len()
    normalized_count = normalized.str.len()
    raw_count = raw.str.len()
    single_mask = gold_count.eq(1) & normalized_count.eq(1)
    errors = pd.Series(
        [
            abs(g[0] - p[0]) if use else np.nan
            for g, p, use in zip(gold, normalized, single_mask, strict=True)
        ],
        index=subset.index,
        dtype=float,
    )
    return {
        **point_metrics(subset),
        "gold_multi_timestamp_rows": int(gold_count.gt(1).sum()),
        "raw_multi_timestamp_rows": int(raw_count.gt(1).sum()),
        "normalized_multi_timestamp_rows": int(normalized_count.gt(1).sum()),
        "raw_cardinality_match_rate": raw_count.eq(gold_count).mean(),
        "normalized_cardinality_match_rate": normalized_count.eq(gold_count).mean(),
        "single_gold_rows": int(gold_count.eq(1).sum()),
        "single_gold_with_parsed_prediction_rows": int(single_mask.sum()),
        "single_gold_median_absolute_error_sec": errors.median(),
        "single_gold_p90_absolute_error_sec": errors.quantile(0.90),
        "single_gold_within_5_sec_rate": errors.le(5).sum() / single_mask.sum()
        if single_mask.sum()
        else np.nan,
        "single_gold_within_30_sec_rate": errors.le(30).sum() / single_mask.sum()
        if single_mask.sum()
        else np.nan,
        "single_gold_within_60_sec_rate": errors.le(60).sum() / single_mask.sum()
        if single_mask.sum()
        else np.nan,
        "single_gold_within_300_sec_rate": errors.le(300).sum() / single_mask.sum()
        if single_mask.sum()
        else np.nan,
    }


def build_time_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        time_frame = frames[dataset][frames[dataset]["answer_format"] == "time"].copy()
        time_frame["gold_cardinality"] = time_frame["answer"].apply(
            lambda value: "multiple" if len(parse_timestamps(value)) > 1 else "single"
        )
        rows.append(
            {
                "dataset": dataset,
                "scope": "dataset",
                "name": "all_time",
                **time_metrics(time_frame),
            }
        )
        for leaf, subset in time_frame.groupby("primary", sort=False):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "primary_capability",
                    "name": leaf,
                    **time_metrics(subset),
                }
            )
        for cardinality, subset in time_frame.groupby("gold_cardinality", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "gold_cardinality",
                    "name": cardinality,
                    **time_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def timestamp_seconds(value: Any) -> int:
    parsed = parse_timestamps(value)
    if len(parsed) != 1:
        raise ValueError(f"expected one timestamp, got {value!r}")
    return parsed[0]


def compare_timestamps(gold: list[int], prediction: list[int], threshold: float) -> bool:
    return len(gold) == len(prediction) and all(
        abs(answer - predicted) <= threshold
        for answer, predicted in zip(sorted(gold), sorted(prediction), strict=True)
    )


def build_time_adapter_counterfactual(
    frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Re-score stored raw time text with the v2 all-timestamp normalizer.

    This isolates normalization only. It is not an exact container rerun because
    the v2 prompt could cause the model to generate different raw text.
    """
    rows = []
    for dataset in DATASETS:
        time_frame = frames[dataset][frames[dataset]["answer_format"] == "time"].copy()
        durations = time_frame.apply(
            lambda row: timestamp_seconds(row["timestamp_end"])
            - timestamp_seconds(row["timestamp_start"]),
            axis=1,
        )
        thresholds = durations.apply(lambda duration: min(5.0, 1 + duration * (4 / 360)))
        gold = time_frame["answer"].apply(parse_timestamps)
        recorded_prediction = time_frame["prediction"].apply(parse_timestamps)
        raw_all_timestamps = time_frame["raw_model_output"].apply(parse_timestamps)
        simulated_recorded = pd.Series(
            [
                compare_timestamps(g, p, threshold)
                for g, p, threshold in zip(
                    gold, recorded_prediction, thresholds, strict=True
                )
            ],
            index=time_frame.index,
        )
        if not simulated_recorded.eq(time_frame["correctness"]).all():
            mismatches = int(
                simulated_recorded.ne(time_frame["correctness"]).sum()
            )
            raise RuntimeError(
                f"{dataset}: {mismatches} recorded time results were not reproduced"
            )
        time_frame["v2_raw_normalizer_correctness"] = [
            compare_timestamps(g, p, threshold)
            for g, p, threshold in zip(gold, raw_all_timestamps, thresholds, strict=True)
        ]

        scopes: list[tuple[str, str, pd.DataFrame]] = [
            ("dataset", "all_time", time_frame)
        ]
        scopes.extend(
            ("primary_capability", leaf, subset)
            for leaf, subset in time_frame.groupby("primary", sort=False)
        )
        for scope, name, subset in scopes:
            recorded = point_metrics(subset)
            counterfactual = point_metrics(
                subset.assign(
                    correctness=subset["v2_raw_normalizer_correctness"]
                )
            )
            old = subset["correctness"].astype(bool)
            new = subset["v2_raw_normalizer_correctness"].astype(bool)
            rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "name": name,
                    "rows": len(subset),
                    "recorded_correct": int(old.sum()),
                    "posthoc_v2_normalizer_correct": int(new.sum()),
                    "recorded_micro_accuracy": recorded["micro_accuracy"],
                    "posthoc_v2_normalizer_micro_accuracy": counterfactual[
                        "micro_accuracy"
                    ],
                    "micro_delta": counterfactual["micro_accuracy"]
                    - recorded["micro_accuracy"],
                    "recorded_video_macro_accuracy": recorded[
                        "video_macro_accuracy"
                    ],
                    "posthoc_v2_normalizer_video_macro_accuracy": counterfactual[
                        "video_macro_accuracy"
                    ],
                    "video_macro_delta": counterfactual["video_macro_accuracy"]
                    - recorded["video_macro_accuracy"],
                    "newly_correct_rows": int((~old & new).sum()),
                    "newly_incorrect_rows": int((old & ~new).sum()),
                }
            )
    return pd.DataFrame(rows)


def numeric_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_numeric)
    pred = subset["prediction"].apply(parse_numeric)
    parsed = gold.notna() & pred.notna()
    gold_values = gold[parsed].astype(float)
    pred_values = pred[parsed].astype(float)
    error = pred_values - gold_values
    return {
        **point_metrics(subset),
        "parsed_rows": int(parsed.sum()),
        "gold_mean": gold_values.mean(),
        "prediction_mean": pred_values.mean(),
        "mean_absolute_error": error.abs().mean(),
        "median_absolute_error": error.abs().median(),
        "undercount_rate": error.lt(0).mean(),
        "exact_numeric_rate": error.eq(0).mean(),
        "overcount_rate": error.gt(0).mean(),
    }


def number_bucket(value: Any) -> str:
    number = parse_numeric(value)
    if number is None:
        return "unparsed"
    integer = int(number)
    return str(integer) if integer <= 5 else "6+"


def build_numeric_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for answer_format in ("number", "percentage"):
            numeric = frame[frame["answer_format"] == answer_format].copy()
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": answer_format,
                    "scope": "dataset",
                    "name": "all",
                    **numeric_metrics(numeric),
                }
            )
            for leaf, subset in numeric.groupby("primary", sort=False):
                rows.append(
                    {
                        "dataset": dataset,
                        "answer_format": answer_format,
                        "scope": "primary_capability",
                        "name": leaf,
                        **numeric_metrics(subset),
                    }
                )
            if answer_format == "number":
                numeric["gold_bucket"] = numeric["answer"].apply(number_bucket)
                order = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6+": 6}
                for bucket, subset in sorted(
                    numeric.groupby("gold_bucket"), key=lambda item: order.get(item[0], 99)
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "answer_format": answer_format,
                            "scope": "gold_value_bucket",
                            "name": bucket,
                            **numeric_metrics(subset),
                        }
                    )
    return pd.DataFrame(rows)


def fo_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_fo_set)
    pred = subset["prediction"].apply(parse_fo_set)
    return {
        **point_metrics(subset),
        "cardinality_match_rate": pd.Series(
            [len(g) == len(p) for g, p in zip(gold, pred, strict=True)],
            index=subset.index,
        ).mean(),
        "mean_set_precision": np.mean(
            [set_precision(g, p) for g, p in zip(gold, pred, strict=True)]
        ),
        "mean_set_recall": np.mean(
            [set_recall(g, p) for g, p in zip(gold, pred, strict=True)]
        ),
    }


def build_fo_diagnostics(
    frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cardinality_rows = []
    label_rows = []
    for dataset in DATASETS:
        frame = frames[dataset][frames[dataset]["answer_format"] == "fo_class"].copy()
        frame["gold_set"] = frame["answer"].apply(parse_fo_set)
        frame["pred_set"] = frame["prediction"].apply(parse_fo_set)
        frame["gold_cardinality"] = frame["gold_set"].apply(
            lambda value: "none" if len(value) == 0 else "single" if len(value) == 1 else "multiple"
        )
        cardinality_rows.append(
            {
                "dataset": dataset,
                "scope": "dataset",
                "name": "all_fo_class",
                **fo_metrics(frame),
            }
        )
        for cardinality, subset in frame.groupby("gold_cardinality", sort=True):
            cardinality_rows.append(
                {
                    "dataset": dataset,
                    "scope": "gold_cardinality",
                    "name": cardinality,
                    **fo_metrics(subset),
                }
            )
        labels = sorted(set().union(*frame["gold_set"], *frame["pred_set"]))
        for label in labels:
            gold_positive = frame["gold_set"].apply(lambda value: label in value)
            pred_positive = frame["pred_set"].apply(lambda value: label in value)
            support = int(gold_positive.sum())
            predicted = int(pred_positive.sum())
            true_positive = int((gold_positive & pred_positive).sum())
            label_rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "gold_support_rows": support,
                    "predicted_positive_rows": predicted,
                    "true_positive_rows": true_positive,
                    "label_recall": true_positive / support if support else np.nan,
                    "label_precision": true_positive / predicted if predicted else np.nan,
                    "exact_row_accuracy_on_gold_positive_rows": frame.loc[
                        gold_positive, "correctness"
                    ].mean()
                    if support
                    else np.nan,
                }
            )
    return pd.DataFrame(cardinality_rows), pd.DataFrame(label_rows)


def build_categorical_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        binary = frame[frame["answer_format"] == "binary"].copy()
        binary["gold_value"] = binary["answer"].str.strip().str.lower()
        for value, subset in binary.groupby("gold_value", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": "binary",
                    "subtype": "gold_value",
                    "value": value,
                    **point_metrics(subset),
                }
            )

        multiple_choice = frame[frame["answer_format"] == "multiple_choice"].copy()
        multiple_choice["selection_type"] = multiple_choice["question"].apply(
            lambda value: "multi_select"
            if MULTI_SELECT_RE.search(str(value))
            else "single_select"
        )
        for value, subset in multiple_choice.groupby("selection_type", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": "multiple_choice",
                    "subtype": "selection_type",
                    "value": value,
                    **point_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def build_provenance(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    manifest_path = RUN_ROOT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_files: dict[str, str] = {
        str(manifest_path.relative_to(REPO)): sha256(manifest_path)
    }
    for dataset in DATASETS:
        for path in (
            RUN_ROOT / dataset / "eval/results.csv",
            RUN_ROOT / dataset / "eval/summary.csv",
            RUN_ROOT / dataset / "predictions.parquet",
            annotation_path(dataset),
        ):
            source_files[str(path.relative_to(REPO))] = sha256(path)

    return {
        "report": "Procedure Qwen3-VL-4B LoRA both-official epoch-8 full test",
        "generated_on": "2026-07-31",
        "source_run": str(RUN_ROOT.relative_to(REPO)),
        "run_manifest": manifest,
        "orena_focus_version": "0.3.4",
        "datasets": {
            dataset: {
                "rows": len(frames[dataset]),
                "videos": frames[dataset]["video"].nunique(),
                "correct": int(frames[dataset]["correctness"].sum()),
            }
            for dataset in DATASETS
        },
        "official_metric": {
            "point_estimate": "mean of per-video question-level correctness means",
            "confidence_interval": (
                "official two-level hierarchical bootstrap: resample videos, then "
                "questions within video; 1000 samples, seed 42, percentile 95%"
            ),
            "primary_source": "each dataset's eval/summary.csv",
            "judge_formats": ["open_ended", "multiple_choice"],
            "judge_model": "os-models/Qwen3.5-4B",
            "procedure_timeout_seconds": 30,
        },
        "descriptive_tables": {
            "point_estimate": "mean of per-video question-level correctness means",
            "confidence_intervals": "not computed; main official slices retain evaluator intervals",
            "secondary_capability_note": "overlapping tags; not an official leaderboard slice",
            "gap_decomposition_note": (
                "symmetric exact decomposition of the question-micro LapChole-minus-HeiCo "
                "gap into capability-group composition and within-group performance"
            ),
        },
        "coverage_notes": {
            "ood_rows": int(sum(frame["ood"].sum() for frame in frames.values())),
            "object_attributes_primary_rows": int(
                sum((frame["primary"] == "object_attributes").sum() for frame in frames.values())
            ),
            "clinical_rows": int(
                sum(frame["clinical_relevance"].sum() for frame in frames.values())
            ),
            "multi_timestamp_reference_rows": int(
                sum(
                    frame.loc[frame["answer_format"] == "time", "answer"]
                    .apply(lambda value: len(parse_timestamps(value)) > 1)
                    .sum()
                    for frame in frames.values()
                )
            ),
            "time_v2_normalizer_posthoc_correctness_changes": 0,
            "time_v2_normalizer_note": (
                "Re-normalizing stored raw time outputs by retaining all timestamps "
                "changed no recorded correctness outcomes; this is not a v2 prompt/container rerun"
            ),
        },
        "source_sha256": source_files,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = {dataset: load_dataset(dataset) for dataset in DATASETS}
    frames = {dataset: loaded[dataset][0] for dataset in DATASETS}
    summaries = {dataset: loaded[dataset][1] for dataset in DATASETS}

    dataset_overall = build_dataset_overall(frames, summaries)
    capability_groups = build_capability_groups(frames, summaries)
    capability_leaves = build_capability_leaves(frames, summaries)
    answer_formats = build_answer_formats(frames, summaries)
    fo_cardinality, fo_labels = build_fo_diagnostics(frames)

    write_csv(dataset_overall, "dataset-overall.csv")
    write_csv(capability_groups, "capability-group.csv")
    write_csv(capability_leaves, "capability-leaf.csv")
    write_csv(
        build_capability_gap(capability_groups, capability_leaves),
        "capability-dataset-gap.csv",
    )
    write_csv(
        build_gap_decomposition(capability_groups, frames),
        "dataset-gap-decomposition.csv",
    )
    write_csv(answer_formats, "answer-format.csv")
    write_csv(
        build_capability_answer_format(frames), "capability-answer-format.csv"
    )
    write_csv(build_secondary_capabilities(frames), "secondary-capability.csv")
    write_csv(build_slices(frames), "slice-performance.csv")
    write_csv(build_video_performance(frames), "video-performance.csv")
    write_csv(build_runtime(frames), "runtime.csv")
    write_csv(build_time_diagnostics(frames), "time-diagnostics.csv")
    write_csv(
        build_time_adapter_counterfactual(frames),
        "time-adapter-counterfactual.csv",
    )
    write_csv(build_numeric_diagnostics(frames), "numeric-diagnostics.csv")
    write_csv(fo_cardinality, "fo-class-cardinality.csv")
    write_csv(fo_labels, "fo-class-label.csv")
    write_csv(build_categorical_diagnostics(frames), "categorical-diagnostics.csv")

    provenance = build_provenance(frames, summaries)
    (OUT_ROOT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(dataset_overall.to_string(index=False))
    print(f"\nWrote {len(list(OUT_ROOT.iterdir()))} artifacts to {OUT_ROOT}")


if __name__ == "__main__":
    main()
