#!/usr/bin/env python
"""Build the Segment epoch-6 submission-checkpoint subset report tables.

The source evaluation is the fixed, seeded 500-row official-test subset used
after each training epoch. It is not a full-test run. Official evaluator
summaries remain authoritative for video-macro scores and confidence intervals;
the additional tables are descriptive slices of the same evaluated rows.
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
RUN_BASE = (
    REPO
    / "track-segment/lora-finetune/logs/"
    / "Qwen3-VL-4B-Instruct-both-official"
)
RUN_ROOT = RUN_BASE / "eval_epoch_6"
OUT_ROOT = (
    REPO
    / "result-summary/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset"
)
SUBMISSION_ROOT = (
    REPO
    / "submissions/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-20260801"
)

DATASETS = ("heico", "lapchole")
FULL_TEST_ROWS = {"heico": 4000, "lapchole": 2254}

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

FO_NAMES = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
    "Mesh",
    "Absorbable Hemostatic Agent",
)

TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]?\d):([0-5]?\d)(?!\d)")
MULTI_SELECT_RE = re.compile(
    r"select (?:none, )?one or multiple answers?", re.IGNORECASE
)
MC_TAIL_RE = re.compile(
    r"select (?:one answer|(?:none, )?one or multiple answers?)[:\s]*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def annotation_path(dataset: str) -> Path:
    if dataset == "heico":
        return REPO / "data/parquet/segment/test/0000.parquet"
    return REPO / "data/parquet/lapchole/segment/test/0000.parquet"


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


def load_subset_ids() -> dict[str, set[str]]:
    records = json.loads((RUN_BASE / "eval_subset_qids.json").read_text())
    output = {dataset: set() for dataset in DATASETS}
    for record in records:
        output[str(record["dataset"])].add(str(record["qID"]))
    return output


def load_dataset(
    dataset: str, subset_ids: dict[str, set[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(RUN_ROOT / dataset / "predictions.parquet").copy()
    predictions["qID"] = predictions["sample_id"].astype(str)
    predictions = predictions.drop(columns=["sample_id"])
    predictions["secondary_capabilities"] = predictions[
        "secondary_capabilities"
    ].apply(as_list)

    results = pd.read_csv(
        RUN_ROOT / dataset / "eval/results.csv", dtype={"qID": str}
    ).rename(
        columns={
            "video": "evaluated_video",
            "ood": "evaluated_ood",
            "clinical": "evaluated_clinical",
            "primary": "evaluated_primary",
            "answer_format": "evaluated_answer_format",
            "latency": "evaluated_latency_sec",
        }
    )

    annotations = pd.read_parquet(
        annotation_path(dataset),
        columns=["id", "generation", "procedure_type", "track"],
    ).copy()
    annotations["qID"] = annotations["id"].astype(str)
    annotations = annotations.drop(columns=["id"])

    merged = predictions.merge(results, on="qID", validate="one_to_one")
    merged = merged.merge(annotations, on="qID", validate="one_to_one")
    merged["dataset"] = dataset
    merged["primary"] = merged["primary_capability"].map(CODE_TO_LEAF)
    merged["group"] = merged["primary"].map(LEAF_TO_GROUP)

    if set(merged["qID"]) != subset_ids[dataset]:
        raise RuntimeError(f"{dataset}: evaluated qIDs differ from fixed subset")
    if merged["qID"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate evaluated qIDs")
    if not merged["primary"].eq(merged["evaluated_primary"]).all():
        raise RuntimeError(f"{dataset}: primary capability mismatch")
    if not merged["answer_format"].eq(merged["evaluated_answer_format"]).all():
        raise RuntimeError(f"{dataset}: answer-format mismatch")
    if not merged["video"].eq(merged["evaluated_video"]).all():
        raise RuntimeError(f"{dataset}: video mismatch")
    if not merged["ood"].eq(merged["evaluated_ood"]).all():
        raise RuntimeError(f"{dataset}: OOD mismatch")
    if not merged["clinical_relevance"].eq(merged["evaluated_clinical"]).all():
        raise RuntimeError(f"{dataset}: clinical flag mismatch")
    if not np.allclose(
        merged["latency_sec"], merged["evaluated_latency_sec"], rtol=0, atol=1e-9
    ):
        raise RuntimeError(f"{dataset}: prediction/evaluator latency mismatch")
    if not merged["track"].eq("segment").all():
        raise RuntimeError(f"{dataset}: non-Segment annotation joined")
    if merged["primary"].isna().any() or merged["group"].isna().any():
        raise RuntimeError(f"{dataset}: unknown primary capability code")
    if merged["error"].fillna("").astype(str).str.len().gt(0).any():
        raise RuntimeError(f"{dataset}: inference errors are present")
    fo_prompts = merged.loc[merged["answer_format"].eq("fo_class"), "prompt"]
    if not fo_prompts.astype(str).str.contains("EXACTLY ONE", regex=False).all():
        raise RuntimeError(f"{dataset}: unexpected source FO-class prompt")

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


def build_dataset_overall(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        pre = summaries[dataset].loc[
            summaries[dataset]["level"] == "pre_evaluation", "accuracy"
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "full_test_rows": FULL_TEST_ROWS[dataset],
                "evaluated_fraction": len(frame) / FULL_TEST_ROWS[dataset],
                **point_metrics(frame),
                **(official_row(summaries[dataset], "overall", "MEAN") or {}),
                "pre_evaluation_score": float(pre),
                "clinical_rows": int(frame["clinical_relevance"].sum()),
                "ood_rows": int(frame["ood"].sum()),
                "timed_out_rows": int(frame["timed_out"].sum()),
                "inference_error_rows": int(
                    frame["error"].fillna("").astype(str).str.len().gt(0).sum()
                ),
                "empty_output_rows": int(
                    frame["raw_model_output"].fillna("").astype(str).str.strip().eq("").sum()
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
            rows.append(
                {
                    "dataset": dataset,
                    "group_code": code,
                    "capability_group": group,
                    "row_share": len(subset) / len(frame),
                    **point_metrics(subset),
                    **(official_row(summaries[dataset], "group", group) or {}),
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
            rows.append(
                {
                    "dataset": dataset,
                    "capability_code": code,
                    "capability_group": group,
                    "primary_capability": leaf,
                    "row_share": len(subset) / len(frame),
                    **point_metrics(subset),
                    **(official_row(summaries[dataset], "leaf", leaf) or {}),
                }
            )
    return pd.DataFrame(rows)


def build_answer_formats(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    formats = sorted(set().union(*(set(frame["answer_format"]) for frame in frames.values())))
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for answer_format in formats:
            subset = frame[frame["answer_format"] == answer_format]
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": answer_format,
                    "row_share": len(subset) / len(frame),
                    **point_metrics(subset),
                    **(
                        official_row(
                            summaries[dataset], "answer_format", answer_format
                        )
                        or {}
                    ),
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
        exploded = frames[dataset].explode("secondary_capabilities")
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


def build_capability_gap(
    groups: pd.DataFrame, leaves: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for level, table, name_column in (
        ("group", groups, "capability_group"),
        ("leaf", leaves, "primary_capability"),
    ):
        for name, subset in table.groupby(name_column, sort=False):
            indexed = subset.set_index("dataset")
            heico = indexed.loc["heico"]
            lapchole = indexed.loc["lapchole"]
            h_score = heico.get("official_video_macro_accuracy", np.nan)
            l_score = lapchole.get("official_video_macro_accuracy", np.nan)
            rows.append(
                {
                    "level": level,
                    "capability": name,
                    "heico_rows": int(heico["rows"]),
                    "lapchole_rows": int(lapchole["rows"]),
                    "heico_official_video_macro_accuracy": h_score,
                    "lapchole_official_video_macro_accuracy": l_score,
                    "lapchole_minus_heico": l_score - h_score
                    if heico["rows"] and lapchole["rows"]
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


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
    latency = subset["latency_sec"].astype(float)
    frames = subset["n_frames"].astype(int)
    return {
        "rows": len(subset),
        "mean_latency_sec": latency.mean(),
        "median_latency_sec": latency.median(),
        "p95_latency_sec": latency.quantile(0.95),
        "max_latency_sec": latency.max(),
        "over_5_sec_rows": int(latency.gt(5).sum()),
        "over_15_sec_rows": int(latency.gt(15).sum()),
        "timed_out_rows": int(subset["timed_out"].sum()),
        "inference_error_rows": int(
            subset["error"].fillna("").astype(str).str.len().gt(0).sum()
        ),
        "mean_frames": frames.mean(),
        "median_frames": frames.median(),
        "max_frames": frames.max(),
        "rows_at_64_frame_cap": int(frames.eq(64).sum()),
    }


def build_runtime(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        rows.append(
            {"dataset": dataset, "scope": "dataset", "name": "all", **latency_metrics(frame)}
        )
        for answer_format, subset in frame.groupby("answer_format", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "answer_format",
                    "name": answer_format,
                    **latency_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def parse_timestamps(value: Any) -> list[int]:
    return sorted(
        int(hour) * 3600 + int(minute) * 60 + int(second)
        for hour, minute, second in TIME_RE.findall(str(value))
    )


def parse_numeric(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_fo_set(value: Any) -> frozenset[str]:
    parts = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1 and parts[0] == "none":
        return frozenset()
    return frozenset(parts)


def set_precision(gold: frozenset[str], pred: frozenset[str]) -> float:
    return len(gold & pred) / len(pred) if pred else float(not gold)


def set_recall(gold: frozenset[str], pred: frozenset[str]) -> float:
    return len(gold & pred) / len(gold) if gold else float(not pred)


def time_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_timestamps)
    pred = subset["prediction"].apply(parse_timestamps)
    raw = subset["raw_model_output"].apply(parse_timestamps)
    gold_count = gold.str.len()
    pred_count = pred.str.len()
    raw_count = raw.str.len()
    single_mask = gold_count.eq(1) & pred_count.eq(1)
    errors = pd.Series(
        [
            abs(g[0] - p[0]) if use else np.nan
            for g, p, use in zip(gold, pred, single_mask, strict=True)
        ],
        index=subset.index,
        dtype=float,
    )
    return {
        **point_metrics(subset),
        "gold_multi_timestamp_rows": int(gold_count.gt(1).sum()),
        "raw_multi_timestamp_rows": int(raw_count.gt(1).sum()),
        "normalized_multi_timestamp_rows": int(pred_count.gt(1).sum()),
        "raw_cardinality_match_rate": raw_count.eq(gold_count).mean(),
        "normalized_cardinality_match_rate": pred_count.eq(gold_count).mean(),
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
    }


def build_time_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        time_frame = frames[dataset][frames[dataset]["answer_format"] == "time"].copy()
        rows.append(
            {"dataset": dataset, "scope": "dataset", "name": "all_time", **time_metrics(time_frame)}
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
    return pd.DataFrame(rows)


def numeric_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_numeric)
    pred = subset["prediction"].apply(parse_numeric)
    parsed = gold.notna() & pred.notna()
    error = pred[parsed].astype(float) - gold[parsed].astype(float)
    return {
        **point_metrics(subset),
        "parsed_rows": int(parsed.sum()),
        "mean_absolute_error": error.abs().mean(),
        "median_absolute_error": error.abs().median(),
        "underprediction_rate": error.lt(0).mean(),
        "exact_numeric_rate": error.eq(0).mean(),
        "overprediction_rate": error.gt(0).mean(),
    }


def build_numeric_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        for answer_format in ("number", "percentage"):
            numeric = frame[frame["answer_format"] == answer_format]
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
    return pd.DataFrame(rows)


def fo_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    gold = subset["answer"].apply(parse_fo_set)
    pred = subset["prediction"].apply(parse_fo_set)
    return {
        **point_metrics(subset),
        "cardinality_match_rate": np.mean(
            [len(g) == len(p) for g, p in zip(gold, pred, strict=True)]
        )
        if len(subset)
        else np.nan,
        "mean_set_precision": np.mean(
            [set_precision(g, p) for g, p in zip(gold, pred, strict=True)]
        )
        if len(subset)
        else np.nan,
        "mean_set_recall": np.mean(
            [set_recall(g, p) for g, p in zip(gold, pred, strict=True)]
        )
        if len(subset)
        else np.nan,
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
            lambda value: "none" if not value else "single" if len(value) == 1 else "multiple"
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


def recognized_fo_values(value: Any) -> frozenset[str]:
    text = " ".join(str(value).lower().split())
    values: set[str] = set()
    masked = text
    for name in sorted(FO_NAMES, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        if re.search(pattern, masked):
            values.add(name.lower())
            masked = re.sub(pattern, " ", masked)
    return frozenset(values)


def mc_options(question: str) -> list[str]:
    match = MC_TAIL_RE.search(question)
    tail = match.group(1) if match else question
    return [item.strip(" .") for item in re.split(r"[;\n]", tail) if item.strip(" .")]


def recognized_mc_values(question: str, value: Any) -> frozenset[str]:
    text = " ".join(str(value).lower().split())
    matched: set[str] = set()
    masked = text
    for option in sorted(mc_options(question), key=len, reverse=True):
        if option.lower() in masked:
            matched.add(option.lower())
            masked = masked.replace(option.lower(), " ")
    return frozenset(matched)


def cardinality_values(row: pd.Series, column: str) -> frozenset[Any]:
    value = row[column]
    if row["answer_format"] == "fo_class":
        return recognized_fo_values(value)
    if row["answer_format"] == "time":
        return frozenset(parse_timestamps(value))
    return recognized_mc_values(str(row["question"]), value)


def build_cardinality_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = frames[dataset]
        scopes = [
            ("fo_class", frame[frame["answer_format"] == "fo_class"]),
            ("time", frame[frame["answer_format"] == "time"]),
            (
                "multiple_choice_multi_select",
                frame[
                    frame["answer_format"].eq("multiple_choice")
                    & frame["question"].astype(str).str.contains(MULTI_SELECT_RE)
                ],
            ),
        ]
        for answer_format, subset in scopes:
            gold = subset.apply(lambda row: cardinality_values(row, "answer"), axis=1)
            raw = subset.apply(
                lambda row: cardinality_values(row, "raw_model_output"), axis=1
            )
            normalized = subset.apply(
                lambda row: cardinality_values(row, "prediction"), axis=1
            )
            rows.append(
                {
                    "dataset": dataset,
                    "answer_format": answer_format,
                    "eligible_rows": len(subset),
                    "gold_multi_rows": int(gold.str.len().gt(1).sum()),
                    "raw_multi_rows": int(raw.str.len().gt(1).sum()),
                    "normalized_multi_rows": int(normalized.str.len().gt(1).sum()),
                    "raw_to_normalized_multi_rows_lost": int(
                        (raw.str.len().gt(1) & normalized.str.len().le(1)).sum()
                    ),
                    "normalized_cardinality_match_rows": int(
                        normalized.str.len().eq(gold.str.len()).sum()
                    ),
                    "normalized_cardinality_match_rate": normalized.str.len()
                    .eq(gold.str.len())
                    .mean()
                    if len(subset)
                    else np.nan,
                    "correct_rows": int(subset["correctness"].sum()),
                }
            )
    return pd.DataFrame(rows)


def build_checkpoint_selection_history() -> pd.DataFrame:
    metrics = pd.read_csv(RUN_BASE / "epoch_metrics.csv").set_index("epoch")
    rows = []
    for epoch_dir in sorted(
        RUN_BASE.glob("eval_epoch_*"), key=lambda path: int(path.name.rsplit("_", 1)[1])
    ):
        epoch = int(epoch_dir.name.rsplit("_", 1)[1])
        summaries = {}
        for dataset in DATASETS:
            path = epoch_dir / dataset / "eval/summary.csv"
            if path.is_file():
                summaries[dataset] = pd.read_csv(path)
        if len(summaries) != len(DATASETS):
            continue

        official = {
            dataset: float(
                summaries[dataset].loc[
                    summaries[dataset]["level"].eq("overall"), "accuracy"
                ].iloc[0]
            )
            for dataset in DATASETS
        }
        pre = {
            dataset: float(
                summaries[dataset].loc[
                    summaries[dataset]["level"].eq("pre_evaluation"), "accuracy"
                ].iloc[0]
            )
            for dataset in DATASETS
        }
        rows.append(
            {
                "epoch": epoch,
                "global_step": int(metrics.loc[epoch, "global_step"]),
                "heico_official_video_macro_accuracy": official["heico"],
                "lapchole_official_video_macro_accuracy": official["lapchole"],
                "mean_official_video_macro_accuracy": np.mean(list(official.values())),
                "heico_pre_evaluation_score": pre["heico"],
                "lapchole_pre_evaluation_score": pre["lapchole"],
                "mean_pre_evaluation_score": np.mean(list(pre.values())),
                "selected_for_submission": epoch == 6,
            }
        )
    return pd.DataFrame(rows)


def build_epoch6_epoch8_paired() -> pd.DataFrame:
    """Tabulate paired correctness changes on the shared selection subset."""
    rows = []
    for dataset in DATASETS:
        epoch_6 = pd.read_csv(
            RUN_BASE / f"eval_epoch_6/{dataset}/eval/results.csv",
            dtype={"qID": str},
        )[["qID", "video", "primary", "answer_format", "correctness"]]
        epoch_6["group"] = epoch_6["primary"].map(LEAF_TO_GROUP)
        if epoch_6["group"].isna().any():
            raise RuntimeError(f"{dataset}: unknown epoch-6 capability")
        epoch_6 = epoch_6.drop(columns=["primary"]).rename(
            columns={"correctness": "epoch_6_correct"}
        )
        epoch_8 = pd.read_csv(
            RUN_BASE / f"eval_epoch_8/{dataset}/eval/results.csv",
            dtype={"qID": str},
        )[["qID", "video", "primary", "answer_format", "correctness"]].rename(
            columns={
                "correctness": "epoch_8_correct",
                "video": "epoch_8_video",
                "primary": "epoch_8_primary",
                "answer_format": "epoch_8_answer_format",
            }
        )
        paired = epoch_6.merge(epoch_8, on="qID", validate="one_to_one")
        if len(paired) != len(epoch_6):
            raise RuntimeError(f"{dataset}: epoch 6/8 qID coverage differs")
        if not paired["video"].eq(paired["epoch_8_video"]).all():
            raise RuntimeError(f"{dataset}: epoch 6/8 video metadata differs")
        if not paired["answer_format"].eq(paired["epoch_8_answer_format"]).all():
            raise RuntimeError(f"{dataset}: epoch 6/8 answer formats differ")
        if not paired["group"].eq(paired["epoch_8_primary"].map(LEAF_TO_GROUP)).all():
            raise RuntimeError(f"{dataset}: epoch 6/8 capabilities differ")

        scopes = [("dataset", "all", paired)]
        scopes.extend(
            ("capability_group", group, paired[paired["group"].eq(group)])
            for _, group in GROUPS
        )
        scopes.extend(
            (
                "answer_format",
                answer_format,
                paired[paired["answer_format"].eq(answer_format)],
            )
            for answer_format in sorted(paired["answer_format"].unique())
        )
        for scope, name, subset in scopes:
            epoch_6_correct = subset["epoch_6_correct"].astype(bool)
            epoch_8_correct = subset["epoch_8_correct"].astype(bool)
            epoch_6_macro = (
                subset.assign(_correct=epoch_6_correct)
                .groupby("video", sort=False)["_correct"]
                .mean()
                .mean()
            )
            epoch_8_macro = (
                subset.assign(_correct=epoch_8_correct)
                .groupby("video", sort=False)["_correct"]
                .mean()
                .mean()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "name": name,
                    "rows": len(subset),
                    "both_correct_rows": int((epoch_6_correct & epoch_8_correct).sum()),
                    "epoch_6_only_correct_rows": int(
                        (epoch_6_correct & ~epoch_8_correct).sum()
                    ),
                    "epoch_8_only_correct_rows": int(
                        (~epoch_6_correct & epoch_8_correct).sum()
                    ),
                    "both_wrong_rows": int((~epoch_6_correct & ~epoch_8_correct).sum()),
                    "epoch_6_micro_accuracy": float(epoch_6_correct.mean()),
                    "epoch_8_micro_accuracy": float(epoch_8_correct.mean()),
                    "epoch_8_minus_epoch_6_micro_accuracy": float(
                        epoch_8_correct.mean() - epoch_6_correct.mean()
                    ),
                    "epoch_6_video_macro_accuracy": float(epoch_6_macro),
                    "epoch_8_video_macro_accuracy": float(epoch_8_macro),
                    "epoch_8_minus_epoch_6_video_macro_accuracy": float(
                        epoch_8_macro - epoch_6_macro
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_provenance(
    frames: dict[str, pd.DataFrame], cardinality: pd.DataFrame
) -> dict[str, Any]:
    source_paths = [
        RUN_BASE / "eval_subset_qids.json",
        RUN_BASE / "epoch_metrics.csv",
        RUN_BASE / "train_meta.json",
        RUN_BASE / "checkpoint-5160/adapter_config.json",
        RUN_BASE / "checkpoint-5160/adapter_model.safetensors",
        SUBMISSION_ROOT / "provenance.json",
        Path(__file__).resolve(),
    ]
    for summary_path in sorted(RUN_BASE.glob("eval_epoch_*/*/eval/summary.csv")):
        source_paths.append(summary_path)
    for dataset in DATASETS:
        source_paths.extend(
            [
                RUN_ROOT / dataset / "eval/results.csv",
                RUN_ROOT / dataset / "eval/summary.csv",
                RUN_ROOT / dataset / "predictions.parquet",
                RUN_BASE / f"eval_epoch_8/{dataset}/eval/results.csv",
                annotation_path(dataset),
            ]
        )

    source_hashes = {
        str(path.relative_to(REPO)): sha256(path)
        for path in dict.fromkeys(source_paths)
    }
    return {
        "report": "Segment Qwen3-VL-4B LoRA both-official epoch-6 seeded official-test subset",
        "generated_on": "2026-08-02",
        "scope": "Fixed seeded subset used for per-epoch checkpoint selection; not a full-test evaluation",
        "source_run": str(RUN_ROOT.relative_to(REPO)),
        "submission_bundle": str(SUBMISSION_ROOT.relative_to(REPO)),
        "checkpoint": {
            "path": str((RUN_BASE / "checkpoint-5160").relative_to(REPO)),
            "epoch": 6,
            "global_step": 5160,
            "adapter_sha256": "92556248ad9e8a82b7f1d295bac3c32f0984e7e247ef1a74499e7134032b73cb",
        },
        "subset": {
            "seed": 42,
            "sampling": "500 rows selected without replacement from the concatenated HeiCo and LapChole official test rows using numpy.default_rng(42)",
            "rows": 500,
            "full_test_rows": 6254,
            "coverage_fraction": 500 / 6254,
            "datasets": {
                dataset: {
                    "evaluated_rows": len(frames[dataset]),
                    "full_test_rows": FULL_TEST_ROWS[dataset],
                    "videos": frames[dataset]["video"].nunique(),
                    "correct": int(frames[dataset]["correctness"].sum()),
                }
                for dataset in DATASETS
            },
        },
        "inference": {
            "base_model": "os-models/Qwen3-VL-4B-Instruct",
            "sampling_stride": {"heico": 25, "lapchole": 30},
            "max_frames": 64,
            "max_pixels": 262144,
            "max_new_tokens": 64,
            "dtype": "bfloat16",
            "judge_model": "os-models/Qwen3.5-4B",
            "inference_errors": 0,
            "latency_limit_enforced_by_source_eval": False,
            "note": "Evaluator.run was called without track/max_latency; all recorded generation latencies were nevertheless below 2.1 seconds.",
        },
        "official_metric": {
            "point_estimate": "mean of per-video question-level correctness means",
            "confidence_interval": "official two-level hierarchical bootstrap: resample videos, then questions within video; 1000 samples, seed 42, percentile 95%",
            "pre_evaluation_score": "unweighted mean over populated primary capability-group x OOD buckets; only five in-distribution buckets were populated",
            "primary_source": "each dataset's eval/summary.csv",
            "orena_focus_version": "0.3.4",
        },
        "submission_inference_difference": {
            "weights_identical": True,
            "exact_container_rerun": False,
            "source_eval_prompt": "The stored epoch-6 evaluation prompt requested exactly one FO class.",
            "submission_prompt": "The packaged adapter permits all applicable FO classes and runtime-defined class names.",
            "normalization_observation": "Stored raw-to-normalized multi-value losses are tabulated in cardinality-audit.csv.",
        },
        "coverage_notes": {
            "ood_rows": int(sum(frame["ood"].sum() for frame in frames.values())),
            "clinical_rows": int(
                sum(frame["clinical_relevance"].sum() for frame in frames.values())
            ),
            "multi_value_gold_rows_by_scope": {
                f"{row.dataset}:{row.answer_format}": int(row.gold_multi_rows)
                for row in cardinality.itertuples()
            },
            "full_test_evaluation_pending": True,
        },
        "source_sha256": source_hashes,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    subset_ids = load_subset_ids()
    loaded = {
        dataset: load_dataset(dataset, subset_ids) for dataset in DATASETS
    }
    frames = {dataset: loaded[dataset][0] for dataset in DATASETS}
    summaries = {dataset: loaded[dataset][1] for dataset in DATASETS}

    if sum(len(frame) for frame in frames.values()) != 500:
        raise RuntimeError("fixed evaluation subset does not contain 500 rows")

    overall = build_dataset_overall(frames, summaries)
    groups = build_capability_groups(frames, summaries)
    leaves = build_capability_leaves(frames, summaries)
    answer_formats = build_answer_formats(frames, summaries)
    fo_cardinality, fo_labels = build_fo_diagnostics(frames)
    cardinality = build_cardinality_audit(frames)

    write_csv(overall, "dataset-overall.csv")
    write_csv(groups, "capability-group.csv")
    write_csv(leaves, "capability-leaf.csv")
    write_csv(build_capability_gap(groups, leaves), "capability-dataset-gap.csv")
    write_csv(answer_formats, "answer-format.csv")
    write_csv(
        build_capability_answer_format(frames), "capability-answer-format.csv"
    )
    write_csv(build_secondary_capabilities(frames), "secondary-capability.csv")
    write_csv(build_slices(frames), "slice-performance.csv")
    write_csv(build_video_performance(frames), "video-performance.csv")
    write_csv(build_runtime(frames), "runtime.csv")
    write_csv(build_time_diagnostics(frames), "time-diagnostics.csv")
    write_csv(build_numeric_diagnostics(frames), "numeric-diagnostics.csv")
    write_csv(fo_cardinality, "fo-class-cardinality.csv")
    write_csv(fo_labels, "fo-class-label.csv")
    write_csv(build_categorical_diagnostics(frames), "categorical-diagnostics.csv")
    write_csv(cardinality, "cardinality-audit.csv")
    write_csv(build_checkpoint_selection_history(), "checkpoint-selection.csv")
    write_csv(
        build_epoch6_epoch8_paired(),
        "checkpoint-epoch6-vs-epoch8-paired.csv",
    )

    provenance = build_provenance(frames, cardinality)
    (OUT_ROOT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(overall.to_string(index=False))
    print(f"\nWrote {len(list(OUT_ROOT.iterdir()))} artifacts to {OUT_ROOT}")


if __name__ == "__main__":
    main()
