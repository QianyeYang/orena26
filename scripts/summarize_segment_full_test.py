#!/usr/bin/env python
"""Build the authoritative local full-test report for the Segment submission.

This consumes every official HeiCo and LapChole Segment test row from the
epoch-6 H200 run. Official evaluator summaries remain authoritative for
video-macro scores and confidence intervals; the other tables are descriptive
slices over exactly the same evaluated rows.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import summarize_segment_epoch6_subset as report


REPO = report.REPO
RUN_BASE = report.RUN_BASE
RUN_ROOT = RUN_BASE / "full_test_epoch_6"
OUT_ROOT = (
    REPO
    / "result-summary/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-full-test"
)
REPORT_PATH = REPO / "result-summary/segment/qwen3-vl-4b-lora-both-official-epoch6-full-test.md"
INDEX_PATH = REPO / "result-summary/segment/README.md"
SUBSET_OUT_ROOT = (
    REPO
    / "result-summary/segment/"
    / "qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset"
)
SUBMISSION_ROOT = report.SUBMISSION_ROOT
DATASETS = report.DATASETS
FULL_TEST_ROWS = report.FULL_TEST_ROWS
EXPECTED_ADAPTER_SHA256 = (
    "92556248ad9e8a82b7f1d295bac3c32f0984e7e247ef1a74499e7134032b73cb"
)

# The shared table builders write through this module-level destination.
report.OUT_ROOT = OUT_ROOT


def _expected_ids(annotations: pd.DataFrame) -> set[str]:
    ids = annotations["id"].astype(str)
    if ids.duplicated().any():
        raise RuntimeError("official annotation IDs are not unique")
    return set(ids)


def _load_manifest(dataset: str) -> dict[str, Any]:
    dataset_root = RUN_ROOT / dataset
    if not (dataset_root / "COMPLETE").is_file():
        raise RuntimeError(f"{dataset}: full-test COMPLETE marker is absent")
    manifest = json.loads((dataset_root / "run_manifest.json").read_text())
    expected = {
        "track": "segment",
        "split": "test",
        "dataset": dataset,
        "rows": FULL_TEST_ROWS[dataset],
        "checkpoint": "checkpoint-5160",
        "epoch": 6,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"{dataset}: run manifest {key}={manifest.get(key)!r}, "
                f"expected {value!r}"
            )
    gpu = str(manifest.get("slurm", {}).get("gpu", ""))
    if "H200" not in gpu.upper():
        raise RuntimeError(f"{dataset}: run manifest does not identify an H200: {gpu}")
    return manifest


def load_dataset(
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = _load_manifest(dataset)
    prediction_path = RUN_ROOT / dataset / "predictions.parquet"
    result_path = RUN_ROOT / dataset / "eval/results.csv"
    summary_path = RUN_ROOT / dataset / "eval/summary.csv"

    predictions = pd.read_parquet(prediction_path).copy()
    if len(predictions) != FULL_TEST_ROWS[dataset]:
        raise RuntimeError(
            f"{dataset}: {len(predictions)} predictions, expected "
            f"{FULL_TEST_ROWS[dataset]}"
        )
    predictions["qID"] = predictions["sample_id"].astype(str)
    if predictions["qID"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate prediction IDs")
    predictions = predictions.drop(columns=["sample_id"])
    predictions["secondary_capabilities"] = predictions[
        "secondary_capabilities"
    ].apply(report.as_list)

    results = pd.read_csv(result_path, dtype={"qID": str}).rename(
        columns={
            "video": "evaluated_video",
            "ood": "evaluated_ood",
            "clinical": "evaluated_clinical",
            "primary": "evaluated_primary",
            "answer_format": "evaluated_answer_format",
            "latency": "evaluated_latency_sec",
        }
    )
    if len(results) != FULL_TEST_ROWS[dataset]:
        raise RuntimeError(
            f"{dataset}: {len(results)} evaluated rows, expected "
            f"{FULL_TEST_ROWS[dataset]}"
        )
    if results["qID"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate evaluator IDs")

    annotations = pd.read_parquet(
        report.annotation_path(dataset),
        columns=["id", "generation", "procedure_type", "track"],
    ).copy()
    expected_ids = _expected_ids(annotations)
    if len(expected_ids) != FULL_TEST_ROWS[dataset]:
        raise RuntimeError(
            f"{dataset}: annotation row constant is stale: "
            f"{len(expected_ids)} != {FULL_TEST_ROWS[dataset]}"
        )
    if set(predictions["qID"]) != expected_ids:
        raise RuntimeError(f"{dataset}: predictions do not cover the full test set")
    if set(results["qID"]) != expected_ids:
        raise RuntimeError(f"{dataset}: evaluator output does not cover the full test set")
    annotations["qID"] = annotations["id"].astype(str)
    annotations = annotations.drop(columns=["id"])

    merged = predictions.merge(results, on="qID", validate="one_to_one")
    merged = merged.merge(annotations, on="qID", validate="one_to_one")
    merged["dataset"] = dataset
    merged["primary"] = merged["primary_capability"].map(report.CODE_TO_LEAF)
    merged["group"] = merged["primary"].map(report.LEAF_TO_GROUP)

    if len(merged) != FULL_TEST_ROWS[dataset]:
        raise RuntimeError(f"{dataset}: merge lost full-test rows")
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
    if merged["timed_out"].astype(bool).any():
        raise RuntimeError(f"{dataset}: timed-out inference rows are present")
    if merged["raw_model_output"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{dataset}: empty raw model outputs are present")

    fo_prompts = merged.loc[merged["answer_format"].eq("fo_class"), "prompt"]
    if not fo_prompts.astype(str).str.contains("one or more", regex=False).all():
        raise RuntimeError(f"{dataset}: source FO prompts are not multi-value aware")
    time_prompts = merged.loc[merged["answer_format"].eq("time"), "prompt"]
    if not time_prompts.astype(str).str.contains(
        "one or more timestamps", regex=False
    ).all():
        raise RuntimeError(f"{dataset}: source time prompts are not multi-value aware")

    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise RuntimeError(f"{dataset}: evaluator summary is empty")
    return merged, summary, manifest


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing provenance sources: {missing}")
    return {
        str(path.relative_to(REPO)): report.sha256(path)
        for path in dict.fromkeys(paths)
    }


def write_aggregate_manifest(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    code_paths = [
        REPO / "track-segment/baseline/src/runner.py",
        REPO / "src/prompts.py",
        REPO / "src/adapter.py",
        REPO / "track-segment/lora-finetune/scripts/full_test_h200.slurm",
    ]
    payload = {
        "track": "segment",
        "split": "test",
        "scope": "all official local test rows for HeiCo and LapChole",
        "rows": sum(FULL_TEST_ROWS.values()),
        "datasets": manifests,
        "checkpoint": "checkpoint-5160",
        "epoch": 6,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "inference_job": "2373",
        "hardware_requirement": "NVIDIA H200",
        "code_sha256_at_validation": _source_hashes(code_paths),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    (RUN_ROOT / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RUN_ROOT / "COMPLETE").touch()
    return payload


def build_provenance(
    frames: dict[str, pd.DataFrame],
    manifests: dict[str, dict[str, Any]],
    aggregate_manifest: dict[str, Any],
    cardinality: pd.DataFrame,
) -> dict[str, Any]:
    source_paths = [
        RUN_ROOT / "run_manifest.json",
        RUN_BASE / "eval_subset_qids.json",
        RUN_BASE / "epoch_metrics.csv",
        RUN_BASE / "train_meta.json",
        RUN_BASE / "checkpoint-5160/adapter_config.json",
        RUN_BASE / "checkpoint-5160/adapter_model.safetensors",
        SUBMISSION_ROOT / "provenance.json",
        REPO / "src/prompts.py",
        REPO / "src/adapter.py",
        REPO / "track-segment/baseline/src/runner.py",
        REPO / "track-segment/lora-finetune/scripts/full_test_h200.slurm",
        Path(__file__).resolve(),
        Path(report.__file__).resolve(),
        SUBSET_OUT_ROOT / "dataset-overall.csv",
        SUBSET_OUT_ROOT / "capability-group.csv",
        SUBSET_OUT_ROOT / "answer-format.csv",
    ]
    source_paths.extend(sorted(RUN_BASE.glob("eval_epoch_*/*/eval/summary.csv")))
    for dataset in DATASETS:
        source_paths.extend(
            [
                RUN_ROOT / dataset / "run_manifest.json",
                RUN_ROOT / dataset / "predictions.parquet",
                RUN_ROOT / dataset / "responses.json",
                RUN_ROOT / dataset / "eval/results.csv",
                RUN_ROOT / dataset / "eval/summary.csv",
                RUN_BASE / f"eval_epoch_6/{dataset}/eval/results.csv",
                RUN_BASE / f"eval_epoch_8/{dataset}/eval/results.csv",
                report.annotation_path(dataset),
            ]
        )

    return {
        "report": "Segment Qwen3-VL-4B LoRA both-official epoch-6 full test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "All 6,254 official local Segment test rows: 4,000 HeiCo and "
            "2,254 LapChole. This is an offline local evaluation, not an online "
            "leaderboard result."
        ),
        "source_run": str(RUN_ROOT.relative_to(REPO)),
        "submission_bundle": str(SUBMISSION_ROOT.relative_to(REPO)),
        "checkpoint": {
            "path": str((RUN_BASE / "checkpoint-5160").relative_to(REPO)),
            "epoch": 6,
            "global_step": 5160,
            "adapter_sha256": EXPECTED_ADAPTER_SHA256,
            "selection_basis": (
                "highest equal-dataset mean pre-evaluation score on the fixed "
                "seed-42 500-row checkpoint-selection subset"
            ),
        },
        "coverage": {
            "rows": sum(len(frame) for frame in frames.values()),
            "full_test_rows": sum(FULL_TEST_ROWS.values()),
            "coverage_fraction": 1.0,
            "datasets": {
                dataset: {
                    "evaluated_rows": len(frames[dataset]),
                    "full_test_rows": FULL_TEST_ROWS[dataset],
                    "videos": int(frames[dataset]["video"].nunique()),
                    "correct": int(frames[dataset]["correctness"].sum()),
                    "ood_rows": int(frames[dataset]["ood"].sum()),
                    "clinical_rows": int(
                        frames[dataset]["clinical_relevance"].sum()
                    ),
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
            "slurm_array_job_id": "2373",
            "dataset_run_manifests": manifests,
            "aggregate_run_manifest": aggregate_manifest,
            "inference_errors": 0,
            "timed_out_rows": 0,
            "multi_value_behavior": (
                "FO-class prompts request all applicable classes and the "
                "normalizer preserves all recognized classes; time prompts and "
                "normalization preserve all returned timestamps."
            ),
        },
        "official_metric": {
            "point_estimate": "mean of per-video question-level correctness means",
            "confidence_interval": (
                "official two-level hierarchical bootstrap: resample videos, "
                "then questions within video; 1000 samples, seed 42, percentile 95%"
            ),
            "pre_evaluation_score": (
                "unweighted mean over populated primary capability-group x OOD buckets"
            ),
            "primary_source": "each dataset's eval/summary.csv",
            "orena_focus_version": importlib.metadata.version("orena-focus"),
        },
        "submission_equivalence": {
            "weights_identical": True,
            "prompt_and_cardinality_normalization_aligned": True,
            "exact_container_rerun": False,
            "note": (
                "The repository runner and packaged submission share the selected "
                "weights, sampling limits, prompts, and answer normalization. This "
                "run used the repository evaluation path rather than invoking the "
                "built container contract."
            ),
        },
        "checkpoint_selection_artifacts": {
            "scope": "fixed seed-42 500-row subset only",
            "history_table": "checkpoint-selection.csv",
            "epoch_6_vs_8_paired_table": "checkpoint-epoch6-vs-epoch8-paired.csv",
        },
        "cardinality": {
            "multi_value_gold_rows_by_scope": {
                f"{row.dataset}:{row.answer_format}": int(row.gold_multi_rows)
                for row in cardinality.itertuples()
            }
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_sha256": _source_hashes(source_paths),
    }


def build_subset_comparison(
    overall: pd.DataFrame,
    groups: pd.DataFrame,
    answer_formats: pd.DataFrame,
) -> pd.DataFrame:
    """Compare full-test scores with the historical checkpoint-selection subset."""
    specs = (
        ("dataset", overall, pd.read_csv(SUBSET_OUT_ROOT / "dataset-overall.csv"), []),
        (
            "capability_group",
            groups,
            pd.read_csv(SUBSET_OUT_ROOT / "capability-group.csv"),
            ["capability_group"],
        ),
        (
            "answer_format",
            answer_formats,
            pd.read_csv(SUBSET_OUT_ROOT / "answer-format.csv"),
            ["answer_format"],
        ),
    )
    rows: list[dict[str, Any]] = []
    for level, full, subset, name_columns in specs:
        keys = ["dataset", *name_columns]
        full_columns = [*keys, "rows", "official_video_macro_accuracy"]
        subset_columns = [*keys, "rows", "official_video_macro_accuracy"]
        paired = full[full_columns].merge(
            subset[subset_columns],
            on=keys,
            how="outer",
            suffixes=("_full", "_subset"),
            validate="one_to_one",
        )
        for row in paired.itertuples(index=False):
            values = row._asdict()
            name = "all" if not name_columns else str(values[name_columns[0]])
            subset_score = values["official_video_macro_accuracy_subset"]
            full_score = values["official_video_macro_accuracy_full"]
            rows.append(
                {
                    "dataset": values["dataset"],
                    "level": level,
                    "name": name,
                    "subset_rows": values["rows_subset"],
                    "full_rows": values["rows_full"],
                    "subset_official_video_macro_accuracy": subset_score,
                    "full_official_video_macro_accuracy": full_score,
                    "full_minus_subset": (
                        full_score - subset_score
                        if pd.notna(full_score) and pd.notna(subset_score)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _label(value: Any) -> str:
    return str(value).replace("_", " ").replace("lapchole", "LapChole").replace(
        "heico", "HeiCo"
    )


def _number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def _score_ci(row: pd.Series) -> str:
    score = row.get("official_video_macro_accuracy")
    if pd.isna(score):
        return "—"
    return (
        f"{float(score):.4f} [{float(row['ci_low']):.4f}, "
        f"{float(row['ci_high']):.4f}]"
    )


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def write_markdown_report(
    frames: dict[str, pd.DataFrame],
    manifests: dict[str, dict[str, Any]],
    overall: pd.DataFrame,
    groups: pd.DataFrame,
    leaves: pd.DataFrame,
    answer_formats: pd.DataFrame,
    cardinality: pd.DataFrame,
    time_diagnostics: pd.DataFrame,
    numeric_diagnostics: pd.DataFrame,
    categorical: pd.DataFrame,
    slices: pd.DataFrame,
    runtime: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    detail_dir = OUT_ROOT.name
    overall_index = overall.set_index("dataset")
    official_mean = overall["official_video_macro_accuracy"].mean()
    pre_eval_mean = overall["pre_evaluation_score"].mean()

    group_means = (
        groups.groupby("capability_group", sort=False)[
            "official_video_macro_accuracy"
        ]
        .mean()
        .sort_values()
    )
    weakest_group = str(group_means.index[0])
    strongest_group = str(group_means.index[-1])

    outcome_rows = []
    for dataset in DATASETS:
        row = overall_index.loc[dataset]
        outcome_rows.append(
            [
                _label(dataset),
                f"{_integer(row['rows'])} / {_integer(row['full_test_rows'])}",
                _integer(row["videos"]),
                _integer(row["correct"]),
                _score_ci(row),
                _number(row["micro_accuracy"]),
                _number(row["pre_evaluation_score"]),
            ]
        )

    comparison_rows = []
    for row in comparison[comparison["level"].eq("dataset")].itertuples():
        comparison_rows.append(
            [
                _label(row.dataset),
                _integer(row.subset_rows),
                _number(row.subset_official_video_macro_accuracy),
                _integer(row.full_rows),
                _number(row.full_official_video_macro_accuracy),
                f"{row.full_minus_subset:+.4f}",
            ]
        )

    group_rows = []
    for _, group in report.GROUPS:
        values: list[Any] = [_label(group)]
        for dataset in DATASETS:
            match = groups[
                groups["dataset"].eq(dataset)
                & groups["capability_group"].eq(group)
            ].iloc[0]
            values.extend([_integer(match["rows"]), _score_ci(match)])
        values.append(_number(group_means[group]))
        group_rows.append(values)

    leaf_rows = []
    for code, leaf, _ in report.LEAVES:
        values = [code, _label(leaf)]
        for dataset in DATASETS:
            match = leaves[
                leaves["dataset"].eq(dataset)
                & leaves["primary_capability"].eq(leaf)
            ].iloc[0]
            values.extend([_integer(match["rows"]), _score_ci(match)])
        leaf_rows.append(values)

    format_rows = []
    for answer_format in sorted(answer_formats["answer_format"].unique()):
        values = [_label(answer_format)]
        for dataset in DATASETS:
            match = answer_formats[
                answer_formats["dataset"].eq(dataset)
                & answer_formats["answer_format"].eq(answer_format)
            ]
            if match.empty:
                values.extend(["0", "—"])
            else:
                values.extend(
                    [_integer(match.iloc[0]["rows"]), _score_ci(match.iloc[0])]
                )
        format_rows.append(values)

    cardinality_rows = []
    for row in cardinality.itertuples():
        cardinality_rows.append(
            [
                _label(row.dataset),
                _label(row.answer_format),
                _integer(row.eligible_rows),
                _integer(row.gold_multi_rows),
                _integer(row.raw_multi_rows),
                _integer(row.normalized_multi_rows),
                _integer(row.raw_to_normalized_multi_rows_lost),
                _number(row.normalized_cardinality_match_rate),
            ]
        )

    time_rows = []
    for row in time_diagnostics.itertuples():
        time_rows.append(
            [
                _label(row.dataset),
                _label(row.name),
                _integer(row.rows),
                _number(row.video_macro_accuracy),
                _integer(row.gold_multi_timestamp_rows),
                _number(row.single_gold_median_absolute_error_sec, 1),
                _number(row.single_gold_p90_absolute_error_sec, 1),
                _number(row.single_gold_within_5_sec_rate),
            ]
        )

    numeric_rows = []
    for row in numeric_diagnostics[numeric_diagnostics["rows"].gt(0)].itertuples():
        numeric_rows.append(
            [
                _label(row.dataset),
                _label(row.answer_format),
                _label(row.name),
                _integer(row.rows),
                _number(row.video_macro_accuracy),
                _number(row.median_absolute_error, 2),
                _number(row.exact_numeric_rate),
            ]
        )

    categorical_rows = []
    for row in categorical.itertuples():
        categorical_rows.append(
            [
                _label(row.dataset),
                _label(row.answer_format),
                _label(row.value),
                _integer(row.rows),
                _number(row.video_macro_accuracy),
            ]
        )

    generation_rows = []
    generations = slices[slices["dimension"].eq("generation")]
    for row in generations.itertuples():
        generation_rows.append(
            [
                _label(row.dataset),
                _label(row.value),
                _integer(row.rows),
                _number(row.rows / len(frames[row.dataset])),
                _number(row.video_macro_accuracy),
            ]
        )

    runtime_rows = []
    dataset_runtime = runtime[
        runtime["scope"].eq("dataset") & runtime["name"].eq("all")
    ]
    for row in dataset_runtime.itertuples():
        manifest = manifests[row.dataset]
        slurm = manifest["slurm"]
        runtime_rows.append(
            [
                _label(row.dataset),
                slurm.get("job_id", "—"),
                slurm.get("node", "—"),
                slurm.get("gpu", "—"),
                _number(row.mean_latency_sec, 3),
                _number(row.p95_latency_sec, 3),
                _number(row.max_latency_sec, 3),
                _integer(row.over_15_sec_rows),
                _integer(row.rows_at_64_frame_cap),
            ]
        )

    multi_gold_total = int(cardinality["gold_multi_rows"].sum())
    normalization_losses = int(
        cardinality["raw_to_normalized_multi_rows_lost"].sum()
    )
    generated_date = datetime.now(timezone.utc).date().isoformat()

    lines = [
        "# Segment epoch-6 full-test analysis",
        "",
        "> **Authoritative local full-test evaluation.** This report covers all "
        "6,254 official Segment test questions: 4,000 HeiCo and 2,254 "
        "LapChole rows across all 38 test videos. It is an offline local result, "
        "not an online leaderboard score or a container-contract rerun.",
        "",
        "## Outcome",
        "",
        _table(
            [
                "Dataset",
                "Evaluated / full rows",
                "Videos",
                "Correct",
                "Official video-macro accuracy (95% CI)",
                "Question-micro accuracy",
                "Pre-evaluation score",
            ],
            outcome_rows,
        ),
        "",
        f"The equal-dataset mean is **{official_mean:.4f}** for official "
        f"video-macro accuracy and **{pre_eval_mean:.4f}** for the "
        "capability/OOD-bucket-balanced pre-evaluation score. Dataset values "
        "remain the primary results because HeiCo and LapChole have different "
        "question distributions.",
        "",
        f"Across the five capability groups, **{_label(weakest_group)} is the "
        f"weakest ability** by equal-dataset mean full-test score "
        f"({_number(group_means.iloc[0])}); {_label(strongest_group)} is the "
        f"strongest ({_number(group_means.iloc[-1])}). Per-dataset support and "
        "uncertainty below should be used when interpreting sparse groups.",
        "",
        "## Full test versus the 500-row selection subset",
        "",
        _table(
            [
                "Dataset",
                "Subset rows",
                "Subset official",
                "Full rows",
                "Full official",
                "Full − subset",
            ],
            comparison_rows,
        ),
        "",
        "The fixed subset remains useful only for explaining checkpoint "
        "selection. The full-test figures above supersede it for model "
        "performance claims. Capability- and format-level deltas are in "
        f"[`subset-vs-full.csv`]({detail_dir}/subset-vs-full.csv).",
        "",
        "## Capability groups",
        "",
        _table(
            [
                "Capability group",
                "HeiCo n",
                "HeiCo score (95% CI)",
                "LapChole n",
                "LapChole score (95% CI)",
                "Equal-dataset mean",
            ],
            group_rows,
        ),
        "",
        "## All primary leaf capabilities",
        "",
        _table(
            [
                "Code",
                "Primary capability",
                "HeiCo n",
                "HeiCo score (95% CI)",
                "LapChole n",
                "LapChole score (95% CI)",
            ],
            leaf_rows,
        ),
        "",
        "Secondary capability tags overlap rather than partitioning the rows. "
        "Their complete distribution is in "
        f"[`secondary-capability.csv`]({detail_dir}/secondary-capability.csv); "
        "capability × answer-format counts are in "
        f"[`capability-answer-format.csv`]({detail_dir}/capability-answer-format.csv).",
        "",
        "## Answer formats",
        "",
        _table(
            [
                "Answer format",
                "HeiCo n",
                "HeiCo score (95% CI)",
                "LapChole n",
                "LapChole score (95% CI)",
            ],
            format_rows,
        ),
        "",
        "Binary and multiple-choice subtype results are:",
        "",
        _table(
            ["Dataset", "Format", "Subtype", "Rows", "Video-macro score"],
            categorical_rows,
        ),
        "",
        "## Cardinality audit",
        "",
        _table(
            [
                "Dataset",
                "Format",
                "Eligible",
                "Gold multi-value",
                "Raw multi-value",
                "Normalized multi-value",
                "Lost in normalization",
                "Cardinality match rate",
            ],
            cardinality_rows,
        ),
        "",
        f"The audited scopes contain **{multi_gold_total:,} multi-value gold "
        f"answers**. The submission-aligned normalizers discarded values from "
        f"**{normalization_losses:,} raw outputs**. FO prompts request every "
        "applicable class, and time prompts request every applicable timestamp.",
        "",
        "Per-cardinality FO exact-set scores and per-label precision/recall are "
        f"in [`fo-class-cardinality.csv`]({detail_dir}/fo-class-cardinality.csv) "
        f"and [`fo-class-label.csv`]({detail_dir}/fo-class-label.csv).",
        "",
        "## Temporal and numeric diagnostics",
        "",
        _table(
            [
                "Dataset",
                "Time scope",
                "Rows",
                "Video-macro score",
                "Multi-time gold",
                "Median abs. error (s)",
                "p90 abs. error (s)",
                "Within 5 s",
            ],
            time_rows,
        ),
        "",
        _table(
            [
                "Dataset",
                "Format",
                "Scope",
                "Rows",
                "Video-macro score",
                "Median abs. error",
                "Exact numeric rate",
            ],
            numeric_rows,
        ),
        "",
        "## Dataset and generation distribution",
        "",
        _table(
            ["Dataset", "Generation source", "Rows", "Row share", "Video-macro score"],
            generation_rows,
        ),
        "",
        "All test rows are in-distribution and non-clinical according to the "
        "official annotation flags, so this test split cannot measure OOD or "
        "clinical-subset generalization. HeiCo contains only sigmoid-resection "
        "videos; LapChole contains only laparoscopic-cholecystectomy videos.",
        "",
        "## Runtime and validity",
        "",
        _table(
            [
                "Dataset",
                "Slurm task",
                "Node",
                "GPU",
                "Mean latency (s)",
                "p95 latency (s)",
                "Max latency (s)",
                ">15 s rows",
                "64-frame-cap rows",
            ],
            runtime_rows,
        ),
        "",
        "Both H200 tasks passed strict ID, count, duplicate, empty-output, "
        "inference-error, and evaluator-coverage checks. Model-generation "
        "latency excludes model loading and may exclude video decoding/frame "
        "preparation.",
        "",
        "## Checkpoint, scoring, and provenance",
        "",
        "The evaluated checkpoint is epoch 6, global step 5,160, with adapter "
        f"SHA-256 `{EXPECTED_ADAPTER_SHA256}`. It was selected using the highest "
        "equal-dataset mean pre-evaluation score on the fixed seed-42 500-row "
        "subset. The full checkpoint history and paired epoch-6 versus epoch-8 "
        f"selection-subset analysis remain in [`checkpoint-selection.csv`]({detail_dir}/checkpoint-selection.csv) "
        f"and [`checkpoint-epoch6-vs-epoch8-paired.csv`]({detail_dir}/checkpoint-epoch6-vs-epoch8-paired.csv).",
        "",
        "The official point estimate is the mean of per-video question-level "
        "correctness means. Its 95% interval is the official two-level "
        "hierarchical bootstrap (videos, then questions within video; 1,000 "
        "samples, seed 42). The pre-evaluation score is the unweighted mean over "
        "populated primary capability-group × OOD buckets.",
        "",
        "The repository evaluation uses the packaged checkpoint, sampling "
        "limits, multi-value prompts, and answer normalization, but it invokes "
        "the repository runner rather than the built container contract. Exact "
        f"source hashes, per-dataset run manifests, and artifact lineage are in "
        f"[`provenance.json`]({detail_dir}/provenance.json).",
        "",
        "## Artifact index",
        "",
        "The numerical directory contains dataset, capability group/leaf and "
        "gap, answer-format and capability-format, secondary capability, "
        "generation/flag slice, per-video, runtime, temporal, numeric, FO, "
        "categorical, cardinality, checkpoint-selection, subset-comparison, and "
        "provenance artifacts. Raw predictions and evaluator outputs remain "
        f"under `{RUN_ROOT.relative_to(REPO)}`.",
        "",
        f"Report generated {generated_date}.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def write_segment_index(overall: pd.DataFrame) -> None:
    values = overall.set_index("dataset")
    heico = values.loc["heico", "official_video_macro_accuracy"]
    lapchole = values.loc["lapchole", "official_video_macro_accuracy"]
    content = f"""# Segment results

## Submission-checkpoint reports

| Report | Checkpoint | Evaluation scope | Main result | Status |
| --- | --- | --- | --- | --- |
| [Qwen3-VL-4B LoRA epoch 6 full test](qwen3-vl-4b-lora-both-official-epoch6-full-test.md) | Epoch 6, step 5,160 | All 6,254 official local test rows; 38 videos | HeiCo {heico:.4f}, LapChole {lapchole:.4f} official video-macro | **Authoritative local full-test report** |
| [Qwen3-VL-4B LoRA epoch 6 selection subset](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset.md) | Epoch 6, step 5,160 | Fixed seed-42 subset: 500 / 6,254 rows | Checkpoint-selection evidence only | Historical / superseded for performance claims |

The full-test report corresponds to the checkpoint packaged in
`submissions/segment/qwen3-vl-4b-lora-both-official-epoch6-20260801`. It
includes dataset, capability-group and leaf, answer-format, cardinality,
numeric, temporal, runtime, video, generation-source, checkpoint-selection,
and subset-versus-full distributions with exact provenance.

The 500-row report is retained to explain checkpoint selection and the earlier
single-value prompt/normalizer audit. Do not use its scores as the primary
Segment result now that full-test inference is available.

Historical baseline and provisional LoRA results remain in the
[cross-track snapshot](../../result-summary.md#segment). Do not compare that
snapshot's differently scoped values directly with the full-test report.
"""
    INDEX_PATH.write_text(content)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = {dataset: load_dataset(dataset) for dataset in DATASETS}
    frames = {dataset: loaded[dataset][0] for dataset in DATASETS}
    summaries = {dataset: loaded[dataset][1] for dataset in DATASETS}
    manifests = {dataset: loaded[dataset][2] for dataset in DATASETS}

    if sum(len(frame) for frame in frames.values()) != sum(FULL_TEST_ROWS.values()):
        raise RuntimeError("full evaluation does not contain all 6,254 rows")

    aggregate_manifest = write_aggregate_manifest(manifests)
    overall = report.build_dataset_overall(frames, summaries)
    groups = report.build_capability_groups(frames, summaries)
    leaves = report.build_capability_leaves(frames, summaries)
    answer_formats = report.build_answer_formats(frames, summaries)
    fo_cardinality, fo_labels = report.build_fo_diagnostics(frames)
    cardinality = report.build_cardinality_audit(frames)
    slices = report.build_slices(frames)
    runtime = report.build_runtime(frames)
    time_diagnostics = report.build_time_diagnostics(frames)
    numeric_diagnostics = report.build_numeric_diagnostics(frames)
    categorical = report.build_categorical_diagnostics(frames)
    comparison = build_subset_comparison(overall, groups, answer_formats)

    report.write_csv(overall, "dataset-overall.csv")
    report.write_csv(groups, "capability-group.csv")
    report.write_csv(leaves, "capability-leaf.csv")
    report.write_csv(
        report.build_capability_gap(groups, leaves), "capability-dataset-gap.csv"
    )
    report.write_csv(answer_formats, "answer-format.csv")
    report.write_csv(
        report.build_capability_answer_format(frames),
        "capability-answer-format.csv",
    )
    report.write_csv(
        report.build_secondary_capabilities(frames), "secondary-capability.csv"
    )
    report.write_csv(slices, "slice-performance.csv")
    report.write_csv(report.build_video_performance(frames), "video-performance.csv")
    report.write_csv(runtime, "runtime.csv")
    report.write_csv(time_diagnostics, "time-diagnostics.csv")
    report.write_csv(numeric_diagnostics, "numeric-diagnostics.csv")
    report.write_csv(fo_cardinality, "fo-class-cardinality.csv")
    report.write_csv(fo_labels, "fo-class-label.csv")
    report.write_csv(categorical, "categorical-diagnostics.csv")
    report.write_csv(cardinality, "cardinality-audit.csv")
    report.write_csv(comparison, "subset-vs-full.csv")
    report.write_csv(
        report.build_checkpoint_selection_history(), "checkpoint-selection.csv"
    )
    report.write_csv(
        report.build_epoch6_epoch8_paired(),
        "checkpoint-epoch6-vs-epoch8-paired.csv",
    )

    provenance = build_provenance(
        frames, manifests, aggregate_manifest, cardinality
    )
    (OUT_ROOT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    write_markdown_report(
        frames,
        manifests,
        overall,
        groups,
        leaves,
        answer_formats,
        cardinality,
        time_diagnostics,
        numeric_diagnostics,
        categorical,
        slices,
        runtime,
        comparison,
    )
    write_segment_index(overall)

    print(overall.to_string(index=False))
    print(f"\nWrote {len(list(OUT_ROOT.iterdir()))} artifacts to {OUT_ROOT}")


if __name__ == "__main__":
    main()
