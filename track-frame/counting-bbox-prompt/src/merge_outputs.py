#!/usr/bin/env python
"""Merge counting-only replacements into a completed full FRAME run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.counting import classify_counting_question  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _unique_records(items: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        qid = str(item["qID"])
        if qid in indexed:
            raise ValueError(f"{source}: duplicate qID {qid}")
        indexed[qid] = item
    return indexed


def merge_responses(
    baseline: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace response rows by qID while preserving full baseline order."""

    baseline_by_id = _unique_records(baseline, "baseline responses")
    replacement_by_id = _unique_records(replacements, "replacement responses")
    unknown = sorted(set(replacement_by_id).difference(baseline_by_id))
    if unknown:
        raise ValueError(f"replacement qIDs absent from baseline: {unknown[:5]}")
    return [
        replacement_by_id.get(str(response["qID"]), response)
        for response in baseline
    ]


def merge_predictions(
    baseline: pd.DataFrame,
    replacements: pd.DataFrame,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Replace rich prediction rows and retain direct-baseline audit columns."""

    baseline_records = baseline.to_dict("records")
    replacement_records = replacements.to_dict("records")
    baseline_ids = [str(record["sample_id"]) for record in baseline_records]
    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("baseline predictions contain duplicate sample_id values")

    replacement_by_id: dict[str, dict[str, Any]] = {}
    for record in replacement_records:
        qid = str(record["sample_id"])
        if qid in replacement_by_id:
            raise ValueError(f"replacement predictions contain duplicate sample_id {qid}")
        replacement_by_id[qid] = record

    unknown = sorted(set(replacement_by_id).difference(baseline_ids))
    if unknown:
        raise ValueError(f"replacement prediction IDs absent from baseline: {unknown[:5]}")

    merged: list[dict[str, Any]] = []
    for baseline_record in baseline_records:
        qid = str(baseline_record["sample_id"])
        output = dict(baseline_record)
        output["baseline_model_name"] = baseline_record.get("model_name", "")
        output["baseline_prompt"] = baseline_record.get("prompt", "")
        output["baseline_raw_model_output"] = baseline_record.get("raw_model_output", "")
        output["baseline_prediction"] = baseline_record.get("prediction", "")
        output["baseline_latency_sec"] = baseline_record.get("latency_sec")
        output["changed_by_prompt"] = qid in replacement_by_id
        if qid in replacement_by_id:
            output.update(replacement_by_id[qid])
        else:
            output.setdefault("prompt_strategy", "direct")
            output.setdefault("counting_mode", "")
            output.setdefault("counting_target", "")
            output.setdefault("structured_output_valid", None)
            output.setdefault("structured_output_status", "")
            output.setdefault("detected_objects", "")
            output.setdefault("derived_count", None)
        output["model_name"] = model_name
        merged.append(output)
    return pd.DataFrame(merged)


def _validate_replacements(predictions: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "question",
        "answer_format",
        "prompt_strategy",
        "prediction",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"replacement predictions missing columns: {missing}")
    if not predictions["prompt_strategy"].eq("bbox-json").all():
        raise ValueError("every replacement must use prompt_strategy=bbox-json")

    unrecognized = [
        str(row.sample_id)
        for row in predictions.itertuples(index=False)
        if classify_counting_question(str(row.question), str(row.answer_format)) is None
    ]
    if unrecognized:
        raise ValueError(f"non-counting replacement rows found: {unrecognized[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="full direct-run dataset directory")
    parser.add_argument("--counting", required=True, help="counting-only runner directory")
    parser.add_argument("--out", required=True, help="merged full dataset directory")
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--expected-replacements",
        type=int,
        default=None,
        help="fail unless exactly this many counting rows were produced",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline).resolve()
    counting_dir = Path(args.counting).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_predictions = pd.read_parquet(baseline_dir / "predictions.parquet")
    counting_predictions = pd.read_parquet(counting_dir / "predictions.parquet")
    _validate_replacements(counting_predictions)
    if (
        args.expected_replacements is not None
        and len(counting_predictions) != args.expected_replacements
    ):
        raise ValueError(
            f"expected {args.expected_replacements} replacements, "
            f"found {len(counting_predictions)}"
        )

    baseline_responses = _load_json(baseline_dir / "responses.json")
    counting_responses = _load_json(counting_dir / "responses.json")
    if len(counting_predictions) != len(counting_responses):
        raise ValueError(
            "counting predictions/responses length mismatch: "
            f"{len(counting_predictions)} != {len(counting_responses)}"
        )

    prediction_ids = set(counting_predictions["sample_id"].astype(str))
    response_ids = {str(response["qID"]) for response in counting_responses}
    if prediction_ids != response_ids:
        raise ValueError("counting prediction and response qID sets differ")

    merged_predictions = merge_predictions(
        baseline_predictions,
        counting_predictions,
        model_name=args.model_name,
    )
    merged_responses = merge_responses(baseline_responses, counting_responses)

    if len(merged_predictions) != len(baseline_predictions):
        raise RuntimeError("merged prediction cardinality changed")
    if len(merged_responses) != len(baseline_responses):
        raise RuntimeError("merged response cardinality changed")

    merged_predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    _write_json(merged_responses, out_dir / "responses.json")
    _write_json(_load_json(baseline_dir / "requests.json"), out_dir / "requests.json")

    baseline_meta_path = baseline_dir / "meta.json"
    counting_meta_path = counting_dir / "meta.json"
    meta = {
        "model_name": args.model_name,
        "design": "direct baseline with bbox-json replacements on counting rows",
        "baseline_dir": str(baseline_dir),
        "counting_dir": str(counting_dir),
        "full_rows": len(merged_predictions),
        "replacement_rows": len(counting_predictions),
        "baseline_meta": (
            _load_json(baseline_meta_path) if baseline_meta_path.exists() else None
        ),
        "counting_meta": (
            _load_json(counting_meta_path) if counting_meta_path.exists() else None
        ),
    }
    _write_json(meta, out_dir / "meta.json")
    print(
        f"merged {len(counting_predictions)}/{len(merged_predictions)} replacements "
        f"into {out_dir}"
    )


if __name__ == "__main__":
    main()
