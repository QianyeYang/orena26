#!/usr/bin/env python3
"""Build the deterministic, priority-ranked Frame annotation pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from guide import CLASS_NAMES, canonical_class, classes_in_answer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "data/annotations/frame-annotation-tool"
SELECTION_VERSION = "frame-priority-v1"
POOL_SIZE = 4_000
DATASET_QUOTA = 2_000
BATCH_SIZE = 100

SOURCES = {
    "heico": {
        "display_name": "HeiCo",
        "parquet": REPO_ROOT / "data/parquet/frame/train/0000.parquet",
        "frames": REPO_ROOT / "data/focus/heico/frames",
        "fps": 25,
    },
    "lapchole": {
        "display_name": "LapChole",
        "parquet": REPO_ROOT
        / "data/parquet/lapchole/frame/train/0000.parquet",
        "frames": REPO_ROOT / "data/focus/lapchole/frames",
        "fps": 30,
    },
}

_COUNT_QUESTION = re.compile(
    r"^how many (?P<label>.+?) appear in this frame\?", re.IGNORECASE
)


def _clean_scalar(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible Python scalars."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return [_clean_scalar(item) for item in value.tolist()]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, 0o664)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _seconds(timestamp: str) -> int:
    hours, minutes, seconds = (int(part) for part in timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _image_path(dataset: str, video: str, timestamp: str) -> Path:
    source = SOURCES[dataset]
    frame_number = round(_seconds(timestamp) * source["fps"])
    return source["frames"] / Path(video).stem / f"frame{frame_number:07d}.jpg"


def _case_id(dataset: str, video: str, timestamp: str) -> str:
    key = f"{dataset}\0{video}\0{timestamp}"
    return f"{dataset}-{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def _stable_key(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()


def _question_kind(question: str) -> tuple[str, str | None]:
    lower = question.casefold()
    if (
        "combination of foreign object classes" in lower
        or "list all foreign objects" in lower
    ):
        return "class_inventory", None
    if "how many different foreign object instances" in lower:
        return "total_instance_count", None
    if "how many different foreign object classes" in lower:
        return "distinct_class_count", None
    count_match = _COUNT_QUESTION.match(question)
    if count_match:
        label = count_match.group("label")
        return "class_count", canonical_class(label)
    if "there is one surgical foreign object visible" in lower:
        return "single_object_identity", None
    if "centre closest" in lower or "center closest" in lower:
        return "closest_to_center", None
    if "located" in lower or "quadrant" in lower:
        return "spatial_location", None
    if "co-occur" in lower or "also visible" in lower:
        return "cooccurrence", None
    return "other", None


def _number(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _normalise_question(row: dict[str, Any]) -> dict[str, Any]:
    kind, target_class = _question_kind(str(row["question"]))
    question = {
        "id": str(_clean_scalar(row["id"])),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "answer_format": str(row["answer_format"]),
        "generation": str(row["generation"]),
        "clinical_relevance": bool(row["clinical_relevance"]),
        "ood": bool(row["ood"]),
        "primary_capability": str(row["primary_capability"]),
        "secondary_capabilities": _clean_scalar(row["secondary_capabilities"]) or [],
        "kind": kind,
    }
    if target_class:
        question["target_class"] = target_class
    if kind in {"total_instance_count", "distinct_class_count", "class_count"}:
        question["numeric_answer"] = _number(row["answer"])
    if kind in {"class_inventory", "single_object_identity"}:
        question["answer_classes"] = list(classes_in_answer(row["answer"]))
    return question


def _priority(case: dict[str, Any]) -> tuple[int, str, list[str]]:
    questions = case["questions"]
    kinds = Counter(question["kind"] for question in questions)
    class_evidence: set[str] = set()
    numeric_answers: list[int] = []
    per_class_targets: set[str] = set()

    for question in questions:
        class_evidence.update(question.get("answer_classes", []))
        if question.get("target_class"):
            per_class_targets.add(question["target_class"])
            if (question.get("numeric_answer") or 0) > 0:
                class_evidence.add(question["target_class"])
        if question.get("numeric_answer") is not None:
            numeric_answers.append(question["numeric_answer"])

    score = 0
    reasons: list[tuple[int, str]] = []

    def award(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append((points, reason))

    if kinds["total_instance_count"]:
        award(28, "exact total-instance counting")
    if "Clip" in per_class_targets:
        award(28, "Clip counting (current major weakness)")
    elif kinds["class_count"]:
        award(12, "per-class instance counting")
    if kinds["distinct_class_count"]:
        award(8, "distinct-class counting")
    if kinds["class_inventory"]:
        award(15, "complete class inventory")
    if kinds["class_inventory"] and (
        kinds["total_instance_count"]
        or kinds["distinct_class_count"]
        or kinds["class_count"]
    ):
        award(20, "inventory and count QA cross-check")

    maximum = max(numeric_answers, default=0)
    if maximum >= 7:
        award(38, f"very crowded frame (QA count {maximum})")
    elif maximum >= 5:
        award(29, f"crowded frame (QA count {maximum})")
    elif maximum >= 3:
        award(17, f"multiple instances (QA count {maximum})")
    elif maximum == 2:
        award(7, "two visible instances")

    class_count = len(class_evidence)
    if class_count >= 4:
        award(27, f"{class_count} known foreign-object classes")
    elif class_count == 3:
        award(20, "three known foreign-object classes")
    elif class_count == 2:
        award(11, "two known foreign-object classes")

    rare_weights = {
        "Gallstone": 36,
        "Needle": 20,
        "External Drain": 9,
        "Silicone Loop": 8,
        "Specimen Bag": 7,
        "Specimen": 6,
        "Sponge": 4,
    }
    rare_hits = [
        (rare_weights[name], name)
        for name in class_evidence
        if name in rare_weights
    ]
    if rare_hits:
        rare_hits.sort(reverse=True)
        points = min(44, sum(points for points, _ in rare_hits))
        names = ", ".join(name for _, name in rare_hits)
        award(points, f"valuable class coverage: {names}")

    if kinds["closest_to_center"] or kinds["spatial_location"]:
        award(8, "spatial grounding QA")
    if any(
        question["primary_capability"] == "1c"
        for question in questions
    ):
        award(5, "object attribute QA")
    if kinds["cooccurrence"]:
        award(5, "object co-occurrence QA")
    if len(questions) > 1:
        award(min(12, 2 * (len(questions) - 1)), f"{len(questions)} QA records")
    if any(question["ood"] for question in questions):
        award(9, "out-of-distribution question")
    if any(question["clinical_relevance"] for question in questions):
        award(3, "clinically relevant question")
    if case["dataset"] == "lapchole":
        award(6, "LapChole domain coverage")

    if (
        maximum >= 7
        or "Gallstone" in class_evidence
        or score >= 105
    ):
        tier = "critical"
    elif maximum >= 5 or score >= 75:
        tier = "high"
    elif score >= 45:
        tier = "medium"
    else:
        tier = "coverage"

    reasons.sort(key=lambda item: (-item[0], item[1]))
    return score, tier, [reason for _, reason in reasons]


def _load_cases() -> tuple[list[dict[str, Any]], list[Path]]:
    import pandas as pd

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    source_paths: list[Path] = []

    for dataset, config in SOURCES.items():
        parquet = config["parquet"]
        if not parquet.is_file():
            raise FileNotFoundError(f"missing official Frame training data: {parquet}")
        source_paths.append(parquet)
        frame = pd.read_parquet(parquet)
        required = {
            "id",
            "video",
            "procedure_type",
            "question",
            "answer",
            "answer_format",
            "generation",
            "clinical_relevance",
            "ood",
            "timestamp_start",
            "primary_capability",
            "secondary_capabilities",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{parquet} lacks columns: {sorted(missing)}")
        for row in frame.to_dict("records"):
            key = (dataset, str(row["video"]), str(row["timestamp_start"]))
            grouped[key].append(row)

    cases: list[dict[str, Any]] = []
    missing_images: list[Path] = []
    for (dataset, video, timestamp), rows in grouped.items():
        questions = sorted(
            (_normalise_question(row) for row in rows),
            key=lambda item: (item["kind"], item["id"]),
        )
        image_path = _image_path(dataset, video, timestamp)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            missing_images.append(image_path)
            continue
        case = {
            "id": _case_id(dataset, video, timestamp),
            "dataset": dataset,
            "dataset_name": SOURCES[dataset]["display_name"],
            "split": "train",
            "video": video,
            "timestamp": timestamp,
            "procedure_type": str(rows[0]["procedure_type"]),
            "image_relative_path": str(image_path.relative_to(REPO_ROOT)),
            "questions": questions,
        }
        score, tier, reasons = _priority(case)
        case["priority_score"] = score
        case["priority_tier"] = tier
        case["priority_reasons"] = reasons
        cases.append(case)

    if missing_images:
        preview = "\n".join(f"  - {path}" for path in missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} referenced training frame(s) are missing or "
            f"empty; repair extraction before building the pool:\n{preview}"
        )
    return cases, source_paths


def _diverse_order(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by score while spreading equal-score cases across source videos."""
    video_position: Counter[str] = Counter()
    decorated: list[tuple[int, int, str, dict[str, Any]]] = []
    for case in sorted(
        cases,
        key=lambda item: (
            -item["priority_score"],
            _stable_key(item["id"]),
        ),
    ):
        video_key = f"{case['dataset']}::{case['video']}"
        position = video_position[video_key]
        video_position[video_key] += 1
        decorated.append(
            (
                -case["priority_score"],
                position,
                _stable_key(video_key + "\0" + case["id"]),
                case,
            )
        )
    return [item[-1] for item in sorted(decorated, key=lambda item: item[:3])]


def _select(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in SOURCES:
        candidates = [case for case in cases if case["dataset"] == dataset]
        if len(candidates) < DATASET_QUOTA:
            raise ValueError(
                f"{dataset} has only {len(candidates)} unique training frames; "
                f"cannot select quota {DATASET_QUOTA}"
            )
        selected.extend(_diverse_order(candidates)[:DATASET_QUOTA])

    selected = _diverse_order(selected)
    if len(selected) != POOL_SIZE:
        raise AssertionError(f"expected {POOL_SIZE} cases, selected {len(selected)}")
    for index, case in enumerate(selected, start=1):
        case["rank"] = index
        case["batch"] = 1 + (index - 1) // BATCH_SIZE
    return selected


def _example_strength(case: dict[str, Any], class_name: str) -> int:
    strength = 0
    for question in case["questions"]:
        answer_classes = question.get("answer_classes", [])
        numeric_answer = question.get("numeric_answer")
        if question["kind"] == "single_object_identity" and answer_classes == [
            class_name
        ]:
            strength = max(strength, 100)
        if question["kind"] == "class_inventory" and answer_classes == [class_name]:
            strength = max(strength, 90)
        if (
            question["kind"] == "class_count"
            and question.get("target_class") == class_name
            and numeric_answer == 1
        ):
            strength = max(strength, 80)
        if class_name in answer_classes:
            strength = max(strength, 60 - 5 * max(0, len(answer_classes) - 1))
        if (
            question.get("target_class") == class_name
            and isinstance(numeric_answer, int)
            and numeric_answer > 0
        ):
            strength = max(strength, 50)
    return strength


def _guide_examples(
    selected: list[dict[str, Any]], all_cases: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Choose verified released-data examples, preferring selected pool cases."""
    result: dict[str, list[str]] = {}
    selected_ids = {case["id"] for case in selected}
    for class_name in CLASS_NAMES:
        candidates = []
        for case in all_cases:
            strength = _example_strength(case, class_name)
            if strength:
                candidates.append(
                    (
                        -strength,
                        0 if case["id"] in selected_ids else 1,
                        0 if case["dataset"] == "heico" else 1,
                        _stable_key(case["id"]),
                        case,
                    )
                )
        examples: list[str] = []
        used_videos: set[str] = set()
        for _, _, _, _, case in sorted(candidates, key=lambda item: item[:4]):
            video_key = f"{case['dataset']}::{case['video']}"
            if video_key in used_videos and len(examples) < 2:
                continue
            examples.append(case["id"])
            used_videos.add(video_key)
            if len(examples) == 3:
                break
        result[class_name] = examples
    return result


def _inspect_dimensions(cases: list[dict[str, Any]]) -> None:
    from PIL import Image

    for index, case in enumerate(cases, start=1):
        path = REPO_ROOT / case["image_relative_path"]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"invalid selected frame {path}: {exc}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions for selected frame: {path}")
        case["image_width"] = width
        case["image_height"] = height
        if index % 500 == 0:
            print(f"verified {index}/{len(cases)} selected JPEGs", flush=True)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "datasets": dict(Counter(case["dataset"] for case in cases)),
        "tiers": dict(Counter(case["priority_tier"] for case in cases)),
        "batches": math.ceil(len(cases) / BATCH_SIZE),
        "unique_videos": len(
            {(case["dataset"], case["video"]) for case in cases}
        ),
        "score": {
            "maximum": max(case["priority_score"] for case in cases),
            "minimum": min(case["priority_score"] for case in cases),
            "median": sorted(case["priority_score"] for case in cases)[
                len(cases) // 2
            ],
        },
    }


def build(output_dir: Path) -> Path:
    cases, source_paths = _load_cases()
    selected = _select(cases)
    _inspect_dimensions(selected)
    selected_by_id = {case["id"]: case for case in selected}

    # Guide examples outside the 4,000-case pool are served read-only, so include
    # just their metadata and verify their JPEGs as well.
    examples = _guide_examples(selected, cases)
    example_ids = {
        case_id for case_ids in examples.values() for case_id in case_ids
    }
    all_by_id = {case["id"]: case for case in cases}
    extra_examples = [
        all_by_id[case_id]
        for case_id in sorted(example_ids - selected_by_id.keys())
    ]
    _inspect_dimensions(extra_examples)

    payload = {
        "schema_version": 1,
        "selection_version": SELECTION_VERSION,
        "generated_at": _utc_now(),
        "source_fingerprint": _source_fingerprint(source_paths),
        "source_files": [
            str(path.relative_to(REPO_ROOT)) for path in source_paths
        ],
        "selection_policy": {
            "size": POOL_SIZE,
            "dataset_quota": DATASET_QUOTA,
            "batch_size": BATCH_SIZE,
            "split": "train only",
            "ordering": "priority_score descending, deterministic diverse ties",
        },
        "summary": _summary(selected),
        "guide_examples": examples,
        "cases": selected,
        "extra_guide_cases": extra_examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "pool.json"
    _atomic_json(pool_path, payload)

    csv_path = output_dir / "pool.csv"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=output_dir
    )
    try:
        os.fchmod(fd, 0o664)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "batch",
                    "case_id",
                    "priority_score",
                    "priority_tier",
                    "dataset",
                    "video",
                    "timestamp",
                    "qa_count",
                    "priority_reasons",
                    "image_relative_path",
                ],
            )
            writer.writeheader()
            for case in selected:
                writer.writerow(
                    {
                        "rank": case["rank"],
                        "batch": case["batch"],
                        "case_id": case["id"],
                        "priority_score": case["priority_score"],
                        "priority_tier": case["priority_tier"],
                        "dataset": case["dataset"],
                        "video": case["video"],
                        "timestamp": case["timestamp"],
                        "qa_count": len(case["questions"]),
                        "priority_reasons": " | ".join(
                            case["priority_reasons"]
                        ),
                        "image_relative_path": case["image_relative_path"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, csv_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return pool_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="reuse an existing manifest with the current selection version",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if an existing manifest is current",
    )
    args = parser.parse_args()

    pool_path = args.output_dir / "pool.json"
    if args.ensure and pool_path.is_file() and not args.force:
        try:
            existing = json.loads(pool_path.read_text(encoding="utf-8"))
            if (
                existing.get("selection_version") == SELECTION_VERSION
                and len(existing.get("cases", [])) == POOL_SIZE
                and existing.get("source_fingerprint")
                == _source_fingerprint(
                    config["parquet"] for config in SOURCES.values()
                )
            ):
                print(f"using existing annotation pool: {pool_path}")
                return
        except (OSError, json.JSONDecodeError):
            pass

    result = build(args.output_dir)
    payload = json.loads(result.read_text(encoding="utf-8"))
    summary = payload["summary"]
    print(
        f"wrote {len(payload['cases'])} ranked cases to {result}; "
        f"datasets={summary['datasets']}, tiers={summary['tiers']}"
    )


if __name__ == "__main__":
    main()
