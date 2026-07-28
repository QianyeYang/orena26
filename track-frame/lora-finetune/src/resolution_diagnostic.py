#!/usr/bin/env python3
"""Paired native-resolution diagnostic for the epoch-30 Frame LoRA.

The current model was trained and evaluated with ``max_pixels=602112``.  This
runner compares that cap with a larger cap on:

* every current official Frame test question whose answer format is ``number``
  or ``fo_class`` and whose native frame exceeds the baseline cap; and
* a deterministic control sample of frames that do not exceed the baseline cap.

Only the processor pixel cap changes between conditions.  No source image is
upscaled, tiled, cropped, or otherwise modified.  The runner uses the
submission-compatible prompt and multi-label FO-class normalisation, records
processed image geometry, and measures CUDA peak allocated and reserved memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("frame-resolution-diagnostic")

MAX_TEXT_LENGTH = 300
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a single endoscopic "
    "frame from colorectal surgery. Look carefully at the image and answer the "
    "question. Respond with ONLY the answer in the exact requested format — no "
    "explanation, no extra words."
)
DEFAULT_FO_NAMES = (
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
TRACKED_FORMATS = ("number", "fo_class")


@dataclass(frozen=True)
class Condition:
    name: str
    max_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--previous-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-max-pixels", type=int, default=602_112)
    parser.add_argument("--high-max-pixels", type=int, default=1_003_520)
    parser.add_argument("--control-per-stratum", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=2732)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value).split())[:MAX_TEXT_LENGTH]


def parse_fo_names(definitions: str) -> tuple[str, ...]:
    lines = [line.strip() for line in definitions.splitlines()]
    names: list[str] = []
    for index in range(len(lines) - 1):
        name, underline = lines[index], lines[index + 1]
        if name and re.fullmatch(r"-{3,}", underline):
            names.append(name)
    return tuple(dict.fromkeys(names)) or DEFAULT_FO_NAMES


def build_instruction(
    question: str, answer_format: str, fo_names: tuple[str, ...]
) -> str:
    if answer_format == "number":
        return (
            f"{question}\n\nAnswer with a single non-negative integer (digits only)."
        )
    if answer_format == "fo_class":
        names = ", ".join(fo_names)
        return (
            f"{question}\n\nAnswer with only one or more of these foreign-object "
            f"names, separated by a comma and a space: {names}. If none is visible, "
            "answer: none. Output nothing else."
        )
    raise ValueError(f"Unsupported diagnostic answer format: {answer_format!r}")


def normalize_number(raw: Any) -> str:
    match = re.search(r"\d+", str(raw))
    return match.group(0) if match else clean_text(raw)


def normalize_fo_class(raw: Any, fo_names: tuple[str, ...]) -> str:
    text = clean_text(raw)
    lower = text.lower()
    if not lower:
        return ""

    matched: set[str] = set()
    masked = lower
    for name in sorted(fo_names, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        if re.search(pattern, masked):
            matched.add(name)
            masked = re.sub(pattern, " ", masked)
    if matched:
        return ", ".join(name for name in fo_names if name in matched)
    if re.search(
        r"\b(none|no foreign object|no object|nothing|absent|not present|n/a)\b",
        lower,
    ):
        return "none"
    return text


def normalize_answer(
    raw: Any, answer_format: str, fo_names: tuple[str, ...]
) -> str:
    if answer_format == "number":
        return normalize_number(raw)
    if answer_format == "fo_class":
        return normalize_fo_class(raw, fo_names)
    raise ValueError(f"Unsupported diagnostic answer format: {answer_format!r}")


def fo_set(value: Any, fo_names: tuple[str, ...]) -> frozenset[str]:
    normalized = normalize_fo_class(value, fo_names)
    return frozenset(
        part.strip().casefold()
        for part in normalized.split(",")
        if part.strip()
    )


def score_answer(
    prediction: Any,
    answer: Any,
    answer_format: str,
    fo_names: tuple[str, ...],
) -> bool:
    if answer_format == "number":
        return normalize_number(prediction) == clean_text(answer)
    if answer_format == "fo_class":
        return fo_set(prediction, fo_names) == fo_set(answer, fo_names)
    raise ValueError(f"Unsupported diagnostic answer format: {answer_format!r}")


def read_definitions(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, str):
        raise TypeError(f"{path} must contain a JSON string")
    return value


def _parquet_path(repo: Path, dataset: str) -> Path:
    if dataset == "heico":
        return repo / "data/parquet/frame/test/0000.parquet"
    if dataset == "lapchole":
        return repo / "data/parquet/lapchole/frame/test/0000.parquet"
    raise ValueError(dataset)


def _image_dimensions(paths: list[str]) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    for index, raw_path in enumerate(dict.fromkeys(paths)):
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty diagnostic frame: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions[raw_path] = image.size
        if (index + 1) % 500 == 0:
            log.info("validated dimensions for %d unique frames", index + 1)
    return dimensions


def build_selection(
    repo: Path,
    previous_root: Path,
    baseline_max_pixels: int,
    control_per_stratum: int,
    seed: int,
    fo_names: tuple[str, ...],
) -> pd.DataFrame:
    """Build a complete affected set plus deterministic unaffected controls."""
    dataset_frames: list[pd.DataFrame] = []
    for dataset in ("heico", "lapchole"):
        annotations = pd.read_parquet(_parquet_path(repo, dataset)).copy()
        annotations["sample_id"] = annotations["id"].astype(str)
        annotations = annotations[
            annotations["answer_format"].isin(TRACKED_FORMATS)
        ].copy()

        previous_path = previous_root / dataset / "predictions.parquet"
        previous = pd.read_parquet(
            previous_path,
            columns=["sample_id", "image_path", "raw_model_output"],
        ).copy()
        previous["sample_id"] = previous["sample_id"].astype(str)
        merged = annotations.merge(
            previous,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if merged["image_path"].isna().any():
            missing = merged.loc[merged["image_path"].isna(), "sample_id"].tolist()
            raise RuntimeError(
                f"{dataset}: previous run lacks {len(missing)} diagnostic rows"
            )
        merged["dataset"] = dataset
        dataset_frames.append(merged)

    candidates = pd.concat(dataset_frames, ignore_index=True)
    dimensions = _image_dimensions(candidates["image_path"].astype(str).tolist())
    candidates["source_width"] = [
        dimensions[str(path)][0] for path in candidates["image_path"]
    ]
    candidates["source_height"] = [
        dimensions[str(path)][1] for path in candidates["image_path"]
    ]
    candidates["source_pixels"] = (
        candidates["source_width"] * candidates["source_height"]
    )
    candidates["affected_by_baseline_cap"] = (
        candidates["source_pixels"] > baseline_max_pixels
    )
    candidates["prior_correct"] = [
        score_answer(raw, answer, fmt, fo_names)
        for raw, answer, fmt in zip(
            candidates["raw_model_output"],
            candidates["answer"],
            candidates["answer_format"],
        )
    ]
    candidates = candidates.rename(
        columns={"raw_model_output": "prior_raw_model_output"}
    )

    affected = candidates[candidates["affected_by_baseline_cap"]].copy()
    affected["selection_group"] = "affected_full"
    affected["selection_stratum"] = (
        affected["dataset"].astype(str)
        + "/"
        + affected["answer_format"].astype(str)
    )

    controls: list[pd.DataFrame] = []
    used_images: set[str] = set()
    stratum_index = 0
    for dataset in ("heico", "lapchole"):
        for answer_format in TRACKED_FORMATS:
            pool = candidates[
                (~candidates["affected_by_baseline_cap"])
                & (candidates["dataset"] == dataset)
                & (candidates["answer_format"] == answer_format)
                & (~candidates["image_path"].astype(str).isin(used_images))
            ].copy()
            pool = pool.drop_duplicates("image_path")
            if len(pool) < control_per_stratum:
                raise RuntimeError(
                    f"Only {len(pool)} controls for {dataset}/{answer_format}; "
                    f"requested {control_per_stratum}"
                )
            sample = pool.sample(
                n=control_per_stratum,
                random_state=seed + stratum_index,
            ).copy()
            sample["selection_group"] = "control_sample"
            sample["selection_stratum"] = f"{dataset}/{answer_format}"
            controls.append(sample)
            used_images.update(sample["image_path"].astype(str))
            stratum_index += 1

    selection = pd.concat([affected, *controls], ignore_index=True)
    selection = selection.sort_values(
        ["affected_by_baseline_cap", "source_pixels", "dataset", "sample_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selection.insert(0, "selection_index", range(len(selection)))
    selection["answer"] = selection["answer"].astype(str)
    selection["image_path"] = selection["image_path"].astype(str)
    selection["instruction"] = [
        build_instruction(question, fmt, fo_names)
        for question, fmt in zip(
            selection["question"], selection["answer_format"]
        )
    ]

    keep = [
        "selection_index",
        "selection_group",
        "selection_stratum",
        "dataset",
        "sample_id",
        "question",
        "answer",
        "answer_format",
        "primary_capability",
        "secondary_capabilities",
        "video",
        "timestamp_start",
        "timestamp_end",
        "image_path",
        "source_width",
        "source_height",
        "source_pixels",
        "affected_by_baseline_cap",
        "prior_correct",
        "prior_raw_model_output",
        "instruction",
    ]
    selection = selection[keep]
    log.info(
        "selection: %d affected questions + %d controls = %d",
        len(affected),
        sum(len(frame) for frame in controls),
        len(selection),
    )
    return selection


def model_messages(instruction: str, system_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def infer_batch(
    model: Any,
    processor: Any,
    batch: pd.DataFrame,
    system_prompt: str,
    max_new_tokens: int,
    device: Any,
) -> tuple[list[dict[str, Any]], float]:
    import torch

    started = time.monotonic()
    images: list[Image.Image] = []
    prompts: list[str] = []
    for row in batch.itertuples(index=False):
        with Image.open(row.image_path) as image:
            images.append(image.convert("RGB"))
        prompts.append(
            processor.apply_chat_template(
                model_messages(row.instruction, system_prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    grids = inputs["image_grid_thw"].detach().cpu().tolist()
    prompt_length = int(inputs["input_ids"].shape[1])

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    decoded = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    elapsed = time.monotonic() - started

    image_processor = processor.image_processor
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    records: list[dict[str, Any]] = []
    for raw, grid in zip(decoded, grids):
        grid_t, grid_h, grid_w = (int(value) for value in grid)
        records.append(
            {
                "raw_model_output": raw.strip(),
                "processed_width": grid_w * patch_size,
                "processed_height": grid_h * patch_size,
                "processed_pixels": grid_w * grid_h * patch_size * patch_size,
                "merged_visual_tokens": (
                    grid_t * grid_h * grid_w // (merge_size * merge_size)
                ),
            }
        )

    del generated, inputs, images
    return records, elapsed


def run_condition(
    condition: Condition,
    model: Any,
    base_path: Path,
    selection: pd.DataFrame,
    system_prompt: str,
    fo_names: tuple[str, ...],
    batch_size: int,
    max_new_tokens: int,
    device: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch
    from transformers import AutoProcessor

    log.info(
        "condition=%s max_pixels=%d: loading processor",
        condition.name,
        condition.max_pixels,
    )
    processor = AutoProcessor.from_pretrained(
        base_path,
        max_pixels=condition.max_pixels,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    warmup = selection.iloc[: min(batch_size, len(selection))]
    log.info("condition=%s: warm-up batch=%d", condition.name, len(warmup))
    with torch.inference_mode():
        infer_batch(
            model,
            processor,
            warmup,
            system_prompt,
            max_new_tokens,
            device,
        )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    idle_allocated = torch.cuda.memory_allocated(0)
    idle_reserved = torch.cuda.memory_reserved(0)
    torch.cuda.reset_peak_memory_stats(0)

    output_records: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(selection), batch_size):
            batch = selection.iloc[start : start + batch_size]
            generated, batch_elapsed = infer_batch(
                model,
                processor,
                batch,
                system_prompt,
                max_new_tokens,
                device,
            )
            per_question = batch_elapsed / max(len(batch), 1)
            for row, generated_row in zip(
                batch.itertuples(index=False), generated
            ):
                normalized = normalize_answer(
                    generated_row["raw_model_output"],
                    row.answer_format,
                    fo_names,
                )
                output_records.append(
                    {
                        "selection_index": row.selection_index,
                        "condition": condition.name,
                        "max_pixels": condition.max_pixels,
                        "normalized_prediction": normalized,
                        "correct": score_answer(
                            normalized,
                            row.answer,
                            row.answer_format,
                            fo_names,
                        ),
                        "latency_sec": per_question,
                        **generated_row,
                    }
                )
            completed = min(start + len(batch), len(selection))
            if completed % 128 == 0 or completed == len(selection):
                log.info(
                    "condition=%s progress=%d/%d",
                    condition.name,
                    completed,
                    len(selection),
                )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    current_allocated = torch.cuda.memory_allocated(0)
    current_reserved = torch.cuda.memory_reserved(0)

    predictions = selection.merge(
        pd.DataFrame(output_records),
        on="selection_index",
        how="left",
        validate="one_to_one",
    )
    if predictions["normalized_prediction"].isna().any():
        raise RuntimeError(f"{condition.name}: missing predictions")

    memory = {
        "condition": condition.name,
        "max_pixels": condition.max_pixels,
        "questions": len(predictions),
        "batch_size": batch_size,
        "wall_seconds": elapsed,
        "questions_per_second": len(predictions) / elapsed,
        "idle_allocated_bytes": idle_allocated,
        "idle_reserved_bytes": idle_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "current_allocated_bytes": current_allocated,
        "current_reserved_bytes": current_reserved,
        "idle_allocated_gib": idle_allocated / 1024**3,
        "idle_reserved_gib": idle_reserved / 1024**3,
        "peak_allocated_gib": peak_allocated / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "current_allocated_gib": current_allocated / 1024**3,
        "current_reserved_gib": current_reserved / 1024**3,
    }
    log.info(
        "condition=%s done: %.1fs %.2f Q/s peak_allocated=%.2f GiB "
        "peak_reserved=%.2f GiB",
        condition.name,
        elapsed,
        memory["questions_per_second"],
        memory["peak_allocated_gib"],
        memory["peak_reserved_gib"],
    )

    del processor
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, memory


def accuracy_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, condition_frame in predictions.groupby("condition", sort=False):
        for group_name, group_frame in condition_frame.groupby(
            "selection_group", sort=False
        ):
            slices = [("all", group_frame)]
            slices.extend(
                (fmt, part)
                for fmt, part in group_frame.groupby("answer_format", sort=False)
            )
            for answer_format, part in slices:
                rows.append(
                    {
                        "condition": condition,
                        "selection_group": group_name,
                        "answer_format": answer_format,
                        "count": len(part),
                        "correct": int(part["correct"].sum()),
                        "accuracy": float(part["correct"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def comparison_rows(
    predictions: pd.DataFrame, baseline_name: str, high_name: str
) -> pd.DataFrame:
    static_columns = [
        "selection_index",
        "selection_group",
        "answer_format",
    ]
    baseline = predictions[predictions["condition"] == baseline_name][
        static_columns + ["correct", "normalized_prediction"]
    ].rename(
        columns={
            "correct": "baseline_correct",
            "normalized_prediction": "baseline_prediction",
        }
    )
    high = predictions[predictions["condition"] == high_name][
        ["selection_index", "correct", "normalized_prediction"]
    ].rename(
        columns={
            "correct": "high_correct",
            "normalized_prediction": "high_prediction",
        }
    )
    paired = baseline.merge(
        high, on="selection_index", how="inner", validate="one_to_one"
    )
    paired["answer_changed"] = (
        paired["baseline_prediction"] != paired["high_prediction"]
    )

    rows: list[dict[str, Any]] = []
    for group_name, group_frame in paired.groupby("selection_group", sort=False):
        slices = [("all", group_frame)]
        slices.extend(
            (fmt, part)
            for fmt, part in group_frame.groupby("answer_format", sort=False)
        )
        for answer_format, part in slices:
            baseline_correct = part["baseline_correct"].astype(bool)
            high_correct = part["high_correct"].astype(bool)
            baseline_accuracy = float(baseline_correct.mean())
            high_accuracy = float(high_correct.mean())
            rows.append(
                {
                    "selection_group": group_name,
                    "answer_format": answer_format,
                    "count": len(part),
                    "baseline_accuracy": baseline_accuracy,
                    "high_accuracy": high_accuracy,
                    "delta_percentage_points": (
                        high_accuracy - baseline_accuracy
                    )
                    * 100.0,
                    "fixed": int((~baseline_correct & high_correct).sum()),
                    "regressed": int((baseline_correct & ~high_correct).sum()),
                    "unchanged_correct": int(
                        (baseline_correct & high_correct).sum()
                    ),
                    "unchanged_wrong": int(
                        (~baseline_correct & ~high_correct).sum()
                    ),
                    "answer_changed": int(part["answer_changed"].sum()),
                }
            )
    return pd.DataFrame(rows)


def geometry_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "selection_group",
        "source_width",
        "source_height",
        "processed_width",
        "processed_height",
        "processed_pixels",
        "merged_visual_tokens",
    ]
    return (
        predictions[columns]
        .groupby(columns, dropna=False)
        .size()
        .reset_index(name="question_count")
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = [column.replace("_", " ") for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_summary(
    output: Path,
    selection: pd.DataFrame,
    accuracy: pd.DataFrame,
    comparison: pd.DataFrame,
    geometry: pd.DataFrame,
    memory: list[dict[str, Any]],
    conditions: list[Condition],
) -> None:
    memory_frame = pd.DataFrame(memory)
    affected_count = int(
        (selection["selection_group"] == "affected_full").sum()
    )
    control_count = int(
        (selection["selection_group"] == "control_sample").sum()
    )
    lines = [
        "# Frame native-resolution diagnostic",
        "",
        (
            f"Paired BF16 inference on {affected_count} questions in the complete "
            "current test subset where native frames exceed the baseline pixel cap, "
            f"plus {control_count} deterministic unaffected controls. Only `number` "
            "and `fo_class` are included, so scoring is deterministic and does not "
            "use an LLM judge."
        ),
        "",
        (
            f"Conditions: `{conditions[0].name}` = {conditions[0].max_pixels:,} "
            f"max pixels; `{conditions[1].name}` = "
            f"{conditions[1].max_pixels:,} max pixels. Source images were never "
            "upscaled, cropped, or tiled."
        ),
        "",
        "## Paired comparison",
        "",
        markdown_table(
            comparison,
            [
                "selection_group",
                "answer_format",
                "count",
                "baseline_accuracy",
                "high_accuracy",
                "delta_percentage_points",
                "fixed",
                "regressed",
                "answer_changed",
            ],
        ),
        "",
        "## Accuracy",
        "",
        markdown_table(
            accuracy,
            [
                "condition",
                "selection_group",
                "answer_format",
                "count",
                "correct",
                "accuracy",
            ],
        ),
        "",
        "## Runtime and CUDA memory",
        "",
        markdown_table(
            memory_frame,
            [
                "condition",
                "max_pixels",
                "questions",
                "batch_size",
                "wall_seconds",
                "questions_per_second",
                "peak_allocated_gib",
                "peak_reserved_gib",
            ],
        ),
        "",
        "## Processor geometry",
        "",
        markdown_table(
            geometry,
            [
                "condition",
                "selection_group",
                "source_width",
                "source_height",
                "processed_width",
                "processed_height",
                "processed_pixels",
                "merged_visual_tokens",
                "question_count",
            ],
        ),
        "",
        (
            "The affected result is complete for the two targeted answer formats, "
            "but it is not the challenge-wide 6,252-question Frame score. The "
            "control group is sampled and should be used only as a no-change check."
        ),
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for path_name in ("base", "adapter", "definitions", "previous_root"):
        path = getattr(args, path_name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path_name} not found: {path}")
        setattr(args, path_name, path)
    if args.baseline_max_pixels >= args.high_max_pixels:
        raise ValueError("high-max-pixels must exceed baseline-max-pixels")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    definitions = read_definitions(args.definitions)
    fo_names = parse_fo_names(definitions)
    system_prompt = (
        SYSTEM_PROMPT
        + "\n\nUse these challenge-provided foreign-object definitions:\n"
        + definitions
    )
    selection = build_selection(
        repo=repo,
        previous_root=args.previous_root,
        baseline_max_pixels=args.baseline_max_pixels,
        control_per_stratum=args.control_per_stratum,
        seed=args.seed,
        fo_names=fo_names,
    )
    selection.to_parquet(output / "selection.parquet", index=False)
    selection.to_csv(output / "selection.csv", index=False)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    properties = torch.cuda.get_device_properties(0)
    log.info(
        "loading model: base=%s adapter=%s dtype=%s GPU=%s %.2f GiB",
        args.base,
        args.adapter,
        args.dtype,
        properties.name,
        properties.total_memory / 1024**3,
    )
    load_started = time.monotonic()
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
        local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.monotonic() - load_started
    log.info(
        "model loaded in %.2fs; allocated=%.2f GiB reserved=%.2f GiB",
        model_load_seconds,
        torch.cuda.memory_allocated(0) / 1024**3,
        torch.cuda.memory_reserved(0) / 1024**3,
    )

    conditions = [
        Condition(f"baseline_{args.baseline_max_pixels}", args.baseline_max_pixels),
        Condition(f"native_high_{args.high_max_pixels}", args.high_max_pixels),
    ]
    predictions_by_condition: list[pd.DataFrame] = []
    memory: list[dict[str, Any]] = []
    for condition in conditions:
        predictions, condition_memory = run_condition(
            condition=condition,
            model=model,
            base_path=args.base,
            selection=selection,
            system_prompt=system_prompt,
            fo_names=fo_names,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        predictions.to_parquet(
            output / f"{condition.name}-predictions.parquet", index=False
        )
        predictions.to_csv(
            output / f"{condition.name}-predictions.csv", index=False
        )
        predictions_by_condition.append(predictions)
        memory.append(condition_memory)
        (output / "memory.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )

    all_predictions = pd.concat(predictions_by_condition, ignore_index=True)
    accuracy = accuracy_rows(all_predictions)
    comparison = comparison_rows(
        all_predictions, conditions[0].name, conditions[1].name
    )
    geometry = geometry_rows(all_predictions)
    accuracy.to_csv(output / "accuracy.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    geometry.to_csv(output / "geometry.csv", index=False)
    write_summary(
        output=output,
        selection=selection,
        accuracy=accuracy,
        comparison=comparison,
        geometry=geometry,
        memory=memory,
        conditions=conditions,
    )

    meta = {
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_revision": git_revision(repo),
        "repo": str(repo),
        "base": str(args.base),
        "adapter": str(args.adapter),
        "base_source": os.environ.get("ORENA_BASE_SOURCE", ""),
        "adapter_source": os.environ.get("ORENA_ADAPTER_SOURCE", ""),
        "definitions": str(args.definitions),
        "previous_root": str(args.previous_root),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "control_per_stratum": args.control_per_stratum,
        "seed": args.seed,
        "conditions": [asdict(condition) for condition in conditions],
        "model_load_seconds": model_load_seconds,
        "gpu_name": properties.name,
        "gpu_total_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "selection_rows": len(selection),
        "affected_rows": int(
            (selection["selection_group"] == "affected_full").sum()
        ),
        "control_rows": int(
            (selection["selection_group"] == "control_sample").sum()
        ),
    }
    (output / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (output / "COMPLETE").touch()
    log.info("complete -> %s", output)


if __name__ == "__main__":
    main()
