#!/usr/bin/env python
"""FRAME LoRA inference runner: parquet -> frame -> base + LoRA adapter -> rich rows.

Mirrors the zero-shot sweep runner's outputs so the existing
``track-frame/baseline/src/evaluate.py`` and the visualiser work unchanged:
  - ``predictions.parquet`` : one rich row per sample (raw + normalised + meta)
  - ``responses.json``      : focus-compatible (qID, content, latency)
  - ``requests.json`` / ``meta.json``

``answer_format`` is input metadata (ships in the question parquet), so this needs
no reference answers — it works on the hidden test too. Run on a GPU node
(see ../scripts/infer.slurm).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # shared `src`
sys.path.insert(0, str(HERE))  # sibling modules (infer)

from focus import save_items  # noqa: E402
from src import adapter, counting, data, frames  # noqa: E402
from src import prompts as P  # noqa: E402
from src.paths import DATASET, DATASETS  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("lora-infer")


def _sec_caps(value) -> str:
    if value is None:
        return ""
    try:
        return "|".join(str(v) for v in list(value))
    except TypeError:
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME LoRA inference runner")
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=DATASET, choices=DATASETS)
    ap.add_argument("--base", required=True, help="base model dir (os-models or /dev/shm stage)")
    ap.add_argument("--adapter", default=None, help="trained PEFT adapter dir (omit = base only)")
    ap.add_argument("--model-name", default="Qwen2.5-VL-7B-Instruct-lora")
    ap.add_argument(
        "--prompt-strategy",
        default="direct",
        choices=P.PROMPT_STRATEGIES,
        help="direct answer or structured localize-then-count prompt",
    )
    ap.add_argument(
        "--question-filter",
        default="all",
        choices=("all", "counting"),
        help="run every row or only recognized FRAME counting rows",
    )
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None, help="cap image tokens, e.g. 602112")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="run the first selected row once without recording it before timed inference",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = data.read_parquet(a.track, a.split, a.dataset).to_dict("records")
    if a.question_filter == "counting":
        rows = [
            row
            for row in rows
            if counting.classify_counting_question(
                str(row["question"]), str(row["answer_format"])
            )
            is not None
        ]
    if a.limit:
        rows = rows[: a.limit]
    reqs = [data.row_to_request(r) for r in rows]
    save_items(reqs, out / "requests.json")
    system_prompt = P.system_prompt(a.track, a.prompt_strategy)
    (out / "meta.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(a).items()},
                "selected_rows": len(rows),
                "system_prompt": system_prompt,
            },
            indent=2,
        )
    )

    import torch  # noqa: E402

    log.info("torch %s | cuda=%s | dev=%s | adapter=%s", torch.__version__,
             torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", a.adapter)

    import infer as I  # noqa: E402 — defer heavy import until after arg parsing

    vlm = I.LoRAVLM(
        a.base,
        adapter_path=a.adapter,
        dtype=a.dtype,
        max_new_tokens=a.max_new_tokens,
        min_pixels=a.min_pixels,
        max_pixels=a.max_pixels,
        trust_remote_code=a.trust_remote_code,
    )

    if a.warmup and rows:
        warmup_req = reqs[0]
        warmup_row = rows[0]
        warmup_image = frames.request_frame_paths(
            warmup_req,
            frames_folder=a.frames_folder,
            dataset=a.dataset,
        )[0]
        if warmup_image.exists():
            warmup_instruction, _ = P.build_instruction(
                warmup_req.question,
                warmup_row["answer_format"],
                prompt_strategy=a.prompt_strategy,
            )
            log.info("running one unscored warm-up generation")
            warmup_raw, warmup_latency = vlm.answer(
                str(warmup_image),
                warmup_instruction,
                system=system_prompt,
            )
            log.info(
                "warm-up complete in %.2fs: %r",
                warmup_latency,
                warmup_raw[:80],
            )
        else:
            log.warning("warm-up skipped; frame is missing: %s", warmup_image)

    records: list[dict] = []
    responses = []
    n_parse = n_err = n_structured = n_structured_valid = 0
    t0 = time.time()

    for i, (req, row) in enumerate(zip(reqs, rows)):
        fmt = row["answer_format"]
        img = frames.request_frame_paths(
            req, frames_folder=a.frames_folder, dataset=a.dataset
        )[0]
        count_question = counting.classify_counting_question(req.question, fmt)
        instr, opts = P.build_instruction(
            req.question, fmt, prompt_strategy=a.prompt_strategy
        )

        raw, lat, err = "", 0.0, ""
        if not img.exists():
            err = f"missing frame: {img}"
            log.warning("qID=%s %s", req.qID, err)
        else:
            try:
                raw, lat = vlm.answer(str(img), instr, system=system_prompt)
            except Exception as e:  # one bad sample must not kill the run
                err = f"{type(e).__name__}: {e}"
                log.warning("qID=%s inference error: %s", req.qID, err)

        structured_result = None
        if a.prompt_strategy == "bbox-json" and count_question is not None:
            structured_result = counting.derive_structured_count(raw, count_question)
            resp = adapter.build_response(
                req.qID,
                structured_result.answer,
                fmt,
                latency=lat,
                options=opts,
            )
            n_structured += 1
            n_structured_valid += int(structured_result.schema_valid)
        else:
            resp = adapter.build_response(req.qID, raw, fmt, latency=lat, options=opts)
        responses.append(resp)
        n_err += int(bool(err))
        n_parse += int(adapter.is_parseable(fmt, resp.content))

        records.append(
            {
                "dataset": a.dataset,
                "sample_id": req.qID,
                "model_name": a.model_name,
                "question": req.question,
                "answer": str(row["answer"]),
                "answer_format": fmt,
                "primary_capability": str(row["primary_capability"]),
                "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                "clinical_relevance": bool(row["clinical_relevance"]),
                "ood": bool(row["ood"]),
                "video": row["video"],
                "timestamp_start": str(row["timestamp_start"]),
                "timestamp_end": str(row["timestamp_end"]),
                "image_path": str(img),
                "prompt": instr,
                "prompt_strategy": a.prompt_strategy,
                "raw_model_output": raw,
                "normalized_prediction": resp.content,
                "prediction": resp.content,
                "counting_mode": (
                    count_question.mode.value if count_question is not None else ""
                ),
                "counting_target": (
                    count_question.target
                    if count_question is not None and count_question.target is not None
                    else ""
                ),
                "structured_output_valid": (
                    structured_result.schema_valid
                    if structured_result is not None
                    else None
                ),
                "structured_output_status": (
                    structured_result.status if structured_result is not None else ""
                ),
                "detected_objects": (
                    structured_result.detections_json
                    if structured_result is not None
                    else ""
                ),
                "derived_count": (
                    structured_result.count if structured_result is not None else None
                ),
                "latency_sec": lat,
                "error": err,
            }
        )

        if i % 100 == 0:
            structured_log = (
                f" structured={structured_result.status}"
                if structured_result is not None
                else ""
            )
            log.info(
                "%d/%d fmt=%s raw=%r -> %r (%.2fs)%s%s",
                i + 1,
                len(reqs),
                fmt,
                raw[:80],
                resp.content[:40],
                lat,
                structured_log,
                f" ERR={err[:40]}" if err else "",
            )

    pd.DataFrame(records).to_parquet(out / "predictions.parquet", index=False)
    save_items(responses, out / "responses.json")

    dt = time.time() - t0
    log.info(
        "done: %d rows | %.1fs | %.2fs/Q | parseable=%d/%d | errors=%d | "
        "structured-valid=%d/%d -> %s",
        len(records),
        dt,
        dt / max(len(reqs), 1),
        n_parse,
        len(reqs),
        n_err,
        n_structured_valid,
        n_structured,
        out / "predictions.parquet",
    )


if __name__ == "__main__":
    main()
