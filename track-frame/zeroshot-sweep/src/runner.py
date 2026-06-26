#!/usr/bin/env python
"""FRAME zero-shot sweep runner: parquet -> frame -> <any VLM> -> rich rows.

Unlike the baseline runner (which saved only the normalised answer), this
preserves the RAW model output alongside the normalised prediction and full
per-sample metadata, for later error analysis.

Writes per model under ``--out``:
  - ``predictions.parquet`` : one rich row per sample (raw + normalised + meta)
  - ``responses.json``      : focus-compatible (qID, content, latency) so the
                              existing baseline ``evaluate.py`` scores it as-is
  - ``requests.json`` / ``meta.json``

``answer_format`` is input metadata (it ships in the question parquet), so this
needs no reference answers to run — but we DO read the parquet's gold ``answer``
and capability columns to embed them in the rich rows for analysis.

Run on a GPU node (see ../scripts/run_inference.slurm).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # shared `src`
sys.path.insert(0, str(HERE))  # sibling modules (vlm)

import pandas as pd  # noqa: E402

from focus import save_items  # noqa: E402
from src import adapter, data, frames  # noqa: E402
from src import prompts as P  # noqa: E402
from src.paths import OS_MODELS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("sweep")


def _sec_caps(value) -> str:
    """Normalise the (list | ndarray | None) secondary_capabilities to a '|' string."""
    if value is None:
        return ""
    try:
        return "|".join(str(v) for v in list(value))
    except TypeError:
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME zero-shot multi-VLM runner")
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True, help="local path to the model dir")
    ap.add_argument("--model-name", required=True, help="label stored in rows / output")
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None, help="cap image tokens, e.g. 602112")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = data.read_parquet(a.track, a.split).to_dict("records")
    if a.limit:
        rows = rows[: a.limit]
    reqs = [data.row_to_request(r) for r in rows]
    save_items(reqs, out / "requests.json")
    (out / "meta.json").write_text(
        json.dumps(
            {**{k: str(v) for k, v in vars(a).items()}, "system_prompt": P.SYSTEM_PROMPT},
            indent=2,
        )
    )

    import torch  # noqa: E402

    log.info(
        "torch %s | cuda=%s | device=%s | model=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        a.model_name,
    )

    import vlm as V  # noqa: E402 — defer heavy import until after arg parsing

    model = V.VLM(
        a.model,
        dtype=a.dtype,
        max_new_tokens=a.max_new_tokens,
        min_pixels=a.min_pixels,
        max_pixels=a.max_pixels,
        trust_remote_code=a.trust_remote_code,
    )

    records: list[dict] = []
    responses = []
    n_parse = n_err = 0
    t0 = time.time()

    for i, (req, row) in enumerate(zip(reqs, rows)):
        fmt = row["answer_format"]
        img = frames.request_frame_paths(req, frames_folder=a.frames_folder)[0]
        instr, opts = P.build_instruction(req.question, fmt)

        raw, lat, err = "", 0.0, ""
        if not img.exists():
            err = f"missing frame: {img}"
            log.warning("qID=%s %s", req.qID, err)
        else:
            try:
                raw, lat = model.answer(str(img), instr, system=P.SYSTEM_PROMPT)
            except Exception as e:  # one bad sample must not kill the sweep
                err = f"{type(e).__name__}: {e}"
                log.warning("qID=%s inference error: %s", req.qID, err)

        resp = adapter.build_response(req.qID, raw, fmt, latency=lat, options=opts)
        responses.append(resp)
        n_err += int(bool(err))
        n_parse += int(adapter.is_parseable(fmt, resp.content))

        records.append(
            {
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
                "raw_model_output": raw,           # NEVER overwritten by the normaliser
                "normalized_prediction": resp.content,
                "prediction": resp.content,        # evaluator-compatible alias
                "latency_sec": lat,
                "error": err,
            }
        )

        if i % 100 == 0:
            log.info(
                "%d/%d fmt=%s raw=%r -> %r (%.2fs)%s",
                i + 1, len(reqs), fmt, raw[:40], resp.content[:40], lat,
                f" ERR={err[:40]}" if err else "",
            )

    pd.DataFrame(records).to_parquet(out / "predictions.parquet", index=False)
    save_items(responses, out / "responses.json")

    dt = time.time() - t0
    log.info(
        "done: %d rows | %.1fs | %.2fs/Q | parseable=%d/%d | errors=%d | -> %s",
        len(records), dt, dt / max(len(reqs), 1), n_parse, len(reqs), n_err,
        out / "predictions.parquet",
    )


if __name__ == "__main__":
    main()
