#!/usr/bin/env python
"""Run the multi-view counting pass and save one row per (question, view).

Kept separate from `benchmark.py` because the two answer different questions.
`benchmark.py` measures a model as it would be deployed; this measures whether
*more views* beat one view, on the ~2,100 `number` rows that carry ~38% of the
final score.

The expensive part — one forward pass per view — is done once and stored
per-view, so `count_combine.py` can sweep combination rules offline instead of
re-running inference for each. That matters: with tiles and augmentations a row
costs 8 passes, and the rule that wins is not obvious in advance.

Latency stays inside budget by construction. A 20-question batch is allowed
120 s of setup plus 20x5 s; a 27B answers in ~0.5 s per view on an H200, so even
8 views per question is ~4 s, and only `number` rows need them at all.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import adapter, data, frames  # noqa: E402
from src import prompts as P  # noqa: E402
from src.counting import CountMode, classify_counting_question  # noqa: E402

import counting_v2 as CV  # noqa: E402
from generate import generate  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("views")


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME v2 multi-view counting pass")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt-strategy", default="v2", choices=list(P.PROMPT_STRATEGIES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"])
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N counting rows per dataset")
    ap.add_argument("--no-tiles", dest="tiles", action="store_false", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = data.read_parquets("frame", a.split, a.datasets)
    df = df[df["answer_format"] == "number"].reset_index(drop=True)
    if a.limit:
        df = df.groupby("_dataset", group_keys=False).head(a.limit).reset_index(drop=True)
    log.info("counting rows: %d %s", len(df), df["_dataset"].value_counts().to_dict())

    sys.path.insert(0, str(REPO / "track-frame" / "lora-finetune" / "src"))
    from infer import LoRAVLM  # noqa: E402

    t_load = time.time()
    model = LoRAVLM(
        a.model, adapter_path=a.adapter, dtype=a.dtype,
        max_new_tokens=a.max_new_tokens, max_pixels=a.max_pixels,
    )
    log.info("model loaded in %.1fs", time.time() - t_load)

    records: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(df.to_dict("records")):
        req = data.row_to_request(row)
        cq = classify_counting_question(req.question, row["answer_format"])
        if cq is None:  # not a shape we know how to decompose
            continue
        img_path = frames.request_frame_paths(
            req, frames_folder=a.frames_folder, dataset=row["_dataset"]
        )[0]
        if not img_path.exists():
            log.warning("qID=%s missing frame %s", req.qID, img_path)
            continue

        system = (
            P.system_prompt("frame", "v2", procedure_type=req.procedure_type)
            if a.prompt_strategy == "v2"
            else P.system_prompt("frame", a.prompt_strategy)
        )
        views = CV.build_views(str(img_path), tiles=a.tiles, augment=a.augment)
        for view in views:
            instr = CV.build_view_instruction(req.question, cq, view)
            try:
                answer, raw, lat = generate(
                    model.model, model.processor, model.device,
                    view.image, instr, system=system,
                    max_new_tokens=a.max_new_tokens, enable_thinking=a.thinking,
                )
                err = ""
            except Exception as e:
                answer, raw, lat, err = "", "", 0.0, f"{type(e).__name__}: {e}"
                log.warning("qID=%s view=%s error: %s", req.qID, view.view_id, err)

            if cq.mode is CountMode.CLASSES:
                value = adapter.normalize_fo_class(answer)
            else:
                value = adapter.normalize_number(answer)

            records.append(
                {
                    "sample_id": req.qID,
                    "dataset": row["_dataset"],
                    "video": row["video"],
                    "question": req.question,
                    "answer": str(row["answer"]),
                    "count_mode": cq.mode.value,
                    "target": cq.target or "",
                    "view_id": view.view_id,
                    "view_kind": view.kind,
                    "raw_model_output": raw,
                    "answer_text": answer,
                    "value": value,
                    "latency_sec": lat,
                    "error": err,
                }
            )

        if i % 50 == 0:
            log.info("%d/%d rows | %d view-records | %.1fs", i + 1, len(df), len(records), time.time() - t0)

    pd.DataFrame(records).to_parquet(out / "views.parquet", index=False)
    dt = time.time() - t0
    n_rows = df["id"].nunique() if "id" in df else len(df)
    (out / "meta.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(a).items()},
                "n_questions": int(n_rows),
                "n_view_records": len(records),
                "infer_sec": round(dt, 2),
                "sec_per_question": round(dt / max(n_rows, 1), 3),
            },
            indent=2,
        )
    )
    log.info("done: %d view-records over %d questions | %.1fs | -> %s",
             len(records), n_rows, dt, out / "views.parquet")


if __name__ == "__main__":
    main()
