#!/usr/bin/env python
"""FRAME baseline runner: parquet -> frame -> Qwen2.5-VL -> normalised responses.json.

`answer_format` is treated as input metadata (it is in the question parquet), so
this does not need the reference answers — it works for the real hidden test too.
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
sys.path.insert(0, str(HERE))  # sibling modules (prompts, model)

from focus import save_items  # noqa: E402
from src import adapter, data, frames  # noqa: E402
from src import prompts as P  # noqa: E402
from src.paths import OS_MODELS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("runner")


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME baseline VLM runner")
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=str(OS_MODELS_DIR / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-pixels", type=int, default=None, help="cap image tokens, e.g. 602112")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = data.read_parquet(a.track, a.split).to_dict("records")
    if a.limit:
        rows = rows[: a.limit]
    reqs = [data.row_to_request(r) for r in rows]
    fmts = [r["answer_format"] for r in rows]
    save_items(reqs, out / "requests.json")
    (out / "meta.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    import torch  # noqa: E402

    log.info("torch %s | cuda.is_available=%s | device=%s", torch.__version__,
             torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    import model as M  # noqa: E402  — defer heavy torch import until after arg parsing

    vlm = M.QwenVL(
        a.model, dtype=a.dtype, max_new_tokens=a.max_new_tokens, max_pixels=a.max_pixels
    )

    responses = []
    n_parse = 0
    t0 = time.time()
    for i, (req, fmt) in enumerate(zip(reqs, fmts)):
        img = frames.request_frame_paths(req, frames_folder=a.frames_folder)[0]
        instr, opts = P.build_instruction(req.question, fmt)
        if img.exists():
            raw, lat = vlm.answer(str(img), instr, system=P.SYSTEM_PROMPT)
        else:
            raw, lat = "", 0.0
            log.warning("missing frame for qID=%s: %s", req.qID, img)
        resp = adapter.build_response(req.qID, raw, fmt, latency=lat, options=opts)
        responses.append(resp)
        n_parse += int(adapter.is_parseable(fmt, resp.content))
        if i % 100 == 0:
            log.info("%d/%d fmt=%s raw=%r -> %r (%.2fs)",
                     i + 1, len(reqs), fmt, raw[:40], resp.content[:40], lat)

    save_items(responses, out / "responses.json")
    dt = time.time() - t0
    log.info(
        "done: %d responses | %.1fs total | %.2fs/Q | parseable=%d/%d | -> %s",
        len(responses), dt, dt / max(len(reqs), 1), n_parse, len(reqs), out / "responses.json",
    )


if __name__ == "__main__":
    main()
