#!/usr/bin/env python
"""SEGMENT zero-shot baseline: clip window -> timestamped frames -> VLM -> rich rows.

Thin CLI over the shared ``src.videovqa.run_video_qa`` loop with SEGMENT
defaults: 1-fps extraction grid (stride 25 @ 25 fps), <= 64 evenly-subsampled
frames per clip, ``max_pixels`` 262144 (960x540 -> 672x384 -> 252 tokens/frame).

Outputs (``predictions.parquet`` / ``responses.json`` / ``predictions.jsonl``
resume log) follow the zeroshot-sweep schema, so the shared evaluator and the
visualiser work unchanged. Run on a GPU node (see ../scripts/).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # shared `src`

from src.videovqa import run_video_qa  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="SEGMENT zero-shot multi-frame VLM runner")
    ap.add_argument("--track", default="segment")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True, help="local path to the model dir")
    ap.add_argument("--model-name", required=True, help="label stored in rows / output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=25, help="grid: every Nth source frame")
    ap.add_argument("--max-frames", type=int, default=64, help="frames per clip after subsample")
    ap.add_argument("--max-pixels", type=int, default=262144, help="processor image-token cap")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    a = ap.parse_args()

    run_video_qa(
        track=a.track, split=a.split, model_path=a.model, model_name=a.model_name,
        out_dir=a.out, stride=a.stride, max_frames=a.max_frames,
        frames_folder=a.frames_folder, limit=a.limit, max_new_tokens=a.max_new_tokens,
        dtype=a.dtype, min_pixels=a.min_pixels, max_pixels=a.max_pixels,
        adapter_path=a.adapter,
    )


if __name__ == "__main__":
    main()
