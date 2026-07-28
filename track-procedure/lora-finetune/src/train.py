#!/usr/bin/env python
"""PROCEDURE LoRA SFT entry — thin config-driven wrapper over ``src.videovqa_sft``.

All logic (dataset with train/inference prompt parity, label-masking collator,
LoRA target auto-discovery, per-epoch seeded-subset eval) lives in the shared
module; this file only parses args + the YAML config. See ../architecture.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # shared `src`

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="PROCEDURE LoRA SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--track", default="procedure")
    ap.add_argument("--limit", type=int, default=None, help="first N train rows (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--base", default=None, help="override base path (e.g. a /dev/shm stage)")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume training: bare flag = latest checkpoint in --out; "
                         "or pass a specific checkpoint dir")
    ap.add_argument("--judge-model", default=None,
                    help="judge model dir; enables per-epoch subset inference + scoring")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="override the per-epoch eval subset size (smoke tests)")
    ap.add_argument("--frames-folder", default="frames")
    a = ap.parse_args()

    from src.videovqa_sft import run_training  # noqa: E402 — after sys.path insert

    cfg = yaml.safe_load(Path(a.config).read_text())
    run_training(
        config=cfg, track=a.track, out_dir=a.out, base_override=a.base, limit=a.limit,
        epochs_override=a.epochs, resume=a.resume, judge_model=a.judge_model,
        eval_limit=a.eval_limit, frames_folder=a.frames_folder,
    )


if __name__ == "__main__":
    main()
