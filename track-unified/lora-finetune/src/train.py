#!/usr/bin/env python
"""UNIFIED (frame+segment+procedure) LoRA SFT entry — thin wrapper.

All logic (per-track dataset dispatch with train/inference prompt parity,
per-item pixel budgets, label-masking collator, LoRA target auto-discovery,
per-epoch per-track seeded-subset eval) lives in ``src.videovqa_unified``;
this file only parses args + the YAML config. See ../architecture.md.
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
    ap = argparse.ArgumentParser(description="UNIFIED multi-track LoRA SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-track", type=int, default=None,
                    help="first N train rows per (track, dataset) shard (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--per-device-batch-size", type=int, default=None,
                    help="override per-GPU microbatch size")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override gradient accumulation steps")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="override optimizer-step checkpoint interval")
    ap.add_argument("--base", default=None, help="override base path (e.g. a /dev/shm stage)")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume training: bare flag = latest checkpoint in --out; "
                         "or pass a specific checkpoint dir")
    ap.add_argument("--judge-model", default=None,
                    help="judge model dir; enables per-epoch subset inference + scoring")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="override every track's per-epoch eval subset size (smoke tests)")
    ap.add_argument("--frames-folder", default="frames")
    a = ap.parse_args()

    from src.videovqa_unified import run_unified_training  # noqa: E402 — after sys.path insert

    cfg = yaml.safe_load(Path(a.config).read_text())
    run_unified_training(
        config=cfg, out_dir=a.out, base_override=a.base,
        limit_per_track=a.limit_per_track, epochs_override=a.epochs,
        per_device_batch_size_override=a.per_device_batch_size,
        grad_accum_override=a.grad_accum, save_steps_override=a.save_steps,
        resume=a.resume, judge_model=a.judge_model,
        eval_limit=a.eval_limit, frames_folder=a.frames_folder,
    )


if __name__ == "__main__":
    main()
