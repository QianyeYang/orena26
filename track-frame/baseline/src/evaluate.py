#!/usr/bin/env python
"""Evaluate FRAME baseline responses with ``focus.Evaluator``.

Open_ended / multiple_choice are graded by an LLM judge (Qwen3.5-4B) — run on a
GPU node. Writes results.csv + summary.csv (per-capability / per-format / overall
with bootstrap CIs) to ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from focus import Evaluator, load_responses  # noqa: E402
from src import data  # noqa: E402
from src.paths import DATASET, DATASETS, OS_MODELS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate FRAME baseline responses")
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=DATASET, choices=DATASETS)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N questions (smoke tests)")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--judge-model", default=None,
                    help="default: os-models/Qwen3.5-4B if present, else the HF id")
    a = ap.parse_args()

    reqs, refs = data.load_split(a.track, a.split, a.dataset)
    if a.limit is not None:
        reqs, refs = reqs[: a.limit], refs[: a.limit]
    responses = load_responses(a.responses)

    judge_model = a.judge_model
    if judge_model is None:
        local = OS_MODELS_DIR / "Qwen3.5-4B"
        judge_model = str(local) if local.exists() else "Qwen/Qwen3.5-4B"

    ev = Evaluator(judge_kwargs={"model_name": judge_model, "device": a.judge_device})
    results_df, summary_df = ev.run(reqs, refs, responses, output_dir=a.out)

    print(summary_df.to_string(index=False))
    overall = summary_df[summary_df.level == "overall"]
    if not overall.empty:
        r = overall.iloc[0]
        print(f"\nOVERALL accuracy = {r.accuracy:.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]  n={int(r['count'])}")


if __name__ == "__main__":
    main()
