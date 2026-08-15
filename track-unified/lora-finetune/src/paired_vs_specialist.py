"""Paired per-video comparison of the unified LoRA against each specialist.

The official metric is a macro over videos, so the marginal confidence intervals
in `summary.csv` are wide (effective n = videos, not questions) and comparing
them across two runs is far too conservative: both runs answered *the same*
questions on *the same* videos, so the comparison should be paired.

For each (track, dataset, epoch) this reports:

* the macro-over-video accuracy of each run, reconstructed from `results.csv`
  and checked against the stored `overall,MEAN`,
* the mean paired per-video delta with a Wilcoxon signed-rank p-value,
* McNemar's test over the individual questions that flipped.

    python track-unified/lora-finetune/src/paired_vs_specialist.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official"

# Specialist full-test results, and which column carries the corrected
# correctness.  FRAME was scored one day before the multi-label fo_class fix
# landed, so its authoritative column is `correctness_new` in the rescore file
# (see result-summary/frame/epoch30-multilabel-rescore.md).
SPECIALISTS = {
    "frame": (
        REPO / "track-frame/lora-finetune/logs/full-test-comparison/new-epoch30",
        "eval/rescore-results.csv",
        "correctness_new",
    ),
    "segment": (
        REPO / "track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_6",
        "eval/results.csv",
        "correctness",
    ),
    "procedure": (
        REPO / "track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_8",
        "eval/results.csv",
        "correctness",
    ),
}
DATASETS = ("heico", "lapchole")


def load(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"qID": str}, low_memory=False)
    out = frame[["qID", "video", column]].rename(columns={column: "correct"})
    out["correct"] = out["correct"].astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    )
    if out["correct"].isna().any():
        raise SystemExit(f"unparsed correctness values in {path}")
    return out


def macro(frame: pd.DataFrame) -> float:
    return float(frame.groupby("video")["correct"].mean().mean())


def stored_overall(summary_path: Path) -> float | None:
    if not summary_path.is_file():
        return None
    summary = pd.read_csv(summary_path)
    row = summary[(summary["level"] == "overall") & (summary["name"] == "MEAN")]
    return None if row.empty else float(row["accuracy"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, nargs="+", default=[4, 8, 12])
    args = ap.parse_args()

    print(
        f"{'track/dataset':<22}{'epoch':>6}{'unified':>9}{'special':>9}"
        f"{'Δmacro':>9}{'wilcoxon':>10}{'mcnemar':>10}{'w>s':>6}{'s>w':>6}{'vids':>6}"
    )
    print("-" * 93)

    for track, (spec_root, spec_rel, spec_col) in SPECIALISTS.items():
        for dataset in DATASETS:
            spec_path = spec_root / dataset / spec_rel
            if not spec_path.is_file():
                print(f"{track}/{dataset:<14} specialist results missing: {spec_path}")
                continue
            spec = load(spec_path, spec_col)
            spec_macro = macro(spec)
            spec_stored = stored_overall(spec_root / dataset / "eval" / "summary.csv")

            for epoch in args.epochs:
                uni_path = RUN / f"full_test/epoch_{epoch}/{track}/{dataset}/eval/results.csv"
                if not uni_path.is_file():
                    continue
                uni = load(uni_path, "correctness")
                uni_macro = macro(uni)

                merged = uni.merge(spec, on="qID", suffixes=("_u", "_s"))
                if len(merged) != len(uni):
                    raise SystemExit(
                        f"{track}/{dataset} epoch {epoch}: qID join lost rows "
                        f"({len(merged)} != {len(uni)})"
                    )

                per_video = merged.groupby("video_u")[["correct_u", "correct_s"]].mean()
                diffs = per_video["correct_u"] - per_video["correct_s"]
                if np.allclose(diffs, 0):
                    w_p = 1.0
                else:
                    w_p = float(stats.wilcoxon(diffs, zero_method="zsplit").pvalue)

                # McNemar over questions: b = unified right / specialist wrong.
                b = int(((merged["correct_u"] == 1) & (merged["correct_s"] == 0)).sum())
                c = int(((merged["correct_u"] == 0) & (merged["correct_s"] == 1)).sum())
                m_p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

                print(
                    f"{track + '/' + dataset:<22}{epoch:>6}{uni_macro:>9.4f}"
                    f"{spec_macro:>9.4f}{float(diffs.mean()):>+9.4f}"
                    f"{w_p:>10.4f}{m_p:>10.2e}{b:>6}{c:>6}{len(per_video):>6}"
                )

            if spec_stored is not None and abs(spec_stored - spec_macro) > 5e-4:
                print(
                    f"  ! {track}/{dataset}: reconstructed specialist macro "
                    f"{spec_macro:.4f} != stored {spec_stored:.4f}"
                )
    print(
        "\nΔmacro = mean over videos of (unified - specialist) per-video accuracy.\n"
        "wilcoxon = signed-rank over videos; mcnemar = binomial over flipped questions.\n"
        "w>s / s>w = questions the unified model got right and the specialist wrong, and vice versa."
    )


if __name__ == "__main__":
    main()
