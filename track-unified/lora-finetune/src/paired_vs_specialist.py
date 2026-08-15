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
import math
from pathlib import Path

import numpy as np
import pandas as pd

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


def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value, normal approximation.

    Zeros are handled like scipy's `zero_method="zsplit"`: they are ranked with
    everything else and their rank mass is split evenly between the two sums.
    Ties in |diff| get average ranks and the variance is tie-corrected.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0 or np.allclose(diffs, 0.0):
        return 1.0

    absolute = np.abs(diffs)
    ranks = pd.Series(absolute).rank(method="average").to_numpy()
    zero_mass = 0.5 * ranks[diffs == 0].sum()
    r_plus = ranks[diffs > 0].sum() + zero_mass
    r_minus = ranks[diffs < 0].sum() + zero_mass

    statistic = min(r_plus, r_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _, tie_counts = np.unique(absolute, return_counts=True)
    var -= (tie_counts**3 - tie_counts).sum() / 48.0
    if var <= 0:
        return 1.0

    z = (statistic - mean) / math.sqrt(var)
    return min(1.0, 2.0 * _norm_sf(abs(z)))


def binom_p(b: int, c: int) -> float:
    """Exact two-sided binomial test of b successes in b+c trials at p=0.5.

    Symmetric around n/2, so the two-sided p is just twice the lower tail.
    Summed in log space so large n does not overflow.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(k + 1)
    ]
    peak = max(log_terms)
    tail = math.exp(peak) * sum(math.exp(term - peak) for term in log_terms)
    return min(1.0, 2.0 * tail)


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
                w_p = wilcoxon_p(diffs.to_numpy())

                # McNemar over questions: b = unified right / specialist wrong.
                b = int(((merged["correct_u"] == 1) & (merged["correct_s"] == 0)).sum())
                c = int(((merged["correct_u"] == 0) & (merged["correct_s"] == 1)).sum())
                m_p = binom_p(b, c)

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
