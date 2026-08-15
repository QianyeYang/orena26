"""Bucket-level head-to-head: unified multi-task LoRA vs each track specialist.

`paired_vs_specialist.py` answers "is the unified model better overall?".  This
answers "*where* is it better or worse?" — the same paired rows, split by
capability group, leaf capability, and answer format.

Both models answered the same questions on the same videos, so every bucket is
compared paired: the delta is over identical rows, and the p-value is McNemar's
exact test over the questions that flipped inside that bucket.

    python track-unified/lora-finetune/src/unified_vs_specialist_buckets.py --epoch 8
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from focus.taxonomy import Capability  # noqa: E402

RUN = REPO / "track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official"

# (root, results file, correctness column).  FRAME was scored one day before the
# multi-label fo_class fix landed, so its authoritative column is
# `correctness_new` in the rescore file.
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

# results.csv carries the leaf capability in `primary`; the group is rolled up
# through the official taxonomy.  (`clinical` is a boolean flag, not a bucket.)
LEVELS = (("group", "group"), ("primary", "leaf"), ("answer_format", "answer_format"))

TRUTHY = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}


def to_group(leaf: str) -> str:
    cap = Capability.from_any(leaf)
    if cap is None:
        return "unknown"
    group = cap.group if cap.is_leaf else cap
    return group.value if group is not None else "unknown"


def binom_p(b: int, c: int) -> float:
    """Exact two-sided binomial test of b successes in b+c trials at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    peak = max(log_terms)
    tail = math.exp(peak) * sum(math.exp(term - peak) for term in log_terms)
    return min(1.0, 2.0 * tail)


def load(path: Path, column: str, with_buckets: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"qID": str}, low_memory=False)
    keep = ["qID", "video", column] + (["primary", "answer_format"] if with_buckets else [])
    out = frame[keep].rename(columns={column: "correct"})
    out["correct"] = out["correct"].astype(str).str.strip().str.lower().map(TRUTHY)
    if out["correct"].isna().any():
        raise SystemExit(f"unparsed correctness values in {path}")
    if with_buckets:
        out["group"] = out["primary"].astype(str).map(to_group)
    return out


def detect_convention(frame: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Return 'micro' or 'macro_video' — whichever reproduces summary.csv.

    Bucket accuracies could plausibly be a plain mean over questions or, like the
    official overall metric, a macro over videos.  Rather than assume, check both
    against the stored summary for this very run.  On HeiCo the two coincide
    (its videos carry equal question counts), so the LapChole runs decide it.
    """
    reference = summary[summary["level"] == "leaf"].set_index("name")["accuracy"]
    if reference.empty:
        return "micro"
    errors = {"micro": 0.0, "macro_video": 0.0}
    for name, expected in reference.items():
        sub = frame[frame["primary"] == name]
        if sub.empty:
            continue
        errors["micro"] += abs(sub["correct"].mean() - expected)
        errors["macro_video"] += abs(sub.groupby("video")["correct"].mean().mean() - expected)
    return min(errors, key=errors.get)


def accuracy(sub: pd.DataFrame, column: str, convention: str) -> float:
    if convention == "micro":
        return float(sub[column].mean())
    return float(sub.groupby("video")[column].mean().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=25,
                    help="hide buckets smaller than this (they are pure noise)")
    args = ap.parse_args()

    print(f"# Unified epoch {args.epoch} vs track specialists — bucket head-to-head\n")
    print(
        f"Paired on qID; Δ and McNemar are over identical rows. "
        f"Buckets with n < {args.min_n} are hidden. Sorted worst Δ first.\n"
    )

    for track, (spec_root, spec_rel, spec_col) in SPECIALISTS.items():
        for dataset in DATASETS:
            uni_dir = RUN / f"full_test/epoch_{args.epoch}/{track}/{dataset}/eval"
            spec_path = spec_root / dataset / spec_rel
            if not (uni_dir / "results.csv").is_file() or not spec_path.is_file():
                continue

            uni = load(uni_dir / "results.csv", "correctness", with_buckets=True)
            spec = load(spec_path, spec_col, with_buckets=False)
            summary = pd.read_csv(uni_dir / "summary.csv")
            convention = detect_convention(uni, summary)

            merged = uni.merge(spec[["qID", "correct"]], on="qID", suffixes=("_u", "_s"))
            if len(merged) != len(uni):
                raise SystemExit(
                    f"{track}/{dataset}: qID join lost rows ({len(merged)} != {len(uni)})"
                )

            print(f"## {track} / {dataset}  ({convention} bucket accuracy)\n")
            for column, level in LEVELS:
                rows = []
                for name, sub in merged.groupby(column):
                    if len(sub) < args.min_n:
                        continue
                    acc_u = accuracy(sub, "correct_u", convention)
                    acc_s = accuracy(sub, "correct_s", convention)
                    b = int(((sub["correct_u"] == 1) & (sub["correct_s"] == 0)).sum())
                    c = int(((sub["correct_u"] == 0) & (sub["correct_s"] == 1)).sum())
                    rows.append((name, len(sub), acc_u, acc_s, acc_u - acc_s,
                                 binom_p(b, c), b, c))
                if not rows:
                    continue
                rows.sort(key=lambda r: r[4])
                print(f"### {level}\n")
                print("| bucket | n | unified | specialist | Δ | McNemar p | uni✓spec✗ | uni✗spec✓ |")
                print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for name, n, acc_u, acc_s, diff, p, b, c in rows:
                    cell = f"**{p:.2g}**" if p < 0.05 else f"{p:.2g}"
                    print(
                        f"| {name} | {n} | {acc_u:.4f} | {acc_s:.4f} | {diff:+.4f} | "
                        f"{cell} | {b} | {c} |"
                    )
                print()


if __name__ == "__main__":
    main()
