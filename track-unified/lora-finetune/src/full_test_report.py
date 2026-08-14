"""Collate the unified full-test sweep into a markdown report.

Reads every `full_test/epoch_<E>/<track>/<dataset>/eval/summary.csv` produced by
the three `full_test_*.slurm` arrays and emits HeiCo/LapChole separately, then
the capability (group) and answer-format breakdowns, then the delta against the
per-track specialist baselines.

    python track-unified/lora-finetune/src/full_test_report.py \
        --run track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official \
        > result-summary/unified/full-test-epoch-sweep.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRACKS = ("frame", "segment", "procedure")
DATASETS = ("heico", "lapchole")

# Per-track specialist baselines on the same official test splits, from
# result-summary/{frame,segment,procedure} (see docs for provenance).
BASELINES = {
    ("frame", "heico"): 0.7067,
    ("frame", "lapchole"): 0.6417,
    ("segment", "heico"): 0.7023,
    ("segment", "lapchole"): 0.7745,
    ("procedure", "heico"): 0.3105,
    ("procedure", "lapchole"): 0.5412,
}


def read_summary(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def overall(summary: pd.DataFrame, path: Path) -> tuple[float, int]:
    row = summary[(summary["level"] == "overall") & (summary["name"] == "MEAN")]
    if row.empty:
        raise SystemExit(f"no overall/MEAN row in {path}")
    return float(row["accuracy"].iloc[0]), int(row["count"].iloc[0])


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def delta(value: float | None, base: float) -> str:
    if value is None:
        return "—"
    return f"{value - base:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="training run directory")
    ap.add_argument("--epochs", type=int, nargs="+", default=[4, 8, 12])
    args = ap.parse_args()

    root = args.run / "full_test"
    summaries: dict[tuple[int, str, str], pd.DataFrame] = {}
    scores: dict[tuple[int, str, str], float] = {}
    counts: dict[tuple[int, str, str], int] = {}

    for epoch in args.epochs:
        for track in TRACKS:
            for dataset in DATASETS:
                path = root / f"epoch_{epoch}" / track / dataset / "eval" / "summary.csv"
                summary = read_summary(path)
                if summary is None:
                    continue
                summaries[(epoch, track, dataset)] = summary
                acc, n = overall(summary, path)
                scores[(epoch, track, dataset)] = acc
                counts[(epoch, track, dataset)] = n

    missing = [
        f"epoch_{e}/{t}/{d}"
        for e in args.epochs
        for t in TRACKS
        for d in DATASETS
        if (e, t, d) not in scores
    ]

    print("# Unified LoRA: full official test, epochs 4 / 8 / 12\n")
    print(
        "One 3-track LoRA (`Qwen3-VL-4B-Instruct-unified-both-official`) run over "
        "every official test question, judged by the local Qwen3.5-4B FOCUS judge. "
        "Headline number is `overall,MEAN` — the official per-video macro accuracy.\n"
    )
    if missing:
        print(f"**Incomplete:** {len(missing)} cell(s) missing — {', '.join(missing)}\n")

    print("## 1. Per dataset, per track\n")
    for track in TRACKS:
        print(f"### {track}\n")
        header = "| dataset | n | " + " | ".join(f"epoch {e}" for e in args.epochs)
        header += " | specialist | best Δ |"
        print(header)
        print("| --- | ---: | " + " | ".join("---:" for _ in args.epochs) + " | ---: | ---: |")
        for dataset in DATASETS:
            base = BASELINES[(track, dataset)]
            vals = [scores.get((e, track, dataset)) for e in args.epochs]
            n = next(
                (counts[(e, track, dataset)] for e in args.epochs if (e, track, dataset) in counts),
                None,
            )
            present = [v for v in vals if v is not None]
            best = max(present) if present else None
            cells = " | ".join(fmt(v) for v in vals)
            print(
                f"| {dataset} | {n if n is not None else '—'} | {cells} | "
                f"{base:.4f} | {delta(best, base)} |"
            )
        print()

    print("## 2. Track means (both datasets)\n")
    print("| track | " + " | ".join(f"epoch {e}" for e in args.epochs) + " | specialist |")
    print("| --- | " + " | ".join("---:" for _ in args.epochs) + " | ---: |")
    for track in TRACKS:
        base = sum(BASELINES[(track, d)] for d in DATASETS) / len(DATASETS)
        cells = []
        for epoch in args.epochs:
            vals = [scores.get((epoch, track, d)) for d in DATASETS]
            cells.append(fmt(sum(vals) / len(vals) if all(v is not None for v in vals) else None))
        print(f"| {track} | " + " | ".join(cells) + f" | {base:.4f} |")
    print()

    print("## 3. Capability breakdown\n")
    for track in TRACKS:
        for dataset in DATASETS:
            frames = []
            for epoch in args.epochs:
                summary = summaries.get((epoch, track, dataset))
                if summary is None:
                    continue
                sub = summary[summary["level"] == "group"][["name", "accuracy", "count"]]
                frames.append(sub.set_index("name").rename(columns={"accuracy": f"epoch {epoch}"}))
            if not frames:
                continue
            merged = frames[0][["count"]].join([f.drop(columns=["count"]) for f in frames])
            print(f"### {track} / {dataset}\n")
            print("| group | n | " + " | ".join(f"epoch {e}" for e in args.epochs) + " |")
            print("| --- | ---: | " + " | ".join("---:" for _ in args.epochs) + " |")
            for name, row in merged.iterrows():
                cells = " | ".join(
                    fmt(row.get(f"epoch {e}")) if pd.notna(row.get(f"epoch {e}")) else "—"
                    for e in args.epochs
                )
                print(f"| {name} | {int(row['count'])} | {cells} |")
            print()

    print("## 4. Answer-format breakdown\n")
    for track in TRACKS:
        for dataset in DATASETS:
            frames = []
            for epoch in args.epochs:
                summary = summaries.get((epoch, track, dataset))
                if summary is None:
                    continue
                sub = summary[summary["level"] == "answer_format"][["name", "accuracy", "count"]]
                frames.append(sub.set_index("name").rename(columns={"accuracy": f"epoch {epoch}"}))
            if not frames:
                continue
            merged = frames[0][["count"]].join([f.drop(columns=["count"]) for f in frames])
            print(f"### {track} / {dataset}\n")
            print("| answer_format | n | " + " | ".join(f"epoch {e}" for e in args.epochs) + " |")
            print("| --- | ---: | " + " | ".join("---:" for _ in args.epochs) + " |")
            for name, row in merged.iterrows():
                cells = " | ".join(
                    fmt(row.get(f"epoch {e}")) if pd.notna(row.get(f"epoch {e}")) else "—"
                    for e in args.epochs
                )
                print(f"| {name} | {int(row['count'])} | {cells} |")
            print()


if __name__ == "__main__":
    main()
