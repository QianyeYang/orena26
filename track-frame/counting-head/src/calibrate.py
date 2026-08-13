#!/usr/bin/env python
"""Zero-GPU null test for the count head, run on already-saved predictions.

The head and this script bet on the same thing: that the model's predicted count
carries more information than exact-match extracts from it. Here that bet is
made in its cheapest possible form — learn ``P(true | predicted, template)`` and
emit the mode instead of the raw number.

If a lookup table recovers nothing, the representation feeding the head is
unlikely to hold signal a two-layer MLP will find, and the head is not worth the
GPU. If it recovers a lot, the head should beat it, because it sees the
representation rather than only the scalar the model happened to emit.

Folds are grouped by video, so a mapping is never fitted on frames from the
video it is scored on.

Usage:
    python calibrate.py --run <benchmark-run-dir> [--run <another>] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from templates import CLASSES_MAX, MAX_COUNT, MIN_COUNT, template_id, template_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("calib")


def parse_int(text) -> int | None:
    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None


def load(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "predictions.parquet")
    df = df[df["answer_format"] == "number"].copy()
    df["template_id"] = df["question"].map(template_id)
    df["target"] = df["answer"].map(parse_int)
    df["pred"] = df["prediction"].map(parse_int)
    df = df.dropna(subset=["template_id", "target"]).copy()
    df["template_id"] = df["template_id"].astype(int)
    df["target"] = df["target"].astype(int)
    # An unparseable number is scored wrong either way; map it to the floor so
    # both arms are penalised identically and the delta stays about calibration.
    df["pred"] = df["pred"].fillna(MIN_COUNT).astype(int)
    df["run"] = run.name
    return df


def clamp(pred: np.ndarray, tid: np.ndarray) -> np.ndarray:
    hi = np.where(tid == 1, CLASSES_MAX, MAX_COUNT)
    return np.clip(pred, MIN_COUNT, hi)


def fit_map(fit: pd.DataFrame) -> dict[tuple[int, int], int]:
    """(template, predicted) -> most frequent true count in the fit rows."""
    g = fit.groupby(["template_id", "pred", "target"]).size().reset_index(name="n")
    g = g.sort_values("n", ascending=False).drop_duplicates(["template_id", "pred"])
    return {(int(r.template_id), int(r.pred)): int(r.target) for r in g.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser(description="cross-fitted count recalibration")
    ap.add_argument("--run", action="append", required=True,
                    help="benchmark run dir containing predictions.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for run in a.run:
        df = load(Path(run)).reset_index(drop=True)
        tid = df["template_id"].to_numpy()
        target = df["target"].to_numpy()
        raw = df["pred"].to_numpy()
        clamped = clamp(raw, tid)

        videos = df["video"].unique()
        rng = np.random.default_rng(a.seed)
        rng.shuffle(videos)
        fold_of = {v: i % a.folds for i, v in enumerate(videos)}
        folds = df["video"].map(fold_of).to_numpy()

        recal = clamped.copy()
        for k in range(a.folds):
            held = folds == k
            if held.sum() == 0 or (~held).sum() == 0:
                continue
            mapping = fit_map(df[~held])
            recal[held] = [
                mapping.get((int(t), int(p)), int(p))
                for t, p in zip(tid[held], raw[held])
            ]
        recal = clamp(recal, tid)

        row = {
            "run": Path(run).name,
            "n": int(len(df)),
            "videos": int(len(videos)),
            "exact_raw": float((raw == target).mean()),
            "exact_clamped": float((clamped == target).mean()),
            "exact_recalibrated": float((recal == target).mean()),
            "within1_clamped": float((np.abs(clamped - target) <= 1).mean()),
            "mean_pred_clamped": float(clamped.mean()),
            "mean_pred_recal": float(recal.mean()),
            "mean_true": float(target.mean()),
        }
        row["gain_vs_clamped"] = row["exact_recalibrated"] - row["exact_clamped"]
        summaries.append(row)
        log.info(
            "%-32s raw=%.4f clamped=%.4f recal=%.4f  (gain %+.4f, ceiling within1=%.4f)",
            row["run"], row["exact_raw"], row["exact_clamped"],
            row["exact_recalibrated"], row["gain_vs_clamped"], row["within1_clamped"],
        )

        # Where the mapping actually moves things, fitted on everything: this is
        # a description of the bias, not a scored prediction.
        full = fit_map(df)
        shifts = pd.DataFrame(
            [
                {
                    "template": template_name(t),
                    "predicted": p,
                    "maps_to": v,
                    "n": int(((tid == t) & (raw == p)).sum()),
                }
                for (t, p), v in sorted(full.items())
            ]
        )
        shifts = shifts[shifts["predicted"] != shifts["maps_to"]]
        shifts.to_csv(out / f"shifts-{Path(run).name}.csv", index=False)
        if len(shifts):
            log.info("mapping moves these (%d rows affected):\n%s",
                     int(shifts["n"].sum()), shifts.to_string(index=False))
        else:
            log.info("mapping is the identity everywhere: no exploitable bias")

    res = pd.DataFrame(summaries)
    res.to_csv(out / "calibration.csv", index=False)
    (out / "results.json").write_text(json.dumps(summaries, indent=2))
    log.info("wrote -> %s", out)


if __name__ == "__main__":
    main()
