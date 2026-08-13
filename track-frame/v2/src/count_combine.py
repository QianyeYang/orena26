#!/usr/bin/env python
"""Sweep view-combination rules over a saved multi-view counting pass.

Rules are selected on held-out **videos**, not held-out rows: frames from one
video share lighting, instrument set and object load, so a rule tuned on some of
its rows and scored on the rest would look better than it is. The same mistake
made post-hoc count recalibration look promising before a group-aware split
showed it losing 1.5 points (see `docs/frame-track-v2-plan.md`).

Cross-dataset transfer is reported separately, because the real metric's
out-of-distribution half is a dataset shift, not a video shift — a rule that only
survives within one dataset is not a rule we can ship.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import counting_v2 as CV  # noqa: E402


def per_question_views(views: pd.DataFrame) -> pd.DataFrame:
    """Pivot the view-level table to one row per question, one column per view."""
    numeric = views[views["count_mode"] != "classes"].copy()
    numeric["n"] = pd.to_numeric(numeric["value"], errors="coerce")
    wide = numeric.pivot_table(
        index=["sample_id", "dataset", "video", "answer", "count_mode"],
        columns="view_id",
        values="n",
        aggfunc="first",
    ).reset_index()
    wide["gt"] = pd.to_numeric(wide["answer"], errors="coerce")
    return wide


def apply_rule(wide: pd.DataFrame, rule: str) -> pd.Series:
    view_cols = [c for c in wide.columns if c == "whole" or c.startswith(("tile:", "hflip", "vflip", "zoom"))]

    def one(row) -> int:
        per_view = {c: int(row[c]) for c in view_cols if pd.notna(row[c])}
        return CV.combine_counts(per_view, rule)

    return wide.apply(one, axis=1)


def _as_set(text: object) -> frozenset[str]:
    s = str(text).strip()
    if not s or s.lower() == "none":
        return frozenset()
    return frozenset(p.strip().lower() for p in s.split(",") if p.strip())


def evaluate_class_counts(views: pd.DataFrame) -> pd.DataFrame:
    """"How many distinct classes" answered through the class-list head.

    |set| is right more often than the set itself — 0.840 versus 0.752 on the
    saved epoch-30 predictions — so the count is read off a predicted class set
    rather than predicted directly. Sets union across views; they never sum.
    """
    cls = views[views["count_mode"] == "classes"]
    if cls.empty:
        return pd.DataFrame()

    rows = []
    for (sid, ds), g in cls.groupby(["sample_id", "dataset"]):
        gt = _as_set(g["answer"].iloc[0])
        whole = g[g["view_id"] == "whole"]["value"]
        rows.append(
            {
                "sample_id": sid,
                "dataset": ds,
                "gt_n": pd.to_numeric(g["answer"].iloc[0], errors="coerce"),
                "whole_n": len(_as_set(whole.iloc[0])) if len(whole) else None,
                "union_n": len(CV.combine_class_sets({r.view_id: _as_set(r.value) for r in g.itertuples()})),
                "gt_set_n": len(gt),
            }
        )
    out = pd.DataFrame(rows)
    # The reference answer for these rows is the number itself, not a class set.
    target = out["gt_n"].fillna(out["gt_set_n"])
    return pd.DataFrame(
        [
            {"estimator": "whole_frame_list", "exact": round(float(out["whole_n"].eq(target).mean()), 4)},
            {"estimator": "union_over_views", "exact": round(float(out["union_n"].eq(target).mean()), 4)},
        ]
    ).assign(n=len(out))


def evaluate(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in CV.COMBINE_RULES:
        pred = apply_rule(wide, rule)
        ok = pred.eq(wide["gt"])
        rows.append(
            {
                "rule": rule,
                "exact": round(float(ok.mean()), 4),
                "within_1": round(float((pred - wide["gt"]).abs().le(1).mean()), 4),
                "mean_signed_err": round(float((pred - wide["gt"]).mean()), 3),
                "exact_gt_ge_4": round(float(ok[wide["gt"] >= 4].mean()), 4),
                "exact_gt_le_3": round(float(ok[wide["gt"] <= 3].mean()), 4),
                "n": int(len(wide)),
            }
        )
    return pd.DataFrame(rows).sort_values("exact", ascending=False)


def group_aware_selection(wide: pd.DataFrame, trials: int = 20, seed: int = 0) -> pd.DataFrame:
    """Fit the rule on half the videos, score it on the other half."""
    videos = sorted(wide["video"].unique())
    rng = np.random.default_rng(seed)
    picked, scores, baselines = [], [], []
    for _ in range(trials):
        perm = rng.permutation(videos)
        half = set(perm[: len(perm) // 2])
        tr, te = wide[wide["video"].isin(half)], wide[~wide["video"].isin(half)]
        if len(tr) < 30 or len(te) < 30:
            continue
        best = max(CV.COMBINE_RULES, key=lambda r: apply_rule(tr, r).eq(tr["gt"]).mean())
        picked.append(best)
        scores.append(float(apply_rule(te, best).eq(te["gt"]).mean()))
        baselines.append(float(apply_rule(te, "whole").eq(te["gt"]).mean()))
    return pd.DataFrame(
        [
            {
                "selected_rule_mode": max(set(picked), key=picked.count) if picked else None,
                "held_out_exact": round(float(np.mean(scores)), 4) if scores else None,
                "whole_baseline": round(float(np.mean(baselines)), 4) if baselines else None,
                "delta": round(float(np.mean(scores) - np.mean(baselines)), 4) if scores else None,
                "trials": len(scores),
            }
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep counting view-combination rules")
    ap.add_argument("--run", required=True, help="dir holding views.parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else run
    views = pd.read_parquet(run / "views.parquet")
    wide = per_question_views(views)

    overall = evaluate(wide)
    print("=== all counting rows (rule fitted and scored on the same rows) ===")
    print(overall.to_string(index=False))

    print("\n=== group-aware: rule selected on held-out videos ===")
    sel = group_aware_selection(wide)
    print(sel.to_string(index=False))

    classes = evaluate_class_counts(views)
    if not classes.empty:
        print("\n=== distinct-class counts via the class-list head ===")
        print(classes.to_string(index=False))

    print("\n=== per dataset ===")
    per_ds = {}
    for ds, g in wide.groupby("dataset"):
        per_ds[ds] = evaluate(g).to_dict("records")
        print(f"\n-- {ds} (n={len(g)})")
        print(evaluate(g).to_string(index=False))

    print("\n=== cross-dataset transfer: pick on one dataset, apply to the other ===")
    transfer = []
    datasets = sorted(wide["dataset"].unique())
    for fit_ds in datasets:
        tr = wide[wide["dataset"] == fit_ds]
        te = wide[wide["dataset"] != fit_ds]
        if tr.empty or te.empty:
            continue
        best = max(CV.COMBINE_RULES, key=lambda r: apply_rule(tr, r).eq(tr["gt"]).mean())
        row = {
            "fit_on": fit_ds,
            "applied_to": "+".join(d for d in datasets if d != fit_ds),
            "rule": best,
            "exact": round(float(apply_rule(te, best).eq(te["gt"]).mean()), 4),
            "whole_baseline": round(float(apply_rule(te, "whole").eq(te["gt"]).mean()), 4),
        }
        row["delta"] = round(row["exact"] - row["whole_baseline"], 4)
        transfer.append(row)
    print(pd.DataFrame(transfer).to_string(index=False))

    (out / "combine.json").write_text(
        json.dumps(
            {
                "overall": overall.to_dict("records"),
                "group_aware": sel.to_dict("records"),
                "per_dataset": per_ds,
                "cross_dataset": transfer,
                "class_counts": classes.to_dict("records") if not classes.empty else [],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out / 'combine.json'}")


if __name__ == "__main__":
    main()
