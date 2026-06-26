#!/usr/bin/env python
"""Cross-model comparison for the FRAME zero-shot sweep.

Scans ``<resp-dir>/<model>/`` folders, each holding ``predictions.parquet`` (rich
rows) and optionally ``eval/{summary,results}.csv`` (focus Evaluator output), and
emits a markdown report:

  * headline   : overall accuracy [CI], invalid-format rate, latency, #errors
  * by answer_format / by capability group : accuracy matrix (models x columns)
  * diagnostics: normalization failures + most common wrong predictions

Accuracy comes from the Evaluator's ``summary.csv``/``results.csv`` (open_ended &
multiple_choice need the LLM judge); everything else is derived from the parquet.

Usage:
    python track-frame/zeroshot-sweep/src/analyze.py \
        --resp-dir track-frame/zeroshot-sweep/logs/resp_test \
        --baseline track-frame/baseline/logs/resp_test \
        --out track-frame/zeroshot-sweep/logs/comparison.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src import adapter  # noqa: E402


def _overall(summary: pd.DataFrame | None) -> tuple[float, float, float] | None:
    if summary is None:
        return None
    row = summary[summary.level == "overall"]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r.accuracy), float(r.ci_low), float(r.ci_high)


def _level_map(summary: pd.DataFrame | None, level: str) -> dict[str, float]:
    if summary is None:
        return {}
    sub = summary[summary.level == level]
    return {r["name"]: float(r.accuracy) for _, r in sub.iterrows()}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load_model(model_dir: Path) -> dict:
    name = model_dir.name
    preds = pd.read_parquet(model_dir / "predictions.parquet")
    summ_p = model_dir / "eval" / "summary.csv"
    res_p = model_dir / "eval" / "results.csv"
    summary = pd.read_csv(summ_p) if summ_p.exists() else None
    results = pd.read_csv(res_p) if res_p.exists() else None

    # invalid-format rate: strict-format parse failures
    invalid = ~preds.apply(
        lambda r: adapter.is_parseable(r.answer_format, r.normalized_prediction), axis=1
    )
    # normalization failures: model said something but it doesn't parse
    norm_fail = invalid & (preds.raw_model_output.str.len() > 0)

    return {
        "name": name,
        "preds": preds,
        "summary": summary,
        "results": results,
        "n": len(preds),
        "overall": _overall(summary),
        "fmt_acc": _level_map(summary, "answer_format"),
        "grp_acc": _level_map(summary, "group"),
        "invalid_rate": float(invalid.mean()),
        "norm_fail": int(norm_fail.sum()),
        "n_err": int((preds.error.str.len() > 0).sum()),
        "lat_mean": float(preds.latency_sec.mean()),
        "lat_p50": float(preds.latency_sec.median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resp-dir", required=True, help="dir of <model>/predictions.parquet")
    ap.add_argument("--baseline", default=None, help="optional baseline resp_test dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dirs = sorted(p for p in Path(a.resp_dir).iterdir() if (p / "predictions.parquet").exists())
    if a.baseline and (Path(a.baseline) / "responses.json").exists():
        # baseline has no predictions.parquet; pull just its summary for the table
        pass
    models = [load_model(d) for d in dirs]
    if not models:
        print("no model prediction dirs found under", a.resp_dir)
        return

    # optional baseline summary (Qwen2.5-VL-7B original run)
    baseline = None
    if a.baseline:
        bs = Path(a.baseline) / "eval" / "summary.csv"
        if bs.exists():
            s = pd.read_csv(bs)
            baseline = {"name": "baseline-Qwen2.5-VL-7B*", "overall": _overall(s),
                        "fmt_acc": _level_map(s, "answer_format"), "grp_acc": _level_map(s, "group")}

    order = sorted(models, key=lambda m: (m["overall"] or (-1,))[0], reverse=True)

    # ---- headline ----
    rows = []
    for m in order:
        ov = f"{m['overall'][0]:.3f} [{m['overall'][1]:.3f},{m['overall'][2]:.3f}]" if m["overall"] else "—"
        rows.append([m["name"], str(m["n"]), ov, f"{m['invalid_rate']*100:.1f}%",
                     str(m["norm_fail"]), str(m["n_err"]),
                     f"{m['lat_mean']:.2f}", f"{m['lat_p50']:.2f}"])
    if baseline and baseline["overall"]:
        b = baseline["overall"]
        rows.append([baseline["name"], "2000", f"{b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]",
                     "—", "—", "—", "—", "—"])
    headline = _md_table(
        ["model", "n", "overall acc [95% CI]", "invalid-fmt", "norm-fail", "errors", "lat mean(s)", "lat p50(s)"],
        rows,
    )

    # ---- by answer_format ----
    fmts = sorted({f for m in models for f in m["fmt_acc"]})
    frows = [[m["name"]] + [f"{m['fmt_acc'].get(f):.3f}" if f in m["fmt_acc"] else "—" for f in fmts]
             for m in order]
    if baseline:
        frows.append([baseline["name"]] + [f"{baseline['fmt_acc'].get(f):.3f}" if f in baseline["fmt_acc"] else "—" for f in fmts])
    fmt_table = _md_table(["model"] + fmts, frows)

    # ---- by capability group ----
    grps = sorted({g for m in models for g in m["grp_acc"]})
    grows = [[m["name"]] + [f"{m['grp_acc'].get(g):.3f}" if g in m["grp_acc"] else "—" for g in grps]
             for m in order]
    if baseline:
        grows.append([baseline["name"]] + [f"{baseline['grp_acc'].get(g):.3f}" if g in baseline["grp_acc"] else "—" for g in grps])
    grp_table = _md_table(["model"] + grps, grows)

    # ---- diagnostics: top wrong predictions (needs results.csv) ----
    diag = []
    for m in order:
        if m["results"] is None:
            continue
        # sample_id (parquet) is str; qID (results.csv) parses as int64 -> cast both
        preds = m["preds"].assign(_sid=m["preds"].sample_id.astype(str))
        res = m["results"][["qID", "correctness"]].assign(_sid=lambda d: d.qID.astype(str))
        merged = preds.merge(res, on="_sid", how="inner")
        wrong = merged[~merged.correctness.astype(bool)]
        common = Counter(wrong.normalized_prediction).most_common(5)
        diag.append(f"- **{m['name']}**: " + ", ".join(f"`{p}`×{c}" for p, c in common))

    report = [
        "# FRAME zero-shot sweep — model comparison\n",
        "_Accuracy from focus Evaluator (Qwen3.5-4B judge for open_ended/MC); "
        "invalid-fmt / norm-fail / latency from predictions.parquet. "
        "norm-fail = non-empty raw output that fails its format parser._\n",
        "`*` baseline = original Qwen2.5-VL-7B run (raw output not preserved).\n",
        "## Overall\n", headline, "",
        "## Accuracy by answer_format\n", fmt_table, "",
        "## Accuracy by capability group\n", grp_table, "",
    ]
    if diag:
        report += ["## Most common wrong predictions (post-normalization)\n", *diag, ""]

    Path(a.out).write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
