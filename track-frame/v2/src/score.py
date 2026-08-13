#!/usr/bin/env python
"""Score a benchmark run the way the leaderboard scores it.

The official FRAME metric is the unweighted mean over (capability group ×
distribution) buckets — for FRAME that is exactly four: object_recognition and
aggregation, each split in-distribution / out-of-distribution. Per-video macro
accuracy, which this repo reported until now, is a different number and ranks
differently: it weights every video equally instead of every bucket equally.

Locally every row carries ``ood=False``, so the honest local analogue is to put
**dataset on the distribution axis** — score HeiCo and LapChole as separate
halves and average the four cells. That mirrors the real metric's shape and, for
a model trained on one dataset and scored on the other, mirrors its meaning too.

``--judge-device`` needs a GPU: open_ended and multiple_choice are graded by an
LLM judge (Qwen3.5-4B by default), not by exact match.

Writes ``results.csv``, ``summary.csv`` (stock focus output) and
``leaderboard.json`` + ``leaderboard.md`` (the four-bucket view) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from focus import Evaluator, load_references, load_requests, load_responses  # noqa: E402
from focus.taxonomy import Capability  # noqa: E402
from src.paths import OS_MODELS_DIR  # noqa: E402

GROUPS = ("object_recognition", "aggregation")


def leaderboard_view(results: pd.DataFrame, dataset_of: dict[str, str]) -> dict:
    """Return the four-bucket (group × dataset) view plus supporting breakdowns."""
    r = results.copy()
    r["dataset"] = r["qID"].astype(str).map(dataset_of)
    r["group"] = r["primary"].map(lambda v: Capability(v).group.value)

    buckets: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for group in GROUPS:
        for ds in sorted(r["dataset"].dropna().unique()):
            cell = r[(r["group"] == group) & (r["dataset"] == ds)]
            key = f"{group}_{ds}"
            buckets[key] = float(cell["correctness"].mean()) if len(cell) else None
            counts[key] = int(len(cell))

    populated = [v for v in buckets.values() if v is not None]
    return {
        "score_leaderboard_shaped": float(np.mean(populated)) if populated else None,
        "buckets": buckets,
        "bucket_counts": counts,
        "by_group": r.groupby("group")["correctness"].mean().round(4).to_dict(),
        "by_dataset": r.groupby("dataset")["correctness"].mean().round(4).to_dict(),
        "by_answer_format": r.groupby("answer_format")["correctness"].mean().round(4).to_dict(),
        "by_answer_format_n": r.groupby("answer_format")["correctness"].size().to_dict(),
        "overall": float(r["correctness"].mean()),
        "n": int(len(r)),
    }


def counting_diagnostic(results: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """Accuracy and signed error by ground-truth count — the range-compression check."""
    m = preds.merge(
        results[["qID", "correctness"]], left_on=preds["sample_id"].astype(str),
        right_on=results["qID"].astype(str), how="inner",
    )
    n = m[m["answer_format"] == "number"].copy()
    if n.empty:
        return {}
    n["g"] = pd.to_numeric(n["answer"], errors="coerce")
    n["p"] = pd.to_numeric(n["normalized_prediction"], errors="coerce")
    n = n.dropna(subset=["g"])
    err = n["p"] - n["g"]
    return {
        "n": int(len(n)),
        "exact": round(float(n["correctness"].mean()), 4),
        "within_1": round(float(err.abs().le(1).mean()), 4),
        "acc_by_gt": {int(k): round(float(v), 3) for k, v in n.groupby("g")["correctness"].mean().items()},
        "n_by_gt": {int(k): int(v) for k, v in n.groupby("g")["correctness"].size().items()},
        "pred_hist": {int(k): int(v) for k, v in n["p"].value_counts().sort_index().items()},
        "max_pred": None if n["p"].isna().all() else int(n["p"].max()),
        "max_gt": int(n["g"].max()),
        "mean_signed_err_by_gt": {
            int(k): round(float(v), 2) for k, v in err.groupby(n["g"]).mean().items()
        },
    }


def render_markdown(name: str, view: dict, counting: dict) -> str:
    lines = [f"# FRAME benchmark — {name}", ""]
    score = view["score_leaderboard_shaped"]
    lines += [f"**Leaderboard-shaped score (mean of 4 buckets): {score:.4f}**", ""]
    lines += ["| bucket | n | accuracy |", "| --- | ---: | ---: |"]
    for key, acc in view["buckets"].items():
        acc_s = "—" if acc is None else f"{acc:.4f}"
        lines.append(f"| {key} | {view['bucket_counts'][key]} | {acc_s} |")
    lines += ["", "| answer_format | n | accuracy |", "| --- | ---: | ---: |"]
    for fmt, acc in sorted(view["by_answer_format"].items()):
        lines.append(f"| {fmt} | {view['by_answer_format_n'][fmt]} | {acc:.4f} |")
    if counting:
        lines += [
            "",
            "## Counting",
            "",
            f"exact **{counting['exact']:.4f}**, within ±1 **{counting['within_1']:.4f}**, "
            f"max prediction {counting['max_pred']} vs max ground truth {counting['max_gt']}",
            "",
            "| GT count | " + " | ".join(str(k) for k in sorted(counting["acc_by_gt"])) + " |",
            "| --- | " + " | ".join("---:" for _ in counting["acc_by_gt"]) + " |",
            "| accuracy | " + " | ".join(f"{counting['acc_by_gt'][k]:.3f}" for k in sorted(counting["acc_by_gt"])) + " |",
            "| n | " + " | ".join(str(counting["n_by_gt"][k]) for k in sorted(counting["acc_by_gt"])) + " |",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a FRAME v2 benchmark run")
    ap.add_argument("--run", required=True, help="benchmark output dir")
    ap.add_argument("--out", default=None, help="default: <run>/eval")
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    a = ap.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else run / "eval"
    out.mkdir(parents=True, exist_ok=True)

    reqs = load_requests(run / "requests.json")
    refs = load_references(run / "references.json")
    responses = load_responses(run / "responses.json")
    preds = pd.read_parquet(run / "predictions.parquet")
    dataset_of = dict(zip(preds["sample_id"].astype(str), preds["dataset"]))

    judge_model = a.judge_model
    if judge_model is None:
        local = OS_MODELS_DIR / "Qwen3.5-4B"
        judge_model = str(local) if local.exists() else "Qwen/Qwen3.5-4B"

    ev = Evaluator(
        judge_kwargs={"model_name": judge_model, "device": a.judge_device},
        num_workers=a.num_workers,
    )
    results_df, summary_df = ev.run(reqs, refs, responses, output_dir=out)

    view = leaderboard_view(results_df, dataset_of)
    counting = counting_diagnostic(results_df, preds)
    name = json.loads((run / "meta.json").read_text()).get("model_name", run.name)

    payload = {"model_name": name, "run": str(run), **view, "counting": counting}
    (out / "leaderboard.json").write_text(json.dumps(payload, indent=2))
    (out / "leaderboard.md").write_text(render_markdown(name, view, counting))

    print(render_markdown(name, view, counting))


if __name__ == "__main__":
    main()
