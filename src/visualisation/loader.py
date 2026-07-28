"""Scoped experiment-result loading for the visualiser.

Each selected result directory is normalised into the same shape so the
server/UI never has to care whether it is a "rich" result (self-contained
``predictions.parquet``) or a "legacy" result (``responses.json`` plus
``requests.json``).

The normal launcher passes an explicit run root, avoiding an expensive scan of
unrelated tracks and historical epochs. Repository-wide discovery remains
available when no roots are supplied.
"""

from __future__ import annotations

# --- login-node thread guard (OpenBLAS/OMP spawn 128 threads and hit RLIMIT_NPROC)
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class RunInfo:
    """One experiment run (a model's outputs for a track/split)."""

    id: str  # unique, == path relative to repo root
    label: str  # human label, e.g. "zeroshot-sweep/Qwen3-VL-8B-Instruct"
    method: str  # e.g. "baseline" | "zeroshot-sweep"
    model_name: str
    track: str
    split: str
    dataset: str
    kind: str  # "rich" | "legacy"
    dir: str
    accuracy: float | None
    count: int

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "method": self.method,
            "model_name": self.model_name,
            "track": self.track,
            "split": self.split,
            "dataset": self.dataset,
            "kind": self.kind,
            "accuracy": self.accuracy,
            "count": self.count,
        }


@dataclass
class SplitData:
    """All runs + shared per-question info for one (track, split, dataset)."""

    track: str
    split: str
    dataset: str
    order: list[str] = field(default_factory=list)  # qID order
    questions: dict[str, dict] = field(default_factory=dict)  # qID -> shared info
    run_ids: list[str] = field(default_factory=list)
    preds: dict[str, dict[str, dict]] = field(default_factory=dict)  # run_id -> qID -> pred
    image_path: dict[str, str] = field(default_factory=dict)  # qID -> abs jpg path


@dataclass
class Dataset:
    runs: list[RunInfo] = field(default_factory=list)
    splits: dict[tuple[str, str, str], SplitData] = field(default_factory=dict)

    def split(self, track: str, split: str, dataset: str) -> SplitData | None:
        return self.splits.get((track, split, dataset))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_str_list(v) -> list[str]:
    """Normalise a secondary-capabilities value (ndarray | list | str | None)."""
    if v is None:
        return []
    values = v if isinstance(v, (list, tuple, np.ndarray)) else [v]
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        for part in re.split(r"[|;,]", str(value)):
            code = part.strip()
            if code and code not in out:
                out.append(code)
    return out


def _read_json(p: Path):
    with p.open() as f:
        return json.load(f)


def _read_meta(run_dir: Path) -> dict:
    p = run_dir / "meta.json"
    return _read_json(p) if p.exists() else {}


def _read_correctness(run_dir: Path) -> dict[str, bool]:
    """qID -> bool from ``eval/results.csv`` (correctness is a 'True'/'False' string)."""
    p = run_dir / "eval" / "results.csv"
    out: dict[str, bool] = {}
    if not p.exists():
        return out
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("qID", "")).strip()
            if qid:
                out[qid] = str(row.get("correctness", "")).strip().lower() == "true"
    return out


def _read_overall_accuracy(run_dir: Path) -> float | None:
    """Overall accuracy from ``eval/summary.csv`` (``overall,MEAN,<acc>,...``)."""
    p = run_dir / "eval" / "summary.csv"
    if not p.exists():
        return None
    with p.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip().lower() == "overall":
                try:
                    return float(row[2])
                except ValueError:
                    return None
    return None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
# A finetune model writes one eval dir per epoch: logs/<base_model>/eval_epoch_<N>/
_EPOCH_RE = re.compile(r"^eval_epoch_(\d+)$")


def _resolve_run_root(repo_root: Path, value: str | Path) -> Path:
    root = Path(value)
    root = root if root.is_absolute() else repo_root / root
    root = root.resolve()
    try:
        root.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"run root must be inside the repository: {root}") from exc
    if not root.is_dir():
        raise FileNotFoundError(f"run root is not a directory: {root}")
    return root


def discover_runs(
    repo_root: Path,
    include_smoke: bool = False,
    run_roots: list[str | Path] | None = None,
) -> list[Path]:
    """Return directories containing ``predictions.parquet`` or ``responses.json``.

    Per-epoch eval dirs are collapsed to the single best-accuracy epoch per base
    model during repository-wide discovery. Explicit run roots load only their
    descendants and are not replaced by historical "best epoch" results.
    """
    dirs: set[Path] = set()
    if run_roots:
        for value in run_roots:
            root = _resolve_run_root(repo_root, value)
            for marker in ("predictions.parquet", "responses.json"):
                direct = root / marker
                if direct.is_file():
                    dirs.add(root)
                dirs.update(p.parent for p in root.rglob(marker))
    else:
        for marker in ("predictions.parquet", "responses.json"):
            for p in repo_root.glob(f"track-*/*/logs/**/{marker}"):
                dirs.add(p.parent)
    if not include_smoke:
        dirs = {d for d in dirs if "/smoke/" not in str(d).replace(os.sep, "/")}
    selected = sorted(dirs) if run_roots else _select_best_epochs(dirs)
    if run_roots and not selected:
        raise FileNotFoundError("no predictions.parquet or responses.json under run root(s)")
    return selected


def _select_best_epochs(dirs: set[Path]) -> list[Path]:
    """Keep only the highest-accuracy ``eval_epoch_<N>`` per base model; pass the rest through."""
    groups: dict[Path, list[tuple[int, Path]]] = {}
    passthrough: list[Path] = []
    for d in dirs:
        m = _EPOCH_RE.match(d.name)
        if m:
            groups.setdefault(d.parent, []).append((int(m.group(1)), d))
        else:
            passthrough.append(d)
    for items in groups.values():
        # max overall accuracy; tie-break on later epoch (more training).
        _, best = max(items, key=lambda it: (_read_overall_accuracy(it[1]) or -1.0, it[0]))
        passthrough.append(best)
    return sorted(passthrough)


def _infer_track(rel: Path) -> str | None:
    """Track from the top-level ``track-<track>`` path component."""
    top = rel.parts[0] if rel.parts else ""
    return top[len("track-"):] if top.startswith("track-") else None


def _infer_split(track: str, sample_ids, split_id_cache: dict) -> str | None:
    """Split whose GT parquet best matches the run's sample_ids (for meta-less runs)."""
    ids = {str(s) for s in sample_ids}
    if not ids:
        return None
    best, best_overlap = None, 0
    for split in paths.SPLITS:
        key = (track, split)
        if key not in split_id_cache:
            p = paths.parquet_path(track, split)
            try:
                split_id_cache[key] = set(pd.read_parquet(p, columns=["id"])["id"].astype(str))
            except Exception:  # noqa: BLE001 — missing/unreadable parquet -> empty set
                split_id_cache[key] = set()
        overlap = len(ids & split_id_cache[key])
        if overlap > best_overlap:
            best, best_overlap = split, overlap
    return best


def _run_identity(
    run_dir: Path, repo_root: Path, meta: dict, rich_model: str | None = None
) -> tuple[str, str, str, str]:
    """Return (run_id, method, model_name, label)."""
    rel = run_dir.relative_to(repo_root)
    parts = rel.parts  # e.g. track-frame, zeroshot-sweep, logs, resp_test, <model>
    method = parts[1] if len(parts) > 1 else rel.parts[0]
    base = (
        meta.get("model_name")
        or (Path(meta["model"]).name if meta.get("model") else None)
        or rich_model
        or run_dir.name
    )
    # Per-epoch finetune dirs carry no model name; surface the chosen epoch instead.
    m = _EPOCH_RE.match(run_dir.name)
    model_name = f"{base} (ep{int(m.group(1))})" if m else base
    return str(rel), method, model_name, f"{method}/{model_name}"


# --------------------------------------------------------------------------- #
# Per-run record loading
# --------------------------------------------------------------------------- #
def _load_rich(
    run_dir: Path, meta: dict
) -> tuple[dict[str, dict], dict[str, dict], dict[str, str], str | None]:
    """Load a self-contained predictions.parquet run.

    Returns (base_info{qID->dict}, preds{qID->dict}, image_paths{qID->str}, model_name).
    ``model_name`` comes from the parquet (set for meta-less finetune runs).
    """
    df = pd.read_parquet(run_dir / "predictions.parquet")
    correct = _read_correctness(run_dir)
    system_prompt = meta.get("system_prompt")
    model_name = None
    if "model_name" in df.columns:
        mn = df["model_name"].dropna()
        model_name = str(mn.iloc[0]) if len(mn) else None
    base: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    images: dict[str, str] = {}
    for r in df.itertuples(index=False):
        qid = str(r.sample_id)
        base[qid] = {
            "question": r.question,
            "answer_format": r.answer_format,
            "primary_capability": r.primary_capability,
            "secondaries": _as_str_list(r.secondary_capabilities),
            "clinical": bool(r.clinical_relevance),
            "ood": bool(r.ood),
            "video": r.video,
            "ts_start": r.timestamp_start,
            "ts_end": r.timestamp_end,
            "gt": r.answer,
            "prompt": r.prompt,
        }
        images[qid] = str(r.image_path)
        preds[qid] = {
            "raw_output": r.raw_model_output,
            "prediction": r.prediction,
            "correct": correct.get(qid),
            "latency": float(r.latency_sec) if pd.notna(r.latency_sec) else None,
            "error": (r.error or "") if isinstance(r.error, str) else "",
            "system_prompt": system_prompt,
        }
    return base, preds, images, model_name


def _load_legacy(
    run_dir: Path, meta: dict, track: str, split: str, gt_cache: dict
) -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    """Load an older responses.json run; reconstruct input from GT parquet + prompts."""
    # Lazy imports — these pull in the `focus` package, only needed for legacy runs.
    from src import data as _data
    from src import frames as _frames
    from src import prompts as _prompts

    responses = _read_json(run_dir / "responses.json")
    req_path = run_dir / "requests.json"
    requests = {str(x["qID"]): x for x in _read_json(req_path)} if req_path.exists() else {}
    correct = _read_correctness(run_dir)
    system_prompt = meta.get("system_prompt")  # baseline has none

    gt_df = gt_cache.get((track, split))
    if gt_df is None:
        gt_df = _data.read_parquet(track, split)
        gt_df = gt_df.set_index(gt_df["id"].astype(str))
        gt_cache[(track, split)] = gt_df

    base: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    images: dict[str, str] = {}
    for item in responses:
        qid = str(item["qID"])
        row = gt_df.loc[qid].to_dict() if qid in gt_df.index else {}
        question = row.get("question") or requests.get(qid, {}).get("question", "")
        fmt = row.get("answer_format", "open_ended")
        try:
            instr, _ = _prompts.build_instruction(question, fmt)
        except Exception:  # noqa: BLE001 — reconstruction is best-effort
            instr = question
        base[qid] = {
            "question": question,
            "answer_format": fmt,
            "primary_capability": row.get("primary_capability"),
            "secondaries": _as_str_list(row.get("secondary_capabilities")),
            "clinical": bool(row.get("clinical_relevance", False)),
            "ood": bool(row.get("ood", False)),
            "video": row.get("video") or requests.get(qid, {}).get("videoID"),
            "ts_start": row.get("timestamp_start"),
            "ts_end": row.get("timestamp_end"),
            "gt": row.get("answer"),
            "prompt": instr,
        }
        if row:
            try:
                req = _data.row_to_request(row)
                images[qid] = str(_frames.request_frame_paths(req)[0])
            except Exception:  # noqa: BLE001
                pass
        content = item.get("content", "")
        preds[qid] = {
            "raw_output": content,
            "prediction": content,  # legacy responses are already normalised at write time
            "correct": correct.get(qid),
            "latency": item.get("latency"),
            "error": "",
            "system_prompt": system_prompt,
        }
    return base, preds, images


# --------------------------------------------------------------------------- #
# Top-level build
# --------------------------------------------------------------------------- #
def build_dataset(
    repo_root: Path | None = None,
    include_smoke: bool = False,
    run_roots: list[str | Path] | None = None,
) -> Dataset:
    """Load selected result roots into an in-memory :class:`Dataset`."""
    repo_root = Path(repo_root) if repo_root else paths.REPO_ROOT
    ds = Dataset()
    gt_cache: dict = {}
    split_id_cache: dict = {}

    for run_dir in discover_runs(repo_root, include_smoke, run_roots):
        try:
            meta = _read_meta(run_dir)
            rel = run_dir.relative_to(repo_root)
            track = meta.get("track") or _infer_track(rel)
            split = meta.get("split")
            dataset = str(meta.get("dataset") or "unknown")
            kind = "rich" if (run_dir / "predictions.parquet").exists() else "legacy"

            rich_model = None
            if kind == "rich":
                base, preds, images, rich_model = _load_rich(run_dir, meta)
                # Meta-less finetune runs: infer split by matching sample_ids to GT.
                if not split and track:
                    split = _infer_split(track, base.keys(), split_id_cache)
            else:
                if not track or not split:
                    logger.warning("skipping legacy run without track/split meta: %s", run_dir)
                    continue
                base, preds, images = _load_legacy(run_dir, meta, track, split, gt_cache)

            if not track or not split:
                logger.warning("skipping run; could not resolve track/split: %s", run_dir)
                continue
            run_id, method, model_name, label = _run_identity(run_dir, repo_root, meta, rich_model)

            key = (track, split, dataset)
            sd = ds.splits.setdefault(
                key,
                SplitData(track=track, split=split, dataset=dataset),
            )
            for qid, info in base.items():
                sd.questions.setdefault(qid, info)
                if qid not in sd.order:
                    sd.order.append(qid)
            sd.image_path.update({q: p for q, p in images.items() if q not in sd.image_path})
            sd.run_ids.append(run_id)
            sd.preds[run_id] = preds

            acc = _read_overall_accuracy(run_dir)
            if acc is None and preds:
                vals = [p["correct"] for p in preds.values() if p["correct"] is not None]
                acc = (sum(vals) / len(vals)) if vals else None
            ds.runs.append(
                RunInfo(
                    id=run_id, label=label, method=method, model_name=model_name,
                    track=track, split=split, dataset=dataset, kind=kind, dir=str(run_dir),
                    accuracy=acc, count=len(preds),
                )
            )
            logger.info("loaded run %s (%s, %d preds, acc=%s)", label, kind, len(preds), acc)
        except Exception:  # noqa: BLE001 — one bad run must not sink the whole tool
            logger.exception("failed to load run dir: %s", run_dir)

    # Stable run ordering: by track, split, dataset, accuracy desc, then label.
    ds.runs.sort(
        key=lambda r: (
            r.track,
            r.split,
            r.dataset,
            -(r.accuracy or 0.0),
            r.label,
        )
    )
    for sd in ds.splits.values():
        order = {r.id: i for i, r in enumerate(ds.runs)}
        sd.run_ids.sort(key=lambda rid: order.get(rid, 1_000_000))
    return ds
