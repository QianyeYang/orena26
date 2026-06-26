"""Parquet -> focus ``Request``/``Reference`` (offline, parquet-direct).

Bypasses ``focus.FocusDataset`` (which loads from the HuggingFace Hub and so
breaks offline and violates the project's parquet rule), while producing
``Request``/``Reference`` objects identical to what ``FocusDataset`` would yield
— so ``focus.Evaluator`` scores them the same way.

Works for all tracks (frame/segment/procedure); the only per-track difference is
the time window, which lives on ``Request.start_time``/``end_time``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from focus import Reference, Request, save_items
from focus.data.formats import ts_to_seconds
from focus.taxonomy import Capability

from .paths import parquet_path

logger = logging.getLogger(__name__)


def read_parquet(track: str, split: str) -> pd.DataFrame:
    """Load a track/split annotation parquet directly (fast path, no HF)."""
    return pd.read_parquet(parquet_path(track, split))


def _capability(code) -> Capability:
    cap = Capability.from_any(code)
    if cap is None:
        raise ValueError(f"Unrecognised capability {code!r}")
    return cap


def row_to_request(row: dict) -> Request:
    """Build a ``Request`` from one parquet row (dict)."""
    return Request(
        qID=str(row["id"]),
        videoID=row["video"],
        start_time=ts_to_seconds(row["timestamp_start"]),
        end_time=ts_to_seconds(row["timestamp_end"]),
        procedure_type=row["procedure_type"],
        question=row["question"],
    )


def row_to_reference(row: dict) -> Reference:
    """Build a ``Reference`` from one parquet row (dict), mirroring focus parsing."""
    fmt = row["answer_format"]
    fmt_kwargs: dict = {}
    if fmt == "time":  # focus' per-question temporal acceptance window
        dur = ts_to_seconds(row["timestamp_end"]) - ts_to_seconds(row["timestamp_start"])
        fmt_kwargs = {"threshold_seconds": min(5.0, 1 + dur * (4 / 360))}

    raw_secs = row["secondary_capabilities"]
    if raw_secs is None:
        raw_secs = []
    secondaries = tuple(
        cap for r in raw_secs if (cap := Capability.from_any(r)) is not None
    )

    return Reference(
        qID=str(row["id"]),
        primary=_capability(row["primary_capability"]),
        _format=fmt,
        answer=row["answer"],
        format_kwargs=fmt_kwargs,
        secondaries=secondaries,
        ood=bool(row["ood"]),
        clinical=bool(row["clinical_relevance"]),
    )


def load_split(track: str, split: str) -> tuple[list[Request], list[Reference]]:
    """Return ``(requests, references)`` for a track/split, in parquet order."""
    df = read_parquet(track, split)
    reqs, refs = [], []
    for row in df.to_dict("records"):
        reqs.append(row_to_request(row))
        refs.append(row_to_reference(row))
    logger.info("loaded %s/%s: %d samples", track, split, len(reqs))
    return reqs, refs


def save_split(track: str, split: str, out_dir: str | Path) -> tuple[list[Request], list[Reference]]:
    """Load a split and write ``requests.json`` + ``references.json`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reqs, refs = load_split(track, split)
    save_items(reqs, out / "requests.json")
    save_items(refs, out / "references.json")
    logger.info("wrote %s and %s", out / "requests.json", out / "references.json")
    return reqs, refs
