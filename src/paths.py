"""Repo paths for the ORena FOCUS project (all env-overridable).

Single source of truth for where data, videos, extracted frames and model
weights live, so every track/method resolves the same locations.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("ORENA_DATA_DIR", REPO_ROOT / "data"))
PARQUET_DIR = DATA_DIR / "parquet"
FOCUS_ROOT = DATA_DIR / "focus"  # focus 'root_dir' — contains the <dataset>/ folder
OS_MODELS_DIR = Path(os.environ.get("ORENA_OS_MODELS", REPO_ROOT / "os-models"))

DATASET = "heico"
TRACKS = ("frame", "segment", "procedure")
SPLITS = ("train", "test")


def dataset_dir() -> Path:
    """`<FOCUS_ROOT>/<dataset>` — holds `videos/` and the frame folders."""
    return FOCUS_ROOT / DATASET


def video_dir() -> Path:
    """Folder of source `.avi` videos."""
    return dataset_dir() / "videos"


def frames_root(folder: str = "frames") -> Path:
    """Root of extracted frames; `folder` selects a variant (e.g. plain vs overlay)."""
    return dataset_dir() / folder


def parquet_path(track: str, split: str) -> Path:
    """Path to a single-shard VQA parquet, e.g. `parquet/frame/test/0000.parquet`."""
    return PARQUET_DIR / track / split / "0000.parquet"
