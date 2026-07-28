#!/usr/bin/env python3
"""Download the official ORena FOCUS videos and parquet annotations.

Downloads are resumable through the Hugging Face cache. By default, only the
new gated LapChole release is downloaded because HeiCo is already present in
this repository.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import (
    HfApi,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    snapshot_download,
)
from huggingface_hub.errors import HfHubHTTPError


REPOSITORIES = {
    "heico": "orena-dkfz/heico-focus-vqa",
    "lapchole": "orena-dkfz/lapchole-focus-vqa",
}
TRACKS = ("frame", "segment", "procedure")
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download official ORena FOCUS data with resume support."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(REPOSITORIES),
        default=["lapchole"],
        help="Dataset releases to download (default: lapchole).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data" / "focus",
        help="Video dataset root (default: <repo>/data/focus).",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=repo_root / "data" / "parquet",
        help="Annotation root (default: <repo>/data/parquet).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data" / ".cache" / "huggingface",
        help="Hugging Face cache on large storage.",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent video downloads (default: 4).",
    )
    parser.add_argument(
        "--skip-videos", action="store_true", help="Download annotations only."
    )
    parser.add_argument(
        "--skip-annotations", action="store_true", help="Download videos only."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check access and report remote contents without downloading.",
    )
    args = parser.parse_args()
    if args.skip_videos and args.skip_annotations:
        parser.error("--skip-videos and --skip-annotations cannot be used together")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    return args


def token_from_bashrc() -> str | None:
    """Read a direct HF_KEY assignment without executing the shell startup file."""
    bashrc = Path.home() / ".bashrc"
    if not bashrc.is_file():
        return None
    assignment = re.compile(r"^\s*(?:export\s+)?HF_KEY\s*=\s*(.+?)\s*$")
    for line in reversed(bashrc.read_text().splitlines()):
        match = assignment.match(line)
        if not match:
            continue
        try:
            values = shlex.split(match.group(1), comments=True)
        except ValueError:
            return None
        return values[0] if len(values) == 1 else None
    return None


def configure_token(repo_root: Path) -> str | None:
    """Prefer the user's .bashrc HF_KEY, then fall back to environment/.env."""
    token = os.environ.get("HF_KEY") or token_from_bashrc()
    load_dotenv(repo_root / ".env", override=False)
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def preflight(
    api: HfApi, dataset: str, token: str | None, revision: str | None
):
    repo_id = REPOSITORIES[dataset]
    if dataset == "lapchole" and not token:
        raise RuntimeError(
            "LapChole is gated and requires HF_TOKEN. Put it in the gitignored "
            ".env or export HF_TOKEN/HF_KEY before running."
        )
    try:
        info = api.dataset_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=token,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Cannot access {repo_id}. Confirm that this Hugging Face account "
            "has accepted the data agreement and its access request was approved."
        ) from exc

    siblings = info.siblings or []
    video_files = [
        item for item in siblings if item.rfilename.startswith("videos/")
    ]
    video_bytes = sum((item.size or 0) for item in video_files)
    parquet_files = [
        item
        for item in siblings
        if item.rfilename.endswith(".parquet")
        and item.rfilename.startswith("data/")
    ]
    access_probe = (
        parquet_files[0]
        if parquet_files
        else (video_files[0] if video_files else None)
    )
    if access_probe is not None:
        try:
            get_hf_file_metadata(
                hf_hub_url(
                    repo_id=repo_id,
                    filename=access_probe.rfilename,
                    repo_type="dataset",
                    revision=revision,
                ),
                token=token,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Metadata is visible, but this token cannot download files from "
                f"{repo_id}. Check gated-repository approval and token permissions."
            ) from exc
    print(
        f"{dataset}: access OK; {len(video_files)} video files"
        f" ({human_bytes(video_bytes) if video_bytes else 'remote size unavailable'});"
        f" {len(parquet_files)} parquet files",
        flush=True,
    )
    return info


def annotation_source(repo_files: set[str], track: str, split: str) -> str:
    candidates = (
        f"data/{track}/{split}.parquet",
        f"{track}/{split}/0000.parquet",
        f"data/{track}/{split}/0000.parquet",
    )
    for candidate in candidates:
        if candidate in repo_files:
            return candidate
    raise RuntimeError(
        f"No parquet file found for {track}/{split}; checked {', '.join(candidates)}"
    )


def annotation_destination(
    parquet_root: Path, dataset: str, track: str, split: str
) -> Path:
    # Preserve the established HeiCo path used throughout this repository.
    base = parquet_root if dataset == "heico" else parquet_root / dataset
    return base / track / split / "0000.parquet"


def download_annotations(
    dataset: str,
    repo_files: set[str],
    token: str | None,
    revision: str | None,
    parquet_root: Path,
    cache_root: Path,
) -> None:
    repo_id = REPOSITORIES[dataset]
    for track in TRACKS:
        for split in SPLITS:
            source = annotation_source(repo_files, track, split)
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=source,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    cache_dir=cache_root / "hub",
                )
            )
            destination = annotation_destination(
                parquet_root, dataset, track, split
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".parquet.tmp")
            shutil.copy2(cached, temporary)
            temporary.replace(destination)
            rows = len(pd.read_parquet(destination))
            print(
                f"{dataset}: {track}/{split}: {rows:,} rows -> {destination}",
                flush=True,
            )


def download_videos(
    dataset: str,
    token: str | None,
    revision: str | None,
    data_root: Path,
    cache_root: Path,
    max_workers: int,
) -> None:
    destination = data_root / dataset
    destination.mkdir(parents=True, exist_ok=True)
    print(f"{dataset}: downloading videos to {destination}", flush=True)
    snapshot_download(
        repo_id=REPOSITORIES[dataset],
        repo_type="dataset",
        local_dir=destination,
        revision=revision,
        token=token,
        cache_dir=cache_root / "hub",
        allow_patterns=["videos/*"],
        max_workers=max_workers,
    )
    print(f"{dataset}: video download complete", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    token = configure_token(repo_root)
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.parquet_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    infos = {}
    for dataset in dict.fromkeys(args.datasets):
        infos[dataset] = preflight(api, dataset, token, args.revision)

    if args.dry_run:
        print("Dry run complete; nothing downloaded.", flush=True)
        return 0

    for dataset in dict.fromkeys(args.datasets):
        info = infos[dataset]
        repo_files = {item.rfilename for item in info.siblings or []}
        if not args.skip_annotations:
            download_annotations(
                dataset=dataset,
                repo_files=repo_files,
                token=token,
                revision=args.revision,
                parquet_root=args.parquet_root,
                cache_root=args.cache_root,
            )
        if not args.skip_videos:
            download_videos(
                dataset=dataset,
                token=token,
                revision=args.revision,
                data_root=args.data_root,
                cache_root=args.cache_root,
                max_workers=args.max_workers,
            )

    print("All requested downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
