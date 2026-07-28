"""Official-compatible sparse frame extraction for all FOCUS tracks.

The official ``orena-focus`` preprocessor decodes with ``decord.VideoReader``
and saves frames with OpenCV as JPEGs at quality 95.  It extracts every source
frame, which is unnecessary for training. This module uses the same decoding
and encoding operations but requests only the unique absolute frame indices
selected from the official train and test parquets.

Output paths retain the official convention::

    <focus_root>/<dataset>/frames/<video_stem>/frame{index:07d}.jpg

Indices and per-dataset FPS match ``focus.data.FocusFrameDataset`` in
``orena-focus==0.3.4``. Window tracks may additionally cap the evenly sampled
frames per question to match the VLM training configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

DATASET_BASE_FPS = {"heico": 25, "lapchole": 30}
TRACKS = ("frame", "segment", "procedure")
SPLITS = ("train", "test")
JPEG_QUALITY = 95
DEFAULT_BATCH_SIZE = 32  # Same batch size as the official implementation.
FRAME_COUNTS_NAME = "_video_frame_counts.json"


def timestamp_to_seconds(value: object) -> float:
    """Parse an annotation timestamp in ``HH:MM:SS[.fraction]`` form."""
    fields = str(value).strip().split(":")
    if len(fields) != 3:
        raise ValueError(f"Invalid timestamp {value!r}; expected HH:MM:SS")
    hours, minutes, seconds = int(fields[0]), int(fields[1]), float(fields[2])
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def annotation_path(
    parquet_root: Path,
    dataset: str,
    track: str,
    split: str,
) -> Path:
    """Return this repository's parquet path for one dataset/track/split."""
    base = parquet_root if dataset == "heico" else parquet_root / dataset
    return base / track / split / "0000.parquet"


def even_subsample(indices: list[int], max_frames: int | None) -> list[int]:
    """Match the training sampler's endpoint-preserving even subsampling."""
    if max_frames is None or len(indices) <= max_frames:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
    return [indices[i] for i in dict.fromkeys(positions.tolist())]


def build_plan(
    dataset: str,
    parquet_root: Path,
    splits: Iterable[str] = SPLITS,
    *,
    track: str = "frame",
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[dict[str, list[int]], int]:
    """Build ``video filename -> sorted unique official-compatible indices``."""
    fps = DATASET_BASE_FPS[dataset]
    stride = max(int(stride), 1)
    requested: dict[str, set[int]] = defaultdict(set)
    row_count = 0

    for split in splits:
        path = annotation_path(parquet_root, dataset, track, split)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {track.upper()} annotations: {path}")
        frame = pd.read_parquet(
            path,
            columns=["video", "track", "timestamp_start", "timestamp_end"],
        )
        row_count += len(frame)

        tracks = {str(track).lower() for track in frame["track"].dropna().unique()}
        if tracks and tracks != {track}:
            raise ValueError(f"{path} contains unexpected tracks: {sorted(tracks)}")

        for row in frame.itertuples(index=False):
            start = round(timestamp_to_seconds(row.timestamp_start) * fps)
            end = round(timestamp_to_seconds(row.timestamp_end) * fps)
            indices = list(range(start, max(end, start) + 1, stride))
            requested[str(row.video)].update(even_subsample(indices, max_frames))

    return {video: sorted(indices) for video, indices in requested.items()}, row_count


def _read_frame_count(video_path_s: str) -> tuple[str, int, str]:
    """Read one source video's exact Decord frame count."""
    video_path = Path(video_path_s)
    try:
        import decord

        reader = decord.VideoReader(
            str(video_path),
            ctx=decord.cpu(0),
            num_threads=1,
        )
        count = len(reader)
        del reader
        return video_path.name, count, "ok"
    except Exception as exc:
        return video_path.name, 0, f"ERROR: {type(exc).__name__}: {exc}"


def clip_plan_to_videos(
    plan: dict[str, list[int]],
    videos_root: Path,
    workers: int,
) -> tuple[dict[str, list[int]], dict[str, int], int]:
    """Drop annotated indices beyond each decoded video while recording counts."""
    paths = [str(videos_root / video) for video in sorted(plan)]
    max_workers = min(max(workers, 1), max(len(paths), 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_read_frame_count, paths))

    failures = [result for result in results if result[2].startswith("ERROR")]
    if failures:
        raise RuntimeError(f"failed to read video frame counts: {failures[:5]}")

    counts = {video: count for video, count, _ in results}
    clipped = 0
    valid_plan = {}
    for video, indices in plan.items():
        valid = [index for index in indices if 0 <= index < counts[video]]
        clipped += len(indices) - len(valid)
        valid_plan[video] = valid
    return valid_plan, counts, clipped


def write_frame_counts(output_root: Path, dataset: str, counts: dict[str, int]) -> Path:
    """Persist source frame counts for identical clipping during training."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / FRAME_COUNTS_NAME
    temporary = output_root / f".{FRAME_COUNTS_NAME}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "decoder": "decord.VideoReader",
                "video_frame_counts": counts,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(path)
    return path


def _extract_video(
    task: tuple[str, str, list[int], int, bool],
) -> tuple[str, int, int, int, str]:
    """Decode and save one video's requested frames with the official operations."""
    video_path_s, output_dir_s, indices, batch_size, overwrite = task
    video_path = Path(video_path_s)
    output_dir = Path(output_dir_s)

    try:
        import cv2
        import decord

        decord.bridge.set_bridge("native")
        cv2.setNumThreads(1)
        if not video_path.is_file():
            return video_path.name, 0, 0, len(indices), "ERROR: source video missing"

        output_dir.mkdir(parents=True, exist_ok=True)
        existing_names = (
            {
                entry.name
                for entry in output_dir.iterdir()
                # A failed/interrupted encoder can leave a zero-byte pathname
                # behind.  Treat it as absent so a normal rerun repairs it.
                if entry.is_file() and entry.stat().st_size > 0
            }
            if not overwrite
            else set()
        )
        todo = [
            index
            for index in indices
            if overwrite or f"frame{index:07d}.jpg" not in existing_names
        ]
        if not todo:
            return video_path.name, 0, len(indices), len(indices), "ok(all-exist)"

        reader = decord.VideoReader(
            str(video_path),
            ctx=decord.cpu(0),
            num_threads=1,
        )
        frame_count = len(reader)
        invalid = [index for index in todo if index < 0 or index >= frame_count]
        if invalid:
            sample = ", ".join(map(str, invalid[:5]))
            return (
                video_path.name,
                0,
                len(indices) - len(todo),
                len(indices),
                f"ERROR: {len(invalid)} indices outside [0, {frame_count}); first: {sample}",
            )

        saved = 0
        for offset in range(0, len(todo), batch_size):
            batch_indices = todo[offset : offset + batch_size]
            frames = reader.get_batch(batch_indices)
            if hasattr(frames, "asnumpy"):
                frames = frames.asnumpy()

            for index, frame in zip(batch_indices, frames, strict=True):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                path = output_dir / f"frame{index:07d}.jpg"
                if not cv2.imwrite(
                    str(path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                ):
                    raise OSError(f"cv2.imwrite failed for {path}")
                saved += 1

        existing = len(indices) - len(todo)
        return video_path.name, saved, existing, len(indices), "ok"
    except Exception as exc:
        return video_path.name, 0, 0, len(indices), f"ERROR: {type(exc).__name__}: {exc}"


def extract_plan(
    plan: dict[str, list[int]],
    videos_root: Path,
    output_root: Path,
    workers: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> list[tuple[str, int, int, int, str]]:
    """Extract a plan in parallel, with one single-threaded decoder per video."""
    tasks = [
        (
            str(videos_root / video),
            str(output_root / Path(video).stem),
            indices,
            batch_size,
            overwrite,
        )
        for video, indices in sorted(plan.items())
    ]
    results: list[tuple[str, int, int, int, str]] = []
    max_workers = min(max(workers, 1), max(len(tasks), 1))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_video, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            video, saved, existing, requested, status = result
            logger.info(
                "%s: saved=%d existing=%d requested=%d %s",
                video,
                saved,
                existing,
                requested,
                status,
            )
    return sorted(results)


def verify_plan(
    plan: dict[str, list[int]],
    output_root: Path,
) -> tuple[list[Path], list[Path], int]:
    """Return missing paths, unexpected frame paths, and expected count."""
    missing: list[Path] = []
    unexpected: list[Path] = []
    expected_count = 0

    for video, indices in plan.items():
        folder = output_root / Path(video).stem
        expected_names = {f"frame{index:07d}.jpg" for index in indices}
        expected_count += len(expected_names)
        existing_names = (
            {
                entry.name
                for entry in folder.iterdir()
                if (
                    entry.name.startswith("frame")
                    and entry.name.endswith(".jpg")
                    and entry.is_file()
                    and entry.stat().st_size > 0
                )
            }
            if folder.is_dir()
            else set()
        )
        missing.extend(folder / name for name in sorted(expected_names - existing_names))
        unexpected.extend(folder / name for name in sorted(existing_names - expected_names))
    return missing, unexpected, expected_count


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def write_manifest(
    output_root: Path,
    dataset: str,
    track: str,
    splits: tuple[str, ...],
    stride: int,
    max_frames: int | None,
    clipped_out_of_range: int,
    row_count: int,
    expected_count: int,
    results: list[tuple[str, int, int, int, str]],
    elapsed_seconds: float,
    batch_size: int,
) -> Path:
    """Write reproducibility metadata beside the extracted frames."""
    import cv2
    import decord

    manifest = {
        "dataset": dataset,
        "track": track,
        "splits": list(splits),
        "annotation_rows": row_count,
        "unique_videos": len(results),
        "unique_frames": expected_count,
        "base_fps": DATASET_BASE_FPS[dataset],
        "stride": stride,
        "max_frames_per_question": max_frames,
        "clipped_out_of_range": clipped_out_of_range,
        "decoder": "decord.VideoReader",
        "encoder": "cv2.imwrite",
        "jpeg_quality": JPEG_QUALITY,
        "batch_size": batch_size,
        "orena_focus_version": package_version("orena-focus"),
        "decord_version": getattr(decord, "__version__", "unknown"),
        "decord_path": str(Path(decord.__file__).resolve()),
        "opencv_version": getattr(cv2, "__version__", "unknown"),
        "opencv_path": str(Path(cv2.__file__).resolve()),
        "python_version": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "elapsed_seconds": elapsed_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "video": video,
                "saved": saved,
                "existing": existing,
                "requested": requested,
                "status": status,
            }
            for video, saved, existing, requested, status in results
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"_{track}_track_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract official-compatible JPEGs needed by a FOCUS track."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_BASE_FPS))
    parser.add_argument("--track", default="frame", choices=TRACKS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--focus-root",
        type=Path,
        default=repo_root / "data" / "focus",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=repo_root / "data" / "parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Exact frames output directory (default: <focus-root>/<dataset>/frames).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.video_limit is not None and args.video_limit < 1:
        parser.error("--video-limit must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(processName)s %(message)s",
    )

    splits = tuple(args.splits)
    plan, row_count = build_plan(
        args.dataset,
        args.parquet_root,
        splits,
        track=args.track,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if args.video_limit:
        selected = sorted(plan)[: args.video_limit]
        plan = {video: plan[video] for video in selected}

    videos_root = args.focus_root / args.dataset / "videos"
    output_root = (
        args.output_root
        if args.output_root is not None
        else args.focus_root / args.dataset / "frames"
    )
    missing_videos = [video for video in plan if not (videos_root / video).is_file()]
    if missing_videos:
        logger.error(
            "%d referenced videos are missing; first: %s",
            len(missing_videos),
            missing_videos[:5],
        )
        return 2

    plan, frame_counts, clipped = clip_plan_to_videos(
        plan, videos_root, workers=args.workers
    )
    frame_counts_path = output_root / FRAME_COUNTS_NAME
    expected_count = sum(len(indices) for indices in plan.values())
    logger.info(
        "plan dataset=%s track=%s splits=%s rows=%d videos=%d unique_frames=%d "
        "base_fps=%d stride=%d max_frames=%s clipped=%d output=%s counts=%s",
        args.dataset,
        args.track,
        splits,
        row_count,
        len(plan),
        expected_count,
        DATASET_BASE_FPS[args.dataset],
        args.stride,
        args.max_frames,
        clipped,
        output_root,
        frame_counts_path,
    )
    if args.dry_run:
        return 0
    write_frame_counts(output_root, args.dataset, frame_counts)

    started = time.monotonic()
    results = extract_plan(
        plan,
        videos_root=videos_root,
        output_root=output_root,
        workers=args.workers,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    elapsed = time.monotonic() - started
    errors = [result for result in results if result[-1].startswith("ERROR")]
    missing, unexpected, verified_count = verify_plan(plan, output_root)
    manifest = write_manifest(
        output_root,
        dataset=args.dataset,
        track=args.track,
        splits=splits,
        stride=args.stride,
        max_frames=args.max_frames,
        clipped_out_of_range=clipped,
        row_count=row_count,
        expected_count=verified_count,
        results=results,
        elapsed_seconds=elapsed,
        batch_size=args.batch_size,
    )
    logger.info(
        "verification expected=%d missing=%d unexpected=%d elapsed=%.1fs manifest=%s",
        verified_count,
        len(missing),
        len(unexpected),
        elapsed,
        manifest,
    )
    if errors:
        logger.error("%d video extraction task(s) failed", len(errors))
    if missing:
        logger.error("missing frame examples: %s", [str(path) for path in missing[:5]])
    if unexpected:
        logger.info(
            "%d frames belong to other extraction plans in the shared folder",
            len(unexpected),
        )
    return 1 if errors or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
