#!/usr/bin/env python
"""Decode-audit every extracted JPEG sampled by the configured training tracks."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.frame_track_extraction import FRAME_COUNTS_NAME, build_plan  # noqa: E402


SAMPLING = {
    "frame": {"heico": (1, 1), "lapchole": (1, 1)},
    "segment": {"heico": (25, 64), "lapchole": (30, 64)},
    "procedure": {"heico": (250, 96), "lapchole": (300, 96)},
}


def inspect_jpeg(path_s: str) -> tuple[str, str | None]:
    """Return ``(path, error)`` after checking size, type, and JPEG structure."""
    path = Path(path_s)
    try:
        if not path.is_file():
            return path_s, "missing"
        if path.stat().st_size == 0:
            return path_s, "empty (0 bytes)"
        with Image.open(path) as image:
            if image.format != "JPEG":
                return path_s, f"unexpected format {image.format!r}"
            image.verify()
        return path_s, None
    except Exception as exc:
        return path_s, f"{type(exc).__name__}: {exc}"


def planned_paths(
    datasets: list[str],
    tracks: list[str],
    splits: tuple[str, ...],
) -> tuple[list[str], dict[str, int]]:
    """Return the unique, frame-count-clipped paths used by the selected plans."""
    paths: set[Path] = set()
    per_plan: dict[str, int] = {}
    parquet_root = REPO / "data" / "parquet"

    for dataset in datasets:
        frames_root = REPO / "data" / "focus" / dataset / "frames"
        counts_doc = json.loads((frames_root / FRAME_COUNTS_NAME).read_text())
        counts = counts_doc["video_frame_counts"]
        for track in tracks:
            stride, max_frames = SAMPLING[track][dataset]
            plan, _ = build_plan(
                dataset,
                parquet_root,
                splits,
                track=track,
                stride=stride,
                max_frames=max_frames,
            )
            count = 0
            for video, indices in plan.items():
                folder = frames_root / Path(video).stem
                valid = (index for index in indices if 0 <= index < counts[video])
                selected = {folder / f"frame{index:07d}.jpg" for index in valid}
                paths.update(selected)
                count += len(selected)
            per_plan[f"{dataset}/{track}"] = count
    return sorted(map(str, paths)), per_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("heico", "lapchole"),
        default=["heico", "lapchole"],
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=tuple(SAMPLING),
        default=list(SAMPLING),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["train", "test"],
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    paths, per_plan = planned_paths(
        args.datasets,
        args.tracks,
        tuple(args.splits),
    )
    print(
        f"auditing {len(paths)} unique JPEGs across "
        f"{len(per_plan)} dataset/track plans with {args.workers} workers",
        flush=True,
    )

    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, (path, error) in enumerate(
            executor.map(inspect_jpeg, paths, chunksize=64),
            start=1,
        ):
            if error:
                failures.append({"path": path, "error": error})
            if done % 10_000 == 0 or done == len(paths):
                print(
                    f"checked={done}/{len(paths)} failures={len(failures)}",
                    flush=True,
                )

    report = {
        "datasets": args.datasets,
        "tracks": args.tracks,
        "splits": args.splits,
        "unique_paths": len(paths),
        "per_plan": per_plan,
        "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report={args.report}", flush=True)
    for failure in failures:
        print(f"BAD {failure['error']}: {failure['path']}", flush=True)
    print(f"audit complete: failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
