"""Frame extraction + path resolution (multi-track ready).

Custom replacement for ``focus.FrameExtractorPreprocessor``: extracts ONLY the
specific frames referenced by a set of ``Request`` objects, saved by their
ABSOLUTE frame index as ``<frames_root>/<video_stem>/frame{idx:07d}.jpg``.

This naming matches :func:`request_frame_paths` below (and
``focus.FocusFrameDataset`` at ``stride=1``), so the runner can look frames up
deterministically.

Decoding uses PyAV (frame-accurate keyframe seek + forward decode). The env's
``decord`` and OpenCV ffmpeg backends cannot decode the dataset's FMP4/mpeg4
videos on aarch64 (swscale "Cannot initialize the conversion context").

Per track:
- FRAME    : ``start_time == end_time`` -> 1 frame/question.
- SEGMENT  /
- PROCEDURE: a window ``[start, end]`` -> frames every ``stride`` source frames.
  Pass the SAME ``stride`` to extraction and to :func:`request_frame_paths` so
  the indices line up.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from focus.config import DATASET_BASE_FPS

from .paths import DATASET
from .paths import frames_root as _frames_root
from .paths import video_dir as _video_dir

logger = logging.getLogger(__name__)

BASE_FPS: int = DATASET_BASE_FPS[DATASET]  # 25 for heico


def dataset_base_fps(dataset: str = DATASET) -> int:
    """Return the official native FPS used for annotation-to-frame indexing."""
    try:
        return int(DATASET_BASE_FPS[dataset])
    except KeyError as exc:
        raise ValueError(f"no official base FPS for dataset {dataset!r}") from exc


def frame_index(seconds: float, base_fps: int = BASE_FPS) -> int:
    """Absolute source-frame index for a timestamp (seconds)."""
    return round(seconds * base_fps)


def _indices_for(start_s: float, end_s: float, base_fps: int, stride: int) -> list[int]:
    sf = frame_index(start_s, base_fps)
    ef = frame_index(end_s, base_fps)
    if ef <= sf:  # single frame (FRAME track)
        return [sf]
    return list(range(sf, ef + 1, max(stride, 1)))


def indices_for_request(request, stride: int = 1, base_fps: int = BASE_FPS) -> list[int]:
    """Absolute frame indices on the extraction grid for one ``Request`` window.

    Public wrapper over the internal grid so inference-time frame selection
    (``src.videovqa``) and extraction share one definition by construction.
    """
    return _indices_for(request.start_time, request.end_time, base_fps, stride)


def needed_frames(requests, base_fps: int = BASE_FPS, stride: int = 1) -> dict[str, list[int]]:
    """Map ``videoID -> sorted unique absolute frame indices`` needed by ``requests``."""
    out: dict[str, set[int]] = defaultdict(set)
    for req in requests:
        out[req.videoID].update(_indices_for(req.start_time, req.end_time, base_fps, stride))
    return {vid: sorted(idxs) for vid, idxs in out.items()}


def request_frame_paths(
    request,
    frames_folder: str = "frames",
    base_fps: int | None = None,
    stride: int = 1,
    dataset: str = DATASET,
) -> list[Path]:
    """Resolve the on-disk frame path(s) for one ``Request`` (single path for FRAME)."""
    fps = base_fps if base_fps is not None else dataset_base_fps(dataset)
    root = _frames_root(frames_folder, dataset)
    stem = Path(request.videoID).stem
    return [
        root / stem / f"frame{i:07d}.jpg"
        for i in _indices_for(request.start_time, request.end_time, fps, stride)
    ]


def _extract_one(args) -> tuple[str, int, int, str]:
    """Worker: extract the given absolute frame indices from one video.

    PyAV keyframe-seeks before each target then decodes forward to it
    (frame-accurate). Never raises — errors are reported per-video.
    """
    video_path, out_dir, indices, resolution, jpeg_quality, skip_existing = args
    try:
        import av
        import cv2

        cv2.setNumThreads(1)  # keep each worker single-threaded (login node caps procs)
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [
            i for i in indices
            if not (skip_existing and (out_dir / f"frame{i:07d}.jpg").exists())
        ]
        if not todo:
            return (video_path.name, 0, len(indices), "ok(all-exist)")
        if not video_path.exists():
            return (video_path.name, 0, len(indices), "ERROR: video missing")

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_count = 1  # parallelism is across videos (process pool); avoid thread blow-up
        fps = float(stream.average_rate)
        tb = stream.time_base

        saved = 0
        for target in sorted(todo):
            container.seek(int((target / fps) / tb), stream=stream, backward=True, any_frame=False)
            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if round(float(frame.pts * tb) * fps) >= target:
                    chosen = frame
                    break
            if chosen is None:
                continue
            img = chosen.to_ndarray(format="bgr24")
            if resolution:
                img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                str(out_dir / f"frame{target:07d}.jpg"),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            saved += 1
        container.close()
        return (video_path.name, saved, len(indices), "ok")
    except Exception as e:  # keep the pool alive; report per-video
        return (video_path.name, 0, len(indices), f"ERROR: {e}")


def _extract_with_retry(task, retries: int = 3, delay: float = 1.0) -> tuple[str, int, int, str]:
    """Run :func:`_extract_one`, retrying transient ``EAGAIN`` (shared-node proc cap).

    ``skip_existing`` makes each retry resume from already-saved frames.
    """
    r = _extract_one(task)
    n = 0
    while r[3].startswith("ERROR") and n < retries:
        time.sleep(delay)
        r = _extract_one(task)
        n += 1
    return r


def extract_frames(
    needed: dict[str, list[int]],
    frames_folder: str = "frames",
    dataset: str = DATASET,
    resolution: tuple[int, int] | None = None,
    jpeg_quality: int = 95,
    skip_existing: bool = True,
    max_workers: int = 1,
    retries: int = 3,
) -> list[tuple[str, int, int, str]]:
    """Extract a ``videoID -> indices`` plan (e.g. from :func:`needed_frames`).

    Frames are saved as ``<frames_root(folder)>/<video_stem>/frame{idx:07d}.jpg``.
    Returns one ``(video, saved, requested, status)`` row per video.

    Runs sequentially in-process when ``max_workers <= 1`` (safe on the login
    node, which caps user processes at ~1900 — forking workers + decoder/BLAS
    threads hits the cap). Use ``max_workers > 1`` only on SLURM compute nodes;
    it is fast enough single-stream (~0.03-0.07 s/frame, whole frame track ~6 min).
    """
    vdir = _video_dir(dataset)
    root = _frames_root(frames_folder, dataset)
    tasks = [
        (vdir / vid, root / Path(vid).stem, idxs, resolution, jpeg_quality, skip_existing)
        for vid, idxs in needed.items()
    ]
    results: list[tuple[str, int, int, str]] = []

    def _record(r: tuple[str, int, int, str]) -> None:
        results.append(r)
        logger.info("extract %s: saved=%d/%d %s", *r)

    worker = partial(_extract_with_retry, retries=retries)
    if max_workers <= 1:
        for task in tasks:
            _record(worker(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for r in ex.map(worker, tasks):
                _record(r)

    errors = [r for r in results if r[3].startswith("ERROR")]
    if errors:
        logger.warning("%d video(s) had errors: %s", len(errors), [e[0] for e in errors])
    return results


def extract_split(
    track: str,
    split: str,
    dataset: str = DATASET,
    frames_folder: str = "frames",
    stride: int = 1,
    resolution: tuple[int, int] | None = None,
    max_workers: int = 1,
    skip_existing: bool = True,
) -> list[tuple[str, int, int, str]]:
    """Convenience: load a track/split and extract exactly its referenced frames."""
    from .data import load_split

    reqs, _ = load_split(track, split, dataset)
    needed = needed_frames(reqs, base_fps=dataset_base_fps(dataset), stride=stride)
    total = sum(len(v) for v in needed.values())
    logger.info(
        "extract_split %s/%s: %d videos, %d unique frames (stride=%d)",
        track, split, len(needed), total, stride,
    )
    return extract_frames(
        needed,
        frames_folder=frames_folder,
        dataset=dataset,
        resolution=resolution,
        max_workers=max_workers,
        skip_existing=skip_existing,
    )


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract only the frames referenced by a track/split."
    )
    ap.add_argument("--track", default="frame", choices=["frame", "segment", "procedure"])
    ap.add_argument("--dataset", default=DATASET, choices=sorted(DATASET_BASE_FPS))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--stride", type=int, default=1, help="frames every N source frames (windows)")
    ap.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--workers", type=int, default=1, help=">1 only on SLURM compute nodes")
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if frame exists")
    a = ap.parse_args()

    res = tuple(a.resolution) if a.resolution else None
    for sp in a.splits:
        extract_split(
            a.track, sp,
            dataset=a.dataset,
            frames_folder=a.frames_folder, stride=a.stride, resolution=res,
            max_workers=a.workers, skip_existing=not a.overwrite,
        )


if __name__ == "__main__":
    _main()
