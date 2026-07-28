"""Multi-frame (video-window) VQA — shared by the SEGMENT and PROCEDURE tracks.

The FRAME-track stack is single-image end to end; this module adds the one
missing piece for window tracks: turning a ``Request`` time window into an
**interleaved, absolute-timestamped multi-image prompt** and running it through
a ``transformers`` VLM.

Design choice (why not the model's native video path): questions and answers
use ABSOLUTE video time ("... at 02:40:06?"), while Qwen3-VL's video pipeline
stamps clip-relative timestamps. So each sampled frame is passed as an image
preceded by a text label ``"Frame at HH:MM:SS:"`` carrying its true video time.

Frame selection reuses the extraction grid (``src.frames`` indices at the same
``stride``), evenly subsampled to ``max_frames`` — selected paths therefore
always exist if extraction ran with the same stride (checked by
:func:`coverage` before any GPU time is spent).

CLI (login-node safe, no GPU/torch needed):
    python -m src.videovqa --track segment --split test --stride 25 --max-frames 64
prints frame-coverage stats for the planned sampling and exits non-zero on gaps.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import adapter, data
from . import prompts as P
from .frames import dataset_base_fps, indices_for_request
from .paths import DATASET, frames_root

logger = logging.getLogger(__name__)
FRAME_COUNTS_NAME = "_video_frame_counts.json"

# meta.json keys that must match for a previous run dir to be resumed
_RESUME_KEYS = (
    "dataset", "track", "split", "model_name", "stride", "max_frames", "frames_folder",
    "min_pixels", "max_pixels", "max_new_tokens", "dtype", "limit", "adapter_path",
)


def seconds_to_ts(seconds: float) -> str:
    """``seconds -> "HH:MM:SS"`` (inverse of focus' ``ts_to_seconds``)."""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def even_subsample(seq: list, k: int) -> list:
    """<= ``k`` items evenly spaced over ``seq``, always keeping both endpoints."""
    n = len(seq)
    if k >= n or n <= 1:
        return list(seq)
    idxs = np.linspace(0, n - 1, max(k, 1)).round().astype(int)
    return [seq[i] for i in dict.fromkeys(idxs.tolist())]


@lru_cache(maxsize=None)
def _video_frame_counts(dataset: str, frames_folder: str) -> dict[str, int]:
    """Load Decord frame counts written by official-compatible extraction."""
    path = frames_root(frames_folder, dataset) / FRAME_COUNTS_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        str(video): int(count)
        for video, count in payload.get("video_frame_counts", {}).items()
    }


def sampled_indices(
    request,
    *,
    dataset: str,
    stride: int,
    max_frames: int,
    frames_folder: str,
    base_fps: int,
) -> list[int]:
    """Return the extraction-matched sample indices, clipped to decoded length."""
    indices = even_subsample(
        indices_for_request(request, stride=stride, base_fps=base_fps), max_frames
    )
    frame_count = _video_frame_counts(dataset, frames_folder).get(request.videoID)
    if frame_count is not None:
        indices = [index for index in indices if 0 <= index < frame_count]
    return indices


@dataclass(frozen=True)
class FrameRef:
    """One sampled frame: absolute source index, video time, on-disk path."""

    index: int
    seconds: float
    ts: str  # "HH:MM:SS" absolute video time
    path: Path


def select_frames(
    request,
    *,
    stride: int,
    max_frames: int,
    frames_folder: str = "frames",
    dataset: str = DATASET,
    base_fps: int | None = None,
    check_exists: bool = True,
) -> list[FrameRef]:
    """Pick <= ``max_frames`` frames on the extraction grid for a request window.

    Missing files are dropped with a warning (robustness at inference time);
    use :func:`coverage` beforehand to guarantee there are none.
    """
    fps = base_fps if base_fps is not None else dataset_base_fps(dataset)
    indices = sampled_indices(
        request, dataset=dataset, stride=stride, max_frames=max_frames,
        frames_folder=frames_folder, base_fps=fps,
    )
    root = frames_root(frames_folder, dataset) / Path(request.videoID).stem
    refs: list[FrameRef] = []
    for i in indices:
        p = root / f"frame{i:07d}.jpg"
        if check_exists and not p.exists():
            logger.warning("qID=%s missing frame %s (skipped)", request.qID, p)
            continue
        secs = i / fps
        refs.append(FrameRef(index=i, seconds=secs, ts=seconds_to_ts(secs), path=p))
    return refs


def build_preamble(track: str, request, frame_refs: list[FrameRef]) -> str:
    """One-sentence orientation: what span the frames cover and how densely."""
    n = len(frame_refs)
    start_ts = seconds_to_ts(request.start_time)
    end_ts = seconds_to_ts(request.end_time)
    dur = max(request.end_time - request.start_time, 0.0)
    interval = dur / (n - 1) if n > 1 else 0.0

    if track == "procedure":
        head = f"The surgical video from its start (00:00:00) up to {end_ts} is shown"
    else:  # segment (and any other window track)
        head = f"This clip covers video time {start_ts} to {end_ts} ({dur:.0f} s of surgery). It is shown"

    if n > 1:
        head += f" as {n} frames in chronological order, about {interval:.0f} s apart"
    else:
        head += " as a single frame"
    return head + ", each labelled with its absolute video timestamp."


def build_messages(
    system: str | None,
    preamble: str,
    frame_refs: list[FrameRef],
    instruction: str,
) -> tuple[list[dict], list[str]]:
    """Chat messages with interleaved ``"Frame at TS:"`` labels + image markers.

    Pure function (no model/IO) so prompt construction is testable on the login
    node. Returns ``(messages, image_paths)``; the image order matches the
    ``{"type": "image"}`` marker order, which is how the processor pairs them.
    """
    content: list[dict] = [{"type": "text", "text": preamble}]
    for ref in frame_refs:
        content.append({"type": "text", "text": f"Frame at {ref.ts}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": instruction})

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})
    return messages, [str(r.path) for r in frame_refs]


class VideoVLM:
    """Multi-image image-text-to-text wrapper (generalises the frame-track ``VLM``).

    Same loading recipe as ``track-frame/zeroshot-sweep/src/vlm.py`` —
    ``AutoModelForImageTextToText`` + ``AutoProcessor``, bf16, sdpa (no
    flash-attn on aarch64), greedy decode — but :meth:`answer` takes prebuilt
    interleaved messages plus N image paths. Optional ``adapter_path`` loads a
    PEFT LoRA adapter on top of the base weights.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        attn_implementation: str = "sdpa",
        adapter_path: str | None = None,
        io_workers: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens

        proc_kwargs: dict = {}
        if min_pixels:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
            logger.info("loaded LoRA adapter from %s", adapter_path)

        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=io_workers)  # hide Lustre latency

    def load_images(self, paths: list[str]):
        from PIL import Image

        return list(self._pool.map(lambda p: Image.open(p).convert("RGB"), paths))

    def answer(self, messages: list[dict], image_paths: list[str]) -> tuple[str, float]:
        """Greedy-decode one answer. Returns ``(text, generate_seconds)``."""
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = self.load_images(image_paths)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            t0 = time.time()
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            latency = time.time() - t0

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return out, latency


def coverage(
    track: str,
    split: str,
    *,
    dataset: str = DATASET,
    stride: int,
    max_frames: int,
    frames_folder: str = "frames",
    limit: int | None = None,
) -> dict:
    """Stat every frame path the sampler would use; report gaps BEFORE GPU time.

    A non-zero ``missing`` almost always means extraction ran with a different
    stride (or not at all) — the grid must match exactly.
    """
    rows = data.read_parquet(track, split, dataset).to_dict("records")
    if limit:
        rows = rows[:limit]
    reqs = [data.row_to_request(r) for r in rows]

    expected_by_folder: dict[Path, set[str]] = {}
    per_q: list[int] = []
    fps = dataset_base_fps(dataset)
    root = frames_root(frames_folder, dataset)
    for req in reqs:
        idxs = sampled_indices(
            req, dataset=dataset, stride=stride, max_frames=max_frames,
            frames_folder=frames_folder, base_fps=fps,
        )
        per_q.append(len(idxs))
        folder = root / Path(req.videoID).stem
        expected_by_folder.setdefault(folder, set()).update(
            f"frame{i:07d}.jpg" for i in idxs
        )

    missing = []
    unique_frames = 0
    for folder, expected in expected_by_folder.items():
        unique_frames += len(expected)
        existing = {entry.name for entry in folder.iterdir()} if folder.is_dir() else set()
        missing.extend(str(folder / name) for name in sorted(expected - existing))
    return {
        "dataset": dataset,
        "track": track,
        "split": split,
        "stride": stride,
        "max_frames": max_frames,
        "questions": len(reqs),
        "mean_frames_per_q": float(np.mean(per_q)) if per_q else 0.0,
        "unique_frames": unique_frames,
        "missing": len(missing),
        "missing_examples": missing[:10],
    }


def _load_done(jsonl_path: Path) -> dict[str, dict]:
    """Read finished records from a (possibly truncated) resume JSONL."""
    done: dict[str, dict] = {}
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["sample_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                logger.warning("skipping corrupt resume line (likely mid-write kill)")
    return done


def _sec_caps(value) -> str:
    """(list | ndarray | None) secondary_capabilities -> '|'-joined string."""
    if value is None:
        return ""
    try:
        return "|".join(str(v) for v in list(value))
    except TypeError:
        return str(value)


def run_video_qa(
    *,
    dataset: str = DATASET,
    track: str,
    split: str,
    model_path: str,
    model_name: str,
    out_dir: str | Path,
    stride: int,
    max_frames: int,
    frames_folder: str = "frames",
    limit: int | None = None,
    max_new_tokens: int = 64,
    dtype: str = "bfloat16",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    adapter_path: str | None = None,
) -> Path:
    """Shared window-track inference loop (rich rows, JSONL resume, OOM fallback).

    Writes under ``out_dir``: ``requests.json``, ``meta.json``,
    ``predictions.jsonl`` (append-as-you-go, makes requeues resumable),
    ``predictions.parquet`` (rich rows, parquet order) and ``responses.json``
    (focus-compatible for the shared evaluator).
    """
    from focus import Response, save_items

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": dataset, "track": track, "split": split, "model_path": str(model_path),
        "model_name": model_name, "stride": stride, "max_frames": max_frames,
        "frames_folder": frames_folder, "limit": limit,
        "max_new_tokens": max_new_tokens, "dtype": dtype,
        "min_pixels": min_pixels, "max_pixels": max_pixels,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "system_prompt": P.system_prompt(track),
    }
    meta_path = out / "meta.json"
    if meta_path.exists():  # resuming — refuse silently mixing incompatible runs
        old = json.loads(meta_path.read_text())
        bad = [k for k in _RESUME_KEYS if old.get(k) != meta.get(k)]
        if bad:
            raise RuntimeError(
                f"{out} holds a run with different settings ({', '.join(bad)}); "
                "use a fresh --out or delete the old dir."
            )
    meta_path.write_text(json.dumps(meta, indent=2))

    rows = data.read_parquet(track, split, dataset).to_dict("records")
    if limit:
        rows = rows[:limit]
    reqs = [data.row_to_request(r) for r in rows]
    save_items(reqs, out / "requests.json")

    cov = coverage(
        track, split, dataset=dataset, stride=stride, max_frames=max_frames,
        frames_folder=frames_folder, limit=limit,
    )
    logger.info("coverage: %s", json.dumps(cov))
    if cov["missing"]:
        raise RuntimeError(
            f"{cov['missing']} sampled frames missing on disk (extraction stride "
            f"mismatch?), e.g. {cov['missing_examples'][:3]} — aborting before GPU."
        )

    jsonl_path = out / "predictions.jsonl"
    done = _load_done(jsonl_path)
    todo = [(req, row) for req, row in zip(reqs, rows) if req.qID not in done]
    logger.info("%d/%d samples already done; %d to run", len(done), len(reqs), len(todo))

    if todo:
        model = VideoVLM(
            str(model_path), dtype=dtype, max_new_tokens=max_new_tokens,
            min_pixels=min_pixels, max_pixels=max_pixels, adapter_path=adapter_path,
        )
        system = P.system_prompt(track)
        t0 = time.time()

        with jsonl_path.open("a") as jf:
            for i, (req, row) in enumerate(todo):
                fmt = row["answer_format"]
                instr, opts = P.build_instruction(req.question, fmt)
                multi = adapter.is_multi_select(req.question)
                refs = select_frames(
                    req, stride=stride, max_frames=max_frames,
                    frames_folder=frames_folder, dataset=dataset,
                )

                raw, lat, err = "", 0.0, ""
                if not refs:
                    err = "no frames available for window"
                    logger.warning("qID=%s %s", req.qID, err)
                else:
                    preamble = build_preamble(track, req, refs)
                    messages, paths = build_messages(system, preamble, refs, instr)
                    try:
                        raw, lat = model.answer(messages, paths)
                    except Exception as e:
                        import torch

                        if isinstance(e, torch.cuda.OutOfMemoryError) and len(refs) > 1:
                            torch.cuda.empty_cache()
                            refs = even_subsample(refs, max(len(refs) // 2, 1))
                            logger.warning(
                                "qID=%s OOM — retrying with %d frames", req.qID, len(refs)
                            )
                            preamble = build_preamble(track, req, refs)
                            messages, paths = build_messages(system, preamble, refs, instr)
                            try:
                                raw, lat = model.answer(messages, paths)
                            except Exception as e2:
                                err = f"{type(e2).__name__}: {e2}"
                        else:
                            err = f"{type(e).__name__}: {e}"
                        if err:
                            logger.warning("qID=%s inference error: %s", req.qID, err)

                resp = adapter.build_response(
                    req.qID, raw, fmt, latency=lat, options=opts, multi=multi
                )
                rec = {
                    "sample_id": req.qID,
                    "model_name": model_name,
                    "question": req.question,
                    "answer": str(row["answer"]),
                    "answer_format": fmt,
                    "primary_capability": str(row["primary_capability"]),
                    "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                    "clinical_relevance": bool(row["clinical_relevance"]),
                    "ood": bool(row["ood"]),
                    "video": row["video"],
                    "timestamp_start": str(row["timestamp_start"]),
                    "timestamp_end": str(row["timestamp_end"]),
                    # middle frame as the representative image (visualiser expects one)
                    "image_path": str(refs[len(refs) // 2].path) if refs else "",
                    "frame_paths": "|".join(str(r.path) for r in refs),
                    "frame_timestamps": "|".join(r.ts for r in refs),
                    "n_frames": len(refs),
                    "prompt": instr,
                    "raw_model_output": raw,  # NEVER overwritten by the normaliser
                    "normalized_prediction": resp.content,
                    "prediction": resp.content,  # evaluator-compatible alias
                    "latency_sec": lat,
                    "error": err,
                }
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                done[req.qID] = rec

                if i % 25 == 0:
                    dt = time.time() - t0
                    rate = dt / (i + 1)
                    logger.info(
                        "%d/%d fmt=%s n_frames=%d raw=%r -> %r (%.2fs, avg %.2fs/Q, "
                        "ETA %.0f min)%s",
                        i + 1, len(todo), fmt, len(refs), raw[:40], resp.content[:40],
                        lat, rate, rate * (len(todo) - i - 1) / 60,
                        f" ERR={err[:40]}" if err else "",
                    )

    # finalise in parquet row order
    import pandas as pd

    records = [done[req.qID] for req in reqs if req.qID in done]
    if len(records) != len(reqs):
        logger.warning("only %d/%d samples have records", len(records), len(reqs))
    pd.DataFrame(records).to_parquet(out / "predictions.parquet", index=False)
    responses = [
        Response(qID=r["sample_id"], content=r["normalized_prediction"],
                 latency=float(r["latency_sec"]))
        for r in records
    ]
    save_items(responses, out / "responses.json")

    n_parse = sum(
        adapter.is_parseable(r["answer_format"], r["normalized_prediction"]) for r in records
    )
    n_err = sum(bool(r["error"]) for r in records)
    lats = [r["latency_sec"] for r in records if r["latency_sec"] > 0]
    logger.info(
        "done: %d rows | mean %.2fs/Q | parseable=%d/%d | errors=%d | -> %s",
        len(records), float(np.mean(lats)) if lats else 0.0, n_parse, len(records),
        n_err, out / "predictions.parquet",
    )
    return out / "predictions.parquet"


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Frame-coverage precheck for window-track sampling (no GPU)."
    )
    ap.add_argument("--dataset", default=DATASET, choices=["heico", "lapchole"])
    ap.add_argument("--track", required=True, choices=["segment", "procedure", "frame"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--max-frames", type=int, required=True)
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cov = coverage(
        a.track, a.split, dataset=a.dataset, stride=a.stride, max_frames=a.max_frames,
        frames_folder=a.frames_folder, limit=a.limit,
    )
    print(json.dumps(cov, indent=2))
    raise SystemExit(1 if cov["missing"] else 0)


if __name__ == "__main__":
    _main()
