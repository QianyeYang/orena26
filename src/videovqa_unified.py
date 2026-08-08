"""Unified multi-track LoRA SFT — FRAME + SEGMENT + PROCEDURE in one run.

One adapter is trained on the concatenation of all three tracks' train rows,
with per-track checkpoint selection afterwards. Every item keeps its own
track's inference regime, built with the SAME shared functions that track's
inference uses, so per-track train/inference prompt parity holds by
construction (see ``tests/test_unified_parity.py``):

- FRAME: 1 image, frame-style messages (no preamble / timestamp label),
  identical to ``track-frame/lora-finetune/src/dataset.py``.
- SEGMENT / PROCEDURE: ``select_frames`` + ``build_preamble`` +
  ``build_messages`` interleaved prompts, identical to ``src.videovqa_sft``.

Pixel budgets differ per track (602112 / 262144 / 131072) but the processor
holds ONE global setting, so :class:`UnifiedCollator` pre-resizes each image
to its item's budget with the model's own smart-resize geometry (factor =
patch_size x merge_size); the processor, loaded with the max budget, then
finds already-conforming images and its own resize is a no-op. The per-epoch
eval path (:class:`UnifiedEpochEvalCallback`) applies the same pre-resize
before ``generate`` — it bypasses the collator, so without this the eval
would silently run every track at the global budget and break parity with
specialist inference.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import adapter, data, frames
from . import prompts as P
from .videovqa import build_messages, build_preamble, coverage, select_frames
from .videovqa_sft import (
    PROJECTOR_CANDIDATES,
    TrainingIntervalSyncCallback,
    VideoEpochEvalCallback,
    VideoSFTCollator,
    _sec_caps,
    detect_projectors,
    discover_targets,
    distributed_rank,
    is_world_process_zero,
    resolve_base,
    stride_for_dataset,
)

log = logging.getLogger("videovqa-unified")

UNIFIED_TRACKS = ("frame", "segment", "procedure")


def smart_resize_dims(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Qwen smart-resize target dims (copied semantics; idempotent under itself)."""
    if height < factor or width < factor:
        raise ValueError(f"image {width}x{height} smaller than factor {factor}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def budget_resize(
    img: Image.Image, *, factor: int, min_pixels: int, max_pixels: int
) -> Image.Image:
    """Resize a PIL image to its track's pixel budget (no-op if already conforming)."""
    h_bar, w_bar = smart_resize_dims(
        img.height, img.width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (img.height, img.width) == (h_bar, w_bar):
        return img
    return img.resize((w_bar, h_bar), Image.BICUBIC)


def processor_pixel_geometry(processor) -> tuple[int, int, int | None]:
    """Return ``(factor, min_pixels, max_pixels)`` from a loaded processor.

    Handles both attribute styles (``min_pixels``/``max_pixels`` and the fast
    processors' ``size={"shortest_edge", "longest_edge"}``).
    """
    ip = processor.image_processor
    factor = int(getattr(ip, "patch_size", 16)) * int(getattr(ip, "merge_size", 2))
    size = getattr(ip, "size", None) or {}
    min_px = getattr(ip, "min_pixels", None) or size.get("shortest_edge") or factor * factor
    max_px = getattr(ip, "max_pixels", None) or size.get("longest_edge")
    return factor, int(min_px), (int(max_px) if max_px else None)


def build_frame_messages(system: str, instruction: str) -> list[dict]:
    """FRAME-track chat structure, byte-identical to the specialist SFT collator."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        },
    ]


class UnifiedSFTDataset(Dataset):
    """Concatenated (track, row) items; ``__getitem__`` dispatches per track.

    ``tracks_cfg`` maps track name -> {stride, max_frames, max_pixels,
    repeat?, eval_n?}. ``repeat`` upsamples a track's rows (cheap for FRAME).
    ``limit_per_track`` keeps the first N rows of each (track, dataset) shard
    for smoke tests.
    """

    def __init__(
        self,
        tracks_cfg: dict[str, dict],
        split: str = "train",
        *,
        datasets: tuple[str, ...] | list[str] = ("heico", "lapchole"),
        frames_folder: str = "frames",
        limit_per_track: int | None = None,
    ) -> None:
        self.frames_folder = frames_folder
        self.track_cfg = {t: tracks_cfg[t] for t in UNIFIED_TRACKS if t in tracks_cfg}
        if not self.track_cfg:
            raise ValueError(f"tracks_cfg selects no track from {UNIFIED_TRACKS}")
        self.systems = {t: P.system_prompt(t) for t in self.track_cfg}
        self.items: list[tuple[str, dict]] = []
        self.counts: dict[str, int] = {}
        for track, tc in self.track_cfg.items():
            rows = data.read_parquets(track, split, tuple(datasets)).to_dict("records")
            if limit_per_track:
                by_ds: dict[str, list[dict]] = {}
                for row in rows:
                    by_ds.setdefault(row["_dataset"], []).append(row)
                rows = [r for ds_rows in by_ds.values() for r in ds_rows[:limit_per_track]]
            repeat = int(tc.get("repeat", 1))
            if repeat < 1:
                raise ValueError(f"tracks.{track}.repeat must be >= 1, got {repeat}")
            for _ in range(repeat):
                self.items.extend((track, row) for row in rows)
            self.counts[track] = len(rows) * repeat
        self.reqs = [data.row_to_request(row) for _, row in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        track, row = self.items[i]
        req = self.reqs[i]
        tc = self.track_cfg[track]
        dataset = row["_dataset"]
        instr, _ = P.build_instruction(req.question, row["answer_format"])
        if track == "frame":
            img = frames.request_frame_paths(
                req, frames_folder=self.frames_folder, dataset=dataset
            )[0]
            messages = build_frame_messages(self.systems["frame"], instr)
            paths = [str(img)]
        else:
            refs = select_frames(
                req, stride=stride_for_dataset(tc["stride"], dataset),
                max_frames=int(tc["max_frames"]), frames_folder=self.frames_folder,
                dataset=dataset, check_exists=False,
            )
            preamble = build_preamble(track, req, refs)
            messages, paths = build_messages(self.systems[track], preamble, refs, instr)
        return {
            "track": track,
            "dataset": dataset,
            "messages": messages,
            "image_paths": paths,
            "answer": str(row["answer"]),
            "max_pixels": int(tc["max_pixels"]),
        }


class UnifiedCollator(VideoSFTCollator):
    """Per-item pixel budgets via pre-resize; masking logic inherited unchanged."""

    def __init__(self, processor, log_first_n: int = 8) -> None:
        super().__init__(processor)
        self.factor, self.min_pixels, self.global_max_pixels = processor_pixel_geometry(
            processor
        )
        self._log_budget = log_first_n

    def load_images(self, ex: dict) -> list:
        budget = int(ex["max_pixels"])
        if self.global_max_pixels and budget > self.global_max_pixels:
            raise ValueError(
                f"track budget {budget} exceeds processor max_pixels "
                f"{self.global_max_pixels}; raise the processor budget in the config"
            )
        images = [
            budget_resize(
                Image.open(p).convert("RGB"),
                factor=self.factor, min_pixels=self.min_pixels, max_pixels=budget,
            )
            for p in ex["image_paths"]
        ]
        if self._log_budget > 0 and images:
            self._log_budget -= 1
            log.info(
                "collate %s: %d image(s) @ %dx%d (budget %d, factor %d)",
                ex.get("track"), len(images), images[0].width, images[0].height,
                budget, self.factor,
            )
        return images


class UnifiedEpochEvalCallback(VideoEpochEvalCallback):
    """Per-epoch eval over (track, dataset) groups.

    Reuses the parent's ``on_epoch_end`` (model state dance, judge scoring,
    CSV append, rank-0 filesystem sync) by keying ``self.groups`` /
    ``self.datasets`` with ``"<track>_<dataset>"`` — the parent only iterates
    those mappings. Deliberately does NOT call ``super().__init__``: groups
    are built per (track, dataset) here, and every attribute the inherited
    methods touch is set below.
    """

    def __init__(
        self,
        *,
        processor,
        out_dir: str,
        judge_model: str,
        tracks_cfg: dict[str, dict],
        datasets: tuple[str, ...] | list[str],
        split: str = "test",
        frames_folder: str = "frames",
        max_new_tokens: int = 64,
        seed: int = 42,
        model_name: str = "lora",
        rank_wait_timeout_seconds: float = 8 * 60 * 60,
    ) -> None:
        self.processor = processor
        self.out = Path(out_dir)
        self.judge_model = judge_model
        self.frames_folder = frames_folder
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.rank_wait_timeout_seconds = rank_wait_timeout_seconds
        self.factor, self.min_pixels, _ = processor_pixel_geometry(processor)

        track_cfgs = {t: tracks_cfg[t] for t in UNIFIED_TRACKS if t in tracks_cfg}
        self.groups: dict[str, dict] = {}
        subset_ids = []
        for track, tc in track_cfgs.items():
            candidates: list[tuple[str, dict]] = []
            for dataset in datasets:
                candidates.extend(
                    (dataset, row)
                    for row in data.read_parquet(track, split, dataset).to_dict("records")
                )
            eval_n = tc.get("eval_n", 500)
            if eval_n and eval_n < len(candidates):
                rng = np.random.default_rng(seed)
                keep = sorted(
                    rng.choice(len(candidates), size=eval_n, replace=False).tolist()
                )
                candidates = [candidates[k] for k in keep]
            for dataset in datasets:
                rows = [row for row_dataset, row in candidates if row_dataset == dataset]
                self.groups[f"{track}_{dataset}"] = {
                    "track": track,
                    "dataset": dataset,
                    "cfg": tc,
                    "rows": rows,
                    "reqs": [data.row_to_request(r) for r in rows],
                    "refs": [data.row_to_reference(r) for r in rows],
                }
                subset_ids.extend(
                    {"track": track, "dataset": dataset, "qID": str(r["id"])} for r in rows
                )
        self.datasets = tuple(self.groups)  # parent iterates this for the CSV columns
        self._evaluator = None  # lazy: load judge only on first eval
        self.metrics_csv = self.out / "epoch_metrics.csv"
        if is_world_process_zero():
            (self.out / "eval_subset_qids.json").write_text(json.dumps(subset_ids, indent=2))
        counts = {key: len(group["rows"]) for key, group in self.groups.items()}
        log.info(
            "unified per-epoch eval armed: %s (seed=%d), judge=%s",
            counts, seed, judge_model,
        )

    def _load_resized(self, paths: list[str], budget: int) -> list:
        return [
            budget_resize(
                Image.open(p).convert("RGB"),
                factor=self.factor, min_pixels=self.min_pixels, max_pixels=budget,
            )
            for p in paths
        ]

    @torch.inference_mode()
    def _generate(self, model, device, messages: list[dict], images: list) -> tuple[str, float]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(device)
        t0 = time.time()
        generated = model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        latency = time.time() - t0
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip(), latency

    def _infer(self, model, key: str, group: dict):
        device = next(model.parameters()).device
        track, dataset, tc = group["track"], group["dataset"], group["cfg"]
        system = P.system_prompt(track)
        budget = int(tc["max_pixels"])
        records, responses = [], []
        n_err = 0
        t0 = time.time()
        for i, (req, row) in enumerate(zip(group["reqs"], group["rows"])):
            fmt = row["answer_format"]
            instr, opts = P.build_instruction(req.question, fmt)
            multi = adapter.is_multi_select(req.question)
            raw, lat, err = "", 0.0, ""
            messages: list[dict] = []
            paths: list[str] = []
            ts_labels: list[str] = []
            if track == "frame":
                img = frames.request_frame_paths(
                    req, frames_folder=self.frames_folder, dataset=dataset
                )[0]
                if img.exists():
                    messages = build_frame_messages(system, instr)
                    paths = [str(img)]
                else:
                    err = f"missing frame: {img}"
            else:
                refs = select_frames(
                    req, stride=stride_for_dataset(tc["stride"], dataset),
                    max_frames=int(tc["max_frames"]), frames_folder=self.frames_folder,
                    dataset=dataset, check_exists=False,
                )
                if refs:
                    preamble = build_preamble(track, req, refs)
                    messages, paths = build_messages(system, preamble, refs, instr)
                    ts_labels = [r.ts for r in refs]
                else:
                    err = "no frames available for window"
            if not err:
                try:
                    raw, lat = self._generate(
                        model, device, messages, self._load_resized(paths, budget)
                    )
                except Exception as e:  # noqa: BLE001 — one bad sample mustn't stop eval
                    err = f"{type(e).__name__}: {e}"
            resp = adapter.build_response(
                req.qID, raw, fmt, latency=lat, options=opts, multi=multi
            )
            responses.append(resp)
            n_err += int(bool(err))
            records.append({
                "track": track, "dataset": dataset, "sample_id": req.qID,
                "model_name": self.model_name, "question": req.question,
                "answer": str(row["answer"]), "answer_format": fmt,
                "primary_capability": str(row["primary_capability"]),
                "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                "clinical_relevance": bool(row["clinical_relevance"]),
                "ood": bool(row["ood"]),
                "video": row["video"], "timestamp_start": str(row["timestamp_start"]),
                "timestamp_end": str(row["timestamp_end"]),
                "image_path": paths[len(paths) // 2] if paths else "",
                "frame_paths": "|".join(paths),
                "frame_timestamps": "|".join(ts_labels),
                "n_frames": len(paths),
                "prompt": instr, "raw_model_output": raw,
                "normalized_prediction": resp.content, "prediction": resp.content,
                "latency_sec": lat, "error": err,
            })
            if i % 100 == 0:
                log.info(
                    "  %s infer %d/%d fmt=%s raw=%r",
                    key, i + 1, len(group["reqs"]), fmt, raw[:32],
                )
        log.info(
            "  %s inference done: %d rows %.1fs errors=%d",
            key, len(records), time.time() - t0, n_err,
        )
        return records, responses


def run_unified_training(
    *,
    config: dict,
    out_dir: str | Path,
    base_override: str | None = None,
    limit_per_track: int | None = None,
    epochs_override: int | None = None,
    per_device_batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    save_steps_override: int | None = None,
    resume=None,
    judge_model: str | None = None,
    eval_limit: int | None = None,
    frames_folder: str = "frames",
) -> None:
    """Config-driven unified SFT loop (mirrors ``videovqa_sft.run_training``)."""
    cfg = deepcopy(config)
    t = cfg["train"]
    overrides = {
        "per_device_batch_size": per_device_batch_size_override,
        "grad_accum": grad_accum_override,
        "save_steps": save_steps_override,
    }
    for key, value in overrides.items():
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} override must be positive, got {value}")
            t[key] = value

    tracks_cfg = cfg["tracks"]
    if eval_limit:
        for tc in tracks_cfg.values():
            tc["eval_n"] = eval_limit
    datasets = tuple(cfg.get("datasets", ["heico", "lapchole"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # Same per-rank ephemeral-root isolation as videovqa_sft.run_training.
        rank = distributed_rank()
        cache_roots = {
            "TMPDIR": "/tmp",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            "TRITON_CACHE_DIR": "/tmp/triton",
            "XDG_CACHE_HOME": "/tmp/xdg",
        }
        for env_name, default in cache_roots.items():
            rank_root = Path(os.environ.get(env_name, default)) / f"rank-{rank}"
            rank_root.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(rank_root)
        (Path(os.environ["XDG_CACHE_HOME"]) / "torch" / "kernels").mkdir(
            parents=True, exist_ok=True
        )
        tempfile.tempdir = os.environ["TMPDIR"]
        log.info("rank %d temp directory: %s", rank, os.environ["TMPDIR"])

    torch.multiprocessing.set_sharing_strategy("file_system")

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    base = base_override or resolve_base(cfg["base_path"])

    coverage_report: dict[str, dict] = {}
    for track, tc in tracks_cfg.items():
        for dataset in datasets:
            for split in ("train", "test"):
                cov = coverage(
                    track, split, dataset=dataset,
                    stride=stride_for_dataset(tc["stride"], dataset),
                    max_frames=int(tc["max_frames"]), frames_folder=frames_folder,
                )
                coverage_report[f"{track}/{dataset}/{split}"] = cov
                if cov["missing"]:
                    raise FileNotFoundError(
                        f"{track}/{dataset}/{split} has {cov['missing']} missing "
                        f"sampled frames; first: {cov['missing_examples'][:3]}"
                    )

    ds = UnifiedSFTDataset(
        tracks_cfg, "train", datasets=datasets, frames_folder=frames_folder,
        limit_per_track=limit_per_track,
    )
    log.info("unified train examples: %d across %s", len(ds), ds.counts)

    log.info("base=%s | torch %s | cuda=%s | dev=%s", base,
             torch.__version__, torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    pcfg = cfg.get("processor", {})
    proc_kwargs: dict = {}
    if pcfg.get("min_pixels"):
        proc_kwargs["min_pixels"] = pcfg["min_pixels"]
    if pcfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = pcfg["max_pixels"]
    processor = AutoProcessor.from_pretrained(base, **proc_kwargs)
    factor, min_px, global_max = processor_pixel_geometry(processor)
    log.info("processor geometry: factor=%d min_pixels=%d max_pixels=%s",
             factor, min_px, global_max)
    for track, tc in tracks_cfg.items():
        budget = int(tc["max_pixels"])
        if global_max and budget > global_max:
            raise ValueError(
                f"tracks.{track}.max_pixels={budget} exceeds processor.max_pixels="
                f"{global_max}; the processor budget must be the max over tracks"
            )
        if budget < min_px:
            raise ValueError(
                f"tracks.{track}.max_pixels={budget} is below the processor min_pixels="
                f"{min_px}; the processor would re-upscale and break the budget"
            )
        if budget % (factor * factor):
            log.warning("tracks.%s.max_pixels=%d is not %d-aligned", track, budget, factor * factor)

    model = AutoModelForImageTextToText.from_pretrained(
        base,
        dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False  # required with gradient checkpointing

    lcfg = cfg["lora"]
    mts = lcfg.get("modules_to_save")
    if not mts:  # auto-detect the multimodal projector to full-train
        mts = detect_projectors(model, lcfg.get("projector_candidates", PROJECTOR_CANDIDATES))
        log.info("auto-detected projector(s): %s", mts)
    exclude = list(lcfg.get("exclude_name_contains", [])) + list(mts or [])
    targets = lcfg.get("target_modules") or discover_targets(model, exclude)
    log.info("LoRA targets (%d): %s | full-train: %s", len(targets), targets, mts)
    lora = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=targets,
        modules_to_save=(mts or None),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # let grads reach LoRA through frozen embeddings

    collator = UnifiedCollator(processor)

    epochs = epochs_override if epochs_override is not None else t["epochs"]
    auto_find_batch_size = t.get("auto_find_batch_size", False)
    if world_size > 1 and auto_find_batch_size:
        auto_find_batch_size = False
        log.info("disabled auto_find_batch_size for distributed training")
    effective_batch_size = (
        int(t["per_device_batch_size"]) * int(t["grad_accum"]) * world_size
    )
    log.info(
        "batching: per_device=%d grad_accum=%d world_size=%d effective=%d",
        t["per_device_batch_size"], t["grad_accum"], world_size, effective_batch_size,
    )
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=(cfg.get("dtype", "bfloat16") == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "epoch"),
        save_steps=int(t.get("save_steps", 500)),
        save_total_limit=None,  # keep every checkpoint for per-track selection
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=t.get("dataloader_persistent_workers", True),
        dataloader_prefetch_factor=t.get("dataloader_prefetch_factor", 4),
        remove_unused_columns=False,  # keep dict examples intact for the collator
        report_to="none",
        seed=t.get("seed", 42),
        optim=t.get("optim", "adamw_torch_fused"),
        tf32=t.get("tf32", True),
        auto_find_batch_size=auto_find_batch_size,
        ddp_find_unused_parameters=t.get("ddp_find_unused_parameters", False),
        ddp_timeout=int(t.get("ddp_timeout", 7200)),
    )
    if is_world_process_zero():
        (out / "train_meta.json").write_text(
            json.dumps({"track": "unified", "base": base, "config": cfg,
                        "targets": targets, "datasets": datasets,
                        "track_counts": ds.counts, "coverage": coverage_report,
                        "epochs": epochs, "n_train": len(ds),
                        "world_size": world_size,
                        "effective_batch_size": effective_batch_size,
                        "processor_geometry": {
                            "factor": factor, "min_pixels": min_px,
                            "max_pixels": global_max,
                        }},
                       indent=2, default=str)
        )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.add_callback(TrainingIntervalSyncCallback())
    if judge_model:
        trainer.add_callback(UnifiedEpochEvalCallback(
            processor=processor, out_dir=str(out), judge_model=judge_model,
            tracks_cfg=tracks_cfg, datasets=datasets, frames_folder=frames_folder,
            max_new_tokens=t.get("max_new_tokens", 64), seed=t.get("seed", 42),
            model_name=cfg.get("model_name", "lora"),
        ))
        log.info("unified per-epoch eval enabled (judge=%s)", judge_model)
    else:
        log.warning("no --judge-model: per-epoch eval disabled")
    if resume:
        log.info("resuming from checkpoint: %s", resume)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(out / "final"))
        log.info("saved adapter -> %s", out / "final")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
