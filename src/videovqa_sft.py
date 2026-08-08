"""Multi-frame LoRA SFT — shared by the SEGMENT and PROCEDURE tracks.

Ports the FRAME-track recipe (``track-frame/lora-finetune/src/{train,dataset,
epoch_eval}.py``) to window tracks. Every prompt is built with the SAME shared
functions inference uses (``src.videovqa.select_frames`` / ``build_preamble`` /
``build_messages`` + ``src.prompts.build_instruction``), so train and inference
prompts agree by construction; sampling parameters (stride / max_frames /
max_pixels) come from the YAML config and are recorded in ``train_meta.json``.

Loss is masked to the assistant answer only. Unlike the frame collator (which
re-processes the image to measure the prompt length), the multi-image prompt is
expensive to re-process, so the collator measures it from the templated text:
the templated full conversation starts with the templated prompt, and the
remaining suffix is pure text (no image tokens), so
``prompt_len = unpadded_len - len(tokenise(suffix))``. This is verified against
the re-process path on the first sample and falls back to it permanently if a
template violates the prefix property.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainerCallback

from . import adapter, data
from . import prompts as P
from .paths import OS_MODELS_DIR, REPO_ROOT
from .videovqa import _sec_caps, build_messages, build_preamble, coverage, select_frames

log = logging.getLogger("videovqa-sft")


def distributed_rank() -> int:
    """Return the global rank before or after torch.distributed initialization."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_world_process_zero() -> bool:
    return distributed_rank() == 0


def stride_for_dataset(stride: int | dict[str, int], dataset: str) -> int:
    """Resolve a scalar or per-dataset source-frame stride."""
    value = stride[dataset] if isinstance(stride, dict) else stride
    value = int(value)
    if value < 1:
        raise ValueError(f"stride for {dataset!r} must be positive, got {value}")
    return value


class VideoSFTDataset(Dataset):
    """One (messages, image_paths, answer) conversation per train row.

    Frame selection + prompt construction happen lazily in ``__getitem__`` (the
    dataloader workers absorb the per-frame stat/IO), via the shared inference
    functions. Supervision target is the gold ``answer`` verbatim, including
    multi-label ``fo_class`` strings accepted by the official set parser.
    """

    def __init__(
        self,
        track: str,
        split: str = "train",
        *,
        datasets: tuple[str, ...] | list[str] = ("heico",),
        stride: int | dict[str, int],
        max_frames: int,
        frames_folder: str = "frames",
        limit: int | None = None,
    ) -> None:
        self.track = track
        self.datasets = tuple(datasets)
        self.stride = stride
        self.max_frames = max_frames
        self.frames_folder = frames_folder
        rows = data.read_parquets(track, split, self.datasets).to_dict("records")
        if limit:
            rows = rows[:limit]
        self.rows = rows
        self.reqs = [data.row_to_request(r) for r in rows]
        self.system = P.system_prompt(track)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row, req = self.rows[i], self.reqs[i]
        dataset = row["_dataset"]
        refs = select_frames(
            req, stride=stride_for_dataset(self.stride, dataset),
            max_frames=self.max_frames, frames_folder=self.frames_folder,
            dataset=dataset, check_exists=False,
        )
        instr, _ = P.build_instruction(req.question, row["answer_format"])
        preamble = build_preamble(self.track, req, refs)
        messages, paths = build_messages(self.system, preamble, refs, instr)
        return {
            "dataset": dataset,
            "messages": messages,
            "image_paths": paths,
            "answer": str(row["answer"]),
        }


class VideoSFTCollator:
    """Chat-format full conversations; mask loss to the assistant answer only."""

    def __init__(self, processor) -> None:
        self.processor = processor
        tok = processor.tokenizer
        tok.padding_side = "right"  # masked-out pads go at the tail
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._prompt_len_mode = "unverified"  # -> "fast" | "slow" after first sample

    def _slow_prompt_len(self, prompt_text: str, images: list) -> int:
        """Frame-track path: re-process the prompt (incl. images) and count tokens."""
        return self.processor(
            text=[prompt_text], images=images or None, return_tensors="pt"
        )["input_ids"].shape[1]

    def load_images(self, ex: dict) -> list:
        """Open one example's images; subclasses may apply per-item resizing."""
        return [Image.open(p).convert("RGB") for p in ex["image_paths"]]

    def __call__(self, examples: list[dict]) -> dict:
        per_images = [self.load_images(ex) for ex in examples]
        flat = [im for ims in per_images for im in ims]
        fulls, prompts = [], []
        for ex in examples:
            full_msgs = ex["messages"] + [
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}
            ]
            fulls.append(self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            ))
            prompts.append(self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            ))

        batch = self.processor(
            text=fulls, images=flat or None, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        tok = self.processor.tokenizer
        for i in range(len(examples)):
            seq_len = int(batch["attention_mask"][i].sum())
            fast_ok = fulls[i].startswith(prompts[i])
            if fast_ok:
                suffix_ids = tok(fulls[i][len(prompts[i]):], add_special_tokens=False)["input_ids"]
                prompt_len = seq_len - len(suffix_ids)
            if self._prompt_len_mode == "unverified":  # one-time cross-check
                slow = self._slow_prompt_len(prompts[i], per_images[i])
                if fast_ok and prompt_len == slow:
                    self._prompt_len_mode = "fast"
                else:
                    self._prompt_len_mode = "slow"
                    log.warning(
                        "fast prompt-length path disagrees with re-processing "
                        "(fast=%s slow=%d) — using the slow path from now on",
                        prompt_len if fast_ok else "n/a", slow,
                    )
                    prompt_len = slow
            elif self._prompt_len_mode == "slow" or not fast_ok:
                prompt_len = self._slow_prompt_len(prompts[i], per_images[i])
            labels[i, :prompt_len] = -100  # mask system + preamble + frames + question
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


@torch.inference_mode()
def generate_video_answer(
    model, processor, device, messages: list[dict], image_paths: list[str],
    max_new_tokens: int = 64,
) -> tuple[str, float]:
    """Greedy-decode one interleaved multi-image prompt -> (text, seconds)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(device)

    t0 = time.time()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - t0

    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return out, latency


class TrainingIntervalSyncCallback(TrainerCallback):
    """Apply the current logging/eval/save intervals after checkpoint restore.

    Transformers restores these derived values from ``trainer_state.json`` and
    otherwise retains the old checkpoint's intervals when the active config
    changes them.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        before = (state.logging_steps, state.eval_steps, state.save_steps)
        state.compute_steps(args, state.max_steps)
        after = (state.logging_steps, state.eval_steps, state.save_steps)
        if after != before:
            log.info(
                "synced checkpoint intervals to active config: "
                "logging/eval/save %s -> %s",
                before,
                after,
            )
        return control


class VideoEpochEvalCallback(TrainerCallback):
    """Per-epoch inference + judge scoring on a SEEDED test subset.

    Full-test per epoch is too heavy for window tracks (~1.5-2 h), so each epoch
    scores the same ``eval_n`` seeded-random test questions -> comparable curve
    across epochs; full test runs only for the chosen best epoch. Structure and
    state handling mirror the frame-track ``PerEpochEvalCallback``.
    """

    def __init__(
        self,
        *,
        processor,
        out_dir: str,
        judge_model: str,
        track: str,
        datasets: tuple[str, ...] | list[str],
        stride: int | dict[str, int],
        max_frames: int,
        split: str = "test",
        frames_folder: str = "frames",
        max_new_tokens: int = 64,
        eval_n: int | None = 500,
        seed: int = 42,
        model_name: str = "lora",
        rank_wait_timeout_seconds: float = 8 * 60 * 60,
    ) -> None:
        self.processor = processor
        self.out = Path(out_dir)
        self.judge_model = judge_model
        self.track = track
        self.datasets = tuple(datasets)
        self.stride = stride
        self.max_frames = max_frames
        self.frames_folder = frames_folder
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.rank_wait_timeout_seconds = rank_wait_timeout_seconds
        self.system = P.system_prompt(track)

        candidates: list[tuple[str, dict]] = []
        for dataset in self.datasets:
            candidates.extend(
                (dataset, row)
                for row in data.read_parquet(track, split, dataset).to_dict("records")
            )
        if eval_n and eval_n < len(candidates):
            rng = np.random.default_rng(seed)
            keep = sorted(rng.choice(len(candidates), size=eval_n, replace=False).tolist())
            candidates = [candidates[k] for k in keep]

        self.groups: dict[str, dict] = {}
        subset_ids = []
        for dataset in self.datasets:
            rows = [row for row_dataset, row in candidates if row_dataset == dataset]
            reqs = [data.row_to_request(row) for row in rows]
            self.groups[dataset] = {
                "rows": rows,
                "reqs": reqs,
                "refs": [data.row_to_reference(row) for row in rows],
            }
            subset_ids.extend({"dataset": dataset, "qID": req.qID} for req in reqs)
        self._evaluator = None  # lazy: load judge only on first eval
        self.metrics_csv = self.out / "epoch_metrics.csv"
        if is_world_process_zero():
            (self.out / "eval_subset_qids.json").write_text(
                json.dumps(subset_ids, indent=2)
            )
        counts = {dataset: len(group["rows"]) for dataset, group in self.groups.items()}
        log.info(
            "per-epoch eval armed: %s %s/%s rows (seed=%d), judge=%s",
            counts, track, split, seed, judge_model,
        )

    def _evaluator_lazy(self):
        if self._evaluator is None:
            from focus import Evaluator
            self._evaluator = Evaluator(
                judge_kwargs={"model_name": self.judge_model, "device": "cuda"}
            )
        return self._evaluator

    def _eval_sync_paths(self, epoch: int) -> tuple[Path, Path]:
        run_id = (
            os.environ.get("SLURM_JOB_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or f"parent_{os.getppid()}"
        )
        run_id = run_id.replace("/", "_")
        sync_dir = self.out / ".eval_sync" / run_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        return (
            sync_dir / f"epoch_{epoch}.started",
            sync_dir / f"epoch_{epoch}.done",
        )

    @staticmethod
    def _write_sync_token(path: Path, token: str) -> None:
        """Atomically publish a small cross-rank filesystem signal."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(token)
        tmp.replace(path)

    def _start_rank0_eval(self, epoch: int) -> str:
        started_path, done_path = self._eval_sync_paths(epoch)
        done_path.unlink(missing_ok=True)
        token = f"{os.getpid()}-{time.time_ns()}"
        self._write_sync_token(started_path, token)
        return token

    def _finish_rank0_eval(self, epoch: int, token: str) -> None:
        _, done_path = self._eval_sync_paths(epoch)
        self._write_sync_token(done_path, token)

    def _wait_for_rank0_eval(self, epoch: int) -> None:
        """Wait without entering an NCCL collective that can hit its watchdog."""
        started_path, done_path = self._eval_sync_paths(epoch)
        entered = time.time()
        # Rank 0 can publish just before this rank enters the callback. The
        # tolerance accepts that race but rejects markers left by an old job.
        fresh_after = entered - 300
        token = None
        deadline = entered + self.rank_wait_timeout_seconds
        log.info(
            "[epoch %d] rank %d waiting for rank-0 evaluation signal",
            epoch,
            torch.distributed.get_rank(),
        )
        while time.time() < deadline:
            if token is None and started_path.exists():
                try:
                    if started_path.stat().st_mtime >= fresh_after:
                        candidate = started_path.read_text().strip()
                        if candidate:
                            token = candidate
                except OSError:
                    pass
            if token is not None and done_path.exists():
                try:
                    if done_path.read_text().strip() == token:
                        log.info(
                            "[epoch %d] rank-0 evaluation complete; rank %d continuing",
                            epoch,
                            torch.distributed.get_rank(),
                        )
                        return
                except OSError:
                    pass
            time.sleep(5)
        raise TimeoutError(
            f"rank 0 did not finish epoch {epoch} evaluation within "
            f"{self.rank_wait_timeout_seconds:.0f}s"
        )

    def _infer(self, model, dataset: str, group: dict):
        device = next(model.parameters()).device
        records, responses = [], []
        n_err = 0
        t0 = time.time()
        stride = stride_for_dataset(self.stride, dataset)
        for i, (req, row) in enumerate(zip(group["reqs"], group["rows"])):
            fmt = row["answer_format"]
            instr, opts = P.build_instruction(req.question, fmt)
            multi = adapter.is_multi_select(req.question)
            refs = select_frames(
                req, stride=stride, max_frames=self.max_frames,
                frames_folder=self.frames_folder, dataset=dataset, check_exists=False,
            )
            raw, lat, err = "", 0.0, ""
            if not refs:
                err = "no frames available for window"
            else:
                preamble = build_preamble(self.track, req, refs)
                messages, paths = build_messages(self.system, preamble, refs, instr)
                try:
                    raw, lat = generate_video_answer(
                        model, self.processor, device, messages, paths, self.max_new_tokens
                    )
                except Exception as e:  # noqa: BLE001 — one bad sample mustn't stop eval
                    err = f"{type(e).__name__}: {e}"
            resp = adapter.build_response(req.qID, raw, fmt, latency=lat,
                                          options=opts, multi=multi)
            responses.append(resp)
            n_err += int(bool(err))
            records.append({
                "dataset": dataset, "sample_id": req.qID,
                "model_name": self.model_name, "question": req.question,
                "answer": str(row["answer"]), "answer_format": fmt,
                "primary_capability": str(row["primary_capability"]),
                "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                "clinical_relevance": bool(row["clinical_relevance"]), "ood": bool(row["ood"]),
                "video": row["video"], "timestamp_start": str(row["timestamp_start"]),
                "timestamp_end": str(row["timestamp_end"]),
                "image_path": str(refs[len(refs) // 2].path) if refs else "",
                "frame_paths": "|".join(str(r.path) for r in refs),
                "frame_timestamps": "|".join(r.ts for r in refs),
                "n_frames": len(refs),
                "prompt": instr, "raw_model_output": raw, "normalized_prediction": resp.content,
                "prediction": resp.content, "latency_sec": lat, "error": err,
            })
            if i % 100 == 0:
                log.info(
                    "  %s infer %d/%d fmt=%s raw=%r",
                    dataset, i + 1, len(group["reqs"]), fmt, raw[:32],
                )
        log.info(
            "  %s inference done: %d rows %.1fs errors=%d",
            dataset, len(records), time.time() - t0, n_err,
        )
        return records, responses

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        epoch = int(round(state.epoch))
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        # Only rank 0 performs generation/judging and writes artifacts. Other
        # ranks poll a filesystem signal. A long-lived NCCL barrier is unsafe
        # here because slow generation can exceed the process-group watchdog.
        if distributed and torch.distributed.get_rank() != 0:
            self._wait_for_rank0_eval(epoch)
            return
        sync_token = self._start_rank0_eval(epoch) if distributed else None
        try:
            was_training = model.training
            prev_cache = getattr(model.config, "use_cache", None)
            gc_was = getattr(model, "is_gradient_checkpointing", False)
            model.eval()
            try:
                model.gradient_checkpointing_disable()  # KV cache needs this off
            except Exception:  # noqa: BLE001
                pass
            try:
                model.config.use_cache = True
            except Exception:  # noqa: BLE001
                pass

            eval_outputs = {}
            total_rows = sum(len(group["rows"]) for group in self.groups.values())
            log.info("[epoch %d] test-subset inference (n=%d)...", epoch, total_rows)
            for dataset, group in self.groups.items():
                eval_outputs[dataset] = self._infer(model, dataset, group)

            # restore training state
            try:
                model.config.use_cache = prev_cache
            except Exception:  # noqa: BLE001
                pass
            if gc_was:
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except Exception:  # noqa: BLE001
                    pass
            if was_training:
                model.train()

            from focus import save_items

            epdir = self.out / f"eval_epoch_{epoch}"
            epdir.mkdir(parents=True, exist_ok=True)
            metrics = {}
            for dataset, group in self.groups.items():
                records, responses = eval_outputs[dataset]
                dataset_dir = epdir / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)
                save_items(responses, dataset_dir / "responses.json")
                pd.DataFrame(records).to_parquet(
                    dataset_dir / "predictions.parquet", index=False
                )

                accuracy = float("nan")
                try:
                    ev = self._evaluator_lazy()
                    _, sum_df = ev.run(
                        group["reqs"], group["refs"], responses,
                        output_dir=str(dataset_dir / "eval"),
                    )
                    ov = sum_df[sum_df.level == "overall"]
                    if not ov.empty:
                        accuracy = float(ov.iloc[0]["accuracy"])
                    log.info(
                        "[epoch %d] %s acc = %.4f -> %s",
                        epoch, dataset, accuracy, dataset_dir,
                    )
                except Exception as e:  # noqa: BLE001 — retain responses for offline scoring
                    log.warning(
                        "[epoch %d] %s judge eval failed: %s (responses saved)",
                        epoch, dataset, e,
                    )
                metrics[dataset] = accuracy

            valid = [value for value in metrics.values() if np.isfinite(value)]
            overall = float(np.mean(valid)) if valid else float("nan")
            log.info("[epoch %d] mean dataset accuracy = %.4f", epoch, overall)

            header = not self.metrics_csv.exists()
            with open(self.metrics_csv, "a") as f:
                if header:
                    f.write(
                        "epoch,overall_acc,"
                        + ",".join(f"{dataset}_acc" for dataset in self.datasets)
                        + ",global_step\n"
                    )
                values = ",".join(str(metrics.get(dataset, float("nan")))
                                  for dataset in self.datasets)
                f.write(f"{epoch},{overall},{values},{state.global_step}\n")
        except Exception as e:  # noqa: BLE001 — never let eval kill training
            log.warning("[epoch %d] per-epoch eval errored (training continues): %s", epoch, e)
            try:  # best effort to leave the model trainable
                model.train()
                model.config.use_cache = False
            except Exception:  # noqa: BLE001
                pass
        finally:
            if sync_token is not None:
                self._finish_rank0_eval(epoch, sync_token)


def resolve_base(p: str) -> str:
    """Resolve a config ``base_path`` (repo-relative, os-models name, or absolute)."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    if (REPO_ROOT / p).exists():
        return str(REPO_ROOT / p)
    return str(OS_MODELS_DIR / path.name)


def discover_targets(model, exclude: list[str]) -> list[str]:
    """All ``nn.Linear`` leaf names whose module path contains none of ``exclude``."""
    import torch.nn as nn

    leaves: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in exclude):
            leaves.add(name.split(".")[-1])
    return sorted(leaves)


PROJECTOR_CANDIDATES = ("merger", "multi_modal_projector", "mlp1", "connector")


def detect_projectors(model, candidates) -> list[str]:
    """Projector module name(s) present in the model (to full-train, not LoRA)."""
    names = [n for n, _ in model.named_modules()]
    return [c for c in candidates if any(c in n for n in names)]


def run_training(
    *,
    config: dict,
    track: str,
    out_dir: str | Path,
    base_override: str | None = None,
    limit: int | None = None,
    epochs_override: int | None = None,
    per_device_batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    save_steps_override: int | None = None,
    resume=None,
    judge_model: str | None = None,
    eval_limit: int | None = None,
    frames_folder: str = "frames",
) -> None:
    """Config-driven multi-frame LoRA SFT loop (port of the frame ``train.py``)."""
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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # Independent ranks must not race over torch_shm_manager sockets or
        # compiler caches. Isolate all ephemeral roots before workers/kernels
        # are created, and reset Python's cached temp root.
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

    # Large prefetched vision batches exposed a resource-sharer socket race in
    # PyTorch's default ``file_descriptor`` strategy on the GPU nodes.  Named
    # shared-memory files remain valid while the receiving process unpickles a
    # batch and do not depend on a worker-owned Unix socket.
    torch.multiprocessing.set_sharing_strategy("file_system")
    log.info(
        "torch multiprocessing sharing strategy: %s",
        torch.multiprocessing.get_sharing_strategy(),
    )

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    base = base_override or resolve_base(cfg["base_path"])
    datasets = tuple(cfg.get("datasets", ["heico"]))
    smp = cfg["sampling"]  # source-frame sampling; stride can differ by dataset FPS
    coverage_by_dataset = {}
    for dataset in datasets:
        coverage_by_dataset[dataset] = {}
        for split in ("train", "test"):
            cov = coverage(
                track, split, dataset=dataset,
                stride=stride_for_dataset(smp["stride"], dataset),
                max_frames=smp["max_frames"], frames_folder=frames_folder,
            )
            coverage_by_dataset[dataset][split] = cov
            if cov["missing"]:
                raise FileNotFoundError(
                    f"{dataset}/{track}/{split} has {cov['missing']} missing sampled "
                    f"frames; first: {cov['missing_examples'][:3]}"
                )

    ds = VideoSFTDataset(
        track, "train", datasets=datasets, stride=smp["stride"],
        max_frames=smp["max_frames"], frames_folder=frames_folder, limit=limit,
    )
    dataset_counts = {
        dataset: sum(row["_dataset"] == dataset for row in ds.rows)
        for dataset in datasets
    }
    log.info(
        "train examples: %d across %s (stride=%s max_frames=%d max_pixels=%s)",
        len(ds), dataset_counts, smp["stride"], smp["max_frames"], smp.get("max_pixels"),
    )

    log.info("track=%s base=%s | torch %s | cuda=%s | dev=%s", track, base,
             torch.__version__, torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    proc_kwargs: dict = {}
    if smp.get("min_pixels"):
        proc_kwargs["min_pixels"] = smp["min_pixels"]
    if smp.get("max_pixels"):
        proc_kwargs["max_pixels"] = smp["max_pixels"]
    processor = AutoProcessor.from_pretrained(base, **proc_kwargs)

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

    collator = VideoSFTCollator(processor)

    epochs = epochs_override if epochs_override is not None else t["epochs"]
    auto_find_batch_size = t.get("auto_find_batch_size", False)
    if world_size > 1 and auto_find_batch_size:
        # Retrying with different local batch sizes independently on each rank
        # can desynchronize DDP. Distributed launchers select a known-good
        # per-device batch explicitly.
        auto_find_batch_size = False
        log.info("disabled auto_find_batch_size for distributed training")
    effective_batch_size = (
        int(t["per_device_batch_size"]) * int(t["grad_accum"]) * world_size
    )
    log.info(
        "batching: per_device=%d grad_accum=%d world_size=%d effective=%d",
        t["per_device_batch_size"], t["grad_accum"], world_size,
        effective_batch_size,
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
        save_total_limit=None,  # keep every epoch checkpoint for eval
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
            json.dumps({"track": track, "base": base, "config": cfg, "targets": targets,
                        "datasets": datasets, "dataset_counts": dataset_counts,
                        "coverage": coverage_by_dataset, "epochs": epochs,
                        "n_train": len(ds), "world_size": world_size,
                        "effective_batch_size": effective_batch_size},
                       indent=2, default=str)
        )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.add_callback(TrainingIntervalSyncCallback())
    if judge_model:
        trainer.add_callback(VideoEpochEvalCallback(
            processor=processor, out_dir=str(out), judge_model=judge_model,
            track=track, datasets=datasets, stride=smp["stride"],
            max_frames=smp["max_frames"],
            frames_folder=frames_folder, max_new_tokens=t.get("max_new_tokens", 64),
            eval_n=(eval_limit or t.get("eval_n", 500)), seed=t.get("seed", 42),
            model_name=cfg.get("model_name", "lora"),
        ))
        log.info("per-epoch subset eval enabled (judge=%s eval_n=%s)",
                 judge_model, eval_limit or t.get("eval_n", 500))
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
