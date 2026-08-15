#!/usr/bin/env python
"""FRAME v2 LoRA SFT trainer.

Deliberately smaller than `track-frame/lora-finetune/src/train.py` in one way:
it has no per-epoch judge-eval callback. Epoch selection now has to happen under
the leaderboard-shaped, dataset-as-distribution-axis protocol
(`track-frame/v2/src/score.py`), which the in-training callback cannot express —
so training just saves an adapter per epoch and `benchmark.slurm` scores the ones
we care about. That also keeps the training job free of judge-model memory.

Everything specific to v2 lives in `dataset.py`: per-row procedure-aware system
prompts, a chat template with thinking suppressed to match inference, and
optional count rebalancing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.paths import OS_MODELS_DIR  # noqa: E402

#: Under ``torchrun`` every rank runs this file. Rank 0 does the talking and the
#: metadata writing; the others stay quiet so the log stays readable and two
#: processes never write ``train_meta.json`` at once. Unset when run directly.
RANK = int(os.environ.get("RANK", "0"))
IS_MAIN = RANK == 0

logging.basicConfig(
    level=logging.INFO if IS_MAIN else logging.WARNING,
    format=f"%(asctime)s %(levelname)s [r{RANK}] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

#: Modules to keep out of LoRA entirely. ``mtp`` is Qwen3.6's multi-token
#: prediction head: it sits outside the normal generate() path, so adapting it
#: spends parameters and memory on weights that never affect an answer.
DEFAULT_EXCLUDE = ("lm_head", "embed_tokens", "mtp")
PROJECTOR_CANDIDATES = ("merger", "multi_modal_projector", "mlp1", "connector")


def resolve_base(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    if (REPO / p).exists():
        return str(REPO / p)
    return str(OS_MODELS_DIR / path.name)


def discover_targets(model, exclude: list[str]) -> list[str]:
    """Leaf names of every ``nn.Linear`` whose full path contains none of *exclude*."""
    import torch.nn as nn

    leaves: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in exclude):
            leaves.add(name.split(".")[-1])
    return sorted(leaves)


def detect_projectors(model, candidates) -> list[str]:
    names = [n for n, _ in model.named_modules()]
    return [c for c in candidates if any(c in n for n in names)]


def excluded_full_names(model, exclude: list[str], targets: list[str]) -> list[str]:
    """Full module paths that a bare leaf-name target would wrongly capture.

    PEFT matches ``target_modules`` by name suffix, so a leaf name shared between
    the language model and an excluded subtree (Qwen3.6 reuses ``down_proj`` in
    both ``model.language_model`` and ``mtp``) would adapt both. Naming the full
    paths in ``exclude_modules`` is what actually keeps them out.
    """
    import torch.nn as nn

    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and any(s in name for s in exclude)
        and name.split(".")[-1] in targets
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME v2 LoRA SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="override base path (e.g. a /dev/shm stage)")
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="train")
    ap.add_argument("--datasets", nargs="+", default=None, help="override config datasets")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    import dataset as D

    torch.multiprocessing.set_sharing_strategy("file_system")

    base = a.base or resolve_base(cfg["base_path"])
    datasets = tuple(a.datasets or cfg.get("datasets", ["heico", "lapchole"]))
    t = cfg["train"]

    ds = D.FrameSFTDatasetV2(
        a.track, a.split, datasets=datasets, frames_folder=a.frames_folder, limit=a.limit,
        prompt_strategy=cfg.get("prompt_strategy", "v2"),
        count_balance_power=float(t.get("count_balance_power", 0.0)),
        count_balance_cap=int(t.get("count_balance_cap", 4)),
    )
    missing = [ex["image_path"] for ex in ds.examples if not Path(ex["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} training frames are missing; first: {missing[:3]}")
    log.info("train examples: %d (from %d rows) across %s", len(ds), len(ds.examples), datasets)
    log.info("counting distribution after balancing: %s", D.count_distribution(ds))

    log.info("base=%s | torch %s | dev=%s", base, torch.__version__,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    proc_kwargs: dict = {}
    for key in ("min_pixels", "max_pixels"):
        if cfg.get(key):
            proc_kwargs[key] = cfg[key]
    if cfg.get("trust_remote_code"):
        proc_kwargs["trust_remote_code"] = True
    processor = AutoProcessor.from_pretrained(base, **proc_kwargs)

    model_kwargs: dict = dict(
        dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    if cfg.get("trust_remote_code"):
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForImageTextToText.from_pretrained(base, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    lcfg = cfg["lora"]
    mts = lcfg.get("modules_to_save") or detect_projectors(
        model, lcfg.get("projector_candidates", PROJECTOR_CANDIDATES)
    )
    exclude = list(lcfg.get("exclude_name_contains", DEFAULT_EXCLUDE)) + list(mts or [])
    targets = lcfg.get("target_modules") or discover_targets(model, exclude)
    exclude_full = excluded_full_names(model, exclude, targets)
    log.info("LoRA targets (%d): %s | full-train: %s | excluded paths: %d",
             len(targets), targets, mts, len(exclude_full))

    lora_kwargs: dict = dict(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=targets, modules_to_save=(mts or None),
        bias="none", task_type="CAUSAL_LM",
    )
    if exclude_full:
        try:
            model_peft = get_peft_model(model, LoraConfig(exclude_modules=exclude_full, **lora_kwargs))
        except TypeError:
            log.warning("peft has no exclude_modules; %d excluded paths will be adapted",
                        len(exclude_full))
            model_peft = get_peft_model(model, LoraConfig(**lora_kwargs))
    else:
        model_peft = get_peft_model(model, LoraConfig(**lora_kwargs))
    model = model_peft
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # let grads reach LoRA through frozen embeddings

    collator = D.SFTCollatorV2(processor, enable_thinking=bool(cfg.get("enable_thinking", False)))
    epochs = a.epochs if a.epochs is not None else t["epochs"]
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
        save_total_limit=None,  # every epoch is a selection candidate
        dataloader_num_workers=t.get("dataloader_num_workers", 8),
        dataloader_pin_memory=t.get("dataloader_pin_memory", False),
        dataloader_persistent_workers=t.get("dataloader_persistent_workers", True),
        dataloader_prefetch_factor=t.get("dataloader_prefetch_factor", 2),
        remove_unused_columns=False,  # keep pixel_values etc. for the collator
        report_to="none",
        seed=t.get("seed", 42),
        optim=t.get("optim", "adamw_torch_fused"),
        tf32=t.get("tf32", True),
        # Only the LoRA adapters and visual.merger carry grads, and every batch
        # exercises all of them, so DDP does not need the unused-parameter scan.
        ddp_find_unused_parameters=t.get("ddp_find_unused_parameters", False),
    )
    if IS_MAIN:
        (out / "train_meta.json").write_text(
            json.dumps(
                {"base": base, "config": cfg, "targets": targets,
                 "excluded_paths": len(exclude_full), "epochs": epochs, "datasets": datasets,
                 "n_examples": len(ds), "n_rows": len(ds.examples),
                 "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                 "effective_batch": (t["per_device_batch_size"] * t["grad_accum"]
                                     * int(os.environ.get("WORLD_SIZE", "1"))),
                 "counts": D.count_distribution(ds)},
                indent=2, default=str,
            )
        )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(out / "final"))
    processor.save_pretrained(str(out / "final"))
    log.info("saved adapter -> %s", out / "final")


if __name__ == "__main__":
    main()
