#!/usr/bin/env python
"""FRAME LoRA SFT trainer (config-driven).

Loads a base VLM via ``AutoModelForImageTextToText``, injects LoRA into the LLM +
vision-encoder linear layers (auto-discovered) and fully trains the multimodal
projector ("merger"), then runs a standard HF ``Trainer`` SFT loop and saves the
PEFT adapter per epoch. Any <=14B VLM is added by a new YAML in ``configs/``.

Run on a GPU node (see ../scripts/train.slurm). bf16, sdpa attention (no flash-attn
on aarch64), single GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # shared `src`
sys.path.insert(0, str(HERE))  # sibling modules (dataset)

from src.paths import OS_MODELS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("train")


def resolve_base(p: str) -> str:
    """Resolve a config ``base_path`` (repo-relative, os-models name, or absolute)."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    if (REPO / p).exists():
        return str(REPO / p)
    return str(OS_MODELS_DIR / path.name)


def discover_targets(model, exclude: list[str]) -> list[str]:
    """All ``nn.Linear`` leaf names whose module path contains none of ``exclude``.

    Covers the LLM (q/k/v/o_proj, gate/up/down_proj) and the vision tower (qkv,
    proj, mlp) in one set, while skipping the head / projector / embeddings.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME LoRA SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--base", default=None, help="override base path (e.g. a /dev/shm stage)")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume training: bare flag = latest checkpoint in --out; "
                         "or pass a specific checkpoint dir")
    ap.add_argument("--judge-model", default=None,
                    help="judge model dir; enables per-epoch test inference + scoring")
    ap.add_argument("--eval-each-epoch", dest="eval_each_epoch", action="store_true", default=True)
    ap.add_argument("--no-eval-each-epoch", dest="eval_each_epoch", action="store_false")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="limit test rows in per-epoch eval (smoke tests)")
    ap.add_argument("--frames-folder", default="frames")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    import dataset as D  # noqa: E402 — defer heavy imports until after arg parsing

    base = a.base or resolve_base(cfg["base_path"])
    log.info("base=%s | torch %s | cuda=%s | dev=%s", base, torch.__version__,
             torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    proc_kwargs: dict = {}
    if cfg.get("trust_remote_code"):
        proc_kwargs["trust_remote_code"] = True
    if cfg.get("min_pixels"):
        proc_kwargs["min_pixels"] = cfg["min_pixels"]
    if cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = cfg["max_pixels"]
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

    ds = D.FrameSFTDataset(a.track, a.split, limit=a.limit)
    collator = D.SFTCollator(processor)
    log.info("train examples: %d", len(ds))

    t = cfg["train"]
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
        save_total_limit=None,  # keep every epoch checkpoint for eval
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        remove_unused_columns=False,  # keep pixel_values etc. for the collator
        report_to="none",
        seed=t.get("seed", 42),
        optim="adamw_torch",
    )
    (out / "train_meta.json").write_text(
        json.dumps({"base": base, "config": cfg, "targets": targets, "epochs": epochs,
                    "n_train": len(ds)}, indent=2, default=str)
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    if a.eval_each_epoch and a.judge_model:
        import epoch_eval as EE
        trainer.add_callback(EE.PerEpochEvalCallback(
            processor=processor, out_dir=str(out), judge_model=a.judge_model,
            track=a.track, split="test", frames_folder=a.frames_folder,
            max_new_tokens=t.get("max_new_tokens", 64),
            eval_limit=a.eval_limit, model_name=cfg.get("model_name", "lora"),
        ))
        log.info("per-epoch eval enabled (judge=%s eval_limit=%s)", a.judge_model, a.eval_limit)
    elif a.eval_each_epoch and not a.judge_model:
        log.warning("--eval-each-epoch but no --judge-model: per-epoch eval disabled")
    if a.resume:
        log.info("resuming from checkpoint: %s", a.resume)
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(out / "final"))
    processor.save_pretrained(str(out / "final"))
    log.info("saved adapter -> %s", out / "final")


if __name__ == "__main__":
    main()
