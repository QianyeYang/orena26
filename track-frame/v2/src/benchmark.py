#!/usr/bin/env python
"""FRAME v2 benchmark runner: any VLM, both datasets, v2 prompts, one output tree.

Differs from ``track-frame/zeroshot-sweep`` in the three things Stage 1 needs:

- **both official datasets in one pass**, each row tagged with ``_dataset`` so
  the scorer can treat one dataset as the out-of-distribution half;
- **per-row system prompts** — the v2 strategy names the row's own
  ``procedure_type``, so the system prompt cannot be hoisted out of the loop the
  way a fixed one can;
- **optional LoRA adapter**, so a fine-tuned checkpoint and a bare base model
  run through byte-identical plumbing.

Prompt strategy is a flag rather than a constant: a fine-tuned adapter must be
evaluated with the prompts it was *trained* on, so comparing a v1-trained
adapter against a v2 zero-shot base means running each at its own setting.

Writes under ``--out``: ``predictions.parquet`` (raw + normalised + metadata),
``responses.json`` / ``requests.json`` / ``references.json`` (focus-compatible,
so ``score.py`` and the stock evaluator both read them), and ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from focus import save_items  # noqa: E402
from src import adapter, data, frames  # noqa: E402
from src import prompts as P  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("bench")


def _sec_caps(value) -> str:
    if value is None:
        return ""
    try:
        return "|".join(str(v) for v in list(value))
    except TypeError:
        return str(value)


def namespaced(item, dataset: str):
    """Return *item* with its ``qID`` prefixed by *dataset*.

    Question ids are unique within a dataset but not across them — HeiCo and
    LapChole share id 2657592 — and ``focus.Evaluator`` refuses duplicate qIDs.
    Nothing outside a single run consumes these ids, so prefixing is safe as long
    as requests, references and responses are all built through here.

    ``Request``/``Reference`` are plain mutable dataclasses, so this rewrites in
    place and returns the same object.
    """
    item.qID = f"{dataset}:{item.qID}"
    return item


def build_prompts(row: dict, req, strategy: str) -> tuple[str, str, list[str] | None]:
    """Return ``(system, instruction, mc_options)`` for one row under *strategy*."""
    fmt = row["answer_format"]
    if strategy in P.V2_FAMILY:
        system = P.system_prompt("frame", strategy, procedure_type=req.procedure_type)
    else:
        system = P.system_prompt("frame", strategy)
    instruction, options = P.build_instruction(req.question, fmt, strategy)
    return system, instruction, options


def main() -> None:
    ap = argparse.ArgumentParser(description="FRAME v2 benchmark runner")
    ap.add_argument("--model", required=True, help="local path to the base model dir")
    ap.add_argument("--model-name", required=True, help="label stored in rows / meta")
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
    ap.add_argument("--prompt-strategy", default="v2", choices=list(P.PROMPT_STRATEGIES))
    ap.add_argument("--track", default="frame")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"])
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N rows per dataset")
    ap.add_argument("--seed", type=int, default=42, help="seed for --limit sampling")
    ap.add_argument("--stratified", action="store_true",
                    help="with --limit, sample per (capability x answer_format) instead of head")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true",
                    help="let a hybrid-thinking model reason before answering")
    ap.add_argument("--thinking-budget", type=int, default=256,
                    help="extra tokens granted to the reasoning block when --thinking")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--quant", default=None,
                    choices=[None, "int8-mlp", "fp8", "fp8-keepvision"],
                    help="quantize weights at load (deployment feasibility check)")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="cap CUDA memory to this fraction of the device, e.g. "
                         "44/141=0.312 to make an H200 fail exactly where a 48 GB "
                         "L40S would; proves the fit rather than estimating it")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = data.read_parquets(a.track, a.split, a.datasets)
    if a.limit:
        if a.stratified:
            # Sample within each stratum by position: pandas 3 dropped
            # `include_groups`, and GroupBy.apply no longer round-trips the
            # grouping columns, so the index is what carries them back.
            strata = ["_dataset", "primary_capability", "answer_format"]
            frac = min(1.0, (a.limit * len(a.datasets)) / len(df))
            picks = (
                df.groupby(strata, group_keys=False)
                .indices.values()
            )
            rng = np.random.default_rng(a.seed)
            keep = np.concatenate(
                [rng.choice(idx, size=max(1, round(len(idx) * frac)), replace=False)
                 for idx in picks]
            )
            df = df.iloc[np.sort(keep)].reset_index(drop=True)
        else:
            df = df.groupby("_dataset", group_keys=False).head(a.limit).reset_index(drop=True)
    rows = df.to_dict("records")
    reqs = [namespaced(data.row_to_request(r), r["_dataset"]) for r in rows]
    refs = [namespaced(data.row_to_reference(r), r["_dataset"]) for r in rows]
    save_items(reqs, out / "requests.json")
    save_items(refs, out / "references.json")
    log.info("rows=%d datasets=%s", len(rows), df["_dataset"].value_counts().to_dict())

    log.info(
        "torch %s | cuda=%s | device=%s | model=%s | adapter=%s | prompts=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        a.model_name,
        a.adapter,
        a.prompt_strategy,
    )

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "track-frame" / "lora-finetune" / "src"))
    from generate import generate  # noqa: E402
    from infer import LoRAVLM  # noqa: E402

    if a.mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        log.info("memory capped at %.1f%% of %.1f GiB = %.1f GiB (L40S proxy)",
                 100 * a.mem_fraction, total, a.mem_fraction * total)

    t_load = time.time()
    model = LoRAVLM(
        a.model,
        adapter_path=a.adapter,
        dtype=a.dtype,
        max_new_tokens=a.max_new_tokens,
        min_pixels=a.min_pixels,
        max_pixels=a.max_pixels,
        attn_implementation=a.attn,
        trust_remote_code=a.trust_remote_code,
        quantization=a.quant,
    )
    load_sec = time.time() - t_load
    weights_gb = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    log.info("model loaded in %.1fs, weights occupy %.1f GiB", load_sec, weights_gb)

    records: list[dict] = []
    responses = []
    n_parse = n_err = 0
    t0 = time.time()

    for i, (req, row) in enumerate(zip(reqs, rows)):
        fmt = row["answer_format"]
        system, instr, opts = build_prompts(row, req, a.prompt_strategy)
        img = frames.request_frame_paths(
            req, frames_folder=a.frames_folder, dataset=row["_dataset"]
        )[0]

        answer, raw, lat, err = "", "", 0.0, ""
        if not img.exists():
            err = f"missing frame: {img}"
            log.warning("qID=%s %s", req.qID, err)
        else:
            try:
                answer, raw, lat = generate(
                    model.model, model.processor, model.device,
                    str(img), instr, system=system,
                    max_new_tokens=a.max_new_tokens,
                    enable_thinking=a.thinking,
                    thinking_budget=a.thinking_budget,
                )
            except Exception as e:  # one bad sample must not kill the sweep
                err = f"{type(e).__name__}: {e}"
                log.warning("qID=%s inference error: %s", req.qID, err)

        resp = adapter.build_response(req.qID, answer, fmt, latency=lat, options=opts)
        responses.append(resp)
        n_err += int(bool(err))
        n_parse += int(adapter.is_parseable(fmt, resp.content))

        records.append(
            {
                "sample_id": req.qID,
                "model_name": a.model_name,
                "dataset": row["_dataset"],
                "question": req.question,
                "answer": str(row["answer"]),
                "answer_format": fmt,
                "primary_capability": str(row["primary_capability"]),
                "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                "procedure_type": row["procedure_type"],
                "clinical_relevance": bool(row["clinical_relevance"]),
                "ood": bool(row["ood"]),
                "video": row["video"],
                "timestamp_start": str(row["timestamp_start"]),
                "image_path": str(img),
                "system_prompt": system,
                "prompt": instr,
                "raw_model_output": raw,        # includes any reasoning block
                "answer_text": answer,          # reasoning stripped
                "normalized_prediction": resp.content,
                "prediction": resp.content,
                "latency_sec": lat,
                "error": err,
            }
        )

        if i % 200 == 0:
            log.info(
                "%d/%d fmt=%s answer=%r -> %r (%.2fs)%s",
                i + 1, len(reqs), fmt, answer[:40], resp.content[:40], lat,
                f" ERR={err[:40]}" if err else "",
            )

    pd.DataFrame(records).to_parquet(out / "predictions.parquet", index=False)
    save_items(responses, out / "responses.json")
    dt = time.time() - t0
    (out / "meta.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(a).items()},
                "n_rows": len(records),
                "load_sec": round(load_sec, 2),
                "infer_sec": round(dt, 2),
                "sec_per_question": round(dt / max(len(reqs), 1), 4),
                "parseable": n_parse,
                "errors": n_err,
                # Deployment budget: 1x L40S 48 GiB, 120 s setup + 20 x 5 s.
                "weights_gib": round(weights_gb, 2),
                "peak_gib": round(
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available()
                    else 0.0,
                    2,
                ),
            },
            indent=2,
        )
    )
    log.info(
        "done: %d rows | %.1fs | %.2fs/Q | parseable=%d/%d | errors=%d | -> %s",
        len(records), dt, dt / max(len(reqs), 1), n_parse, len(reqs), n_err, out,
    )


if __name__ == "__main__":
    main()
