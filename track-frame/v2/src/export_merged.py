#!/usr/bin/env python
"""Export a deployable single-directory model: merge the LoRA, optionally quantize.

Why this exists
---------------
``benchmark.py --adapter X --quant int8-mlp`` quantizes the *base* and then hangs
an unmerged PEFT adapter on top (``infer.LoRAVLM``: ``from_pretrained`` with the
quantization config, *then* ``PeftModel.from_pretrained``). That is fine for
measurement but it is not what should ship:

- the submission would have to carry base + adapter and re-run PEFT wiring
  inside the 120 s setup budget;
- the LoRA stays in bf16 on top of INT8-rounded weights, so the arithmetic
  differs from a merged model that is rounded *after* the deltas are folded in.

``docs/large-model-deployment.md`` checklist item 5 says to merge before
quantizing "so what is measured is what ships". This script does that, which
means the merged artefact **must be re-benchmarked** -- merge-then-quantize is a
different numerical path from the quantize-then-attach run that produced 0.6551,
and the difference is not assumed to be zero.

Stages (either or both)
-----------------------
``--out-merged``  fold the adapter into the base, save bf16 (~52 GiB).
``--out-quant``   load that merged model under ``int8-mlp`` and save it
                  pre-quantized (~36 GiB), so deployment is a plain
                  ``from_pretrained`` with no torchao work at setup time.

Usage::

    python export_merged.py \
        --base os-models/Qwen3.6-27B \
        --adapter track-frame/v2/logs/Qwen3.6-27B-frame-v1prompt/checkpoint-23136 \
        --out-merged os-models/Qwen3.6-27B-frame-ep24-merged \
        --out-quant  os-models/Qwen3.6-27B-frame-ep24-int8mlp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "track-frame" / "lora-finetune" / "src"))

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from infer import MLP_ONLY_EXCLUSIONS, count_quantized_linears  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("export")


def _abs(repo: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else repo / q


def _copy_processor(base: Path, out: Path, trust_remote_code: bool) -> None:
    """Save the processor and carry over any files it does not serialise.

    A merged directory has to be loadable on its own; ``chat_template.jinja``
    and friends live beside the weights and are easy to forget.
    """
    kw = {"trust_remote_code": True} if trust_remote_code else {}
    AutoProcessor.from_pretrained(str(base), **kw).save_pretrained(str(out))
    for name in ("chat_template.jinja", "chat_template.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "generation_config.json"):
        src = base / name
        if src.is_file() and not (out / name).is_file():
            shutil.copy2(src, out / name)
            log.info("copied %s", name)


def merge(base: Path, adapter: Path, out: Path, dtype: str, trust_remote_code: bool) -> None:
    from peft import PeftModel

    kw: dict = {"dtype": getattr(torch, dtype), "device_map": "cpu"}
    if trust_remote_code:
        kw["trust_remote_code"] = True

    t = time.time()
    log.info("loading base %s on CPU", base)
    model = AutoModelForImageTextToText.from_pretrained(str(base), **kw)
    log.info("base loaded in %.1fs", time.time() - t)

    t = time.time()
    log.info("applying adapter %s", adapter)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    log.info("merged in %.1fs", time.time() - t)

    t = time.time()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    _copy_processor(base, out, trust_remote_code)
    log.info("merged model written to %s in %.1fs", out, time.time() - t)
    del model


def quantize(src: Path, out: Path, base: Path, dtype: str, trust_remote_code: bool) -> None:
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig

    kw: dict = {
        "dtype": getattr(torch, dtype),
        "device_map": "cpu",
        "quantization_config": TorchAoConfig(
            quant_type=Int8WeightOnlyConfig(),
            modules_to_not_convert=list(MLP_ONLY_EXCLUSIONS),
        ),
    }
    if trust_remote_code:
        kw["trust_remote_code"] = True

    t = time.time()
    log.info("loading %s under int8-mlp", src)
    model = AutoModelForImageTextToText.from_pretrained(str(src), **kw)
    quant, total = count_quantized_linears(model)
    log.info("quantized %d/%d Linear modules in %.1fs", quant, total, time.time() - t)
    if quant != 192:
        # The fp8-keepvision lesson: an exclusion list that matches nothing runs
        # a whole benchmark before anyone notices. Fail here instead.
        raise SystemExit(f"expected 192 quantized Linears, got {quant}/{total}")

    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        model.save_pretrained(str(out), safe_serialization=True)
    except Exception as err:  # torchao tensor subclasses are not always safetensors-able
        log.warning("safetensors save failed (%s); retrying with torch.save format", err)
        for stale in out.glob("*.safetensors"):
            stale.unlink()
        model.save_pretrained(str(out), safe_serialization=False)
    _copy_processor(base, out, trust_remote_code)
    log.info("quantized model written to %s in %.1fs", out, time.time() - t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model dir (repo-relative or absolute)")
    ap.add_argument("--adapter", help="LoRA checkpoint to merge; omit to quantize only")
    ap.add_argument("--out-merged", help="where to write the merged bf16 model")
    ap.add_argument("--out-quant", help="where to write the int8-mlp model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    if not a.out_merged and not a.out_quant:
        ap.error("nothing to do: pass --out-merged and/or --out-quant")
    if a.adapter and not a.out_merged:
        ap.error("--adapter needs --out-merged (merge before quantizing)")

    base = _abs(REPO, a.base)
    merged = _abs(REPO, a.out_merged) if a.out_merged else None
    quant_out = _abs(REPO, a.out_quant) if a.out_quant else None

    if merged:
        merge(base, _abs(REPO, a.adapter), merged, a.dtype, a.trust_remote_code)
    if quant_out:
        quantize(merged or base, quant_out, base, a.dtype, a.trust_remote_code)

    for d in (merged, quant_out):
        if d:
            gib = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30
            log.info("%s -> %.2f GiB on disk", d, gib)


if __name__ == "__main__":
    main()
