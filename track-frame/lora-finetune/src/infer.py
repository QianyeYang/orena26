"""LoRA-adapter-aware single-image VLM wrapper + reusable generate helper.

``generate_answer`` is the single-image greedy-decode recipe (apply_chat_template +
processor + generate + decode); it is the same recipe the zero-shot sweep proved
across Qwen2.5-VL, Qwen3-VL, InternVL3.5, LLaVA-OneVision and SmolVLM2, so it is
architecture-agnostic. ``LoRAVLM`` wraps it for standalone inference (base + PEFT
adapter); the per-epoch eval callback reuses ``generate_answer`` on the in-memory
training model. Local/offline, sdpa attention (aarch64).
"""

from __future__ import annotations

import logging
import os
import time

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

log = logging.getLogger(__name__)


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    device,
    image_path: str,
    instruction: str,
    system: str | None = None,
    max_new_tokens: int = 64,
) -> tuple[str, float]:
    """Greedy-decode one (image, instruction) -> (text, seconds) with any VLM."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    t0 = time.time()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - t0

    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return out, latency


#: The commit `kernels-community/finegrained-fp8` version 4 resolves to. Pinning
#: it is what makes FP8 work offline: transformers asks for ``version=4``, and
#: turning a version *spec* into a commit needs the Hub even when the snapshot is
#: already cached — so an offline container fails with "not available in the
#: local cache" despite having the files. The loader also accepts ``revision``
#: (``sonic-moe`` is registered that way), which resolves purely locally.
FP8_KERNEL_REVISION = os.environ.get(
    "FOCUS_FP8_KERNEL_REVISION", "7cdb05d472d6c954c7d03182ed836ebfd4610df0"
)


#: Everything that must stay in bf16 when quantizing the language-model MLPs.
#: ``should_convert_module`` matches these with ``re.match``, i.e. as anchored
#: prefixes, so five patterns cover all 415 non-MLP Linears. The list is the
#: lesson from FP8 written down: the 48 linear-attention layers carry ``A_log``,
#: ``dt_bias`` and a short convolution — values that get exponentiated or set an
#: integration timestep, where rounding changes the dynamics rather than the
#: precision. Qwen's own FP8 config exempts exactly these.
MLP_ONLY_EXCLUSIONS: tuple[str, ...] = (
    r"model\.language_model\.layers\.\d+\.linear_attn",
    r"model\.language_model\.layers\.\d+\.self_attn",
    r"model\.visual",
    r"lm_head",
    r"model\.embed_tokens",
)


def count_quantized_linears(model) -> tuple[int, int]:
    """Return (quantized, total) Linear counts.

    Worth logging on every quantized load: the ``fp8-keepvision`` attempt
    silently converted the modules it was told to exempt and reported memory
    byte-identical to plain FP8, which went unnoticed for a full benchmark.
    A count makes that failure loud.
    """
    total = quant = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += 1
            # torchao swaps ``weight`` for a tensor subclass; FP8 swaps the
            # module. Either way the plain-Tensor identity no longer holds.
            weight = getattr(module, "weight", None)
            if weight is not None and type(weight.data) is not torch.Tensor:
                quant += 1
    return quant, total


def _pin_fp8_kernel_revision() -> None:
    """Rewrite the hub-kernel mapping from a version spec to an exact commit."""
    try:
        from transformers.integrations import hub_kernels
    except ImportError:  # transformers without hub-kernel support
        return
    mapping = getattr(hub_kernels, "_HUB_KERNEL_MAPPING", None)
    if not isinstance(mapping, dict) or "finegrained-fp8" not in mapping:
        return  # upstream renamed it; let transformers resolve as it sees fit
    mapping["finegrained-fp8"] = {
        "repo_id": "kernels-community/finegrained-fp8",
        "revision": FP8_KERNEL_REVISION,
    }


class LoRAVLM:
    """Base VLM + optional PEFT adapter, exposing :meth:`answer`."""

    def __init__(
        self,
        base_path: str,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        quantization: str | None = None,
    ) -> None:
        self.max_new_tokens = max_new_tokens

        proc_kwargs: dict = {}
        if trust_remote_code:
            proc_kwargs["trust_remote_code"] = True
        if min_pixels:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(base_path, **proc_kwargs)

        model_kwargs: dict = dict(
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        # Unconditional: a *pre-quantized* checkpoint (Qwen3.6-27B-FP8 and the
        # like) carries its own quantization_config, so ``quantization`` is None
        # here yet the FP8 Triton kernel is still needed at runtime. Pinning only
        # inside the branches below left those runs to fail every single row.
        _pin_fp8_kernel_revision()
        if quantization == "fp8":
            # Runtime round-to-nearest FP8 over *every* Linear. Measured 0.3217
            # bf16 -> 0.2063 on 1,203 stratified rows, because it also quantizes
            # the 48 linear-attention layers' A_log / dt_bias / conv1d — values
            # that get exponentiated, where rounding changes the dynamics rather
            # than the precision. Kept only for the ablation record: prefer a
            # pre-quantized checkpoint, which ships its own exemption list.
            from transformers import FineGrainedFP8Config

            model_kwargs["quantization_config"] = FineGrainedFP8Config()
        elif quantization == "fp8-keepvision":
            # Does not work: `weights_gib` came out byte-identical to plain fp8
            # (27.86), so these prefixes matched nothing. Qwen's own FP8 config
            # spells out all 882 exempt modules as fully-qualified paths.
            from transformers import FineGrainedFP8Config

            model_kwargs["quantization_config"] = FineGrainedFP8Config(
                modules_to_not_convert=["visual", "merger", "lm_head"]
            )
        elif quantization == "int8-mlp":
            # The deployment path. A 27B in bf16 is 50.96 GiB loaded and an
            # L40S gives ~43.5 GiB for weights, so only ~8 GiB has to go --
            # 16%, not the 50% FP8 throws away. The MLPs alone are 31.88 GiB of
            # that 50.96 and are plain matmuls; at INT8 they become 15.94 and
            # the model lands near 35 GiB with everything numerically delicate
            # untouched. Weight-only INT8 also tends to *speed up* decode on a
            # bandwidth-limited card like the L40S rather than slow it.
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import TorchAoConfig

            model_kwargs["quantization_config"] = TorchAoConfig(
                quant_type=Int8WeightOnlyConfig(),
                modules_to_not_convert=list(MLP_ONLY_EXCLUSIONS),
            )
        elif quantization:
            raise ValueError(f"unknown quantization: {quantization}")
        model = AutoModelForImageTextToText.from_pretrained(base_path, **model_kwargs)
        if quantization:
            quant, total = count_quantized_linears(model)
            log.info("quantized %d/%d Linear modules (%s)", quant, total, quantization)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

    def answer(
        self, image_path: str, instruction: str, system: str | None = None
    ) -> tuple[str, float]:
        return generate_answer(
            self.model, self.processor, self.device, image_path, instruction, system,
            self.max_new_tokens,
        )
