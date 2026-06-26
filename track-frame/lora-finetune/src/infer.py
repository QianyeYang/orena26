"""LoRA-adapter-aware single-image VLM wrapper + reusable generate helper.

``generate_answer`` is the single-image greedy-decode recipe (apply_chat_template +
processor + generate + decode); it is the same recipe the zero-shot sweep proved
across Qwen2.5-VL, Qwen3-VL, InternVL3.5, LLaVA-OneVision and SmolVLM2, so it is
architecture-agnostic. ``LoRAVLM`` wraps it for standalone inference (base + PEFT
adapter); the per-epoch eval callback reuses ``generate_answer`` on the in-memory
training model. Local/offline, sdpa attention (aarch64).
"""

from __future__ import annotations

import time

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


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
        model = AutoModelForImageTextToText.from_pretrained(base_path, **model_kwargs)
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
