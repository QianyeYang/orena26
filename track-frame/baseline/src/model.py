"""Qwen2.5-VL inference wrapper for the FRAME baseline.

Single-image, single-turn VQA. Loads from a LOCAL path (``os-models/...``) so it
runs offline. Greedy decoding, short outputs (answers are terse by design).

Avoids the optional ``qwen_vl_utils`` dependency: images are loaded with PIL and
handed to the processor directly alongside an ``{"type": "image"}`` placeholder.
"""

from __future__ import annotations

import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenVL:
    """Thin Qwen2.5-VL wrapper exposing :meth:`answer` (text -> (response, latency))."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        attn_implementation: str = "sdpa",  # flash_attention_2 often unavailable on aarch64
    ) -> None:
        self.max_new_tokens = max_new_tokens
        proc_kwargs: dict = {}
        if min_pixels:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

    @torch.inference_mode()
    def answer(self, image_path: str, instruction: str, system: str | None = None) -> tuple[str, float]:
        """Greedy-decode an answer for one image + instruction. Returns (text, seconds)."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append(
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

        t0 = time.time()
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        latency = time.time() - t0

        trimmed = generated[:, inputs.input_ids.shape[1] :]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return out, latency
