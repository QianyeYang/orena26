"""Generic single-image VLM wrapper for the FRAME zero-shot sweep.

One class covers every native ``transformers`` image-text-to-text model
(Qwen2.5-VL, Qwen3-VL dense + MoE, InternVL3.5, LLaVA-OneVision, SmolVLM2):
load with ``AutoModelForImageTextToText`` + ``AutoProcessor``, prompt with the
model's own chat template plus a single image, greedy-decode a terse answer.

Keeps the same contract as the baseline ``track-frame/baseline/src/model.py``
``QwenVL`` (``answer(image_path, instruction, system) -> (raw_text, latency)``)
so the shared runner / prompts / adapter all apply unchanged.

Loads from a LOCAL ``os-models/`` path (offline). ``sdpa`` attention because
flash-attention-2 is unavailable on aarch64.
"""

from __future__ import annotations

import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLM:
    """Generic image-text-to-text wrapper exposing :meth:`answer`."""

    def __init__(
        self,
        model_path: str,
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
        if min_pixels:  # Qwen-VL family only; harmless to omit otherwise
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        model_kwargs: dict = dict(
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **model_kwargs
        ).eval()

    @torch.inference_mode()
    def answer(
        self, image_path: str, instruction: str, system: str | None = None
    ) -> tuple[str, float]:
        """Greedy-decode an answer for one image + instruction. Returns (text, seconds).

        Two-step prompt build (template text + separate image) is the most
        portable path across VLM families: the chat template inserts the
        model-specific image placeholder tokens from the ``{"type": "image"}``
        marker, then the processor fuses text + pixels.
        """
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
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )

        t0 = time.time()
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        latency = time.time() - t0

        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return out, latency
