"""Generation for hybrid-thinking VLMs, with the reasoning block under our control.

Qwen3.6 ships thinking **on** by default: its chat template opens a `<think>`
block unless `enable_thinking=false`, in which case it prefills an empty one.
Left alone, a smoke run spent 12 s per question and returned truncated
reasoning instead of an answer — every row unparseable.

That matters beyond tidiness, because the track's budget is pooled: a
20-question batch gets 120 s of setup plus 20×5 s, so ~6-9 s per question
survives model loading. Unbounded reasoning does not fit; a bounded budget
might, and thinking is worth measuring rather than assuming. So this module
exposes the toggle and, when reasoning is on, splits the generation budget
explicitly instead of letting the answer be whatever survives truncation.

`strip_thinking` is applied unconditionally — a model that emits a reasoning
block anyway (wrong template, an adapter that learned to) must not leak it into
`fo_class`, where `adapter.normalize_fo_class` substring-matches class names and
would collect every class the reasoning happened to mention.
"""

from __future__ import annotations

import re
import time

from PIL import Image
import torch

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"^.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning block, closed or dangling, and return the answer text."""
    if "</think>" in text:
        text = _THINK_BLOCK.sub("", text)
        if "</think>" in text:  # unmatched opener, e.g. template-prefilled
            text = _THINK_OPEN.sub("", text)
    elif "<think>" in text:
        # Truncated mid-reasoning: there is no answer in here at all.
        return ""
    return text.strip()


def apply_template(processor, messages: list[dict], *, enable_thinking: bool) -> str:
    """Render the chat template, tolerating templates without a thinking toggle."""
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older processors reject unknown kwargs; their templates have no
        # thinking block to suppress anyway.
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_messages(instruction: str, system: str | None) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    )
    return messages


@torch.inference_mode()
def generate(
    model,
    processor,
    device,
    image: str | Image.Image,
    instruction: str,
    system: str | None = None,
    *,
    max_new_tokens: int = 64,
    enable_thinking: bool = False,
    thinking_budget: int = 256,
) -> tuple[str, str, float]:
    """Greedy-decode one (image, instruction) → ``(answer, raw, seconds)``.

    ``raw`` keeps the undecorated model output including any reasoning, so runs
    stay auditable; ``answer`` is what the scorer sees. With thinking enabled the
    token budget is ``thinking_budget + max_new_tokens``, so the answer still has
    room after the model finishes reasoning.
    """
    messages = build_messages(instruction, system)
    text = apply_template(processor, messages, enable_thinking=enable_thinking)
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

    budget = max_new_tokens + (thinking_budget if enable_thinking else 0)
    t0 = time.time()
    generated = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
    latency = time.time() - t0

    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return strip_thinking(raw), raw, latency
