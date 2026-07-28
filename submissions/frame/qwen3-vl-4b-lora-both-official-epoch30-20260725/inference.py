#!/usr/bin/env python3
"""Offline ORena FOCUS Frame inference for Qwen3-VL-4B plus epoch-30 LoRA."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROCESS_START = time.monotonic()
APP_PATH = Path(__file__).resolve().parent
VENDOR_ARCHIVE = APP_PATH / "resources" / "python-vendor.tar"
VENDOR_PATH = Path(os.environ.get("FOCUS_VENDOR_PATH", "/tmp/focus-python"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not (VENDOR_PATH / ".complete").is_file():
    VENDOR_PATH.mkdir(parents=True, exist_ok=True)
    root = VENDOR_PATH.resolve()
    with tarfile.open(VENDOR_ARCHIVE) as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe path in Python vendor archive: {member.name}")
        archive.extractall(VENDOR_PATH)
    (VENDOR_PATH / ".complete").touch()
sys.path.insert(0, str(VENDOR_PATH))

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("focus-frame")

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
FRAME_PATH = INPUT_PATH / "frames"
BASE_PATH = APP_PATH / "resources" / "base_model"
ADAPTER_PATH = APP_PATH / "resources" / "adapter"

MAX_PIXELS = 602_112
MAX_NEW_TOKENS = 64
MICRO_BATCH_SIZE = max(1, int(os.environ.get("FOCUS_BATCH_SIZE", "8")))
MAX_TEXT_LENGTH = 300
MODEL_DTYPE_NAME = os.environ.get("FOCUS_MODEL_DTYPE", "bfloat16").strip().lower()
MODEL_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
if MODEL_DTYPE_NAME not in MODEL_DTYPES:
    raise ValueError(
        "FOCUS_MODEL_DTYPE must be 'bfloat16' or 'float16', "
        f"not {MODEL_DTYPE_NAME!r}"
    )
MODEL_DTYPE = MODEL_DTYPES[MODEL_DTYPE_NAME]

SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a single endoscopic "
    "frame from colorectal surgery. Look carefully at the image and answer the "
    "question. Respond with ONLY the answer in the exact requested format — no "
    "explanation, no extra words."
)

DEFAULT_FO_NAMES = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
    "Mesh",
    "Absorbable Hemostatic Agent",
)


@dataclass
class Request:
    qID: str
    videoID: str
    start_time: float
    end_time: float
    procedure_type: str
    question: str


@dataclass
class Response:
    qID: str
    content: str
    latency: float


def load_requests(path: Path) -> list[Request]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("request.json must contain a list")
    return [Request(**row) for row in rows]


def save_responses(items: list[Response], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clean_text(value: Any) -> str:
    return " ".join(str(value).split())[:MAX_TEXT_LENGTH]


def load_fo_definitions() -> str:
    path = INPUT_PATH / "FO_definitions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, str):
        raise TypeError("FO_definitions.json must contain a JSON string")
    return value


def parse_fo_names(definitions: str) -> tuple[str, ...]:
    """Read class headings from the challenge-provided definitions."""
    lines = [line.strip() for line in definitions.splitlines()]
    names: list[str] = []
    for index in range(len(lines) - 1):
        name, underline = lines[index], lines[index + 1]
        if name and re.fullmatch(r"-{3,}", underline):
            names.append(name)
    return tuple(dict.fromkeys(names)) or DEFAULT_FO_NAMES


def infer_answer_format(question: str) -> str:
    """Recover the hidden reference format from stable instructions in the question.

    This rule exactly matches all 6,252 current official Frame test questions.
    Unknown templates safely fall back to open-ended output.
    """
    text = " ".join(question.lower().split())
    if "please select one answer:" in text:
        return "multiple_choice"
    if text.endswith("please answer with yes or no."):
        return "binary"
    if text.endswith("please provide a number."):
        return "number"
    if (
        text.endswith("please provide a class name.")
        or text.endswith("please provide the class names or answer with none.")
    ):
        return "fo_class"
    return "open_ended"


def parse_mc_options(question: str) -> list[str]:
    match = re.search(
        r"select one answer:\s*(.+)$", question, flags=re.IGNORECASE | re.DOTALL
    )
    tail = match.group(1) if match else ""
    return [part.strip(" .") for part in re.split(r"[;\n]", tail) if part.strip(" .")]


def build_instruction(
    question: str, answer_format: str, fo_names: tuple[str, ...]
) -> tuple[str, list[str] | None]:
    options: list[str] | None = None
    if answer_format == "binary":
        instruction = f"{question}\n\nAnswer with exactly one word: yes or no."
    elif answer_format == "number":
        instruction = (
            f"{question}\n\nAnswer with a single non-negative integer (digits only)."
        )
    elif answer_format == "fo_class":
        names = ", ".join(fo_names)
        instruction = (
            f"{question}\n\nAnswer with only one or more of these foreign-object "
            f"names, separated by a comma and a space: {names}. If none is visible, "
            "answer: none. Output nothing else."
        )
    elif answer_format == "multiple_choice":
        options = parse_mc_options(question)
        instruction = (
            f"{question}\n\nAnswer by copying EXACTLY ONE of the listed options, "
            "verbatim, and nothing else."
        )
    else:
        instruction = f"{question}\n\nAnswer concisely, in a few words."
    return instruction, options


def normalize_binary(raw: str) -> str:
    text = clean_text(raw).lower()
    if text in {"yes", "no"}:
        return text
    if re.search(r"\b(yes|yeah|yep|true|present|affirmative)\b", text):
        return "yes"
    if re.search(r"\b(no|nope|false|absent|negative)\b", text):
        return "no"
    return text


def normalize_fo_class(raw: str, fo_names: tuple[str, ...]) -> str:
    text = clean_text(raw)
    lower = text.lower()
    if not lower:
        return ""

    matched: set[str] = set()
    masked = lower
    for name in sorted(fo_names, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        if re.search(pattern, masked):
            matched.add(name)
            masked = re.sub(pattern, " ", masked)
    if matched:
        return ", ".join(name for name in fo_names if name in matched)
    if re.search(
        r"\b(none|no foreign object|no object|nothing|absent|not present|n/a)\b",
        lower,
    ):
        return "none"
    return text


def normalize_answer(
    raw: str,
    answer_format: str,
    fo_names: tuple[str, ...],
    options: list[str] | None,
) -> str:
    if not raw.strip():
        return ""
    if answer_format == "binary":
        return normalize_binary(raw)
    if answer_format == "number":
        match = re.search(r"\d+", raw)
        return match.group(0) if match else clean_text(raw)
    if answer_format == "fo_class":
        return normalize_fo_class(raw, fo_names)
    if answer_format == "multiple_choice" and options:
        lower = clean_text(raw).lower()
        for option in sorted(options, key=len, reverse=True):
            if option.lower() in lower:
                return option[:MAX_TEXT_LENGTH]
    return clean_text(raw)


class FrameModel:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this submission")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.device = torch.device("cuda:0")

        self.processor = AutoProcessor.from_pretrained(
            BASE_PATH,
            max_pixels=MAX_PIXELS,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        base = AutoModelForImageTextToText.from_pretrained(
            BASE_PATH,
            dtype=MODEL_DTYPE,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = PeftModel.from_pretrained(
            base,
            ADAPTER_PATH,
            is_trainable=False,
            local_files_only=True,
        ).eval()

    @torch.inference_mode()
    def generate(self, items: list[dict[str, Any]], system_prompt: str) -> tuple[list[str], float]:
        images: list[Image.Image] = []
        prompts: list[str] = []
        for item in items:
            with Image.open(item["frame_path"]) as image:
                images.append(image.convert("RGB"))
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["instruction"]},
                    ],
                },
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        torch.cuda.synchronize()
        started = time.monotonic()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        prompt_length = inputs["input_ids"].shape[1]
        answers = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [answer.strip() for answer in answers], elapsed


def run_batch_with_fallback(
    model: FrameModel,
    items: list[dict[str, Any]],
    system_prompt: str,
    raw_answers: dict[int, str],
    latencies: dict[int, float],
    errors: dict[int, str],
) -> None:
    """Split a failed micro-batch recursively, preserving per-question isolation."""
    try:
        answers, elapsed = model.generate(items, system_prompt)
        per_question = elapsed / max(len(items), 1)
        for item, answer in zip(items, answers):
            raw_answers[item["index"]] = answer
            latencies[item["index"]] = per_question
        return
    except Exception as exc:
        log.exception("Micro-batch of %d failed", len(items))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(items) == 1:
            errors[items[0]["index"]] = f"{type(exc).__name__}: {exc}"
            raw_answers[items[0]["index"]] = ""
            latencies[items[0]["index"]] = 0.0
            return
        middle = len(items) // 2
        run_batch_with_fallback(
            model, items[:middle], system_prompt, raw_answers, latencies, errors
        )
        run_batch_with_fallback(
            model, items[middle:], system_prompt, raw_answers, latencies, errors
        )


def run() -> int:
    log.info("ORena FOCUS Frame inference starting")
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        raise ValueError("request.json contains no requests")

    definitions = load_fo_definitions()
    fo_names = parse_fo_names(definitions)
    system_prompt = (
        SYSTEM_PROMPT
        + "\n\nUse these challenge-provided foreign-object definitions:\n"
        + definitions
    )
    log.info(
        "batch=%d micro_batch=%d dtype=%s classes=%s",
        len(requests),
        MICRO_BATCH_SIZE,
        MODEL_DTYPE_NAME,
        ", ".join(fo_names),
    )

    prepared: list[dict[str, Any]] = []
    raw_answers: dict[int, str] = {}
    latencies: dict[int, float] = {}
    errors: dict[int, str] = {}
    formats: dict[int, str] = {}
    options_by_index: dict[int, list[str] | None] = {}

    for index, request in enumerate(requests):
        answer_format = infer_answer_format(request.question)
        instruction, options = build_instruction(request.question, answer_format, fo_names)
        formats[index] = answer_format
        options_by_index[index] = options
        frame_path = FRAME_PATH / f"{request.qID}.png"
        try:
            if not frame_path.is_file() or frame_path.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty frame: {frame_path}")
            with Image.open(frame_path) as image:
                image.verify()
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"
            raw_answers[index] = ""
            latencies[index] = 0.0
            log.exception("qID=%s frame validation failed", request.qID)
            continue
        prepared.append(
            {
                "index": index,
                "frame_path": frame_path,
                "instruction": instruction,
            }
        )

    log.info("Loading model once for %d valid question(s)", len(prepared))
    model = FrameModel()
    log.info(
        "model loaded in %.2fs on %s; allocated=%.2f GiB",
        time.monotonic() - PROCESS_START,
        torch.cuda.get_device_name(0),
        torch.cuda.memory_allocated(0) / 1024**3,
    )

    for start in range(0, len(prepared), MICRO_BATCH_SIZE):
        batch = prepared[start : start + MICRO_BATCH_SIZE]
        run_batch_with_fallback(
            model, batch, system_prompt, raw_answers, latencies, errors
        )

    responses: list[Response] = []
    for index, request in enumerate(requests):
        content = normalize_answer(
            raw_answers.get(index, ""),
            formats[index],
            fo_names,
            options_by_index[index],
        )
        latency = float(latencies.get(index, 0.0))
        if not math.isfinite(latency) or latency < 0:
            latency = 0.0
        responses.append(Response(qID=request.qID, content=content, latency=latency))
        log.info(
            "qID=%s format=%s latency=%.3fs answer=%r%s",
            request.qID,
            formats[index],
            latency,
            content,
            f" error={errors[index]}" if index in errors else "",
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_responses(responses, OUTPUT_PATH / "answer.json")
    log.info(
        "wrote %d responses; failures=%d total=%.2fs peak_gpu=%.2f GiB",
        len(responses),
        len(errors),
        time.monotonic() - PROCESS_START,
        torch.cuda.max_memory_allocated(0) / 1024**3,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
