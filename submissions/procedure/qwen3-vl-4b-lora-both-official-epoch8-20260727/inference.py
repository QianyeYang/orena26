#!/usr/bin/env python3
"""Offline ORena FOCUS Procedure inference for Qwen3-VL-4B + epoch-8 LoRA."""

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
    vendor_root = VENDOR_PATH.resolve()
    with tarfile.open(VENDOR_ARCHIVE) as archive:
        for member in archive.getmembers():
            target = (vendor_root / member.name).resolve()
            if not target.is_relative_to(vendor_root):
                raise RuntimeError(
                    f"unsafe path in Python vendor archive: {member.name}"
                )
        archive.extractall(VENDOR_PATH)
    (VENDOR_PATH / ".complete").touch()
sys.path.insert(0, str(VENDOR_PATH))

# Import torch before decord. Reversing this order can break CUDA initialisation
# with some decord builds.
import torch
from decord import VideoReader, cpu
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("focus-procedure")

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
VIDEO_PATH = INPUT_PATH / "plain"
BASE_PATH = APP_PATH / "resources" / "base_model"
ADAPTER_PATH = APP_PATH / "resources" / "adapter"

MAX_PIXELS = 131_072
MAX_NEW_TOKENS = 64
FRAME_INTERVAL_SECONDS = 10.0
MAX_TEXT_LENGTH = 300
MAX_FRAMES = int(os.environ.get("FOCUS_MAX_FRAMES", "96"))
if not 1 <= MAX_FRAMES <= 96:
    raise ValueError("FOCUS_MAX_FRAMES must be between 1 and 96")

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
    "You are an expert surgical vision assistant analysing a colorectal surgery "
    "video from its start up to the current moment. The video is shown as a "
    "sequence of frames in chronological order, each labelled with its absolute "
    "video timestamp (hh:mm:ss from the start of the video). Timestamps "
    "mentioned in the question refer to that same absolute video time. Look "
    "carefully at the frames and answer the question. Respond with ONLY the "
    "answer in the exact requested format — no explanation, no extra words."
)

FO_NAMES = (
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

_MC_TAIL_RE = re.compile(
    r"select (?:one answer|(?:none, )?one or multiple answers?)[:\s]*(.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Request:
    qID: str
    videoID: str
    start_time: float
    end_time: float
    procedure_type: str
    question: str


@dataclass(frozen=True)
class Response:
    qID: str
    content: str
    latency: float


def load_requests(path: Path) -> list[Request]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("request.json must contain a list")
    requests = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("every request must be an object")
        requests.append(
            Request(
                qID=str(row["qID"]),
                videoID=str(row["videoID"]),
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                procedure_type=str(row["procedure_type"]),
                question=str(row["question"]),
            )
        )
    return requests


def save_responses(items: list[Response], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clean_text(value: Any) -> str:
    return " ".join(str(value).split())[:MAX_TEXT_LENGTH]


def infer_answer_format(question: str) -> str:
    """Recover the hidden format from stable question instructions.

    These rules exactly match all 3,127 current official Procedure test rows.
    Unknown future templates safely fall back to open-ended output.
    """
    text = " ".join(question.lower().split())
    if "please select one answer:" in text:
        return "multiple_choice"
    if text.endswith("please answer with yes or no."):
        return "binary"
    if "format xx%" in text:
        return "percentage"
    if "hh:mm:ss" in text:
        return "time"
    if "please provide a number" in text:
        return "number"
    if (
        "please provide a class name" in text
        or "please provide the class name" in text
    ):
        return "fo_class"
    return "open_ended"


def parse_mc_options(question: str) -> list[str]:
    match = _MC_TAIL_RE.search(question)
    tail = match.group(1) if match else question
    return [
        option.strip(" .")
        for option in re.split(r"[;\n]", tail)
        if option.strip(" .")
    ]


def is_multi_select(question: str) -> bool:
    return bool(
        re.search(
            r"select (?:none, )?one or multiple answers?",
            question,
            flags=re.IGNORECASE,
        )
    )


def build_instruction(
    question: str, answer_format: str
) -> tuple[str, list[str] | None]:
    """Reproduce the format-specific training instruction exactly."""
    options: list[str] | None = None
    if answer_format == "binary":
        instruction = f"{question}\n\nAnswer with exactly one word: yes or no."
    elif answer_format == "number":
        instruction = (
            f"{question}\n\nAnswer with a single non-negative integer (digits only)."
        )
    elif answer_format == "fo_class":
        names = ", ".join(FO_NAMES)
        instruction = (
            f"{question}\n\nAnswer with EXACTLY ONE of these foreign-object names: "
            f"{names}. If none is visible, answer: none. Output only the name."
        )
    elif answer_format == "multiple_choice":
        options = parse_mc_options(question)
        if is_multi_select(question):
            instruction = (
                f"{question}\n\nAnswer by copying the correct option(s) from the "
                "list, verbatim; separate multiple options with a comma and a "
                "space. If none apply, answer: none. Output nothing else."
            )
        else:
            instruction = (
                f"{question}\n\nAnswer by copying EXACTLY ONE of the listed "
                "options, verbatim, and nothing else."
            )
    elif answer_format == "percentage":
        instruction = f"{question}\n\nAnswer with a single number (a percentage)."
    elif answer_format == "time":
        instruction = f"{question}\n\nAnswer with a timestamp as hh:mm:ss."
    else:
        instruction = f"{question}\n\nAnswer concisely, in a few words."
    return instruction, options


def normalize_binary(text: str) -> str:
    lower = clean_text(text).lower()
    if lower in {"yes", "no"}:
        return lower
    if re.search(
        r"\b(yes|yeah|yep|true|correct|present|affirmative)\b", lower
    ) or lower.startswith("yes"):
        return "yes"
    if re.search(
        r"\b(no|nope|false|incorrect|absent|negative)\b", lower
    ) or lower.startswith("no"):
        return "no"
    return lower


def normalize_fo_class(text: str) -> str:
    lower = clean_text(text).lower()
    if not lower:
        return "none"
    best: tuple[tuple[int, int], str] | None = None
    for name in FO_NAMES:
        index = lower.find(name.lower())
        if index != -1:
            key = (index, -len(name))
            if best is None or key < best[0]:
                best = (key, name)
    if best is not None:
        return best[1]
    if re.search(
        r"(^|\W)(none|no foreign object|no object|nothing|absent|not present|n/a)"
        r"(\W|$)",
        lower,
    ):
        return "none"
    return clean_text(text)


def normalize_multiple_choice(
    text: str, options: list[str], *, multi: bool
) -> str:
    cleaned = clean_text(text)
    lower = cleaned.lower()
    if multi:
        matched: set[str] = set()
        masked = lower
        for option in sorted(options, key=len, reverse=True):
            if option.lower() in masked:
                matched.add(option)
                masked = masked.replace(option.lower(), " ")
        if matched:
            return ", ".join(
                option for option in options if option in matched
            )[:MAX_TEXT_LENGTH]
        if re.search(r"(^|\W)none(\W|$)", lower):
            return "none"
        return cleaned
    for option in sorted(options, key=len, reverse=True):
        if option.lower() in lower:
            return option[:MAX_TEXT_LENGTH]
    return cleaned


def normalize_answer(
    raw: str,
    answer_format: str,
    options: list[str] | None,
    *,
    multi: bool,
) -> str:
    if answer_format == "binary":
        return normalize_binary(raw)
    if answer_format == "number":
        match = re.search(r"\d+", raw)
        return match.group(0) if match else clean_text(raw)
    if answer_format == "fo_class":
        return normalize_fo_class(raw)
    if answer_format == "percentage":
        match = re.search(r"\d+(?:\.\d+)?", raw)
        return match.group(0) if match else clean_text(raw)
    if answer_format == "time":
        match = re.search(r"(\d{1,2}):([0-5]?\d):([0-5]?\d)", raw)
        if match:
            hours, minutes, seconds = (int(value) for value in match.groups())
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return clean_text(raw)
    if answer_format == "multiple_choice" and options:
        return normalize_multiple_choice(raw, options, multi=multi)
    return clean_text(raw)


def seconds_to_timestamp(seconds: float) -> str:
    value = int(round(seconds))
    return (
        f"{value // 3600:02d}:"
        f"{(value % 3600) // 60:02d}:"
        f"{value % 60:02d}"
    )


def even_subsample(values: list[int], count: int) -> list[int]:
    if count >= len(values) or len(values) <= 1:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    step = (len(values) - 1) / (count - 1)
    return list(
        dict.fromkeys(values[round(index * step)] for index in range(count))
    )


def sample_indices(video_path: Path, max_frames: int) -> tuple[list[int], float, int]:
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count < 1:
        raise ValueError(f"video contains no frames: {video_path}")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"video has invalid FPS {fps}: {video_path}")
    stride = max(round(fps * FRAME_INTERVAL_SECONDS), 1)
    indices = list(range(0, frame_count, stride)) or [0]
    indices = even_subsample(indices, max_frames)
    del reader
    return indices, fps, frame_count


def decode_frames(video_path: Path, indices: list[int]) -> list[Image.Image]:
    started = time.monotonic()
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    arrays = reader.get_batch(indices).asnumpy()
    del reader
    images = [Image.fromarray(array, mode="RGB") for array in arrays]
    log.info(
        "decoded %d selected frames from %s in %.2fs",
        len(images),
        video_path.name,
        time.monotonic() - started,
    )
    return images


def build_messages(
    request: Request,
    indices: list[int],
    fps: float,
    instruction: str,
) -> list[dict[str, Any]]:
    duration = max(request.end_time - request.start_time, 0.0)
    interval = duration / (len(indices) - 1) if len(indices) > 1 else 0.0
    end_timestamp = seconds_to_timestamp(request.end_time)
    if len(indices) > 1:
        preamble = (
            f"The surgical video from its start (00:00:00) up to {end_timestamp} "
            f"is shown as {len(indices)} frames in chronological order, about "
            f"{interval:.0f} s apart, each labelled with its absolute video "
            "timestamp."
        )
    else:
        preamble = (
            f"The surgical video from its start (00:00:00) up to {end_timestamp} "
            "is shown as a single frame, each labelled with its absolute video "
            "timestamp."
        )

    content: list[dict[str, str]] = [{"type": "text", "text": preamble}]
    for index in indices:
        absolute_seconds = request.start_time + index / fps
        content.append(
            {
                "type": "text",
                "text": f"Frame at {seconds_to_timestamp(absolute_seconds)}:",
            }
        )
        content.append({"type": "image"})
    content.append({"type": "text", "text": instruction})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


class ProcedureModel:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required for this submission")

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
            self.processor.tokenizer.pad_token = (
                self.processor.tokenizer.eos_token
            )

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
    def generate(
        self,
        request: Request,
        video_path: Path,
        instruction: str,
        max_frames: int,
    ) -> tuple[str, int]:
        indices, fps, frame_count = sample_indices(video_path, max_frames)
        log.info(
            "qID=%s video=%d frames @ %.3f fps; selected=%d span=%.1fs",
            request.qID,
            frame_count,
            fps,
            len(indices),
            frame_count / fps,
        )
        images = decode_frames(video_path, indices)
        messages = build_messages(request, indices, fps, instruction)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
        ).to(self.device)
        del images

        generated = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
        prompt_length = inputs["input_ids"].shape[1]
        answer = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, len(indices)


def answer_with_oom_fallback(
    model: ProcedureModel,
    request: Request,
    video_path: Path,
    instruction: str,
) -> tuple[str, int]:
    frames = MAX_FRAMES
    while True:
        try:
            return model.generate(request, video_path, instruction, frames)
        except torch.cuda.OutOfMemoryError:
            if frames <= 1:
                raise
            frames = max(frames // 2, 1)
            torch.cuda.empty_cache()
            log.warning(
                "qID=%s CUDA OOM; retrying with at most %d frames",
                request.qID,
                frames,
            )


def run() -> int:
    log.info("ORena FOCUS Procedure inference starting")
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        raise ValueError("request.json contains no requests")
    definitions_path = INPUT_PATH / "FO_definitions.json"
    if not definitions_path.is_file():
        raise FileNotFoundError(definitions_path)

    allowed_seconds = 120.0 + 30.0 * len(requests)
    log.info(
        "batch=%d dtype=%s max_frames=%d interval=%.1fs max_pixels=%d "
        "pooled_budget=%.1fs",
        len(requests),
        MODEL_DTYPE_NAME,
        MAX_FRAMES,
        FRAME_INTERVAL_SECONDS,
        MAX_PIXELS,
        allowed_seconds,
    )

    formats: dict[int, str] = {}
    instructions: dict[int, str] = {}
    options: dict[int, list[str] | None] = {}
    multi_select: dict[int, bool] = {}
    videos: dict[int, Path] = {}
    errors: dict[int, str] = {}
    raw_answers: dict[int, str] = {}
    latencies: dict[int, float] = {}

    for index, request in enumerate(requests):
        answer_format = infer_answer_format(request.question)
        instruction, question_options = build_instruction(
            request.question, answer_format
        )
        formats[index] = answer_format
        instructions[index] = instruction
        options[index] = question_options
        multi_select[index] = is_multi_select(request.question)
        video_path = VIDEO_PATH / f"{request.qID}.mp4"
        videos[index] = video_path
        if not video_path.is_file() or video_path.stat().st_size == 0:
            errors[index] = f"missing or empty video: {video_path}"
            raw_answers[index] = ""
            latencies[index] = 0.0

    log.info("loading model once for the batch")
    model = ProcedureModel()
    log.info(
        "model ready in %.2fs on %s; allocated=%.2f GiB",
        time.monotonic() - PROCESS_START,
        torch.cuda.get_device_name(0),
        torch.cuda.memory_allocated(0) / 1024**3,
    )

    for index, request in enumerate(requests):
        if index in errors:
            log.error("qID=%s %s", request.qID, errors[index])
            continue
        elapsed = time.monotonic() - PROCESS_START
        if elapsed >= allowed_seconds - 5.0:
            errors[index] = "pooled latency safety stop"
            raw_answers[index] = ""
            latencies[index] = 0.0
            log.error(
                "qID=%s skipped to avoid pooled-budget overrun", request.qID
            )
            continue

        started = time.monotonic()
        try:
            raw, used_frames = answer_with_oom_fallback(
                model,
                request,
                videos[index],
                instructions[index],
            )
            raw_answers[index] = raw
            log.info(
                "qID=%s generated with %d frames raw=%r",
                request.qID,
                used_frames,
                raw[:120],
            )
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"
            raw_answers[index] = ""
            log.exception(
                "qID=%s failed; emitting an empty response", request.qID
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        latencies[index] = time.monotonic() - started

    responses: list[Response] = []
    for index, request in enumerate(requests):
        content = normalize_answer(
            raw_answers.get(index, ""),
            formats[index],
            options[index],
            multi=multi_select[index],
        )
        latency = float(latencies.get(index, 0.0))
        if not math.isfinite(latency) or latency < 0:
            latency = 0.0
        responses.append(
            Response(qID=request.qID, content=content, latency=latency)
        )
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
