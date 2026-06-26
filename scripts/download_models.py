"""Download open-source VLMs / LLMs into ./os-models/<name> for offline use.

Snapshots each repo into a self-contained local dir (safetensors + config +
tokenizer/processor), so models load via ``from_pretrained("os-models/<name>")``
with no HF cache — portable into the offline submission Docker.

Usage:
    python scripts/download_models.py            # all models below
    python scripts/download_models.py Qwen2.5-VL-7B-Instruct   # one, by local name
"""

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

OS_MODELS = Path(__file__).resolve().parents[1] / "os-models"

# repo_id -> local folder name under os-models/
MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",  # baseline VLM
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",                          # focus default judge
    # --- zero-shot FRAME sweep candidates (see tmp/vlm_candidates.md) ---
    # Tier 1
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    # InternVL: use the -HF (native InternVLForConditionalGeneration) variants;
    # the plain repos are InternVLChatModel (trust_remote_code, custom .chat()).
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-14B-HF": "InternVL3_5-14B",
    # Tier 2 (ceiling)
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B-Instruct",  # bf16 sub for AWQ
    "OpenGVLab/InternVL3_5-30B-A3B-HF": "InternVL3_5-30B-A3B",
    # General VLM comparisons
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "SmolVLM2-2.2B-Instruct",
}

# skip duplicate / non-HF weight formats to save bandwidth & disk
IGNORE = ["*.bin", "*.pth", "*.gguf", "original/**", "*.onnx"]


def main(only: list[str] | None = None) -> None:
    OS_MODELS.mkdir(parents=True, exist_ok=True)
    for repo, name in MODELS.items():
        if only and name not in only and repo not in only:
            continue
        dest = OS_MODELS / name
        print(f"=== {repo} -> {dest} ===", flush=True)
        t = time.time()
        try:
            snapshot_download(repo_id=repo, local_dir=str(dest), ignore_patterns=IGNORE, max_workers=4)
            print(f"  OK in {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # one bad repo shouldn't abort the rest
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
