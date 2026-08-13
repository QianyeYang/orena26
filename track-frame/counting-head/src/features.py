#!/usr/bin/env python
"""Extract one hidden-state vector per FRAME counting question, plus the
model's own generated answer for a paired baseline.

Both come from a single ``generate`` call: ``hidden_states[0]`` is the prompt
forward pass, so its last layer at the final position is exactly the state the
model conditions on when it emits the first answer token. Taking the baseline
from the same call rather than a separate benchmark run guarantees the head and
the number it must beat were produced by identical weights on identical inputs.

The base model is never modified. Features are written once and the head is then
trained offline in seconds, so hyperparameters can be swept without touching a
GPU again — and a frozen backbone answers the question that matters first: is
the count information already present in the representation?

Writes ``features.npz`` (hidden states, template ids, targets, baselines) and
``rows.parquet`` (everything else, for slicing the results afterwards).
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

from src import data, frames  # noqa: E402
from src import prompts as P  # noqa: E402

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "track-frame" / "lora-finetune" / "src"))

from templates import template_id, template_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("features")


def parse_int(text) -> int | None:
    """First integer in *text*, or ``None``. Mirrors the scorer's leniency."""
    import re

    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None


@torch.inference_mode()
def run(model, processor, device, image_path: str, instruction: str,
        system: str | None, max_new_tokens: int) -> tuple[np.ndarray, str]:
    """Return (last-layer final-prompt-position hidden state, generated text)."""
    from PIL import Image

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    # hidden_states[0] is the prompt pass: tuple over layers, each (B, L, D).
    prompt_last_layer = out.hidden_states[0][-1]
    vec = prompt_last_layer[0, -1, :].float().cpu().numpy()

    trimmed = out.sequences[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return vec, answer


def main() -> None:
    ap = argparse.ArgumentParser(description="counting-head feature extraction")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"])
    ap.add_argument("--frames-folder", default="frames")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per split (smoke)")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict] = []
    for split in a.splits:
        df = data.read_parquets("frame", split, a.datasets)
        df = df[df["answer_format"] == "number"].copy()
        df["_split"] = split
        if a.limit:
            df = df.head(a.limit)
        frame_rows.extend(df.to_dict("records"))
    log.info("counting rows: %d over splits=%s", len(frame_rows), a.splits)

    from infer import LoRAVLM  # noqa: E402

    t0 = time.time()
    vlm = LoRAVLM(
        a.model, adapter_path=a.adapter, dtype=a.dtype,
        max_new_tokens=a.max_new_tokens, min_pixels=a.min_pixels,
        max_pixels=a.max_pixels, attn_implementation=a.attn,
    )
    log.info("model loaded in %.1fs", time.time() - t0)

    system = P.system_prompt("frame", "direct")

    vecs: list[np.ndarray] = []
    keep: list[dict] = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(frame_rows):
        tid = template_id(row["question"])
        target = parse_int(row["answer"])
        if tid is None or target is None:
            skipped += 1
            continue
        req = data.row_to_request(row)
        img = frames.request_frame_paths(
            req, frames_folder=a.frames_folder, dataset=row["_dataset"]
        )[0]
        if not img.exists():
            skipped += 1
            log.warning("missing frame: %s", img)
            continue

        instruction, _ = P.build_instruction(req.question, "number", "direct")
        try:
            vec, answer = run(
                vlm.model, vlm.processor, vlm.device, str(img), instruction,
                system, a.max_new_tokens,
            )
        except Exception as e:  # one bad row must not lose the whole extraction
            skipped += 1
            log.warning("qID=%s failed: %s: %s", row["id"], type(e).__name__, e)
            continue

        vecs.append(vec)
        keep.append(
            {
                "qID": f"{row['_dataset']}:{row['id']}",
                "dataset": row["_dataset"],
                "split": row["_split"],
                "video": row["video"],
                "procedure_type": row["procedure_type"],
                "ood": bool(row["ood"]),
                "question": row["question"],
                "template_id": tid,
                "template": template_name(tid),
                "target": target,
                "baseline_text": answer,
                "baseline": parse_int(answer),
            }
        )
        if i % 250 == 0:
            log.info("%d/%d t=%s target=%d baseline=%r (%.1fs elapsed)",
                     i + 1, len(frame_rows), template_name(tid), target,
                     answer[:20], time.time() - t0)

    X = np.stack(vecs).astype(np.float32)
    meta = pd.DataFrame(keep)
    np.savez_compressed(
        out / "features.npz",
        X=X,
        template_id=meta["template_id"].to_numpy(np.int64),
        target=meta["target"].to_numpy(np.int64),
        baseline=meta["baseline"].fillna(-1).to_numpy(np.int64),
        is_test=(meta["split"] == "test").to_numpy(),
    )
    meta.to_parquet(out / "rows.parquet", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(a).items()},
                "n_rows": int(len(meta)),
                "n_skipped": skipped,
                "hidden_dim": int(X.shape[1]),
                "extract_sec": round(time.time() - t0, 1),
                "template_counts": meta["template"].value_counts().to_dict(),
                "split_counts": meta["split"].value_counts().to_dict(),
            },
            indent=2,
        )
    )
    log.info("wrote %d rows, dim=%d, skipped=%d -> %s", len(meta), X.shape[1], skipped, out)


if __name__ == "__main__":
    main()
