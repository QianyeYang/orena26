"""FRAME LoRA SFT dataset + label-masking collator.

Reuses the shared pipeline (`src.data` / `src.frames` / `src.prompts`) so the
train-time prompt is byte-identical to what inference sends. One image per FRAME
question; the supervision target is the gold ``answer`` string.

Loss is masked to the assistant answer only: we tokenise the prompt (system +
user-with-image, with the generation prompt) to find the boundary, then the full
sequence (prompt + answer). The image tokens live entirely in the prefix, so the
boundary transfers verbatim into the full sequence — no per-architecture
loss-masking logic, which keeps this collator valid across every VLM family.
"""

from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

from src import data, frames
from src import prompts as P


class FrameSFTDataset(Dataset):
    """One (image_path, instruction, answer) example per FRAME train row."""

    def __init__(
        self,
        track: str = "frame",
        split: str = "train",
        datasets: tuple[str, ...] | list[str] = ("heico",),
        frames_folder: str = "frames",
        limit: int | None = None,
    ) -> None:
        rows = data.read_parquets(track, split, datasets).to_dict("records")
        if limit:
            rows = rows[:limit]
        self.examples: list[dict] = []
        for row in rows:
            req = data.row_to_request(row)
            img = frames.request_frame_paths(
                req, frames_folder=frames_folder, dataset=row["_dataset"]
            )[0]
            instr, _ = P.build_instruction(req.question, row["answer_format"])
            self.examples.append(
                {
                    "dataset": row["_dataset"],
                    "image_path": str(img),
                    "instruction": instr,
                    "answer": str(row["answer"]),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


class SFTCollator:
    """Chat-format a batch and mask loss to the assistant answer (prompt masked)."""

    def __init__(self, processor, system_prompt: str = P.SYSTEM_PROMPT) -> None:
        self.processor = processor
        self.system_prompt = system_prompt
        tok = processor.tokenizer
        tok.padding_side = "right"  # masked-out pads go at the tail
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    def _messages(self, instruction: str, answer: str | None = None) -> list[dict]:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        ]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def __call__(self, examples: list[dict]) -> dict:
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        full_texts = [
            self.processor.apply_chat_template(
                self._messages(ex["instruction"], ex["answer"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in examples
        ]
        batch = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        for i, ex in enumerate(examples):
            prompt_text = self.processor.apply_chat_template(
                self._messages(ex["instruction"]), tokenize=False, add_generation_prompt=True
            )
            prompt_len = self.processor(
                text=[prompt_text], images=[images[i]], return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100  # mask system + question + image tokens
        labels[batch["attention_mask"] == 0] = -100  # mask padding
        batch["labels"] = labels
        return batch
