"""FRAME v2 SFT dataset: v2 prompts, thinking suppressed, counts rebalanced.

Three differences from `track-frame/lora-finetune/src/dataset.py`, each tied to a
measured defect (see `docs/frame-track-v2-plan.md`):

**Per-row system prompt.** The v2 strategy names the row's own
``procedure_type``, so the system prompt varies per example and cannot be hoisted
into the collator as a constant the way a fixed one can.

**Thinking suppressed at both ends.** Qwen3.6 opens a ``<think>`` block unless
the chat template is told otherwise. Training on a template that differs from the
inference template by a prefilled reasoning block would put every answer token at
a different position than it will occupy at test time, so the flag is threaded
through and must match `track-frame/v2/src/generate.py`.

**Count rebalancing.** Counting answers are distributed 669/597/471/368/248/142/
63/45/14/7/4 over counts 1-11, and the fine-tuned model inherits that prior: it
predicted 6 or more just 23 times in 2,094 answers while ground truth reached 12,
scoring 0.011 on the 202 rows with a count of 6+. Replicating high-count rows by
inverse square-root frequency flattens the prior without the 167x oversampling
that full inverse-frequency would demand. Off by default so it stays an ablation.
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import re

from PIL import Image
from torch.utils.data import Dataset

from src import data, frames
from src import prompts as P

log = logging.getLogger(__name__)

_COUNT_ANSWER = re.compile(r"^\d+$")


class FrameSFTDatasetV2(Dataset):
    """One (image, system, instruction, answer) example per FRAME train row."""

    def __init__(
        self,
        track: str = "frame",
        split: str = "train",
        datasets: tuple[str, ...] | list[str] = ("heico", "lapchole"),
        frames_folder: str = "frames",
        limit: int | None = None,
        *,
        prompt_strategy: str = "v2",
        count_balance_power: float = 0.0,
        count_balance_cap: int = 4,
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
            instr, _ = P.build_instruction(req.question, row["answer_format"], prompt_strategy)
            if prompt_strategy in P.V2_FAMILY:
                system = P.system_prompt(track, prompt_strategy, procedure_type=req.procedure_type)
            else:
                system = P.system_prompt(track, prompt_strategy)
            self.examples.append(
                {
                    "dataset": row["_dataset"],
                    "image_path": str(img),
                    "system": system,
                    "instruction": instr,
                    "answer": str(row["answer"]),
                    "answer_format": row["answer_format"],
                }
            )

        self.index = list(range(len(self.examples)))
        if count_balance_power > 0:
            self.index = self._balanced_index(count_balance_power, count_balance_cap)

    def _balanced_index(self, power: float, cap: int) -> list[int]:
        """Replicate counting examples so rare counts are seen more often.

        Only ``number`` rows are touched — replicating an fo_class row would
        change the class prior, which is not the distribution being corrected.
        """
        counts = [
            int(ex["answer"])
            for ex in self.examples
            if ex["answer_format"] == "number" and _COUNT_ANSWER.match(ex["answer"])
        ]
        if not counts:
            return list(range(len(self.examples)))
        freq = Counter(counts)
        top = max(freq.values())

        index: list[int] = []
        replicated = Counter()
        for i, ex in enumerate(self.examples):
            reps = 1
            if ex["answer_format"] == "number" and _COUNT_ANSWER.match(ex["answer"]):
                c = int(ex["answer"])
                reps = min(cap, max(1, round((top / freq[c]) ** power)))
                replicated[c] += reps
            index.extend([i] * reps)
        log.info(
            "count balancing (power=%.2f cap=%d): %d -> %d examples; per-count totals %s",
            power, cap, len(self.examples), len(index), dict(sorted(replicated.items())),
        )
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        return self.examples[self.index[i]]


class SFTCollatorV2:
    """Chat-format a batch and mask loss to the assistant answer.

    The prompt boundary is found by tokenising the prompt alone and reusing its
    length in the full sequence: image tokens live entirely in the prefix, so the
    boundary transfers verbatim and no per-architecture masking logic is needed.
    """

    def __init__(self, processor, *, enable_thinking: bool = False) -> None:
        self.processor = processor
        self.enable_thinking = enable_thinking
        tok = processor.tokenizer
        tok.padding_side = "right"  # masked-out pads go at the tail
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    def _messages(self, ex: dict, with_answer: bool) -> list[dict]:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": ex["system"]}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": ex["instruction"]}],
            },
        ]
        if with_answer:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}
            )
        return msgs

    def _render(self, ex: dict, with_answer: bool) -> str:
        # generate.apply_template must stay byte-identical for the prompt half.
        kwargs = {"tokenize": False, "add_generation_prompt": not with_answer}
        try:
            return self.processor.apply_chat_template(
                self._messages(ex, with_answer),
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.processor.apply_chat_template(self._messages(ex, with_answer), **kwargs)

    def __call__(self, examples: list[dict]) -> dict:
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        full_texts = [self._render(ex, True) for ex in examples]
        batch = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        for i, ex in enumerate(examples):
            prompt_len = self.processor(
                text=[self._render(ex, False)], images=[images[i]], return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100  # mask system + question + image tokens
        labels[batch["attention_mask"] == 0] = -100  # mask padding
        batch["labels"] = labels
        return batch


def count_distribution(ds: FrameSFTDatasetV2) -> dict[int, int]:
    """Realised distribution over counting answers after any rebalancing."""
    out = Counter()
    for i in ds.index:
        ex = ds.examples[i]
        if ex["answer_format"] == "number" and _COUNT_ANSWER.match(ex["answer"]):
            out[int(ex["answer"])] += 1
    return dict(sorted(out.items()))
