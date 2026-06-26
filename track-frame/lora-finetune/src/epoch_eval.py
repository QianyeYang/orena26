"""Per-epoch in-loop evaluation callback.

After every training epoch this runs the *current* model over the public test
split (inference) and scores it with ``focus.Evaluator`` (LLM judge for
open_ended / multiple_choice) — all inside the training job, so no separate
inference/eval task ever needs to be queued. Per epoch it writes
``eval_epoch_<N>/{responses.json,predictions.parquet,eval/}`` and appends the
overall accuracy to ``epoch_metrics.csv``.

The whole body is guarded: an eval failure logs a warning and training continues
(the epoch checkpoint is already saved, so it can always be scored offline).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import torch
from focus import save_items
from transformers import TrainerCallback

from src import adapter, data, frames
from src import prompts as P

log = logging.getLogger("epoch-eval")


def _sec_caps(value) -> str:
    if value is None:
        return ""
    try:
        return "|".join(str(v) for v in list(value))
    except TypeError:
        return str(value)


class PerEpochEvalCallback(TrainerCallback):
    """Inference + judge scoring of the in-memory model at each epoch end."""

    def __init__(
        self,
        processor,
        out_dir: str,
        judge_model: str,
        track: str = "frame",
        split: str = "test",
        frames_folder: str = "frames",
        system_prompt: str = P.SYSTEM_PROMPT,
        max_new_tokens: int = 64,
        eval_limit: int | None = None,
        model_name: str = "lora",
    ) -> None:
        self.processor = processor
        self.out = Path(out_dir)
        self.judge_model = judge_model
        self.frames_folder = frames_folder
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        rows = data.read_parquet(track, split).to_dict("records")
        if eval_limit:
            rows = rows[:eval_limit]
        self.rows = rows
        self.reqs = [data.row_to_request(r) for r in rows]
        self.refs = [data.row_to_reference(r) for r in rows]
        self._evaluator = None  # lazy: load judge only on first eval
        self.metrics_csv = self.out / "epoch_metrics.csv"
        log.info("per-epoch eval armed: %d %s/%s rows, judge=%s",
                 len(self.rows), track, split, judge_model)

    def _evaluator_lazy(self):
        if self._evaluator is None:
            from focus import Evaluator
            self._evaluator = Evaluator(
                judge_kwargs={"model_name": self.judge_model, "device": "cuda"}
            )
        return self._evaluator

    def _infer(self, model) -> list:
        from infer import generate_answer

        device = next(model.parameters()).device
        records, responses = [], []
        n_err = 0
        t0 = time.time()
        for i, (req, row) in enumerate(zip(self.reqs, self.rows)):
            fmt = row["answer_format"]
            img = frames.request_frame_paths(req, frames_folder=self.frames_folder)[0]
            instr, opts = P.build_instruction(req.question, fmt)
            raw, lat, err = "", 0.0, ""
            if not img.exists():
                err = f"missing frame: {img}"
            else:
                try:
                    raw, lat = generate_answer(
                        model, self.processor, device, str(img), instr,
                        self.system_prompt, self.max_new_tokens,
                    )
                except Exception as e:  # noqa: BLE001 — one bad sample mustn't stop eval
                    err = f"{type(e).__name__}: {e}"
            resp = adapter.build_response(req.qID, raw, fmt, latency=lat, options=opts)
            responses.append(resp)
            n_err += int(bool(err))
            records.append({
                "sample_id": req.qID, "model_name": self.model_name, "question": req.question,
                "answer": str(row["answer"]), "answer_format": fmt,
                "primary_capability": str(row["primary_capability"]),
                "secondary_capabilities": _sec_caps(row.get("secondary_capabilities")),
                "clinical_relevance": bool(row["clinical_relevance"]), "ood": bool(row["ood"]),
                "video": row["video"], "timestamp_start": str(row["timestamp_start"]),
                "timestamp_end": str(row["timestamp_end"]), "image_path": str(img),
                "prompt": instr, "raw_model_output": raw, "normalized_prediction": resp.content,
                "prediction": resp.content, "latency_sec": lat, "error": err,
            })
            if i % 200 == 0:
                log.info("  infer %d/%d fmt=%s raw=%r", i + 1, len(self.reqs), fmt, raw[:32])
        log.info("  inference done: %d rows %.1fs errors=%d", len(records), time.time() - t0, n_err)
        return records, responses

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        epoch = int(round(state.epoch))
        try:
            was_training = model.training
            prev_cache = getattr(model.config, "use_cache", None)
            gc_was = getattr(model, "is_gradient_checkpointing", False)
            model.eval()
            try:
                model.gradient_checkpointing_disable()  # KV cache needs this off
            except Exception:  # noqa: BLE001
                pass
            try:
                model.config.use_cache = True
            except Exception:  # noqa: BLE001
                pass

            log.info("[epoch %d] test inference (n=%d)...", epoch, len(self.reqs))
            records, responses = self._infer(model)

            # restore training state
            try:
                model.config.use_cache = prev_cache
            except Exception:  # noqa: BLE001
                pass
            if gc_was:
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except Exception:  # noqa: BLE001
                    pass
            if was_training:
                model.train()

            epdir = self.out / f"eval_epoch_{epoch}"
            epdir.mkdir(parents=True, exist_ok=True)
            save_items(responses, epdir / "responses.json")
            pd.DataFrame(records).to_parquet(epdir / "predictions.parquet", index=False)

            overall = float("nan")
            try:
                ev = self._evaluator_lazy()
                _, sum_df = ev.run(self.reqs, self.refs, responses, output_dir=str(epdir / "eval"))
                ov = sum_df[sum_df.level == "overall"]
                if not ov.empty:
                    overall = float(ov.iloc[0]["accuracy"])
                log.info("[epoch %d] OVERALL acc = %.4f -> %s", epoch, overall, epdir)
            except Exception as e:  # noqa: BLE001 — keep responses for offline scoring
                log.warning("[epoch %d] judge eval failed: %s (responses saved)", epoch, e)

            header = not self.metrics_csv.exists()
            with open(self.metrics_csv, "a") as f:
                if header:
                    f.write("epoch,overall_acc,global_step\n")
                f.write(f"{epoch},{overall},{state.global_step}\n")
        except Exception as e:  # noqa: BLE001 — never let eval kill training
            log.warning("[epoch %d] per-epoch eval errored (training continues): %s", epoch, e)
            try:  # best effort to leave the model trainable
                model.train()
                model.config.use_cache = False
            except Exception:  # noqa: BLE001
                pass
