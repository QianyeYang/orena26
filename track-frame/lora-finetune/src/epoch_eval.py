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

import json
import logging
import time
from pathlib import Path

import numpy as np
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
        datasets: tuple[str, ...] | list[str] = ("heico",),
        eval_n: int | None = None,
        seed: int = 42,
        model_name: str = "lora",
    ) -> None:
        self.processor = processor
        self.out = Path(out_dir)
        self.judge_model = judge_model
        self.frames_folder = frames_folder
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.datasets = tuple(datasets)

        candidates: list[tuple[str, dict]] = []
        for dataset in self.datasets:
            candidates.extend(
                (dataset, row)
                for row in data.read_parquet(track, split, dataset).to_dict("records")
            )
        if eval_n and eval_n < len(candidates):
            rng = np.random.default_rng(seed)
            keep = sorted(rng.choice(len(candidates), size=eval_n, replace=False).tolist())
            candidates = [candidates[i] for i in keep]

        self.groups: dict[str, dict] = {}
        subset_ids = []
        for dataset in self.datasets:
            rows = [row for row_dataset, row in candidates if row_dataset == dataset]
            reqs = [data.row_to_request(row) for row in rows]
            self.groups[dataset] = {
                "rows": rows,
                "reqs": reqs,
                "refs": [data.row_to_reference(row) for row in rows],
            }
            subset_ids.extend({"dataset": dataset, "qID": req.qID} for req in reqs)
        self._evaluator = None  # lazy: load judge only on first eval
        self.metrics_csv = self.out / "epoch_metrics.csv"
        (self.out / "eval_subset_qids.json").write_text(json.dumps(subset_ids, indent=2))
        counts = {dataset: len(group["rows"]) for dataset, group in self.groups.items()}
        log.info(
            "per-epoch eval armed: %s %s/%s rows (seed=%d), judge=%s",
            counts, track, split, seed, judge_model,
        )

    def _evaluator_lazy(self):
        if self._evaluator is None:
            from focus import Evaluator
            self._evaluator = Evaluator(
                judge_kwargs={"model_name": self.judge_model, "device": "cuda"}
            )
        return self._evaluator

    def _infer(self, model, dataset: str, group: dict) -> tuple[list, list]:
        from infer import generate_answer

        device = next(model.parameters()).device
        records, responses = [], []
        n_err = 0
        t0 = time.time()
        for i, (req, row) in enumerate(zip(group["reqs"], group["rows"])):
            fmt = row["answer_format"]
            img = frames.request_frame_paths(
                req, frames_folder=self.frames_folder, dataset=dataset
            )[0]
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
                "dataset": dataset, "sample_id": req.qID,
                "model_name": self.model_name, "question": req.question,
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
                log.info(
                    "  %s infer %d/%d fmt=%s raw=%r",
                    dataset, i + 1, len(group["reqs"]), fmt, raw[:32],
                )
        log.info(
            "  %s inference done: %d rows %.1fs errors=%d",
            dataset, len(records), time.time() - t0, n_err,
        )
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

            eval_outputs = {}
            total_rows = sum(len(group["rows"]) for group in self.groups.values())
            log.info("[epoch %d] test-subset inference (n=%d)...", epoch, total_rows)
            for dataset, group in self.groups.items():
                eval_outputs[dataset] = self._infer(model, dataset, group)

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
            metrics = {}
            for dataset, group in self.groups.items():
                records, responses = eval_outputs[dataset]
                dataset_dir = epdir / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)
                save_items(responses, dataset_dir / "responses.json")
                pd.DataFrame(records).to_parquet(
                    dataset_dir / "predictions.parquet", index=False
                )

                accuracy = float("nan")
                try:
                    ev = self._evaluator_lazy()
                    _, sum_df = ev.run(
                        group["reqs"], group["refs"], responses,
                        output_dir=str(dataset_dir / "eval"),
                    )
                    ov = sum_df[sum_df.level == "overall"]
                    if not ov.empty:
                        accuracy = float(ov.iloc[0]["accuracy"])
                    log.info(
                        "[epoch %d] %s acc = %.4f -> %s",
                        epoch, dataset, accuracy, dataset_dir,
                    )
                except Exception as e:  # noqa: BLE001 — retain responses for offline scoring
                    log.warning(
                        "[epoch %d] %s judge eval failed: %s (responses saved)",
                        epoch, dataset, e,
                    )
                metrics[dataset] = accuracy

            valid = [value for value in metrics.values() if np.isfinite(value)]
            overall = float(np.mean(valid)) if valid else float("nan")
            log.info("[epoch %d] mean dataset accuracy = %.4f", epoch, overall)

            header = not self.metrics_csv.exists()
            with open(self.metrics_csv, "a") as f:
                if header:
                    f.write(
                        "epoch,overall_acc,"
                        + ",".join(f"{dataset}_acc" for dataset in self.datasets)
                        + ",global_step\n"
                    )
                values = ",".join(str(metrics.get(dataset, float("nan")))
                                  for dataset in self.datasets)
                f.write(f"{epoch},{overall},{values},{state.global_step}\n")
        except Exception as e:  # noqa: BLE001 — never let eval kill training
            log.warning("[epoch %d] per-epoch eval errored (training continues): %s", epoch, e)
            try:  # best effort to leave the model trainable
                model.train()
                model.config.use_cache = False
            except Exception:  # noqa: BLE001
                pass
