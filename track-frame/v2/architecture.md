# FRAME v2

Rebuild of the FRAME track aimed at the leaderboard metric rather than at
per-video macro accuracy. Plan and measurements: `docs/frame-track-v2-plan.md`.
Forked from `track-frame/lora-finetune` on 2026-08-10; the unified run continues
untouched on the a100 box.

## Why a fork

Three things change at once, and none of them is a tweak to the existing method:

1. **The target metric changes.** The leaderboard averages four buckets
   (object_recognition and aggregation × in-/out-of-distribution) with equal
   weight. `lora-finetune` optimises and early-stops on per-video macro accuracy
   over in-distribution test rows. Those rank differently: half the real score
   comes from 606 out-of-distribution questions we never measure, and one
   aggregation-OOD question is worth eight object-recognition-ID questions.
2. **The prompts change.** v2 fixes a procedure-type misstatement, fo_class
   arity, and an open_ended contradiction. A v1-trained adapter must keep being
   evaluated with v1 prompts, so the two cannot share one evaluation path.
3. **The base model is under review.** Stage 1 benchmarks Qwen3.6-27B against
   Qwen3-VL-8B and the shipped 4B before anything is trained.

## Layout

    src/benchmark.py   any VLM (+ optional LoRA adapter) over both datasets,
                       per-row v2 system prompts, rich predictions.parquet
    src/score.py       focus.Evaluator + the four-bucket leaderboard view and
                       the counting range-compression diagnostic
    scripts/benchmark.slurm   one h200 single-GPU node: infer then score
    logs/<label>/      per-run outputs; logs/<label>/eval/leaderboard.md

Shared code stays in the repo-level `src/` — `prompts.build_instruction`,
`adapter.build_response`, `data.row_to_request`, `frames.request_frame_paths`
are all reused verbatim, so train- and inference-time prompts cannot drift.

## Evaluation contract

Locally every row carries `ood=False`, so the four-bucket metric has no
distribution axis to split on. `score.py` substitutes **dataset** for that axis:
HeiCo and LapChole are scored as separate halves and the four cells averaged.
This mirrors the real metric's shape, and for a model trained on one dataset and
scored on the other it mirrors its meaning too. The stock focus
`pre_evaluation_score` is still written to `summary.csv` for continuity.

`open_ended` and `multiple_choice` are graded by an LLM judge (Qwen3.5-4B), not
by exact match, so scoring needs a GPU. The judge is told to accept a correct
answer that carries extra text — but `OpenEnded.read` rejects anything over 300
characters first, so length is a hard constraint and verbosity is not.

## Hardware

`h200` is eight **single-GPU** nodes at 141 GB, so a 27B runs in bf16 without
sharding. The a100 box (8×A100-80GB) is held by the unified run. Deployment
target is different again — 1× L40S 48 GB with a pooled budget of 120 s setup
plus 5 s per question for a 20-question batch — so a 27B ships quantised
(FP8 ≈ 27 GB, AWQ-INT4 ≈ 15 GB) and model load must be measured there early.
