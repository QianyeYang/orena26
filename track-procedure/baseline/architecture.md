# PROCEDURE baseline — zero-shot multi-frame VLM (interleaved timestamped images)

## Method

Each PROCEDURE question asks about the video **from the start** up to a moment
`T` (`[00:00:00, T]`, `T` from 00:05:09 to 04:56:20). The prefix is fed to an
off-the-shelf VLM (default **Qwen3-VL-4B-Instruct**) as an **interleaved
multi-image prompt**:

```
[system: PROCEDURE_SYSTEM_PROMPT]
The surgical video from its start (00:00:00) up to 02:40:06 is shown as 128
frames in chronological order, about 75 s apart, each labelled with its
absolute video timestamp.
Frame at 00:00:00: <image>
Frame at 00:01:15: <image>
...
<question + format-pinning instruction>
```

Why not the model's native video path: questions and answers use **absolute**
video time, while Qwen3-VL's video pipeline stamps clip-relative timestamps.
Text-labelling every frame keeps question/answer/frame time in one coordinate
system.

Pipeline (all shared code): `src.videovqa.run_video_qa` → frame selection on the
extraction grid (`src.frames.indices_for_request`) + even subsample →
`src.prompts` format-pinned instruction → greedy decode → `src.adapter`
normalisation → focus-compatible `responses.json` + rich
`predictions.parquet`. Scored by the unchanged
`track-frame/baseline/src/evaluate.py --track procedure` (Qwen3.5-4B judge).
PROCEDURE has two leaderboards: Technical (all rows) and Clinical
(`clinical_relevance=True`, 16 test rows) — both come out of the same eval.

## Sampling / token budget

| setting | value | note |
|---|---|---|
| extraction grid stride | 250 (= every 10 s @ 25 fps source) | `sbatch scripts/extract_frames.slurm procedure test 10 250` |
| max frames / question | 128 (even subsample, endpoints kept) | mean 125.8 frames/Q on test |
| max_pixels | 262144 | 960×540 → 672×384 → **252 tokens/frame** |
| mean prefill | ~33 K tokens | |
| observed latency (GH200, 4B, bf16 sdpa) | mean 2.63 s/Q (full test) | challenge budget 30 s/Q |

Effective frame spacing after subsampling long prefixes: ~40–130 s. Fallback if
latency/OOM: `--max-pixels 131072` (120 tok/frame) before cutting frames. OOM
at runtime auto-retries once with half the frames.

## Run

```bash
sbatch scripts/extract_frames.slurm procedure test 10 250          # once (repo root)
sbatch track-procedure/baseline/scripts/run_smoke.slurm Qwen3-VL-4B-Instruct 12
sbatch track-procedure/baseline/scripts/run_inference.slurm Qwen3-VL-4B-Instruct test
sbatch --dependency=afterok:<J> track-procedure/baseline/scripts/run_eval.slurm Qwen3-VL-4B-Instruct test
```

Inference is requeue-safe (appends to `predictions.jsonl`, resumes by qID;
`meta.json` guards against resuming with different settings). Coverage precheck:
`python -m src.videovqa --track procedure --split test --stride 250 --max-frames 128`.

## Scoring floors to keep in mind when reading numbers

- `percentage` (20 test Q): focus compares with `isclose(abs_tol=1e-9)` →
  effectively exact match; ≈0 expected zero-shot.
- `time` (208 test Q): accepted within ±5 s (threshold caps at 5 for long
  windows) — but the sampled frames are ~40–130 s apart, so time localisation
  ≈0 is a **sampling-resolution floor**, not a model failure.
- `fo_class`: the official parser accepts comma-separated sets; the adapter
  must preserve every recognized class.

## Results

See `result-summary.md` (PROCEDURE section) — filled by Phase 2/4 runs.
