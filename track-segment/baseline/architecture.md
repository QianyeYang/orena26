# SEGMENT baseline — zero-shot multi-frame VLM (interleaved timestamped images)

## Method

Each SEGMENT question asks about a clip `[timestamp_start, timestamp_end]`
(~10 s / 30 s / 2 min / 5 min windows). The clip is fed to an off-the-shelf VLM
(default **Qwen3-VL-4B-Instruct**) as an **interleaved multi-image prompt**:

```
[system: SEGMENT_SYSTEM_PROMPT]
This clip covers video time 01:35:40 to 01:40:39 (299 s of surgery). It is shown
as 64 frames in chronological order, about 5 s apart, each labelled with its
absolute video timestamp.
Frame at 01:35:40: <image>
Frame at 01:35:45: <image>
...
<question + format-pinning instruction>
```

Why not the model's native video path: questions and answers use **absolute**
video time ("… at 02:40:06?"), while Qwen3-VL's video pipeline stamps
clip-relative timestamps. Text-labelling every frame with its true video time
keeps question/answer/frame time in one coordinate system.

Pipeline (all shared code): `src.videovqa.run_video_qa` → frame selection on the
extraction grid (`src.frames.indices_for_request`) + even subsample →
`src.prompts` format-pinned instruction (incl. the multi-select MC variant) →
greedy decode → `src.adapter` normalisation → focus-compatible
`responses.json` + rich `predictions.parquet`. Scored by the unchanged
`track-frame/baseline/src/evaluate.py --track segment` (Qwen3.5-4B judge for
open_ended/multiple_choice/matching).

## Sampling / token budget

| setting | value | note |
|---|---|---|
| extraction grid stride | 25 (= 1 fps @ 25 fps source) | `sbatch scripts/extract_frames.slurm segment test 10 25` |
| max frames / question | 64 (even subsample, endpoints kept) | mean 49.8 frames/Q on test |
| max_pixels | 262144 | 960×540 → 672×384 → **252 tokens/frame** |
| mean prefill | ~13 K tokens | |
| observed latency (GH200, 4B, bf16 sdpa) | mean 0.86 s/Q (full test) | challenge budget 15 s/Q |

Fallback if latency/OOM: `--max-pixels 131072` (120 tok/frame) before cutting
frames. OOM at runtime auto-retries once with half the frames.

## Run

```bash
sbatch scripts/extract_frames.slurm segment test 10 25            # once (repo root)
sbatch track-segment/baseline/scripts/run_smoke.slurm Qwen3-VL-4B-Instruct 12
sbatch track-segment/baseline/scripts/run_inference.slurm Qwen3-VL-4B-Instruct test
sbatch --dependency=afterok:<J> track-segment/baseline/scripts/run_eval.slurm Qwen3-VL-4B-Instruct test
```

Inference is requeue-safe (appends to `predictions.jsonl`, resumes by qID;
`meta.json` guards against resuming with different settings). A coverage
precheck stats every sampled frame path before the model loads:
`python -m src.videovqa --track segment --split test --stride 25 --max-frames 64`.

## Scoring floors to keep in mind when reading numbers

- `percentage` (42 test Q): focus compares with `isclose(abs_tol=1e-9)` →
  effectively exact match; ≈0 is expected zero-shot, not a bug.
- `time` (371 test Q): accepted within ±`min(5, 1 + dur·4/360)` s → ±1.1–4.3 s
  for segment windows; at a 1-fps sampling grid this is genuinely hard.
- `fo_class`: the official parser accepts comma-separated sets; the adapter
  must preserve every recognized class.
- `multiple_choice` (173 test Q): 21 are multi-select ("select none, one or
  multiple answers"); prompts + `src.adapter.normalize_text(multi=True)` handle
  them (comma+space joined in listed-option order, the GT convention).

## Results

See `result-summary.md` (SEGMENT section) — filled by Phase 2/4 runs.
