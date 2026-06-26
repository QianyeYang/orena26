# FRAME zero-shot sweep

Zero-shot evaluation of several open-weight VLMs on the FRAME track, preserving
**raw model output + normalized prediction + full metadata** per sample (the
`baseline/` method discarded raw output, blocking error analysis). No fine-tuning.

## Why a separate method folder
`baseline/` is Qwen2.5-VL-specific and its `logs/` hold the recorded 0.305 result.
This method is a generic multi-family wrapper, so it lives apart (per CLAUDE.md)
but reuses the shared `src/` (`data`, `frames`, `adapter`, `prompts`, `paths`)
and the baseline's `evaluate.py` unchanged.

## Pipeline
parquet row -> frame (`src.frames`) -> format-aware prompt (`src.prompts`) ->
`VLM.answer` (greedy) -> normalize (`src.adapter`) -> rich rows.

- `src/vlm.py` — one `VLM` class over `AutoModelForImageTextToText` +
  `AutoProcessor`. Same `answer(image, instr, system) -> (raw, latency)` contract
  as the baseline `QwenVL`. `sdpa` attention (no flash-attn on aarch64), bf16,
  greedy, loads from local `os-models/` (offline). Verified to cover Qwen2.5-VL,
  Qwen3-VL (dense + MoE), InternVL3.5, LLaVA-OneVision, SmolVLM2 in transformers 5.12.
- `src/runner.py` — orchestration. Per model writes under `--out`:
  - `predictions.parquet` — rich row per sample (raw never overwritten; see fields below)
  - `responses.json` — focus `Response` (qID, content=normalized, latency) for `evaluate.py`
  - `requests.json`, `meta.json` (includes the system prompt)
  - inference errors are caught per sample into the `error` column (sweep never aborts)
- `src/analyze.py` — cross-model comparison (overall / by answer_format / by
  primary_capability accuracy, invalid-format rate, latency, error & normalization cases).

### predictions.parquet columns
`sample_id, model_name, question, answer, answer_format, primary_capability,
secondary_capabilities, clinical_relevance, ood, video, timestamp_start,
timestamp_end, image_path, prompt, raw_model_output, normalized_prediction,
prediction, latency_sec, error`

## Models (see `tmp/vlm_candidates.md`; MiniCPM-V skipped, AWQ -> bf16 32B)
Tier 1: Qwen2.5-VL-7B (rerun), Qwen3-VL-4B, Qwen3-VL-8B, InternVL3_5-8B, InternVL3_5-14B.
Tier 2: Qwen3-VL-30B-A3B, Qwen3-VL-32B, Qwen2.5-VL-32B (bf16), InternVL3_5-30B-A3B.
General: LLaVA-OneVision-7B, SmolVLM2-2.2B.

## Run
```bash
# download (compute node, batched by tier)
sbatch scripts/download_models.slurm <local-name> ...
# smoke (8 samples)
sbatch track-frame/zeroshot-sweep/scripts/run_smoke.slurm SmolVLM2-2.2B-Instruct
# full inference (test split, 2000) then eval with the Qwen3.5-4B judge
sbatch track-frame/zeroshot-sweep/scripts/run_inference.slurm <local-name> test
```

## Notes / risks
- InternVL3.5 native loading is assumed; if a checkpoint needs custom code, pass
  `--trust-remote-code` (smoke test surfaces this).
- 30B-A3B (MoE) and 32B (dense) bf16 fit one 96GB GH200; `device_map="auto"`.
