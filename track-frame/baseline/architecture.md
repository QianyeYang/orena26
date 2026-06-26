# FRAME baseline — zero-shot Qwen2.5-VL

End-to-end, offline, model-agnostic baseline for the FRAME VQA track. One frame
per question, format-constrained output, scored by `focus.Evaluator`.

## Pipeline
```
parquet (data/) ──src.data──▶ Request + answer_format
                              │
   src.frames.request_frame_paths ──▶ frame{idx:07d}.jpg  (extracted by src.frames)
                              │
   prompts.build_instruction(question, answer_format)  ──┐
                              │                           ▼
                         PIL image + instruction ──▶ model.QwenVL.answer ──▶ raw text + latency
                              │
   src.adapter.build_response (normalise to the strict format) ──▶ responses.json
                              │
   focus.Evaluator (+ Qwen3.5-4B judge for open/MC) ──▶ results.csv + summary.csv
```

## Components (`src/`)
- `prompts.py` — per-format instruction + `SYSTEM_PROMPT`. Pins output shape:
  binary→`yes/no`, number→bare int, fo_class→one of 9 FO names or `none`,
  multiple_choice→copy one option verbatim, open_ended→terse.
- `model.py` — `QwenVL`: Qwen2.5-VL-7B-Instruct, single image, greedy decode,
  `sdpa` attention (flash-attn unavailable on aarch64), loads from local path.
- `runner.py` — drives parquet→frames→VLM→`adapter`→`responses.json`. Uses
  `answer_format` as **input metadata** (no reference answers needed → works on
  the hidden test). Logs parse-rate and s/Q.
- `evaluate.py` — `focus.Evaluator` wrapper; judge defaults to local
  `os-models/Qwen3.5-4B`.

## Run
```bash
# weights -> os-models/ (compute node)
sbatch scripts/download_models.slurm                 # repo-level
# frames -> data/focus/heico/frames/ (already done for frame train+test)
sbatch scripts/extract_frames.slurm                  # repo-level
# inference (GPU)
sbatch track-frame/baseline/scripts/run_inference.slurm test --limit 50   # smoke
sbatch track-frame/baseline/scripts/run_inference.slurm test              # full
# eval (GPU, needs judge weights)
sbatch track-frame/baseline/scripts/run_eval.slurm test
```
Outputs under `track-frame/baseline/logs/resp_<split>/`.

## Model / hardware
- VLM: `Qwen2.5-VL-7B-Instruct` (bf16, ~16 GB) — fits one GH200 (~96 GB) easily.
- Judge: `Qwen3.5-4B` on GPU. Both can co-reside on one GPU during train-time val.
- Budget at submission: 48 GB GPU, 5 s/Q, offline. Knob: `--max-pixels` caps image
  tokens to hold the 5 s/Q budget if needed.

## Known limitations / notes
- Local `test` = Sigmoid Resection only (train = Procto + Rectal): a generalisation
  probe, not in-distribution. Interpret the baseline number accordingly.
- Multi-label fo_class answers are unscorable by the stock format → single-label
  only (see `docs/issues-tbd.md`).
- FO class list = all 9 canonical names (`FOType.names()`), not just the 7 in train.
- Eval accuracy is a macro-mean over the 10 test videos with bootstrap CIs.

## Candidates to try next (keep updated)
| # | Model | Why |
|---|-------|-----|
| 1 | Qwen2.5-VL-7B-Instruct | START — speed/quality/VRAM fit |
| 2 | Qwen2.5-VL-32B-Instruct (AWQ) | higher ceiling, watch 5 s/Q |
| 3 | InternVL3-8B | medical-domain A/B |
