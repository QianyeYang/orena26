# FRAME — LoRA fine-tuning

Config-driven LoRA SFT harness for VLMs ≤14B. First target: **Qwen2.5-VL-7B**
(beats the recorded zero-shot baseline of 0.305). Other ≤14B models are added by
dropping a new YAML into `src/configs/` — the trainer/runner are model-agnostic.

## Both-dataset official rerun

`configs/qwen3_vl_4b_both_official.yaml` starts a fresh adapter from the
original `Qwen3-VL-4B-Instruct` weights and trains on all official HeiCo and
LapChole train rows (13,748 total). Official test rows remain evaluation-only.
The B200 recipe keeps the effective batch at 16 with batch size 16, saves every
epoch, evaluates a seeded 1,000-row test subset separately by dataset, and
writes to `logs/Qwen3-VL-4B-Instruct-both-official/`.

## Method
- **Task:** multi-task SFT over *all* FRAME answer formats (fo_class, open_ended,
  number, multiple_choice, binary), matching the evaluation. One frame per question.
- **Prompt:** identical at train and inference time — `src.prompts.SYSTEM_PROMPT` +
  `src.prompts.build_instruction(question, answer_format)`. Target = the gold
  `answer` string.
- **Loss masking:** supervise only the assistant answer. The collator tokenises the
  prompt (system + user-with-image, generation prompt on) to find the boundary,
  then the full sequence; image tokens live entirely in the prefix, so the boundary
  transfers verbatim. Arch-agnostic — no per-model masking code.
- **Trainable scope:** LoRA (r=16, α=32, dropout=0.05) on the LLM (q/k/v/o_proj,
  gate/up/down_proj) **and** the vision encoder (qkv, proj, mlp), auto-discovered as
  every `nn.Linear` leaf except head/projector/embeddings. The multimodal projector
  (`merger`) is **fully trained** via PEFT `modules_to_save`. Surgical endoscopy is
  far out-of-domain visually, so the vision path is adapted, not just the LLM.
- **Precision/HW:** bf16, `sdpa` attention (no flash-attn on aarch64), gradient
  checkpointing, single GH200. bs=1 × grad-accum 16, lr 1e-4 cosine, 3 epochs
  (~750 optimiser steps), checkpoint per epoch. No quantization (96 GB is ample).

## Layout
```
src/
  dataset.py   FrameSFTDataset (parquet -> examples) + SFTCollator (label masking)
  train.py     load base, auto-discover LoRA targets, PEFT + HF Trainer, save adapter/epoch
  infer.py     LoRAVLM: base + PEFT adapter, same answer() contract as the sweep VLM
  runner.py    inference -> predictions.parquet + responses.json (+ requests/meta)
  configs/qwen2_5_vl_7b.yaml
scripts/
  train.slurm  stage base -> /dev/shm, train (reads base_path from the config)
  infer.slurm  stage base, run runner with a given adapter/checkpoint dir
  eval.slurm   reuse track-frame/baseline/src/evaluate.py (focus.Evaluator, judge Qwen3.5-4B)
logs/          <model>/checkpoint-*/ + <model>/final/ ; resp_<split>_<tag>/
```

## Reuse (no duplication)
Data `src.data` · frames `src.frames` · prompts `src.prompts` · normalisation
`src.adapter` · eval `track-frame/baseline/src/evaluate.py` · visualiser
`src.visualisation` (auto-discovers `predictions.parquet`).

## Run
Paths below are repo-relative (the scripts `cd` to the repo root and resolve them).
```bash
LF=track-frame/lora-finetune
# smoke (32 rows, 1 epoch) then a 16-sample inference of the resulting adapter
sbatch $LF/scripts/train.slurm $LF/src/configs/qwen2_5_vl_7b.yaml --limit 32 --epochs 1
sbatch $LF/scripts/infer.slurm $LF/logs/Qwen2.5-VL-7B-Instruct/final test --limit 16

# full: train, then eval each epoch checkpoint on the public test
sbatch $LF/scripts/train.slurm
sbatch $LF/scripts/infer.slurm $LF/logs/Qwen2.5-VL-7B-Instruct/checkpoint-750 test
sbatch $LF/scripts/eval.slurm  $LF/logs/resp_test_checkpoint-750 test
```

## Add a ≤14B model
Copy `configs/qwen2_5_vl_7b.yaml`, set `base_path` (and `trust_remote_code: true`
for InternVL), adjust `max_pixels`. `target_modules: null` auto-discovers; set
`modules_to_save` to that model's projector name (e.g. `multi_modal_projector` for
LLaVA-OneVision) if it differs from `merger`.

## Inference budget note
Test budget is 48 GB GPU / 5 s per question. Per-sample latency with the merged/
adapter model should stay well under 5 s at `max_pixels=602112` (zero-shot Qwen2.5-VL
was ~0.16 s); LoRA adds negligible overhead.
