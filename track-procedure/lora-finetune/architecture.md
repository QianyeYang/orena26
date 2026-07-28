# PROCEDURE LoRA SFT — Qwen3-VL-4B on interleaved timestamped frames

## Both-dataset official rerun

`src/configs/qwen3_vl_4b_both_official.yaml` starts from the original
`Qwen3-VL-4B-Instruct` weights and trains on all 6,873 official HeiCo and
LapChole train rows; official test rows are evaluation-only. Sampling is one
frame per ten source seconds (stride 250 for HeiCo, 300 for LapChole), capped
at 96 frames, using official-compatible Decord/OpenCV JPEG extraction. The
B200 recipe uses batch size 2 × accumulation 8 with automatic OOM fallback,
saves every epoch, and writes to
`logs/Qwen3-VL-4B-Instruct-both-official/`.

## Method

LoRA SFT of **Qwen3-VL-4B-Instruct** on the 2000 PROCEDURE train questions,
using the exact frame-track winning recipe (which took FRAME 0.322 → **0.828**):

- LoRA r=16, α=32, dropout 0.05 on every `nn.Linear` leaf of the LLM + vision
  tower (auto-discovered; `lm_head`/`embed_tokens` excluded); the multimodal
  projector (`merger`) is fully trained via `modules_to_save`.
- Effective batch 16 (bs 1 × grad-accum 16), lr 1e-4 cosine, warmup 0.03,
  bf16, gradient checkpointing, seed 42, save per epoch, auto-resume across
  24 h jobs. 8-epoch target (125 steps/epoch).
- Supervision: gold `answer` verbatim, including multi-label `fo_class` rows
  accepted by the official set parser; loss masked to the assistant answer only.

**Prompt parity by construction, reduced sampling**:
`src.videovqa_sft.VideoSFTDataset` builds every conversation with the same
`select_frames` / `build_preamble` / `build_messages` / `build_instruction` the
zero-shot baseline runner uses. Sampling is **reduced vs the baseline defaults**
(stride 250, **96 frames, max_pixels 131072** → 120 tok/frame, ~11.8 K prefill
tokens/sample): training at the baseline's 128 @ 262144 (~33 K tokens) OOMs in
backward on a 96 GB GH200 — the CE loss materialises full logits (33 K × 151 K
vocab ≈ 10 GB bf16) plus their gradient on top of activations. SFT inference
(`scripts/infer.slurm`) runs at these training settings, and the SFT delta is
measured against a **zero-shot rerun at the same settings**
(`baseline/logs/resp_test/Qwen3-VL-4B-Instruct_sftparity`), keeping the
comparison clean; epochs span 24 h jobs via auto-resume.

The collator finds the prompt/answer boundary from the templated text (the
templated full conversation starts with the templated prompt; the suffix is
pure text), cross-checked against the frame-track re-processing path on the
first sample — avoids re-processing ~126 images per sample per step.

## Per-epoch eval

After each epoch, the in-memory model runs a **seeded 500-question test
subset** (seed 42; qIDs in `logs/<model>/eval_subset_qids.json`) and scores it
in-job with the focus Evaluator (Qwen3.5-4B judge) → `epoch_metrics.csv` +
`eval_epoch_<N>/`. Full-test inference + eval run only for the chosen best
epoch.

## Run

```bash
sbatch scripts/extract_frames.slurm procedure train 20 250     # once (repo root)
# smoke (~2 steps + tiny eval):
sbatch track-procedure/lora-finetune/scripts/train.slurm src/configs/qwen3_vl_4b.yaml --limit 48 --epochs 1 --eval-limit 12
# full (auto-resumes if resubmitted after 24h wall):
sbatch track-procedure/lora-finetune/scripts/train.slurm
# best epoch -> full test:
sbatch track-procedure/lora-finetune/scripts/infer.slurm track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct/checkpoint-<S> test
sbatch track-procedure/lora-finetune/scripts/eval.slurm  track-procedure/lora-finetune/logs/resp_test_checkpoint-<S> test
```

## Results

See `result-summary.md` (PROCEDURE section).
