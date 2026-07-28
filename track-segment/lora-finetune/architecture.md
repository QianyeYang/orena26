# SEGMENT LoRA SFT — Qwen3-VL-4B on interleaved timestamped frames

## Both-dataset official rerun

`src/configs/qwen3_vl_4b_both_official.yaml` starts from the original
`Qwen3-VL-4B-Instruct` weights and trains on all 13,746 official HeiCo and
LapChole train rows; official test rows are evaluation-only. Sampling is one
source second (stride 25 for HeiCo, 30 for LapChole), at most 64 frames, using
official-compatible Decord/OpenCV JPEG extraction. The B200 recipe uses batch
size 2 per GPU × 2 GPUs × accumulation 4 (effective batch 16), saves at every
epoch boundary plus three intermediate recovery points per epoch (every 215
steps), and writes to `logs/Qwen3-VL-4B-Instruct-both-official/`. Distributed
evaluation runs on rank 0 while the other rank waits at a barrier; only rank 0
writes evaluation and metadata artifacts.

## Method

LoRA SFT of **Qwen3-VL-4B-Instruct** on the 4000 SEGMENT train questions, using
the exact frame-track winning recipe (which took FRAME 0.322 → **0.828**):

- LoRA r=16, α=32, dropout 0.05 on every `nn.Linear` leaf of the LLM + vision
  tower (auto-discovered; `lm_head`/`embed_tokens` excluded); the multimodal
  projector (`merger`) is fully trained via `modules_to_save`.
- Effective batch 16 (bs 1 × grad-accum 16), lr 1e-4 cosine, warmup 0.03,
  bf16, gradient checkpointing, seed 42, save per epoch, auto-resume across
  24 h jobs. 12-epoch target (250 steps/epoch).
- Supervision: gold `answer` verbatim, including multi-label `fo_class` rows
  accepted by the official set parser; loss masked to the assistant answer only.

**Train/inference parity by construction**: `src.videovqa_sft.VideoSFTDataset`
builds every conversation with the same `select_frames` / `build_preamble` /
`build_messages` / `build_instruction` the zero-shot baseline runner uses, at
the same sampling settings (stride 25 = 1 fps grid, ≤64 frames, max_pixels
262144 → 252 tok/frame, ~13 K prefill tokens/sample) — so the SFT delta over
the zero-shot baseline is clean.

The collator finds the prompt/answer boundary from the templated text (the
templated full conversation starts with the templated prompt; the suffix is
pure text), cross-checked against the frame-track re-processing path on the
first sample — avoids re-processing ~50 images per sample per step.

## Per-epoch eval

After each epoch, the in-memory model runs a **seeded 500-question test
subset** (seed 42; qIDs in `logs/<model>/eval_subset_qids.json`) and scores it
in-job with the focus Evaluator (Qwen3.5-4B judge) → `epoch_metrics.csv` +
`eval_epoch_<N>/`. Full-test inference + eval run only for the chosen best
epoch (subset ≈ full-test to a few points; final numbers always from full test).
Only rank 0 performs generation and judge scoring. Other DDP ranks poll a
shared filesystem completion token, avoiding a long-lived NCCL collective and
its watchdog timeout while evaluation runs.

## Run

```bash
sbatch scripts/extract_frames.slurm segment train 20 25        # once (repo root)
# smoke (~2 steps + tiny eval):
sbatch track-segment/lora-finetune/scripts/train.slurm src/configs/qwen3_vl_4b.yaml --limit 48 --epochs 1 --eval-limit 12
# 2×A100 DDP smoke, effective batch 16:
RUN_NAME_OVERRIDE=_smoke_ddp_a100 NUM_GPUS=2 \
  sbatch -p a100 --gres=gpu:2 -t 02:00:00 \
  track-segment/lora-finetune/scripts/train_multigpu.slurm \
  track-segment/lora-finetune/src/configs/qwen3_vl_4b_both_official.yaml \
  --limit 32 --epochs 1 --eval-limit 12 --save-steps 1 \
  --per-device-batch-size 1 --grad-accum 8
# full 2×B200 (auto-resumes from the newest checkpoint):
NUM_GPUS=2 sbatch track-segment/lora-finetune/scripts/train_multigpu.slurm \
  track-segment/lora-finetune/src/configs/qwen3_vl_4b_both_official.yaml \
  --per-device-batch-size 2 --grad-accum 4
# best epoch -> full test:
sbatch track-segment/lora-finetune/scripts/infer.slurm track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct/checkpoint-<S> test
sbatch track-segment/lora-finetune/scripts/eval.slurm  track-segment/lora-finetune/logs/resp_test_checkpoint-<S> test
```

## Results

See `result-summary.md` (SEGMENT section).
