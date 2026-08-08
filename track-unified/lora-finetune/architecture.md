# Unified multi-track LoRA SFT (frame + segment + procedure)

One LoRA adapter trained jointly on all three tracks' train data, with
**per-track checkpoint selection**: every track keeps its own inference format,
sampling regime and pixel budget, and the best epoch for each track is picked
independently from one run's `epoch_metrics.csv`. Specialists remain the
fallback — this experiment is leaderboard-risk-free by construction.

## Why joint training

- FRAME data (13,748 rows) donates dense per-frame FO perception to the window
  tracks; SEGMENT and PROCEDURE share time/duration/percentage answer
  conventions; rare classes pool across tracks (Gallstone recall 0.059,
  Hemostatic Agent 0/5 in specialists).
- The unified checkpoint is the prerequisite for the retrieve-then-zoom
  two-stage temporal pipeline: stage 2 (dense window) is segment-shaped input,
  so one model must handle both regimes.
- Specialist baselines to beat (both-official, full test, HeiCo/LapChole):
  FRAME ep30 0.664/0.555 (≈0.709/0.643 after multi-label rescore),
  SEGMENT ep6 0.7023/0.7745, PROCEDURE ep8 0.3105/0.5412.

## What it will NOT fix (separate workstreams)

±1–5 s time localisation over 5 h (needs two-stage zoom), counts ≥6 range
compression (needs map-reduce counting), percentage exact-match convention.

## Design

### Data mixing

`UnifiedSFTDataset` (`src/videovqa_unified.py`) concatenates per-(track,
dataset) row lists — 34,367 rows/epoch at `repeat: 1`:

| track     | heico | lapchole | regime (train == inference) |
|-----------|-------|----------|------------------------------|
| frame     | 8,000 | 5,748    | 1 image @ ≤602,112 px, frame-style prompt (no preamble/timestamp) |
| segment   | 8,000 | 5,746    | ≤64 frames @ 1 s grid (stride 25/30) @ ≤262,144 px, interleaved "Frame at HH:MM:SS:" |
| procedure | 4,000 | 2,873    | ≤96 frames @ 10 s grid (stride 250/300) @ ≤131,072 px, same interleave |

The HF Trainer's seeded shuffle interleaves tracks; each `__getitem__`
dispatches on the item's track and builds its prompt with the SAME shared
functions its track's inference uses (`frames.request_frame_paths` +
frame-style messages for FRAME; `select_frames`/`build_preamble`/
`build_messages` for windows) — per-track train/inference prompt parity is
preserved byte-for-byte (`tests/test_unified_parity.py`).

Optional `repeat: N` per track upsamples cheaply (a frame item is ~0.6 k tokens
vs ~17 k for segment, so extra frame passes cost ~5% wall time each). v1 uses
uniform `repeat: 1`.

### Per-item pixel budgets

The specialists set `max_pixels` on the processor (one global value); a mixed
batch needs per-item budgets. `UnifiedCollator` pre-resizes every image to its
item's budget using the model's own `smart_resize` geometry (factor =
patch 16 × merge 2 = 32; all three budgets are 32²-aligned), then the processor
(loaded with `max_pixels = max over tracks = 602,112`) sees already-conforming
images and its resize is a no-op — so each track's pixels match its specialist
exactly. The per-epoch eval path pre-resizes the same way before `generate`
(the eval callback bypasses the collator, so this is required for parity, not
an optimisation).

### Batching / OOM

`per_device_batch_size: 1`, `grad_accum: 4`, **4× A100 DDP**
(`train_multigpu.slurm`) → effective batch 16, identical to every specialist.
This is the production-validated geometry: the segment both-official run
itself trained at world_size 4 / pdbs 1 / accum 4, and the 2×A100 smoke ran
segment's full 64-frame @ 262 k px items at pdbs 1 on 80 GB — procedure items
(12.3 k vision tokens) are strictly smaller than segment items (16.4 k), so no
per-track frame reduction is needed on A100. Single-sample microbatches also
make mixed-length batches free: no padding waste between a 0.6 k-token frame
item and a 17 k-token segment item, and no mixed-budget processor calls.
34,367 rows / eff 16 = **2,148 optimizer steps/epoch** (independent of world
size); `save_steps: 537` = exact quarter epochs for requeue recovery, plus the
per-epoch checkpoints used for selection. Single-GPU fallback: `train.slurm`
with `--grad-accum 16`.

### Epochs and checkpoint selection

12 epochs (cosine over the full budget, lr 1e-4, warmup 0.03, seed 42 — the
specialist recipe). Segment peaked at 6/12, procedure at 8/8, frame at 30/30:
the window optima sit inside this budget with margin; frame gets 12 passes
instead of 30, partially compensated by transfer — if the frame curve is still
rising at epoch 12, v2 raises `tracks.frame.repeat`. Per-epoch eval writes one
row per epoch with 6 accuracies (track × dataset) + overall mean; each track's
submission candidate = its own argmax epoch.

### Per-epoch eval

`UnifiedEpochEvalCallback` extends the window-track callback (same rank-0
filesystem sync, judge = local Qwen3.5-4B, never kills training): three seeded
500-row test subsets (one per track, split across both datasets), written to
`eval_epoch_<N>/<track>_<dataset>/…` and `epoch_metrics.csv` with columns
`frame_heico … procedure_lapchole`. ~35–45 min/epoch on top of training.

### Wall time (estimate, verify in smoke)

~2,148 steps/epoch on 4× A100 + per-epoch eval → measure the real s/step in
the smoke; the run spans multiple requeued 2-day jobs with auto-resume (same
mechanism as the segment production run). Raise `NUM_GPUS` (with
`--grad-accum` adjusted to keep eff 16) if the projection is too slow.

## Files

- `src/videovqa_unified.py` (shared src): dataset, collator, eval callback,
  `run_unified_training()`.
- `track-unified/lora-finetune/src/train.py`: thin CLI wrapper.
- `src/configs/qwen3_vl_4b_unified.yaml`: the run config.
- `scripts/train_multigpu.slurm`: PRIMARY — 4× A100 torchrun DDP, /dev/shm
  staging, auto-resume, judge in-job.
- `scripts/train.slurm`: single-GPU b200 fallback (`--grad-accum 16`).
- `tests/test_unified_parity.py`: CPU prompt-parity tests vs the specialist
  datasets (run on civo login node).

## Usage

```bash
# smoke (4x A100, minutes):
RUN_NAME_OVERRIDE=_smoke_a100_unified NUM_GPUS=4 \
  sbatch scripts/train_multigpu.slurm src/configs/qwen3_vl_4b_unified.yaml \
    --limit-per-track 8 --epochs 1 --eval-limit 6 --save-steps 2
# real run (auto-resumes across requeues):
sbatch scripts/train_multigpu.slurm
```

## Selection & reporting

After the run: pick per-track best epochs from `epoch_metrics.csv`, run the
full 6,254/6,252-row test for those checkpoints with the existing per-track
infer pipelines (adapter path swapped), and compare against the specialist
baselines in `result-summary/<track>/`. Only adopt per track where the unified
checkpoint wins.
