# ORena FOCUS — final FRAME / SEGMENT / PROCEDURE models

Solution for the Foreign Object Contextual Understanding in Surgery (FOCUS)
challenge, MICCAI 2026: VQA about foreign objects (clips, sponges, needles,
drains, ...) in laparoscopic video. Tracks:
[Frame](https://frame.orena-focus-challenge.org/) ·
[Segment](https://segment.orena-focus-challenge.org/) ·
[Procedure](https://procedure.orena-focus-challenge.org/).

**Approach.** One Qwen3.5-4B LoRA family (r=64, α=128, every linear layer
including ViT and merger) trained jointly on all three tracks. FRAME reads one
still (2,048–3,072 visual tokens). SEGMENT and PROCEDURE use the model's native
video path over 96 / 448 frames at 1 Hz with absolute timestamps.
Stage 1 trains the **champion**; stage 2 **anneals** it on a
capability-rebalanced mix.

## Final submissions

| Track | Model | Docker image tag | Build context (git-ignored) |
| --- | --- | --- | --- |
| FRAME | anneal `checkpoint-2428` (stage 2, warm-started from the champion) | `focus-frame-qwen35-4b-anneal-s2428` | `submissions/frame/qwen3-5-4b-anneal-balanced-res2k-r64-a128-step2428-20260902/` |
| SEGMENT | champion `checkpoint-12504` (stage 1) | `focus-video-qwen35-4b-unified-ep8` | `submissions/video/qwen3-5-4b-unified-video-r64-a128-epoch8-20260830/` |
| PROCEDURE | same model and same image as SEGMENT (track routed per question) | same | same |

```text
adapter_model.safetensors SHA-256
c5dce31015928167284430658f83f27d8c76d70814a794cfea2b4d1a4eb5b654  champion  checkpoint-12504  (SEGMENT, PROCEDURE)
0d49a4fcd3376e4667dad7d2e332509351001798a936f6d7e2628e6add7feac7  anneal    checkpoint-2428   (FRAME)
```

Base model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Neither final model uses external data.

**No clean validation exists.** Training used every released row (train and
test splits), so local scores are memorisation. Each final checkpoint is the end
of its completed cosine schedule; only the online pre-evaluation is an
uncontaminated measurement. Exact provenance (jobs, GPUs, hyperparameters,
data hashes) is in each bundle's `provenance.json`.

## Quick reference: a trained model comes in

| | FRAME | SEGMENT + PROCEDURE |
| --- | --- | --- |
| Put the adapter in | `<bundle>/resources/adapter/` | `<bundle>/resources/adapter/` |
| Base model in | `<bundle>/resources/model/` | `<bundle>/resources/base_model/` |
| Inference script | `<bundle>/inference.py` | `<bundle>/inference.py` |
| Engine | ms-swift `TransformersEngine` + LoRA | HF `AutoModelForImageTextToText` + PEFT |
| Input | `/input/request.json`, `/input/FO_definitions.json`, `/input/frames/<qID>.png` | `/input/request.json`, `/input/FO_definitions.json`, `/input/plain/<qID>.mp4` |
| Output | `/output/answer.json` | `/output/answer.json` |
| Challenge GPU | L40S 48 GB | H100 80 GB |

`<bundle>` is the build context in the table above. `inference.py` is the
container entrypoint; it hard-codes `/input` and `/output`, so it is run through
the container (Section 7), not directly. Swapping in a new adapter of the same
architecture needs no code change: the video script loads
`resources/adapter/adapter_config.json` if present, the frame script always
loads `resources/adapter/`.

## 1. Repository map

| Path | Role |
| --- | --- |
| `track-unified/qwen3-5-4b-video/` | **the final method**: `configs/`, `scripts/` (SLURM), `src/` (dataset builders, ms-swift plugin), `architecture.md` |
| `src/video_geometry.py` | input contract shared by training and the containers: frame plan, resize geometry, prompts |
| `src/frame_track_extraction.py` | official-compatible frame extraction (`extract_plan`) |
| `scripts/` | data and model download, frame extraction (SLURM) |
| `submissions/{frame,video}/<bundle>/` | self-contained Docker / Apptainer build contexts |

`.gitignore` excludes `data/`, `os-models/`, `logs/` (checkpoints), `tmp/` and
`/submissions/*`, so data, weights, checkpoints and bundles are not in git.

## 2. Environment

- **Cluster:** civo (SLURM). The repo root is `/datasets/engs2732/orena`
  (`~/projects/orena` is a symlink to it); the SLURM scripts hard-code this path.
- **Always** `conda activate orena` (`environment.yml`: Python 3.12,
  `orena-focus` 0.3.4, `peft` 0.19.1, `decord`, `pandas`).
- **ms-swift is not in that env.** Training stages a vendored overlay,
  `tmp/qwen3-5-2b-ms-swift/vendor`, to `/dev/shm` and prepends it to
  `PYTHONPATH`. It is stock, unpatched ms-swift 4.3.2 (all 506 `.py` files match
  the PyPI RECORD hashes) plus: transformers 5.12.1, accelerate 1.12.0,
  datasets 3.1.0, trl 0.27.2, qwen-vl-utils 0.0.14, modelscope 1.38.0,
  tensorboard 2.21.0. To recreate it, `pip install --target
  tmp/qwen3-5-2b-ms-swift/vendor` those pins.
- **decord:** the conda-forge build fails ("Set output pixel format error"). Any
  job that decodes video must prepend `PYTHONPATH=$REPO/data/vendor/decord-pypi`
  (the frame-extraction SLURM scripts already do).
- **GPUs:** run `~/check_gpus.sh` first. SLURM QOS caps a user at 4 concurrent
  GPUs. Training scripts require **B200** and abort on any other GPU. Submission
  build and smoke tests prefer A100 → H200 → B200. H200 nodes have 14 CPUs, so
  keep `--cpus-per-task` ≤ 12 there.
- **Docker host:** the civo login node has no Docker (Apptainer is used there).
  Images are built and saved on a separate x86_64 Linux host with an NVIDIA
  V100 32 GB.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate orena
cd /datasets/engs2732/orena          # run every command below from the repo root
```

## 3. Preprocessing

Every step reads only local files; annotations are loaded with
`pd.read_parquet()` (never `load_dataset()`, ~160× slower).

### 3.1 Data and base model

```bash
# HF token with access to the gated datasets: HF_KEY in ~/.bashrc, or HF_TOKEN in .env (see .env.example)
sbatch scripts/download_data.slurm --datasets heico lapchole
#   data/focus/{heico,lapchole}/videos/*.mp4   (30 + 170 videos, ~240 GiB)
#   data/parquet/{frame,segment,procedure}/{train,test}/0000.parquet            (HeiCo)
#   data/parquet/lapchole/{frame,segment,procedure}/{train,test}/0000.parquet   (LapChole)

python scripts/download_models.py Qwen3.5-4B          # -> os-models/Qwen3.5-4B
```

To pin the exact base revision, call `huggingface_hub.snapshot_download(
"Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
local_dir="os-models/Qwen3.5-4B")`. The organizers have pushed silent annotation
revisions before. Training used the parquet downloaded on 2026-08-18; the civo
HF cache records `main` at `d2ce84dd…` (HeiCo) and `1d9a4b66…` (LapChole), so
compare against `commits/main` before trusting a fresh download.

Rows per track (train + test, both datasets): **frame 20,000 · segment 20,000 ·
procedure 10,000** (50,000 in total, all used for training).

### 3.2 Frame grids

```bash
# (a) FRAME track: the still at each question's timestamp. Run from the repo root (the script uses $SLURM_SUBMIT_DIR).
for ds in heico lapchole; do sbatch scripts/extract_frames.slurm $ds frame 1 1; done

# (b) SEGMENT/PROCEDURE: dense 1 s grid over each video, train and test videos separately.
sbatch scripts/dense_extract_train.slurm     # array 0-1 = heico, lapchole
sbatch scripts/dense_extract.slurm           # array 0-1 = heico, lapchole
```

Output for both: `data/focus/<dataset>/{frames,frames-1s}/<video_stem>/frame{index:07d}.jpg`,
JPEG quality 95, where `index` is the absolute source-frame index (HeiCo 25 fps,
LapChole 30 fps). **Never write your own decoder**: extraction goes through
`src.frame_track_extraction.extract_plan` (decord → RGB2BGR → `cv2.imwrite`),
which reproduces the official grid byte-for-byte.

The dataset builder prefers `frames-1s` and falls back to the older
question-driven `frames` grid for an index the dense pass did not write. In the
champion build 15 rows were short by 151 frames in total (present in neither
grid) and were built with fewer frames; none were dropped.

### 3.3 Unified training set (stage 1)

```bash
python track-unified/qwen3-5-4b-video/src/build_dataset.py
#   -> data/derived/unified/qwen3-5-4b-video/train_all.jsonl  (+ train_all.manifest.json)
```

Expect 50,000 rows (frame 20,000 / segment 20,000 / procedure 10,000) and
`dropped_rows: 0`; the builder aborts if the counts differ. Reference SHA-256 of
the file used for the champion: `47d799c62c7817a8…`. Rows are shuffled with
seed 42.

| Row type | Prompt | Media |
| --- | --- | --- |
| frame | system = `track-frame/qwen3-5-2b-res2k/resources/system_prompt.txt`; user = `<image>{question}` | 1 JPEG; 2,048–3,072 visual tokens (`IMAGE_MIN_TOKEN_NUM`/`IMAGE_MAX_TOKEN_NUM`/`MAX_PIXELS=1003520`, exported by the training scripts) |
| segment | system = `build_system_prompt(procedure_type)`; user = window timing + FO definitions + question | 96 frames, 589,824 px/frame (~1024×576), ~25–28k tokens |
| procedure | same builder | 448 frames, 258,048 px/frame (~672×384), ~56k tokens |

Video rows carry a `focusvideo/v1:` JSON frame plan in the `videos` field
(absolute frame indices, source fps, pixel budget). The frame plan is 1 Hz
anchored at the **window start**, evenly sub-sampled to 96 / 448 (even count).
Frames are resized aspect-preserving, snapped to 32 px, never upscaled
(`video_geometry.target_resolution`), by the training plugin
(`qwen3_5_video_plugin.py`). All of this logic lives in `src/video_geometry.py`;
the containers mirror it byte-for-byte, so any change must be made in both.

### 3.4 Anneal set (stage 2)

```bash
python track-unified/qwen3-5-4b-video/src/build_anneal_dataset.py
#   -> data/derived/unified/qwen3-5-4b-anneal/{anneal_train,anneal_holdout}.jsonl (+ *_meta.jsonl, manifest.json)
```

Defaults reproduce the run (seed 20260831): 20,001 training rows (frame 8,000 ·
segment 8,000 · procedure 4,001; SHA-256 `9f78078f37f5afd3…`) and a 1,080-row
holdout. Why: the metric averages capability group × ID/OOD buckets with equal
weight, and groups 4–5 are 5.6% of video rows but 40% of each video track's
score. Row shares per (track, group) follow √(natural share); duplicated video
rows use `jittered_subsample` (a different frame phase, ~3 s segment / ~36 s
procedure) so a duplicate is a different input, not a repeat. The holdout is
carved before duplication and **is contaminated** (the champion trained on those
rows), so it is only a forgetting monitor.

### 3.5 CPU gate before any GPU time

```bash
PYTHONPATH=tmp/qwen3-5-2b-ms-swift/vendor python track-unified/qwen3-5-4b-video/src/verify_encoding.py \
  --jsonl data/derived/unified/qwen3-5-4b-video/train_all.jsonl --rows 3
```

Proves absolute `<T seconds>` stamps, on-budget visual token counts and an
answer-only loss mask. `track-unified/qwen3-5-4b-video/scripts/smoke_train.slurm`
is the optional GPU smoke test (per-track step time and OOM check; expects
`tmp/smoke_{frame,segment,procedure}.jsonl`).

## 4. Training

Method files: `track-unified/qwen3-5-4b-video/{configs,scripts,src}`; design and
measured cost model in its `architecture.md`. Training is driven by ms-swift
(`swift sft`) with `--external_plugins
track-unified/qwen3-5-4b-video/src/qwen3_5_video_plugin.py`, which is required:
stock ms-swift fabricates clip-relative frame indices, so it would train on
wrong timestamps. The launch scripts pass it for you.

### Stage 1 — champion (used for SEGMENT, PROCEDURE, and as the FRAME init)

```bash
bash track-unified/qwen3-5-4b-video/scripts/submit_chain.sh 4 4 8
#                                                           NGPU LINKS EPOCHS [RUN_DIR]
#   -> track-unified/qwen3-5-4b-video/logs/training/qwen3.5-4b-unified-video-r64-a128/
#      the newest 15 checkpoints (save_total_limit 15), ending with checkpoint-12504,
#      and a TRAINING_COMPLETE marker
```

- The B200 walltime is a hard 2 days, so the run is a **chain of resumable
  links** submitted up front with `--dependency=afterany`. Each link resumes from
  the newest checkpoint; once `TRAINING_COMPLETE` exists, surplus links exit
  immediately.
- The submit script sets `--cpus-per-task = 40 × NGPU` and `--mem = 200G × NGPU`.
  The job is **data-bound before it is GPU-bound** (a 448-frame sample costs ~15
  CPU-seconds); at 12 CPUs/GPU the GPUs idled at 0–5%.
- Gradient accumulation restores a global batch of ~32 for any GPU count (4 GPUs
  × accum 8). 8 epochs = 12,504 steps (1,563 steps/epoch), ~17 h/epoch on 4 GPUs.
  The original run started on 2 GPUs and migrated to 4 at step 1032 (SLURM jobs
  6122 → 6393 → 6394 → 6395, ~157 h wall-clock).
- Monitor: `squeue -u $USER`, `tail -f track-unified/qwen3-5-4b-video/logs/train-*.log`.

### Stage 2 — capability-rebalanced anneal (FRAME)

Needs stage 1 finished (`checkpoint-12504`) and the anneal set from 3.4.

```bash
bash track-unified/qwen3-5-4b-video/scripts/submit_anneal_chain.sh 3 4 4
#                                                                  NGPU LINKS EPOCHS [RUN_DIR] [INIT_ADAPTER]
#   INIT_ADAPTER defaults to the champion checkpoint-12504
#   -> .../logs/training/qwen3.5-4b-anneal-balanced-r64-a128/checkpoint-{50,...,2428}/  (final: checkpoint-2428)
```

Link 1 warm-starts from the champion's **weights only**:
`--resume_from_checkpoint <champion> --resume_only_model true --ignore_data_skip true`.
Both flags are required; without `--ignore_data_skip` the trainer tries to
fast-forward the dataloader by the champion's 12,504 steps. Later links do a
normal full-state resume. Check the warm start on link 1 from the first logged
loss: it should be tiny (≈ 2e-4 with token accuracy 1.0 in the 20-step
warm-start smoke test), whereas a cold LoRA starts near loss 1.0.

### Hyperparameters

| | Stage 1 (champion) | Stage 2 (anneal) |
| --- | --- | --- |
| Config | `configs/sft_video.yaml` | `configs/sft_anneal.yaml` |
| Data | `train_all.jsonl`, 50,000 rows | `anneal_train.jsonl`, 20,001 rows |
| Init | base Qwen3.5-4B | champion `checkpoint-12504`, weights only |
| LoRA | r=64, α=128, dropout 0.05, `all-linear` (LM + ViT + merger) | same |
| LR / schedule | 1e-4, cosine, warmup 0.03, weight decay 0.1 | 1e-5, cosine, warmup 0.03, weight decay 0.1 |
| Epochs → steps | 8 → 12,504 | 4 → 2,428 |
| Global batch | 32 (micro 1 × accum 8 × 4 GPUs) | 33 (micro 1 × accum 11 × 3 GPUs) |
| `max_length` | 65,536 (`truncation_strategy: delete`) | same |
| Precision | bf16, sdpa attention, gradient checkpointing (LM + ViT) | same |
| Optimizer | AdamW fused, β=(0.9, 0.95), eps 1e-8, grad-clip 1.0 | same |
| Seeds | `seed` 42, `data_seed` 42 | same |
| Hardware | B200, 2 → 4 GPUs | 3 × B200 (jobs 8081–8090) |
| Wall-clock | ~157 h | ~40.7 h |
| Checkpoints | every 100 steps, keep 15 | every 50 steps, keep 60 |

## 5. Putting a trained model into a submission bundle

Copy **only** the three adapter files (never `optimizer.pt`, `rng_state_*`,
`trainer_state.json`). The result must be byte-identical to the checkpoint.

```bash
# FRAME  <- anneal checkpoint-2428
CK=track-unified/qwen3-5-4b-video/logs/training/qwen3.5-4b-anneal-balanced-r64-a128/checkpoint-2428
B=submissions/frame/qwen3-5-4b-anneal-balanced-res2k-r64-a128-step2428-20260902

# SEGMENT + PROCEDURE  <- champion checkpoint-12504
# CK=track-unified/qwen3-5-4b-video/logs/training/qwen3.5-4b-unified-video-r64-a128/checkpoint-12504
# B=submissions/video/qwen3-5-4b-unified-video-r64-a128-epoch8-20260830

mkdir -p $B/resources/adapter
cp $CK/{adapter_config.json,adapter_model.safetensors,additional_config.json} $B/resources/adapter/
sha256sum $B/resources/adapter/adapter_model.safetensors    # compare with the table at the top
```

Bundle layout (both tracks):

```text
<bundle>/
├── inference.py                    container entrypoint (the inference script)
├── Dockerfile  apptainer.def       image definitions
├── do_build.sh  do_save.sh  do_test_run.sh              Docker: build / save archive / offline smoke test
├── do_build_apptainer.sh  do_test_apptainer.sh  smoke_apptainer.slurm   Apptainer on civo
├── provenance.json                 authoritative record: checkpoint, adapter SHA-256, training run
├── validate_output.py              (frame also: validate_bundle.py, test_adapter_contract.py)
├── test/input/...                  synthetic fixtures
└── resources/
    ├── adapter/                    <- the trained LoRA goes here
    ├── model/  (frame)  |  base_model/  (video)     Qwen3.5-4B, copy of os-models/Qwen3.5-4B
    ├── python-vendor.tar           pinned runtime wheels, unpacked to /tmp/focus-python at start
    ├── system_prompt.txt           (frame only)
    └── weights.sha256              (frame only) checksums of every resource file
```

- **Base model:** `rsync -a --exclude=.cache os-models/Qwen3.5-4B/ $B/resources/model/`
  (`base_model/` for video). The bundled copies hold the same files as
  `os-models/Qwen3.5-4B`.
- **`python-vendor.tar`:** reuse the file from the same track's final bundle
  (copy or hard link); frame and video use different archives.
- **New bundle for a new checkpoint:** make a real copy of the final bundle
  (`cp -a`), replace `resources/adapter/*`, then update `provenance.json`
  (`source_checkpoint`, `checkpoint_global_step`, `checkpoint_adapter_sha256`)
  and, for frame, `resources/weights.sha256`. Packaging rules are in `AGENTS.md`
  ("Submission Packaging").
- **Hard links:** the final video bundle shares `inference.py`, `provenance.json`
  and `guidance.md` (same inode) with
  `submissions/segment/qwen3-5-4b-base-video96-1024x576-20260822/`, and
  `python-vendor.tar` is shared across many bundles. Replace such a file (write
  a new one, then `mv` it over) instead of editing it in place, otherwise the
  change silently lands in the other bundles too.
- **Do not** put datasets, secrets, optimizer states, SIF files or Docker
  archives inside a bundle.
- The frame bundle's `README.md`, `guidance.md`, `resources/training_provenance/`
  and the log strings in `inference.py` still describe the earlier epoch-28
  model. `provenance.json` and `weights.sha256` are authoritative.

## 6. Inference

### FRAME — `submissions/frame/<bundle>/inference.py`

One ms-swift `TransformersEngine` load per batch; per question: system prompt =
the packaged instruction prefix + the batch's runtime `/input/FO_definitions.json`,
user = `<image>{question}`, thinking disabled, greedy, `max_tokens=64`,
micro-batch 8. Failed micro-batches are bisected so one bad question never
loses the batch.

| Env var | Default | Meaning |
| --- | --- | --- |
| `FOCUS_MODEL_DTYPE` | `bfloat16` | `bfloat16` or `float16` |
| `FOCUS_BATCH_SIZE` | `8` | micro-batch size |
| `IMAGE_MIN_TOKEN_NUM` / `IMAGE_MAX_TOKEN_NUM` / `MAX_PIXELS` | `2048` / `3072` / `1003520` | res2k geometry, must match training |

### SEGMENT + PROCEDURE — `submissions/video/<bundle>/inference.py`

One container serves both tracks. `request.json` has no track field, so each
question is routed on its window length `end_time − start_time`: **≤ 305 s →
segment** (96 frames, 589,824 px, 15 s/question budget), **otherwise
procedure** (448 frames, 258,048 px, 30 s/question). The cut sits in the
measured 300.0 s / 309.0 s gap between the tracks (0 misroutes on all 30,000
annotated rows). The clip is already trimmed to the window; frame indices are
offset by the window start so the model sees absolute timestamps. A pooled-time
watchdog degrades to ½ then ¼ of the frames instead of failing, and every
question always receives a format-valid answer. Upload this same image to both
the segment and procedure slots.

| Env var | Default | Meaning |
| --- | --- | --- |
| `FOCUS_SEGMENT_MAX_SECONDS` | `305` | segment/procedure routing cut |
| `FOCUS_MODEL_DTYPE` | `bfloat16` | `bfloat16` or `float16` |
| `FOCUS_MAX_FRAMES` | unset | override the frame target |
| `FOCUS_ZOOM` | `0` | zoom refinement, off in the submission |

### Input and output contract (all tracks)

```text
/input/request.json         [{"qID","videoID","start_time","end_time","procedure_type","question"}, ...]
/input/FO_definitions.json  one JSON string: the foreign-object taxonomy for this batch
/input/frames/<qID>.png     FRAME track            /input/plain/<qID>.mp4   SEGMENT / PROCEDURE (5 fps clip trimmed to the window)
/output/answer.json         [{"qID","content","latency"}, ...]   one entry per request
```

Offline (no network); the model is loaded once per batch; each question is
error-isolated.

## 7. Build, test and save the Docker image

On civo (Apptainer, offline GPU smoke test on the synthetic fixture; builds a
SIF, runs it, validates the output and deletes the SIF):

```bash
sbatch submissions/frame/qwen3-5-4b-anneal-balanced-res2k-r64-a128-step2428-20260902/smoke_apptainer.slurm
sbatch submissions/video/qwen3-5-4b-unified-video-r64-a128-epoch8-20260830/smoke_apptainer.slurm
```

Transfer the bundle to the Docker host (hard links preserved), then on that host,
inside the bundle directory:

```bash
rsync -aHh --info=progress2 submissions/<frame|video>/<bundle> <docker-host>:<dir>/   # from civo

./do_build.sh                                  # docker build --platform=linux/amd64 --tag <tag> .
FOCUS_TEST_MODEL_DTYPE=float16 FOCUS_TEST_BATCH_SIZE=1 ./do_test_run.sh   # offline (--network=none) smoke test
./do_save.sh                                   # -> <tag>_<image-created-UTC>.tar.gz, verified with gzip --test
```

`FOCUS_TEST_BATCH_SIZE` applies to the frame bundle only; the video script runs
both fixtures (`interface_1` all-segment, `mixed` = 5 segment + 4 procedure, which
exercises the routing). The host's V100 has no native BF16, so the smoke test
uses the FP16 override; **the image default stays BF16** and `do_save.sh`
refuses to save unless the baked-in defaults are intact
(`FOCUS_MODEL_DTYPE=bfloat16` plus `FOCUS_BATCH_SIZE=8` for frame, or
`FOCUS_SEGMENT_MAX_SECONDS=305` and `FOCUS_ZOOM=0` for video). Never pass the
FP16 override while saving.

Run the built image on any input directory:

```bash
docker run --rm --gpus=all --network=none --shm-size=2g \
  -v /path/to/input:/input:ro -v /path/to/output:/output -v <tmp-volume>:/tmp \
  focus-video-qwen35-4b-unified-ep8        # or focus-frame-qwen35-4b-anneal-s2428
```

Archives saved for the final bundles (the timestamp is the image creation time,
UTC): `focus-frame-qwen35-4b-anneal-s2428_2026-09-03_06-33-20.tar.gz` (FRAME) and
`focus-video-qwen35-4b-unified-ep8_2026-08-30_08-02-11.tar.gz` (SEGMENT and
PROCEDURE).
