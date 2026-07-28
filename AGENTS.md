# Repository Instructions

Keep this file brief, but do not omit critical information.

## Scope

These instructions apply to the entire repository. More specific `AGENTS.md`
files in subdirectories override them for their directory trees.

## Project Overview

This repository contains solutions for the Foreign Object Contextual
Understanding in Surgery (FOCUS) challenge. It has three tracks:

- Frame: https://frame.orena-focus-challenge.org/
- Segment: https://segment.orena-focus-challenge.org/
- Procedure: https://procedure.orena-focus-challenge.org/

The main work is building vision-language model (VLM) pipelines for
visual-question-answering (VQA) tasks.

Always use the `orena` conda environment for repository work. Do not mix it
with other environments.

## GPU Allocation

Before submitting GPU work, run `~/check_gpus.sh` to inspect current
availability. Prefer GPU partitions in this order: B200, H200, then A100.

## Repository Layout

- `data/`: Dataset storage.
- `src/`: Shared source code used by all methods.
- `docs/`: Documentation, guidelines, important records, and Markdown files.
- `scripts/`: Utility scripts.
- `track-*/`: Track-specific challenge work.
- `submissions/`: Generated submission Docker artifacts.
- `tmp/`: Temporary, one-time scripts, code, documents, and other materials.
- `os-models/`: Open-source VLMs and LLMs.
- `result-summary/`: Track- and experiment-specific numerical result reports.
  Keep its indexes and reports clean and well organized; include dataset and
  capability distributions instead of only headline scores.
- `result-summary.md`: Historical cross-track snapshot retained for provenance.

Each `track-*` directory is organized by method:

- Put each method group in a separate directory because architectures can
  differ substantially, for example `track-*/baseline/`.
- `<method>/architecture.md`: Method architecture documentation.
- `<method>/logs/`: Training and inference logs, plus model checkpoints.
- `<method>/src/`: Method-specific code only. Put reusable code in the
  repository-level `src/`.
- `<method>/scripts/`: SLURM submission scripts for training and testing.

## Submission Packaging

- Keep the upstream reference unchanged in
  `submissions/orena-focus-submission-template/`.
- Put generated bundles directly under
  `submissions/{frame,segment,procedure}/<descriptive-versioned-name>/`.
- Use the discoverable `make-submission` Codex skill for new or rebuilt bundles.
- Every bundle must be self-contained for transfer to an x86_64 Docker host:
  include Docker and Apptainer definitions/scripts, pinned requirements,
  inference code, model resources, licenses, synthetic test input,
  `provenance.json`, output validation, and `guidance.md`. Regular files and
  same-filesystem hardlinks are allowed; external symlinks are not.
- Do not include datasets, secrets, caches, optimizer states, duplicate
  checkpoints, SIF files, or Docker archives in a bundle.
- Preserve the challenge contract: offline inference, one model load per batch,
  one response per request, per-question error isolation, and
  `/output/answer.json`.
- Preserve answer cardinality in every adapter. In particular, `fo_class` is
  multi-label: retain all recognized comma-separated classes, load runtime FO
  definitions, and never reduce to the first match. Test multi-label and
  overlapping names with the official format parser before packaging.
- Workflow: resolve and record the exact evaluated checkpoint; assemble and
  statically validate the bundle; build and smoke-test it locally with
  Apptainer; validate output IDs/count/types and runtime; delete the temporary
  SIF and cache after a passing test; transfer the bundle; then follow
  `guidance.md` to build, offline-smoke-test, validate, and save it with Docker.
- For submission build/smoke testing only, allocate GPUs in this order:
  A100, H200, then B200. This overrides the general GPU order above.
- The Docker handoff machine has 32 GB NVIDIA V100 GPUs. Use its GPU for Docker
  smoke testing whenever the NVIDIA runtime is available. For a final BF16
  image, run the V100 smoke test through a temporary FP16 runtime override, but
  keep and verify BF16 as the image default before saving the final archive.
- Never claim Docker validation from an Apptainer-only test. Record both stages
  separately in `guidance.md`.

## Data

Videos:

- HeiCo: 30 files, 149.6 GiB, in `data/focus/heico/videos/`.
- LapChole: 170 files, 90.3 GiB, in `data/focus/lapchole/videos/`.

VQA annotations:

- HeiCo: `data/parquet/{frame,segment,procedure}/{train,test}/0000.parquet`.
- LapChole:
  `data/parquet/lapchole/{frame,segment,procedure}/{train,test}/0000.parquet`.
- Always load annotations directly from `data/parquet/` with
  `pandas.read_parquet()`. Do not use `load_dataset()`; direct Parquet loading
  is approximately 160 times faster (4 ms versus 650 ms per load).

Train/test row counts:

| Dataset | Frame | Segment | Procedure |
| --- | ---: | ---: | ---: |
| HeiCo | 8000 / 4000 | 8000 / 4000 | 4000 / 2000 |
| LapChole | 5748 / 2252 | 5746 / 2254 | 2873 / 1127 |
