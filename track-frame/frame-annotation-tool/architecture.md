# Frame Annotation Tool

Local, SSH-friendly bounding-box annotation for the FOCUS Frame track.

## Purpose

The tool creates instance-level supervision for the two largest Frame-model
weaknesses: exact object counting and complete foreign-object identification.
It groups all official training VQA rows that refer to the same frame into one
case, displays those rows beside the image, and derives live consistency checks
from the boxes.

The annotation pool contains 4,000 unique official **training** frames: 2,000
HeiCo and 2,000 LapChole. Test frames are excluded. Cases are ranked from
highest to lowest expected research value and grouped into 100-case priority
batches.

## Priority model

The deterministic score emphasises:

- total-instance and Clip-count questions;
- answers of five or more, especially seven or more;
- multi-class inventories and frames that have both inventory and count QA;
- rare classes, especially Gallstone and Needle;
- crowded/spatial questions, QA density, OOD examples, and LapChole cases.

The manifest records the score and human-readable reasons for every case.
Stable tie-breaking spreads adjacent ranks across videos where scores are
otherwise equal.

## Annotation schema

Every visible foreign-object instance receives:

- canonical FOCUS class;
- axis-aligned box in source-image pixel coordinates;
- difficult flag (tiny, blurred, truncated, or heavily occluded);
- uncertain flag (requires review).

Per-case state also records `no_foreign_objects`, notes, a four-item completion
checklist, revision, timestamps, and status.

`complete` is server-validated. It is rejected when boxes are invalid, the
checklist is unfinished, an uncertain box remains, or derived counts/classes
contradict an applicable QA answer. Such cases can be saved as
`needs_review` instead of being silently accepted.

## Storage and recovery

- Pool manifest:
  `data/annotations/frame-annotation-tool/pool.json`
- One atomic JSON record per case:
  `data/annotations/frame-annotation-tool/records/<case_id>.json`
- Append-only audit metadata:
  `data/annotations/frame-annotation-tool/audit.jsonl`
- Timestamped canonical, COCO, and progress exports:
  `data/annotations/frame-annotation-tool/exports/`

The server uses optimistic revisions to prevent two browser tabs from
overwriting one another.

## Interface

- Queue: priority rank, batch, score, dataset, reasons, status, and filters.
- Annotate: zoomable canvas, draw/move/resize boxes, class selector, flags,
  live inventory/counts, QA context, consistency warnings, autosave, and
  next-incomplete navigation.
- Guide: class definitions, recognition/exclusion tips, verified released-data
  examples, and explicit warnings where no released example exists. Mesh and
  Absorbable Hemostatic Agent have no released Frame-training QA example, so
  their external clinical references are visibly distinguished from challenge
  data.
- Progress: status and batch completion summaries plus export.

The server binds to `127.0.0.1` by default. Use SSH port forwarding rather
than exposing annotation writes to a network interface.
