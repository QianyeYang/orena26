# ORena SAVE FOCUS Challenge — Overview

> Canonical reference for the challenge. Compiled 2026-06-18 from the three track
> websites (Info, Track Overview, Evaluation, Data, Rules, Prizes, Timeline tabs),
> the shared landing/timeline pages, the Hugging Face dataset card, and the
> `orena-focus` Python package. See [Sources](#sources) for the exact URLs.

## 1. What it is

A **MICCAI 2026** medical-AI benchmark on **FOCUS = Foreign Object Contextual
Understanding in Surgery**: building **Vision-Language Model (VLM) pipelines for
surgical VQA** that detect and reason about *foreign objects* (sponges, needles,
clips, drains, silicone loops, specimen bags) in laparoscopic surgery.

**Clinical motivation:** retained foreign objects after surgery are rare but
clinically serious adverse events associated with patient harm; the challenge
benchmarks whether VLMs can support intraoperative quality assurance.

- **Organizer:** DKFZ (IMSY-DKFZ). **Sponsors:** Helmholtz, Wellcome LEAP, DKFZ.
- **Platform:** Grand Challenge. **Support:** `orena-support@dkfz.de`.
- **Forum:** `orenaforum.orena-focus-challenge.org`.
- The site expands the **FOCUS** part but does **not** individually define
  "ORena" / "SAVE".

## 2. The three tracks

All tracks share one I/O contract:
**input** = visual context + metadata (procedure name, time point, expected
output, list of foreign-object classes) + a natural-language question →
**output** = a *short text answer*.

| | **FRAME** | **SEGMENT** | **PROCEDURE** |
|---|---|---|---|
| **Input** | Single RGB frame | Video clip ≤ 5 min | Full procedure video |
| **Skill** | Instantaneous perception | Local temporal reasoning | Long-context memory & tracking |
| **GPU / budget** | 48 GB / **5 s per Q** | 80 GB / **15 s per Q** | 80 GB H100 / **30 s per Q** |
| **Prize share** | ~20% | ~40% | ~40% (split tech/clinical) |
| **Leaderboards** | 1 | 1 | **2** (Technical + Clinical) |
| **HF subset rows** | 6k | 6k | 3k |

Recommended development order: **FRAME first** (no temporal modeling), then reuse
the structure for SEGMENT, then PROCEDURE.

## 3. Data — HeiCo-FOCUS VQA

- **Hosted:** `huggingface.co/datasets/orena-dkfz/heico-focus-vqa` (QA annotations
  only), **CC-BY-NC-SA-4.0**. **Videos are NOT on HF** — pulled separately via the
  `focus` package: `download("heico")` → `/data/focus/heico/videos/`.
- **Batch 1 (released 2026-05-15):** HeiCo colorectal (proctocolectomy) surgery,
  **15,000 VQA pairs**, annotated by **39 domain experts**. HF card: 10k train / 5k test.
- **Batch 2 (planned):** 170 laparoscopic cholecystectomy videos, 35,000 VQA pairs.
- **Full plan:** **200 training videos / 50,000 VQA pairs**; validation = 20 videos;
  **test = 200 withheld videos** across diverse procedures.

**Schema fields:** `id, video, procedure_type, question, answer, answer_format,
track, generation` (manual/automatic), `clinical_relevance` (bool), `ood` (bool),
`timestamp_start`, `timestamp_end`, `primary_capability`, `secondary_capabilities[]`.
Stored as Parquet; loadable via `datasets.load_dataset(...)` or the `FocusDataset` wrapper.

## 4. Answer formats (8)

These drive answer parsing/validation.

**Closed-ended** (exact / tolerance match):
- `binary` — yes/no, case-insensitive (parsed boolean)
- `number` — parsed integer, exact match
- `percentage` & `time` — **tolerance-aware**, thresholds from inter-rater variability
- `fo_class` — one or more comma-separated canonical names, compared as a
  case- and order-insensitive set

**Open-ended** (**LLM-as-judge**: up to 3 judge LLMs, majority vote; judges
undisclosed during the challenge):
- `multiple_choice` · `open_ended` · `matching`

> Missing or unparseable answers are scored **incorrect**.

## 5. Evaluation & ranking

- **Metric: Accuracy** (proportion of correctly answered VQA cases) — the single
  ranking metric.
- **Stratified buckets:** 2 robustness levels (**ID / OOD**) × **5 core
  capabilities** = up to **10 buckets**.
  - 5 core capabilities: object recognition & identity matching · temporal
    grounding · aggregation · event/procedural understanding · complex reasoning.
  - FRAME uses a **reduced taxonomy**: object identification · object attributes &
    state · spatial localization (camera) · spatial localization (situs) · object aggregation.
  - HF stores **15 fine-grained** `primary_capability` labels that roll up into the 5 groups.
- **Ranking pipeline:** (1) per-bucket mean accuracy with **cluster-aware
  bootstrapping** to collapse statistically insignificant gaps → (2) **Copeland
  method** (count pairwise bucket dominance across models) → (3) **bootstrap
  win-rate** tie-break for the top 3.
- **PROCEDURE** has two leaderboards: **Technical** (all VQA pairs) and **Clinical**
  (only `clinical_relevance = true` pairs).

## 6. Baselines to beat

Two baselines:
1. A **SOTA frontier closed VLM, zero-shot**.
2. A **strong open-source VLM fine-tuned on challenge data**.

You must beat **both** to advance from pre-evaluation
(PROCEDURE: beat both on *at least one* of its two leaderboards). The
fine-tuned-open baseline is the realistic bar.

## 7. Submission mechanics & rules

- **Docker container, fully offline (no internet at inference)**, single GPU,
  **fully automated** (no manual interaction), plus a required technical **method description**.
- **Up to 10 pre-evaluation submissions per team**; no multi-accounting.
- **Training is permissive:** challenge data + any public/private datasets +
  **open or closed models** allowed; resources need not predate launch.
- ⚠️ Because inference is offline, **closed APIs cannot be called at test time** —
  closed models are only usable for *training / data generation*. All weights must
  be baked into the container.
- Co-authorship on the challenge paper for baseline-beaters (≤3 co-authors/team).
  Organizer-lab members are not prize-eligible.
- **Prizes:** total **$50k+**. Track split ~20% / 40% / 40% (PROCEDURE split ~evenly
  across its Technical/Clinical leaderboards). Per leaderboard top 3:
  1st 50% / 2nd 30% / 3rd 20%.

## 8. Timeline (compiled 2026-06-18)

| Date | Milestone |
|---|---|
| 2026-03-13 | Challenge accepted at MICCAI |
| 2026-05-15 | Registration + **Batch-1 training data** released |
| 2026-05-28 | Kick-off webinar |
| 2026-06-15 | Extra training materials + **submission examples** |
| **2026-07-15** | **Pre-evaluation phase + public leaderboard opens** |
| 2026-08-15 | Final submission opens for baseline-beaters |
| 2026-09-01 | Registration / pre-evaluation cutoff |
| **2026-09-08** | **Final submission deadline** (Docker + method description) |
| 2026-10-01 | Results announced at MICCAI 🏆 |

## 9. Tooling — `orena-focus` package

Official toolkit (`pip install orena-focus`, MIT license). Key components:

- **Data access:** `FocusDataset("heico", DatasetSplit.{TRAIN,TEST}, Track.{FRAME,SEGMENT,PROCEDURE})`;
  `request, reference = ds[0]` → `request.question`, `reference.answer`.
  Config via `FocusConfig(root_dir=...)` + `set_config(...)`; `download("heico")`.
- **Preprocessing:** `FrameExtractorPreprocessor`, `VideoTimestampOverlayPreprocessor`.
- **Evaluation:** `Evaluator` produces capability-grouped metrics (mirrors the real scorer).
- **Answer formats:** `Binary, Number, Percentage, Time, FOClass, OpenEnded, Matching, MultipleChoice`.

## 10. Open discrepancies to confirm

1. **Batch-1 video count:** challenge pages say **30** colorectal videos; the HF
   card metadata says **20 unique videos**. Likely the HF metadata reflects a
   released/train subset — confirm once videos finish downloading.
2. **Capability taxonomy** appears in three granularities (FRAME's reduced 5 / the
   5 core groups used for bucketing / 15 fine-grained HF labels). Ranking buckets
   use the **5 core groups × ID/OOD**.

## 11. Strategic read

- Build the FRAME loop first:
  `load sample → extract frame → VLM(image + question + metadata) → parse to answer_format → score with Evaluator → save prediction`.
- Winning move: a **fine-tuned open VLM** packaged fully offline, with robust
  per-format answer parsing and attention to **ID/OOD generalization** (Batch-2
  cholecystectomy is the OOD shift vs. Batch-1 colorectal).
- Prepare all weights/resources offline before submission — the Docker
  environment has no internet.

## Sources

- FRAME: <https://frame.orena-focus-challenge.org/> (+ `/track-overview/`, `/evaluation/`)
- SEGMENT: <https://segment.orena-focus-challenge.org/> (+ `/evaluation/`)
- PROCEDURE: <https://procedure.orena-focus-challenge.org/> (+ `/track-overview/`, `/evaluation/`, `/data/`, `/rules/`, `/prizes/`)
- Landing & timeline: <https://orena-focus-challenge.org/> , <https://orena-focus-challenge.org/timeline/>
- Dataset: <https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa>
- Package: <https://github.com/IMSY-DKFZ/orena-focus>
