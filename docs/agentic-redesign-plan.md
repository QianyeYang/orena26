# Agentic redesign — design record & roadmap (segment + procedure)

Status: **agreed design**, 2026-08-17 (discussion between user and Claude).
Companion implementation plan for step 1: [`bucket-eval-plan.md`](bucket-eval-plan.md).
Evidence base: [`../result-summary/leaderboard/2026-08-17-pre-eval-leaderboard-analysis.md`](../result-summary/leaderboard/2026-08-17-pre-eval-leaderboard-analysis.md).

## 1. Why (three measured facts)

1. **The hidden metric is an unweighted mean of capability×OOD buckets**, not micro
   accuracy. Procedure: 2 complex-reasoning-OOD questions = 10% of the score; 55
   small-bucket questions = 50%. Our V.U "regression" answered **+38 more questions
   correctly** (micro 0.391→0.429, 4th-best in field) yet lost −0.02 macro. We have
   been selecting checkpoints on the wrong objective.
2. **SFT destroyed general reasoning exactly where the metric is expensive**:
   cplxR_ood 0/2 (zero-shot API baseline: 2/2; every team above 14th: ≥1/2),
   eventU_ood 3/19 on segment (worst in top 10).
3. **We spend ~0 of the latency budget** (segment 15 s/q, procedure 30 s/q; #1 spends
   26.9 s/q; a rival runs Qwen3.6-27B bf16 at 7.2 s/q on the eval GPU).

## 2. Decisions

**D1 — Frozen general VLM orchestrator (Qwen3.6-27B class), never fine-tuned.**
Router + planner + synthesiser + default route for anything unroutable/OOD. Its whole
value is being undamaged by our SFT. Weights already on civo `os-models/`:
`Qwen3.6-27B` (+`-FP8`), `Qwen3.8-27B` as alternative.

**D2 — 4B multi-adapter probe on one shared base; adapters never merged.**
One Qwen3-VL-4B base + LoRA adapters toggled per call (`set_adapter`/
`disable_adapter`). Gotcha: our recipe fully retrains the projector via
`modules_to_save('merger')` — adapter-off must restore the *original* merger.
**Required equivalence test**: adapter-off outputs == vanilla base on a fixed prompt
set. Future probe adapters: pure LoRA (no `modules_to_save`) so on/off is clean.

**D3 — Segment + procedure are ONE system.** Track enters as parameters (window
[start,end] vs [0,T]; 15 vs 30 s/q). Practical driver: seg+proc share **one pool of
10 pre-eval submissions** (3 spent → ~7 left; cutoff 2026-09-01, docker deadline
2026-09-08), so shared components double the information per submission.

**D4 — Unified multi-scale probe is the end state; per-track adapters are
transitional floors only.** The last unified run's segment regression (tempG −0.07,
p=3e-11) was two absolute-timestamp granularities fighting over one output head.
Fix, when we train it: (a) **scale-free output = frame-index pointing** (code maps
index→seconds; no clock arithmetic — our worst skill); (b) **continuous scale
augmentation**: from each (video, event-time) synthesise windows 30 s→2 h containing
the event, ≤64 frames, supervise the frame index; preamble states span+interval =
scale conditioning; (c) augment *pointing* only — durations are computed from two
located boundaries, never eyeballed. Probe context is flat (≤64 frames) at any
scale; each scan level refines 64× (2 h → ~2 s precision in 2 levels).

**D5 — Routing: one batched 27B zero-shot call per 20-question batch.** System
prompt = 15 taxonomy definitions (`src/visualisation/capabilities.py`) + per-question
`answer_format` hint (given in input; cross-tab of 20,619 rows shows format nearly
determines capability: time→2a/2b, percentage→2b, MC→1d, number→3a/3b,
fo_class→1a+tails, binary→1a/1b/4a/4c, open_ended→groups 4/5). Output per question:
`{capability, ood_suspect, reason}`. No trained/lexical router (immune to hidden
generator phrasing). **Validate once** on the 34k labelled local rows.

**D6 — OOD = the router's `ood_suspect` flag; no video-level machinery.** Prompt
enumerates the 8 FO classes seen in training + the 2 never seen (Mesh, Absorbable
Hemostatic Agent). Flag → CoT route; final answer by **27B arbitration**: floor
answer + CoT answer + evidence frames presented symmetrically, verdict must cite
evidence (guards: self-preference, position bias, deference to confident SFT tone).
**Validate locally** on rows where floor≠CoT (gold known): any slice where
arbitration < max(floor, CoT) gets a fixed policy. Video procedure-type is prompt
*context* only.

**D7 — Budget: per-question step limits from the window length** (<15 min → small,
≥15 min → large; general form: scan depth ≈ ⌈log₆₄(window/precision)⌉). Pool per
batch ≈ 120 s setup + 20×15 s (seg) / 20×30 s (proc). Non-negotiable rails:
(1) **floor pass writes a complete valid output file before any upgrade** — forfeits
(bucket-stratified!) become impossible; (2) wall-clock watchdog + max_new_tokens
caps; (3) step limits sized so the worst-case batch fits the pool, using a measured
per-step cost table.

**D8 — Bounded workflow library, iterations must acquire new pixels** (no rhetorical
self-refinement):
- *first-occurrence* (2a): coarse presence timeline → anchor earliest **confirmed**
  positive / last negative → zoom ×2 levels. NOT naive binary search — visibility is
  non-monotone (naive 3-chunk retrieval publicly scored 0.115 tempG).
- *duration/percentage* (2b): locate both boundaries finely, subtract/divide in code.
- *inventory / counting* (3a/3b): map-reduce over windows with **typed observations**
  `{object, ts, position, conf}` + explicit dedup at window seams (free-text
  summaries amplify double-counting). Attacks the ≥6-count collapse (3% correct).
- *ordering* (4c): locate each event, compare timestamps **in code**.
- *interaction/purpose/reasoning* (4a/4b/5): gather evidence clips → frozen 27B CoT.

**D9 — Probe API = typed functions rendering EXACT training templates.** Workflows
never freestyle prompts to SFT adapters; only the frozen 27B gets free-form text.
Transitional toolbox (adapters used strictly in-distribution):
- per-frame `inventory(frame)` / `presence(frame, obj)` = **frame adapter** with
  frame-track templates — per-frame presence is the only *scale-free* primitive we
  own (works at 156 s spacing that no video adapter ever saw); coarse scans = ~64
  batched single-frame calls;
- `first_appearance(window, obj)` at ≤few-minute windows @1 fps = **segment adapter**
  (its literal training task);
- frame-adapter accuracy ~0.79 → a 64-frame timeline WILL contain errors → never
  anchor on a single unconfirmed positive (27B verify or consecutive-positives).
The toolbox's function list **is** the sub-task SFT curriculum for D4's unified
probe; the toolbox retires function-by-function when the trained probe beats it.

**D10 — Tier-1 is inference-only.** Everything above runs on existing checkpoints.
Training is triggered by simulator evidence, in this order of likelihood:
(1) unified multi-scale probe (D4) once the toolbox's seams are measured;
(2) convention patches (segment `percentage` ≈ 0 on exact match) riding along;
(3) 27B probe adapter only if probe *accuracy* is the binding error source;
(4) trajectory distillation / synthetic group-4/5 data after the pipeline exists.

## 3. Worked example (agreed walkthrough)

Video 00:00:00–02:46:00, question "When is the first time the clip was placed?"
(format `time`): floor pass writes fallback answer (~2 s) → batched router:
{2a, first-occurrence, large step limit} → shared index: 64 frame-adapter inventory
calls → timeline: first clip evidence frame #12 (00:31:12), #11 negative → Level 1:
64 frames over [00:28:36, 00:31:12] (~2.4 s spacing), probe points at transition →
Level 2: 1 fps dense → 00:30:07 → 27B verifies transition frames (stapler vs clip
applier; placement-event vs mere visibility) → arbitration (floor 00:35:12 vs
workflow 00:30:07 + evidence) → emit `00:30:07`. ~13 s of a ~35 s/q pool share.
Failure paths: empty timeline → one offset scan → "never"/floor via arbitration;
watchdog → stored answer stands; ood_suspect → CoT + arbitration.

## 4. Interface facts — verified & to verify

Verified: input rows carry `answer_format` (runner builds prompts from it);
submission = offline docker, /input videos, /output answers; 8 answer formats.
**To verify from past submission logs/metrics (no submission cost):** batch == one
video's 20 questions? budget pooled per batch vs hard per-question? eval GPU model
(procedure evidence suggests 80 GB-class)? exact input JSON fields. Also re-measure
27B cold-load in-container (~89 s from shared FS vs 120 s setup allowance).

## 5. Roadmap

1. **Bucket-macro evaluation** (step 1, plan in `bucket-eval-plan.md`) — CPU-only:
   all needed full tests already have per-question `eval/results.csv` on civo.
2. Interface verification (§4) from existing submission artifacts.
3. Tier-1 pipeline + simulator: workflows on existing checkpoints, per-step latency
   cost table, bucket-by-bucket comparison vs floors. Local validation experiments:
   router accuracy (D5), arbitration accuracy on disagreements (D6).
4. **Prompt/sampling playground** (user-requested): interactive interface to pick a
   video+question, edit the system prompt, choose the sampling strategy (uniform N /
   window+density / zoom level), and see the sampled frames, the **exact rendered
   conversation**, output, and latency; every trial logged as a structured record so
   findings promote to batch simulator runs. Extends the existing vis-tool infra.
5. First submission spends only after tier-1 beats floors in the simulator; segment
   board entry (1 Aug) is stale — banking current-best segment is a candidate spend.
6. Training tiers per D10.

Frame track: continues separately per `frame-track-v2-plan.md` (no agent loop; its
analog = enumerate-then-count/tiled). Note: `os-models/` already holds frame-ep24
27B variants (merged/int8/int4/FP8) from the v2 workstream — bucket-eval sweep will
score those runs too. Quantisation warning stands: a rival's int8/4-bit 27B scored
0.137 vs 0.459 bf16 — always accuracy-validate quantised variants locally.
