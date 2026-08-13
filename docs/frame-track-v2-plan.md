# FRAME track v2 — plan to pass the leaderboard leader

Status: proposal, 2026-08-10. Leader `pre_evaluation_score` **0.6235** (Qwen3.6-27B).
Ours **0.5310** (Qwen3-VL-4B LoRA, epoch 30). Gap **0.0925**.

## 1. What the score actually is

FRAME populates only 4 of the 10 buckets, each worth exactly 25%:

| bucket | questions | leader | ours | gap | value of 1 question |
| --- | ---: | ---: | ---: | ---: | ---: |
| object_recognition_id | 747 | 0.7416 | 0.6560 | +0.086 | 3.3e-4 |
| object_recognition_ood | 512 | 0.5430 | 0.4121 | **+0.131** | 4.9e-4 |
| aggregation_id | 553 | 0.5606 | 0.5027 | +0.058 | 4.5e-4 |
| aggregation_ood | **94** | 0.6489 | 0.5532 | +0.096 | **2.7e-3** |

(Bucket sizes recovered from the exact accuracy fractions; both score files
reproduce to 1e-15 as the mean of these four.)

Three consequences drive everything below:

- **OOD is 606 questions but half the score.** Our worst bucket is
  object_recognition_ood, and it is also where the leader beats us hardest.
- **One aggregation_ood question is worth 8 object_recognition_id questions.**
  Polishing strong ID buckets is close to worthless.
- Composition by answer_format (from the official annotations):
  aggregation = 76% `number` + 23% `binary`; object_recognition = 77%
  `fo_class` + 16% `open_ended` + 7% `multiple_choice`. So **counting alone
  carries ~38% of the final score** and fo_class ~38%.

## 2. Diagnosis (measured, epoch-30 full test, 6,252 rows)

Rescored buckets: aggregation 0.5675, object_recognition 0.7793.

**a. Counting is the dominant failure and it is range compression, not noise.**
`number` exact 0.478, within ±1 **0.798**. Accuracy by ground-truth count:

| GT | 1 | 2 | 3 | 4 | 5 | ≥6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| acc | 0.792 | 0.465 | 0.272 | 0.289 | 0.238 | 0.011 |
| n | 737 | 527 | 276 | 201 | 151 | 202 |

The model **almost never predicts above 5** — 18 sixes, 4 sevens, 1 nine out of
2,094 answers, against ground truth reaching 12. Errors split 40% in the 1–2
regime, 42% in 3–5, 18% at ≥6, so this is a perception limit across the whole
range, not just a tail problem.

**b. Two of ten canonical classes are absent from all training data.**
`FOType.names()` has 10 classes; the official FRAME answers only ever use 8.
**Mesh** and **Absorbable Hemostatic Agent** never appear in HeiCo or LapChole.
A 30-epoch LoRA cannot emit a label it has never produced, so every hidden-OOD
question involving those two is lost by construction — a prime suspect for the
object_recognition_ood gap.

**c. Prompt defects** (all three confirmed against the annotations; fixed in §3).

**d. We spend 7% of the latency budget.** Batches are 20 questions → the
allowance is 120 s setup + 20×5 s = **220 s**. We use 15.3 s, the leader 42 s.
Roughly 14× unspent compute.

### Ideas tested and rejected — do not relitigate

| idea | result |
| --- | --- |
| Post-hoc recalibration of predicted counts | −0.015 group-aware, −0.074 cross-dataset. Predictions are already argmax-calibrated (`pred=k` → modal GT `k` for k≤5). |
| Higher resolution (602k → 1003k px) | +0.26 pp overall, +0.73 pp counting. Noise. |
| Decompose instance-count into per-class counts | Per-class counts are distributed almost identically to instance totals (max 10 vs 11), so the sub-questions are no easier. |
| bbox-prompt counting on the 4B LoRA | Rejected 2026-07-27; systematic undercount at ≥2 objects. |

### One free win found

"How many different foreign object **classes**?" is better answered through the
class-**list** head: |predicted set| is right **0.840** of the time versus
**0.752** for the direct number question (n=1,313 vs 436). The size can be right
when the set is wrong. That is +8.8 pp on 19% of `number` rows, no training.

## 3. Prompt corrections (done — `prompt_strategy="v2"`)

Added additively to `src/prompts.py`; every existing caller uses positional
defaults, so the in-flight unified run is untouched. 74 tests pass, including
`test_unified_parity.py` and 15 new v2 tests.

1. **Procedure type was a lie.** The system prompt hard-coded *"a single
   endoscopic frame from colorectal surgery"* for every row — false for all
   8,000 LapChole rows (Laparoscopic Cholecystectomy) and for whatever the
   hidden OOD split holds. v2 takes the row's own `procedure_type`, falling back
   to "minimally invasive" rather than guessing.
2. **fo_class arity was unconditioned.** v1 always said "one or more … separate
   with a comma", including on the 4,365 of 8,969 FRAME fo_class rows (48.7%)
   that are single-answer by construction. The questions state their own arity —
   *"Please provide a class name"* vs *"the class names"* — a rule with **zero
   violations across all 16,190 fo_class rows in all three tracks**. v2 pins
   single-answer templates to one class and keeps the set instruction for
   `List all` / `Which combination` (28.3% of fo_class answers really are
   multi-class).
3. **open_ended contradicted the question.** The 646 positional-enumeration rows
   ask for *"1. Sponge: top/left 2. Needle: bottom/left"* and v1 appended
   "Answer concisely, in a few words". v2 detects self-formatting questions and
   only adds the 300-character guard (`OpenEnded.read` rejects longer answers
   before the judge ever sees them).
4. **Class roster in scope.** v2 names all ten canonical classes with one-line
   visual descriptors faithful to the challenge's own `FO_definitions.txt` — the
   same text the official judge reads — so Mesh and Absorbable Hemostatic Agent
   stay emittable. Costs ~400 tokens of system prompt.

Note on multiple_choice: the user's suspicion does not apply there. All 842
FRAME MC rows are single-answer quadrant questions, so "exactly one" is correct.
But MC and open_ended are **LLM-judged, not exact-matched**, and the judge is
told to accept a right answer that carries extra text — so terseness is a
constraint we impose on ourselves, not one the scorer imposes. Only the 300-char
cap is real.

## 4. Plan

**Stage 1 — find the capacity ceiling (inference only, 1 H200 node, ~½ day).**
Zero-shot `Qwen3.6-27B` (natively multimodal, apache-2.0, bf16 fits an H200 NVL)
with v2 prompts on our full local test, plus `Qwen3-VL-8B` and the current
4B LoRA as controls, all scored with the 4-bucket leaderboard metric. This is
the one measurement that decides the rest: how much of the leader's 0.0925 is
capacity and how much is task engineering.

**Stage 2 — make local evaluation OOD-honest.** We currently early-stop on
in-distribution test while OOD is half the score. Switch to leave-one-dataset-out
(train HeiCo → select on LapChole and vice versa) and report the 4-bucket metric
locally instead of per-video macro accuracy.

**Stage 3 — train.** LoRA on the chosen base, **6–10 epochs, not 30**, with
v2 prompts, count-balanced sampling to attack range compression, and the class
roster held in scope.

**Stage 4 — spend the latency budget** (~14× unspent):
- tiled counting: 2×2 crops with a "count only objects whose centre lies in this
  crop" rule, summed, plus the whole frame as a check — converts one hard
  count of 8 into four easy counts of 0–3;
- TTA self-consistency on `number` (flips/crops; median vs max is an open
  question given the systematic undercount);
- route class-count questions through the list head (+8.8 pp, validated);
- allow brief reasoning **only** on judge-scored formats — never on `fo_class`,
  where `normalize_fo_class` substring-matches class names and would pick up
  every class mentioned in the reasoning.

**Stage 5 — package for 1× L40S 48 GB.** 27B needs FP8 (~27 GB) or AWQ-INT4
(~15 GB) plus vLLM; merge LoRA to bf16, then quantize. Model load eats the 120 s
setup allowance, so measure it on L40S-class hardware **early** — this is the
main deployment risk.

## 4b. Stage 1 operational findings (2026-08-10)

Recorded before the accuracy numbers land, because they change how any Qwen3.6
model must be run.

**Qwen3.6 is a hybrid-thinking model and ships thinking on.** Its chat template
opens a `<think>` block unless `enable_thinking=false`. Run as-is, the first
smoke took **12.0 s per question** and returned truncated reasoning instead of an
answer — every row unparseable. With thinking suppressed: **0.48 s per question,
40/40 parseable, 0 errors**. The leader's 42 s for a 20-question batch implies
they run the same way. `track-frame/v2/src/generate.py` owns the toggle and
strips any reasoning block defensively, because `adapter.normalize_fo_class`
substring-matches class names and would otherwise harvest every class the
reasoning mentioned.

**Model load: 88.9 s cold from the shared filesystem, 8.6 s warm.** The setup
allowance is 120 s, so cold load alone would eat it. Deployment must stage
weights on container-local disk; measure on L40S-class hardware before
submitting.

**HeiCo and LapChole share question id 2657592.** Ids are unique within a
dataset but not across them, so *any* run that scores both datasets through one
`focus.Evaluator` dies on `Duplicate response for qID`. This never surfaced
before because the existing full test scores each dataset in its own directory.
`track-frame/v2/src/benchmark.py` prefixes qIDs with the dataset;
`tmp/frame-rethink/repair_qids.py` backfills runs inferred before the fix.

**The official train/test splits are video-disjoint** — 0 shared videos and 0
shared (video, timestamp) frames, in both datasets. So local test numbers are an
honest held-out-video measure, and the distance to the leaderboard is the hidden
set being harder, not leakage.

**Scale does nothing zero-shot; adaptation does everything.** From the earlier
zero-shot sweep (HeiCo test, v1 prompts, two-bucket mean): Qwen3-VL-4B 0.371,
8B 0.363, **32B 0.362**, best-of-sweep InternVL3.5-30B-A3B 0.517. The shipped
fine-tuned 4B scores **0.701** on the same HeiCo two-bucket view (0.661 over both
datasets, four buckets). Fine-tuning that 4B is worth **+0.33**; scaling it to
32B zero-shot is worth **-0.01**. The leader's 27B is therefore near-certainly
fine-tuned, and the marginal value of a fine-tuned 27B over a fine-tuned 4B is
an open question rather than an established 0.0925.

Shipped-model reference, in the four-bucket view (Qwen3-VL-4B epoch 30, v1
prompts, 6,252 local test rows):

| | HeiCo | LapChole |
| --- | ---: | ---: |
| object_recognition | 0.7920 | 0.7585 |
| aggregation | 0.6101 | 0.4838 |

Four-bucket mean **0.6611**.

**Zero-shot 27B is heavily conservative on this domain.** On the 40-row smoke it
answered "none" 13 times and counted 0 where ground truth ran 1-6, scoring 0.245.
Quadrant multiple-choice (0.60) and open-ended (0.71) held up; `fo_class` (0.19)
and `number` (0.00) did not. It does emit Mesh — the class roster in the v2
prompt works — but misapplies it. Provisional reading: the leader's 0.6235 is
**not** zero-shot, and fine-tuning is doing most of the work in this track. Full
stratified runs will confirm or overturn this.

## 5. Open decisions

1. **Base model.** `Qwen3.6-27B` (matches the leader; needs FP8/INT4 at
   inference) vs `Qwen3.6-35B-A3B` (MoE, 3B active → far faster, ~35 GB FP8) vs
   staying at 8B and winning on task engineering.
2. **Compute.** The a100 box is held by the unified run (job 3450). The h200
   partition is 8 free single-GPU nodes at 141 GB each — enough to LoRA a 27B on
   one node, and enough to run several experiments in parallel.
3. **Fork or converge.** Does FRAME leave the unified model, or does the unified
   run continue alongside?
