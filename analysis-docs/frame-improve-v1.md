# Frame Track Performance and Improvement Analysis (v1)

Analysis date: 2026-07-26

## Scope

This document analyses the current Frame-track model:

- Base model: `Qwen3-VL-4B-Instruct`
- Adapter: `track-frame/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/final`
- Training endpoint: epoch 30
- Training data: the official HeiCo and LapChole Frame training sets
- Evaluation run:
  `track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/`
- Evaluation coverage: all 4,000 HeiCo and 2,252 LapChole test questions
- Image/inference failures: zero

The challenge's headline accuracy is the mean of per-video accuracies. The
capability and answer-format tables below instead show transparent row-level
exact accuracy (`correct / cases`). For these test sets, the difference between
the two aggregations is small.

## Headline performance

| Dataset | Correct / questions | Row accuracy | Challenge per-video accuracy |
| --- | ---: | ---: | ---: |
| HeiCo | 2,657 / 4,000 | 66.4% | 0.664 |
| LapChole | 1,250 / 2,252 | 55.5% | 0.555 |
| Combined | 3,907 / 6,252 | 62.5% | — |
| Equal-dataset mean | — | — | 0.610 |

The model is genuinely better than the previous epoch-24 checkpoint on the
same current test data: +3.4 percentage points on HeiCo and +16.6 points on
LapChole. The historical result around 0.83 came from the older 2,000-row HeiCo
test release and is not directly comparable.

## Performance by official capability

### Capability groups

| Group | HeiCo | LapChole | Combined | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 1. Object recognition and identity matching | 1,513 / 2,125 (71.2%) | 788 / 1,296 (60.8%) | 2,301 / 3,421 (67.3%) | Moderately strong, but the score is substantially depressed by a multi-label output-adapter defect described below. |
| 2. Temporal grounding | 0 / 0 | 0 / 1 (0.0%) | 0 / 1 (0.0%) | One question is not enough for a conclusion. |
| 3. Aggregation | 1,144 / 1,875 (61.0%) | 462 / 955 (48.4%) | 1,606 / 2,830 (56.7%) | The main genuine weakness, dominated by exact instance and clip counting. |
| 4. Event and procedural understanding | 0 / 0 | 0 / 0 | 0 / 0 | Not represented as a primary Frame-test capability. |
| 5. Complex reasoning | 0 / 0 | 0 / 0 | 0 / 0 | Not represented as a primary Frame-test capability. |

### Sub-capabilities

The zero-case rows are important: the current Frame evaluation provides no
evidence about those abilities.

| Code | Official capability | HeiCo | LapChole | Combined |
| --- | --- | ---: | ---: | ---: |
| 1a | Object identification | 1,097 / 1,591 (69.0%) | 461 / 866 (53.2%) | 1,558 / 2,457 (63.4%) |
| 1b | Object instance identity matching | 0 / 0 | 0 / 0 | 0 / 0 |
| 1c | Object attributes and state | 18 / 21 (85.7%) | 119 / 192 (62.0%) | 137 / 213 (64.3%) |
| 1d | Object spatial localization relative to the camera | 345 / 427 (80.8%) | 187 / 213 (87.8%) | 532 / 640 (83.1%) |
| 1e | Object spatial localization relative to the situs | 53 / 86 (61.6%) | 21 / 25 (84.0%) | 74 / 111 (66.7%) |
| 2a | Temporal localization | 0 / 0 | 0 / 1 (0.0%) | 0 / 1 (0.0%) |
| 2b | Duration estimation | 0 / 0 | 0 / 0 | 0 / 0 |
| 3a | Object aggregation | 1,144 / 1,875 (61.0%) | 462 / 955 (48.4%) | 1,606 / 2,830 (56.7%) |
| 3b | Event aggregation | 0 / 0 | 0 / 0 | 0 / 0 |
| 4a | Foreign-object interaction recognition | 0 / 0 | 0 / 0 | 0 / 0 |
| 4b | Foreign-object usage purpose | 0 / 0 | 0 / 0 | 0 / 0 |
| 4c | Temporal ordering | 0 / 0 | 0 / 0 | 0 / 0 |
| 5a | Functional reasoning | 0 / 0 | 0 / 0 | 0 / 0 |
| 5b | Causal and consequence reasoning | 0 / 0 | 0 / 0 | 0 / 0 |
| 5c | Multi-step compositional reasoning | 0 / 0 | 0 / 0 | 0 / 0 |

The official taxonomy is described on the
[FOCUS data page](https://procedure.orena-focus-challenge.org/data/). Secondary
tags exist, but they overlap heavily; primary assignments are used above so
each question contributes to exactly one sub-capability.

## Performance by answer format

| Answer format | HeiCo | LapChole | Combined |
| --- | ---: | ---: | ---: |
| Binary | 448 / 548 (81.8%) | 154 / 176 (87.5%) | 602 / 724 (83.1%) |
| Foreign-object class | 1,216 / 1,755 (69.3%) | 501 / 920 (54.5%) | 1,717 / 2,675 (64.2%) |
| Multiple choice | 102 / 112 (91.1%) | 84 / 90 (93.3%) | 186 / 202 (92.1%) |
| Number | 696 / 1,326 (52.5%) | 304 / 768 (39.6%) | 1,000 / 2,094 (47.8%) |
| Open ended | 195 / 259 (75.3%) | 207 / 298 (69.5%) | 402 / 557 (72.2%) |

Binary and multiple-choice questions are strong. Number questions are the
largest real weakness. The foreign-object-class score contains a major
evaluation-path defect and should not be interpreted at face value.

## Priority 0: repair the multi-label foreign-object adapter

The installed `focus` package in the `orena` environment now defines
`FOClass` as accepting one or more comma-separated class names and comparing
them as an order-insensitive set. The repository adapter is stale:

- `src/adapter.py::normalize_fo_class()` returns only the first recognised
  class.
- `src/prompts.py` tells the model to answer with exactly one class.
- The corresponding comments still claim that `fo_class` is single-label.

This creates a deterministic scoring failure:

| Multi-label target subset | Cases | Currently correct | Raw model output already exactly correct |
| --- | ---: | ---: | ---: |
| HeiCo | 277 | 0 | 179 (64.6%) |
| LapChole | 333 | 0 | 199 (59.8%) |
| Combined | 610 | 0 | 378 (62.0%) |

For example, when the target and raw output are both `Clip, Sponge`, the
adapter changes the submitted prediction to `Clip`, making it wrong.

If the adapter merely preserved those 378 already-correct raw answers without
changing any other result, the counterfactual scores would be:

| Scope | Current | Recoverable after adapter-only correction |
| --- | ---: | ---: |
| HeiCo row/per-video accuracy | 66.4% | 70.9% |
| LapChole row accuracy | 55.5% | 64.3% |
| LapChole per-video accuracy | 55.5% | 64.3% |
| Combined row accuracy | 62.5% | 68.5% |
| Capability 1a combined | 63.4% | 78.8% |
| `fo_class` combined | 64.2% | 78.3% |

These are recoverable counterfactuals, not a newly evaluated run. The correct
next action is to update the prompt and normaliser, add single-label,
multi-label, ordering, duplicate, `Specimen`/`Specimen Bag`, and `none` tests,
and then rerun the full evaluation.

## Main genuine weakness: number and instance aggregation

### Accuracy by counting task

| Counting task | Correct / cases | Accuracy |
| --- | ---: | ---: |
| Count all visible object instances | 329 / 830 | 39.6% |
| Count Clips | 210 / 681 | 30.8% |
| Count distinct object classes | 328 / 436 | 75.2% |
| Count Sponges | 77 / 83 | 92.8% |
| Count External drains | 38 / 45 | 84.4% |
| Count Needles | 8 / 9 | 88.9% |
| Count Specimens | 6 / 6 | 100.0% |
| Count Specimen bags | 4 / 4 | 100.0% |

The model can usually identify which classes are present, and it can count
large or isolated objects such as sponges and drains. It fails when it must
separate many visually similar instances, especially small clips. This is
evidence of insufficient instance grounding rather than a general inability
to understand numbers.

### Counting error behaviour

Across all 2,094 number questions:

- Exact accuracy: 47.8%
- Mean absolute error: 0.86 objects
- Signed bias: -0.39 objects
- Undercount: 33.9% of questions
- Overcount: 18.3% of questions

| Ground-truth count | Correct / cases | Exact accuracy | Mean absolute error | Undercount rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 584 / 737 | 79.2% | 0.26 | 0.0% |
| 2–3 | 320 / 803 | 39.9% | 0.74 | 35.0% |
| 4–5 | 94 / 352 | 26.7% | 1.28 | 65.3% |
| 6 or more | 2 / 202 | 1.0% | 2.80 | 98.5% |

For clip-count questions alone, accuracy is 30.8%, MAE is 1.22, signed bias is
-0.62, and 45.1% are undercounts. Only 1 of 103 clip-count questions with a
ground-truth count of at least six is correct.

The HeiCo model never predicts more than six on number questions despite
ground truth reaching twelve. This strongly suggests learned answer-range
compression and visual under-segmentation, not occasional arithmetic errors.

## Insights for the remaining capabilities

### 1a — Object identification

Single-class targets are already comparatively strong: 1,717 of 2,065
single-class `fo_class` cases are correct (83.1%). Multi-object completeness is
weaker even before the adapter defect: the raw answer contains the complete
correct set on 378 of 610 multi-label cases (62.0%).

After fixing the adapter, improvement should focus on missed secondary
objects, especially small clips, and on rare classes such as gallstones.
Multi-label supervision should use an order-invariant objective or canonical
class ordering rather than teaching one free-form string.

### 1c — Object attributes and state

| State task | Correct / cases | Accuracy |
| --- | ---: | ---: |
| Grasp state | 62 / 97 | 63.9% |
| Occlusion by an instrument | 48 / 64 | 75.0% |
| Occlusion by anatomy | 17 / 40 | 42.5% |
| Object condition/state | 9 / 11 | 81.8% |

Anatomical occlusion is much harder than instrument occlusion, probably
because boundaries between tissue and a partially hidden object are
ambiguous. Targeted hard examples, occlusion labels, and segmentation masks
would be more useful than simply adding more generic VQA epochs.

### 1d — Camera-relative localization

This is the strongest well-supported capability at 83.1%. Quadrant
multiple-choice questions reach 92.1%; class answers associated with
localization reach 73.2%. The model understands coarse image geometry, but
identifying the correct member of a crowded scene remains harder. A detector
would provide both class and coordinates and should make this capability more
deterministic.

### 1e — Situs-relative localization

The combined result is 66.7%, but only 111 questions exist and the two datasets
are imbalanced. Many questions combine anatomy with phrases such as “before
being not visible for more than one minute,” even though the Frame model sees
one selected image. This mixes anatomical recognition with temporal context
and makes causal attribution difficult. It should be evaluated by question
template and with manual review before changing the training recipe.

### Unsupported capabilities

The Frame test has no primary examples for 1b, 2b, 3b, any group-4 capability,
or any group-5 capability. High or low accuracy on their sparse secondary tags
would not be reliable evidence. Research claims should be restricted to the
capabilities with meaningful primary support.

## Why the present model behaves this way

1. **A stale output contract hides correct recognition.** Multi-label answers
   are collapsed before evaluation.
2. **SFT supervises the final answer, not the visual instances.** A correct
   count produces one token-level target but does not teach where each object
   is or how instances should be separated.
3. **Small clips are compressed by the image-token budget.** The current
   602,112-pixel cap is adequate for global scene understanding but can erase
   the boundaries between adjacent clips.
4. **Exact counting compounds missed instances.** One missed clip makes the
   whole answer wrong; high-count frames therefore collapse rapidly.
5. **Answer-frequency learning encourages conservative counts.** The negative
   bias and truncated prediction range indicate that the model has learned
   common counts more strongly than a general enumeration operation.
6. **LapChole adds domain and composition shift.** It has more multi-label
   class targets proportionally, weaker number accuracy, different anatomy,
   and different object combinations. The adapter defect explains part, but
   not all, of its lower score.

## Prioritised improvement plan

### P0 — Correctness of the current pipeline

1. Make `fo_class` prompting and normalisation multi-label aware.
2. Add regression tests against the installed `focus.FOClass`.
3. Rerun the same 6,252 predictions through the corrected adapter first, then
   rerun full model inference only if necessary.
4. Refresh the visualiser and this document with corrected official scores.

### P1 — Instance-grounded counting

1. Annotate a case-disjoint pilot containing boxes, points, or masks for every
   visible FOCUS object instance.
2. Deliberately oversample clips, counts of four or more, overlapping
   instances, partial occlusion, blur, blood, smoke, and both datasets.
3. Train a high-resolution detector baseline and report exact count accuracy,
   count MAE, undercount rate, small-object recall, and per-class AP.
4. Use detector outputs as structured evidence:
   `class, confidence, centre, box/mask, count`.
5. Answer simple count, presence, co-occurrence, and quadrant questions
   deterministically from detections; give the same evidence to the VLM for
   open-ended questions.

The detector literature and a proposed experimental design are summarised in
[frame-detection-review.md](frame-detection-review.md).

### P2 — Count-aware VLM training

After a detector baseline establishes that the instances are visually
recoverable:

- Add grounding examples in which the assistant produces a canonical object
  list before the final count.
- Balance batches by object class and ground-truth count.
- Use high-count curriculum and hard-example mining.
- Train on high-resolution crops or tiled regions for clip clusters.
- Consider an auxiliary box, point, density-map, or count loss only after the
  separate detector baseline is understood.

A new loss is therefore not the first required step. Separate detection is
easier to diagnose and can establish whether the bottleneck is visual
resolution, annotation ambiguity, or VLM reasoning.

### P3 — State and anatomical reasoning

- Build focused subsets for grasping and instrument-vs-anatomy occlusion.
- Add masks or occlusion fractions where feasible.
- Report performance per question template, not only pooled capability.
- Review the temporal wording of Frame questions before attributing 1e errors
  to spatial reasoning.

## Reproducibility notes

The statistics were derived directly from:

- `../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/heico/predictions.parquet`
- `../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/heico/eval/results.csv`
- `../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/lapchole/predictions.parquet`
- `../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/lapchole/eval/results.csv`

Related comparison and integrity checks are in
[frame-model-full-test-comparison.md](../docs/frame-model-full-test-comparison.md).
