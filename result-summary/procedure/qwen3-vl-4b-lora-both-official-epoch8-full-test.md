# PROCEDURE epoch-8 full-test capability analysis

## Outcome

The selected submission model is the Qwen3-VL-4B LoRA adapter trained on both
official datasets for 8/8 epochs, using `checkpoint-3440`. Its complete local
test run covered all 3,127 official Procedure questions with zero inference
errors and zero 30-second timeouts.

The model is strongest on object recognition, identity matching, binary
questions, and open-ended reasoning. Its dominant weakness is exact temporal
grounding. Long-horizon aggregation is the second major failure mode on HeiCo,
especially event counts above five. These two groups account for 76.2% of the
HeiCo test rows and largely explain the low HeiCo headline score.

| Dataset | Rows / videos | Correct | Official accuracy (95% CI) | Question-micro accuracy | Pre-evaluation score |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 2,000 / 10 | 621 | 0.3105 [0.2825, 0.3395] | 0.3105 | 0.5282 |
| LapChole | 1,127 / 28 | 610 | 0.5412 [0.4869, 0.5970] | 0.5413 | 0.5789 |

The two dataset scores should remain separate. They cover different procedure
types and substantially different capability and answer-format mixtures. The
mean of the two challenge-style pre-evaluation scores is 0.5535, but this is a
convenient cross-dataset summary, not a single official leaderboard score.

The full test completed on 2026-07-29. This analysis was produced on
2026-07-31.

## Scoring and scope

The official point estimate is the mean of per-video correctness means. The
reported 95% intervals come directly from `orena-focus` 0.3.4: a two-level
hierarchical bootstrap first resamples videos and then questions within each
selected video, using 1,000 samples and seed 42. Consequently, a point score
can differ slightly from `correct / rows`, particularly for LapChole, where
videos contain 40–42 questions.

The official evaluator used exact-format comparison for binary, FO-class,
number, percentage, and time answers. Open-ended and multiple-choice answers
were judged with the local Qwen3.5-4B judge. Time answers used a per-question
acceptance window between one and five seconds. Procedure responses had a
30-second timeout.

## Capability groups

| Capability group | HeiCo rows / share | HeiCo score (95% CI) | LapChole rows / share | LapChole score (95% CI) | LapChole − HeiCo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object recognition and identity matching | 374 / 18.70% | 0.6585 [0.5794, 0.7388] | 564 / 50.04% | 0.7584 [0.6963, 0.8193] | +0.0999 |
| Temporal grounding | 795 / 39.75% | 0.1162 [0.0684, 0.1737] | 426 / 37.80% | 0.2870 [0.2099, 0.3782] | +0.1709 |
| Aggregation | 728 / 36.40% | 0.2992 [0.2313, 0.3767] | 95 / 8.43% | 0.6330 [0.5083, 0.7631] | +0.3338 |
| Event and procedural understanding | 52 / 2.60% | 0.8200 [0.6633, 0.9433] | 17 / 1.51% | 0.4615 [0.1923, 0.7308] | −0.3585 |
| Complex reasoning | 51 / 2.55% | 0.7148 [0.5000, 0.9186] | 25 / 2.22% | 0.8929 [0.7381, 1.0000] | +0.1780 |

The scalable strength is object recognition: it has hundreds of questions in
both datasets and scores 0.66–0.76. Event understanding and complex reasoning
look stronger, but together they represent only 4.2–5.2% of each dataset and
their leaf composition differs sharply, so their headline values are not
stable cross-dataset comparisons.

Temporal grounding is consistently weak and heavily represented in both test
sets. Aggregation is also weak on HeiCo, where it is more than four times as
prevalent as in LapChole. LapChole's high aggregation score is partly a
question-type effect: its event-aggregation questions are open-ended, whereas
all 326 HeiCo event-aggregation questions require exact numeric answers.

### Why LapChole is 23 points higher

The question-micro gap is 0.2308. A symmetric decomposition by capability
group attributes 0.0836 of the gap to the different group mixture and 0.1472
to within-group performance differences.

| Capability group | Composition contribution | Within-group contribution |
| --- | ---: | ---: |
| Object recognition | +0.2181 | +0.0262 |
| Temporal grounding | −0.0037 | +0.0562 |
| Aggregation | −0.1211 | +0.0702 |
| Event understanding | −0.0072 | −0.0077 |
| Complex reasoning | −0.0026 | +0.0023 |
| **Total** | **+0.0836** | **+0.1472** |

About 36% of the observed gap is therefore associated with capability mix and
64% with within-group accuracy. This is descriptive, not causal: dataset,
procedure type, video duration, question templates, and capability mix all
change together.

## All primary leaf capabilities

Every primary leaf is shown below. A dagger marks fewer than 30 questions;
those scores are coverage facts rather than reliable rankings.

| Code | Primary capability | HeiCo score (95% CI), n | LapChole score (95% CI), n |
| --- | --- | ---: | ---: |
| 1a | Object identification | 0.6180 [0.5048, 0.7293], 261 | 0.7633 [0.6874, 0.8312], 402 |
| 1b | Instance matching | 0.7990 [0.6791, 0.9100], 61 | 0.8601 [0.7797, 0.9435], 99 |
| 1c | Object attributes | Not evaluated, 0 | Not evaluated, 0 |
| 1d | Spatial localization — camera | 0.7582 [0.5463, 0.9273], 50 | 0.6661 [0.4750, 0.8418], 53 |
| 1e | Spatial localization — situs | 1.0000 [1.0000, 1.0000], 2† | 0.7857 [0.5000, 1.0000], 10† |
| 2a | Temporal localization | 0.1144 [0.0654, 0.1770], 701 | 0.2891 [0.2052, 0.3779], 391 |
| 2b | Duration estimation | 0.1374 [0.0600, 0.2329], 94 | 0.1604 [0.0000, 0.3751], 35 |
| 3a | Object aggregation | 0.4078 [0.3295, 0.4914], 402 | 0.5580 [0.4135, 0.7185], 77 |
| 3b | Event aggregation | 0.2008 [0.0903, 0.3422], 326 | 0.8529 [0.6765, 1.0000], 18† |
| 4a | FO-interaction recognition | 0.7000 [0.4500, 0.9000], 23† | 0.5714 [0.2857, 0.8571], 8† |
| 4b | FO-usage purpose | 1.0000 [1.0000, 1.0000], 1† | 0.7500 [0.2500, 1.0000], 4† |
| 4c | Temporal ordering | 0.9630 [0.8519, 1.0000], 28† | 0.2000 [0.0000, 0.6000], 5† |
| 5a | Functional reasoning | Not evaluated, 0 | 0.4000 [0.0000, 0.8000], 6† |
| 5b | Causal/consequence reasoning | Not evaluated, 0 | 1.0000 [1.0000, 1.0000], 17† |
| 5c | Multi-step reasoning | 0.7148 [0.5147, 0.9074], 51 | 1.0000 [1.0000, 1.0000], 2† |

Among leaves with at least 30 questions, the clearest strengths are instance
matching in both datasets, HeiCo camera localization and multi-step reasoning,
and LapChole object identification. The clear weaknesses are temporal
localization and duration estimation in both datasets, plus HeiCo event
aggregation.

No primary object-attributes question exists in either test set. Functional
and causal reasoning are also absent as HeiCo primaries, so this full test does
not establish comprehensive capability coverage despite covering every
official test row.

## Capability-by-format details

The broad capability values hide major differences in the requested answer
type.

| Primary capability / format | HeiCo score, n | LapChole score, n |
| --- | ---: | ---: |
| Temporal localization / time | 0.1148, 699 | 0.2850, 388 |
| Duration estimation / time | 0.0325, 68 | 0.1146, 26 |
| Object aggregation / number | 0.3591, 299 | 0.4444, 11 |
| Object aggregation / FO class | 0.6368, 101 | 0.6875, 12 |
| Event aggregation / number | 0.2008, 326 | Not present |
| Event aggregation / open-ended | Not present | 0.8529, 18 |
| Object identification / FO class | 0.5925, 231 | 0.7713, 356 |
| Spatial localization — camera / multiple choice | 0.6278, 32 | 0.3500, 37 |

Duration estimation is an especially instructive example. Its overall score
is raised by 12 perfect FO-class questions across the datasets, while its
actual time-answer scores are only 0.0325 and 0.1146. Similarly, the
event-aggregation dataset gap is not an apples-to-apples comparison: HeiCo
requires exact event counts; LapChole uses judged open-ended responses.

The complete 44-cell table is in
[`capability-answer-format.csv`](qwen3-vl-4b-lora-both-official-epoch8-full-test/capability-answer-format.csv).

## Answer-format profile

| Answer format | HeiCo score (95% CI), n | LapChole score (95% CI), n |
| --- | ---: | ---: |
| Binary | 0.8324 [0.7402, 0.9135], 129 | 0.8173 [0.7362, 0.8929], 146 |
| FO class | 0.6055 [0.4966, 0.7099], 380 | 0.7526 [0.6715, 0.8273], 371 |
| Multiple choice | 0.6278 [0.3111, 0.8838], 32 | 0.3500 [0.1000, 0.6333], 37 |
| Number | 0.2526 [0.1870, 0.3247], 625 | 0.4444 [0.1111, 0.8333], 11 |
| Open-ended | 0.7067 [0.4800, 0.9000], 50 | 0.7071 [0.6024, 0.8095], 142 |
| Percentage | 0.0000 [0.0000, 0.0000], 17 | 0.0000 [0.0000, 0.0000], 6 |
| Time | 0.1073 [0.0594, 0.1683], 767 | 0.2807 [0.1992, 0.3742], 414 |

Binary and open-ended answers transfer well across datasets. Time and
percentage are the consistent floors. LapChole's number score is based on only
11 questions and should not be compared directly with HeiCo's 625-question
number distribution.

### Temporal grounding

Across 1,181 time questions, only 189 were correct. For questions with one
ground-truth timestamp:

| Dataset | Single-time rows | Median absolute error | 90th-percentile error | Within 5 s | Within 30 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 750 | 136 s | 2,030 s | 11.1% | 27.5% |
| LapChole | 397 | 18 s | 264 s | 26.7% | 61.5% |

The error scale is far larger than the evaluator's one-to-five-second
acceptance window. Limiting a many-minute or multi-hour context to at most 96
frames provides enough evidence for coarse event presence, but not reliable
second-level localization. A retrieve-then-zoom stage is the highest-priority
architectural improvement.

There are also 34 multi-timestamp references, 17 per dataset, and all 34 were
incorrect. The source full-test normalizer retained only the first normalized
timestamp. The v2 submission adapter now preserves every generated timestamp.
However, post-hoc application of that v2 normalizer to the stored raw model
text changed zero answers from correct to incorrect or vice versa: the raw
outputs still failed complete cardinality-and-timing matching. This post-hoc
check isolates normalization only and is not an exact rerun of the v2
container prompt.

### Counts and percentages

HeiCo's 625 number questions have only 0.2352 question-micro accuracy. The two
aggregation leaves behave differently:

- Object aggregation: 0.3591 video-macro accuracy on 299 number rows, mean
  absolute error 1.66.
- Event aggregation: 0.2008 on 326 rows, mean absolute error 6.62.
- Ground-truth counts of six or more: 260 rows, only 8 correct (3.1% micro;
  4.1% video-macro), median absolute error 7, with 65.0% undercounted.
- Ground-truth zero and one: approximately 47–49% video-macro accuracy.

The failure is therefore concentrated in long-horizon repeated-event counts,
not merely numeric formatting. LapChole has only 11 number questions and is
too small for the same analysis.

All 23 percentage questions were wrong under the evaluator's exact numeric
comparison. The predictions were not random—the mean absolute errors were
2.55 percentage points on HeiCo and 3.50 on LapChole—but approximate values
receive no credit.

### Foreign-object classes

| Ground-truth cardinality | HeiCo score, n | LapChole score, n |
| --- | ---: | ---: |
| None | 0.7356, 73 | 0.8519, 29 |
| One class | 0.6749, 212 | 0.8487, 208 |
| Multiple classes | 0.3044, 95 | 0.5907, 134 |

Exact-set performance drops sharply for multi-label answers. The model often
recognizes part of the set but omits or adds a class: mean set recall remains
0.738 on HeiCo and 0.852 on LapChole multi-label rows while exact accuracy is
only 0.274 and 0.537 at the question-micro level.

Class-level findings include:

- Sponge recall is 0.960 on HeiCo and 0.963 on LapChole, the most transferable
  class result.
- Clip recall is 0.657 on HeiCo versus 0.933 on LapChole.
- Specimen recall is 0.368 on HeiCo versus 0.839 on LapChole.
- Gallstone recall is only 0.059 on 17 LapChole-positive rows.
- Absorbable Hemostatic Agent is missed on all five positive LapChole rows.
- Silicone Loop has no positive test reference, yet is predicted on 18 HeiCo
  and three LapChole rows, so its apparent behavior is false-positive-only.

These figures are label presence precision/recall, while the official
FO-class metric remains exact set equality. Full values are in
[`fo-class-label.csv`](qwen3-vl-4b-lora-both-official-epoch8-full-test/fo-class-label.csv).

### Binary and multiple choice

Binary answers are robust but asymmetric. "No" scores 0.8988 on HeiCo and
0.9167 on LapChole, versus 0.7650 and 0.6852 for "yes", indicating a meaningful
negative-answer advantage.

Every multiple-choice question in this full test is single-select. The test
therefore provides no empirical coverage of the submission adapter's
multi-select behavior. Multiple-choice accuracy also varies substantially by
dataset, but the samples span only five HeiCo and ten LapChole videos.

## Secondary capabilities

Secondary tags overlap and are not official leaderboard slices, but they show
which skills co-occur with difficult questions:

- Secondary event aggregation remains very weak: 0.046 on 49 HeiCo rows and
  0.105 on 37 LapChole rows.
- Secondary object identification is extremely common—1,737 HeiCo and 709
  LapChole rows—but scores only 0.265 and 0.447 because it frequently
  accompanies harder temporal and aggregation tasks.
- Secondary temporal localization scores 0.576 on HeiCo and 0.771 on
  LapChole. This does not contradict the poor primary temporal score: these
  questions involve temporal context but do not necessarily demand an exact
  timestamp as the answer.

The complete overlapping distribution is in
[`secondary-capability.csv`](qwen3-vl-4b-lora-both-official-epoch8-full-test/secondary-capability.csv).

## Dataset and annotation slices

| Generation source | HeiCo score, n | LapChole score, n |
| --- | ---: | ---: |
| Anchor | 0.3685, 802 | 0.6533, 452 |
| Automatic | 0.2989, 1,188 | 0.4683, 549 |
| Manual | 0.5333, 10 | 0.6740, 126 |

Automatic questions are harder in both datasets, but this slice is confounded
by capability and answer-format composition. HeiCo's manual result has only
ten rows.

There are no OOD rows in either test set, so the model's OOD performance is
unknown. The clinical subset contains 20 HeiCo questions (15 correct) and one
LapChole question (incorrect); it is far too small for a cross-dataset claim.

Per-video accuracy ranges from 0.240 to 0.345 across the ten equally sized
HeiCo videos. LapChole is much more heterogeneous, ranging from 0.300 to 0.775
across 28 videos. This variation is why the official uncertainty is
video-aware.

## Runtime and validity

| Dataset | Rows | Mean latency | Median | p95 | Maximum | Over 5 s | Over 30 s / timed out | Inference errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 2,000 | 2.318 s | 2.120 s | 2.700 s | 8.841 s | 17 | 0 / 0 | 0 |
| LapChole | 1,127 | 2.329 s | 2.190 s | 2.745 s | 7.132 s | 10 | 0 / 0 | 0 |

All 3,127 responses were produced, parsed, and evaluated. The latency recorded
in each response is model-generation latency; end-to-end container timing can
also include model loading, decoding, and frame preparation.

## Priorities before further training

1. Add coarse-to-fine temporal retrieval: locate a candidate interval with
   sparse frames, then resample densely enough to satisfy the one-to-five-second
   tolerance.
2. Train and decode explicitly for long-horizon event counts, with extra weight
   on counts of six or more and repeated appearances.
3. Calibrate percentage outputs to the dataset's exact target convention;
   approximate numeric answers receive no partial credit.
4. Improve exact multi-label FO-set completion, especially HeiCo specimen and
   LapChole gallstone / hemostatic-agent cases.
5. Add evaluation coverage for OOD questions, primary object attributes, and
   multi-select multiple choice. Sparse high scores should not drive model
   selection.

## Reproducibility and artifacts

- Training method:
  [`architecture.md`](../../track-procedure/lora-finetune/architecture.md)
- Exact completed run:
  [`full_test_epoch_8/`](../../track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_8/)
- Successful full-test log:
  [`full-test-a100-1735.log`](../../track-procedure/lora-finetune/logs/full-test-a100-1735.log)
- Submission checkpoint provenance:
  [`provenance.json`](../../submissions/procedure/qwen3-vl-4b-lora-both-official-epoch8-v2-20260727/provenance.json)
- Reproducible analysis script:
  [`summarize_procedure_full_test.py`](../../scripts/summarize_procedure_full_test.py)
- Machine-readable analysis tables and source hashes:
  [`qwen3-vl-4b-lora-both-official-epoch8-full-test/`](qwen3-vl-4b-lora-both-official-epoch8-full-test/)

The artifact directory includes dataset overall, capability group, all primary
leaves, dataset gaps, group-gap decomposition, capability-by-format, secondary
capability, answer-format, time, numeric, FO-class, categorical, annotation
slice, per-video, runtime, adapter-counterfactual, and provenance tables.
