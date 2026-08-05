# Segment epoch-6 submission-checkpoint analysis

> **Provisional checkpoint-selection evidence.** This report covers the fixed
> seed-42 subset used during training: 500 of 6,254 official Segment test rows
> (7.99%). It is not a full-test run, an online score, or an exact rerun of the
> packaged submission. The subset nevertheless includes all 10 HeiCo and all 28
> LapChole test videos.

## Outcome

The Segment submission uses Qwen3-VL-4B-Instruct with the LoRA adapter from
epoch 6, global step 5,160:
`track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/checkpoint-5160`.
Its adapter SHA-256 is
`92556248ad9e8a82b7f1d295bac3c32f0984e7e247ef1a74499e7134032b73cb`.
Training ultimately completed all 12 epochs, but the submission deliberately
uses this earlier checkpoint.

| Dataset | Evaluated / full rows | Videos | Correct | Official video-macro accuracy (95% CI) | Question-micro accuracy | Pre-evaluation score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 318 / 4,000 (7.95%) | 10 / 10 | 225 | 0.6971 [0.6056, 0.7873] | 0.7075 | 0.7309 |
| LapChole | 182 / 2,254 (8.07%) | 28 / 28 | 135 | 0.7544 [0.6648, 0.8389] | 0.7418 | 0.7953 |

The equal-dataset mean is 0.7257 for official video-macro accuracy and 0.7631
for the challenge-style pre-evaluation score. These are useful checkpoint
selection summaries, not a combined official leaderboard score. The two
dataset results should remain separate because their question mixtures differ
substantially.

The strongest well-represented capability is object recognition. Temporal
grounding is the main weakness in both datasets. HeiCo also exposes weaker
aggregation and multi-select multiple-choice behavior; the latter result is
reversed on LapChole. Sparse event-understanding and reasoning scores are too
small to rank reliably.

The source evaluation completed on 2026-07-29. This report was generated on
2026-08-02.

## Why epoch 6 was submitted

“Best epoch” depends on the selection metric. Epoch 8 has the highest
equal-dataset mean of the official video-macro point estimates, 0.7335. Epoch 6
has the highest mean pre-evaluation score, 0.7631 versus epoch 8's 0.7591. The
submission provenance selected epoch 6 using that capability/OOD-bucket-balanced
pre-evaluation metric.

| Epoch | Step | HeiCo official | LapChole official | Mean official | HeiCo pre-eval | LapChole pre-eval | Mean pre-eval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 860 | 0.5708 | 0.6530 | 0.6119 | 0.5915 | 0.5009 | 0.5462 |
| 2 | 1,720 | 0.5971 | 0.7024 | 0.6497 | 0.6356 | 0.8176 | 0.7266 |
| 3 | 2,580 | 0.6317 | 0.6888 | 0.6603 | 0.6621 | 0.8195 | 0.7408 |
| 4 | 3,440 | 0.6620 | 0.7252 | 0.6936 | 0.7233 | 0.7532 | 0.7383 |
| **6** | **5,160** | **0.6971** | **0.7544** | **0.7257** | **0.7309** | **0.7953** | **0.7631** |
| 7 | 6,020 | 0.6803 | 0.7541 | 0.7172 | 0.6688 | 0.7607 | 0.7147 |
| 8 | 6,880 | 0.6562 | 0.8108 | **0.7335** | 0.6998 | 0.8185 | 0.7591 |
| 10 | 8,600 | 0.6531 | 0.7984 | 0.7258 | 0.6418 | 0.8251 | 0.7335 |
| 11 | 9,460 | 0.6553 | 0.7898 | 0.7226 | 0.6600 | 0.8234 | 0.7417 |
| 12 | 10,320 | 0.6508 | 0.7823 | 0.7165 | 0.6567 | 0.8198 | 0.7382 |

Epochs 5 and 9 have no completed per-epoch evaluation artifact and are therefore
absent rather than treated as zero.

The direct epoch-6 versus epoch-8 comparison uses identical question IDs.
Epoch 6 uniquely answered 31 HeiCo rows correctly while epoch 8 uniquely
answered 16; on LapChole those counts were 3 and 12. Thus epoch 8 traded a
4.09-point HeiCo video-macro decrease for a 5.64-point LapChole increase. Its
largest HeiCo regression was aggregation (−16.28 video-macro points), which
helps explain why bucket-balanced selection favored epoch 6. Full paired
dataset, capability, and answer-format outcomes are in
[`checkpoint-epoch6-vs-epoch8-paired.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/checkpoint-epoch6-vs-epoch8-paired.csv).

## Scoring, training, and inference scope

The official point estimate is the mean of per-video question-correctness
means. The 95% intervals come from `orena-focus` 0.3.4's two-level bootstrap:
videos are resampled, then questions within each video, with 1,000 samples,
seed 42, and percentile intervals. The pre-evaluation score is the unweighted
mean of populated primary-capability-group × OOD buckets. This subset contains
only the five in-distribution buckets and no OOD rows.

Exact-format comparison was used for binary, FO-class, number, percentage, and
time answers. Open-ended and multiple-choice responses were scored by the local
Qwen3.5-4B judge. Time correctness uses the evaluator's duration-dependent
one-to-five-second tolerance.

The LoRA run trained on all 13,746 official training rows: 8,000 HeiCo and
5,746 LapChole. Epoch-6 inference used BF16, stride 25 for HeiCo and 30 for
LapChole, at most 64 frames, 262,144 pixels per frame, and 64 generated tokens.
The base model was loaded from `os-models/Qwen3-VL-4B-Instruct`.

The selected checkpoint was written by resumed job 1633 on two B200 GPUs. The
subsequent job that completed the 12-epoch run, job 2090, used four A100 GPUs
and ran for 1 day, 3 hours, 30 minutes, 35 seconds. These are training-lineage
facts; the latter duration is not the isolated cost of producing epoch 6.

The source evaluator was invoked without `track` or `max_latency`, so it did
not enforce the Segment 15-second limit. All recorded model-generation
latencies were nevertheless below 2.1 seconds. These timings exclude model
load and may exclude video decode and frame preparation.

## Capability groups

| Capability group | HeiCo rows / share | HeiCo score (95% CI) | LapChole rows / share | LapChole score (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Object recognition | 102 / 32.08% | 0.8444 [0.7183, 0.9542] | 113 / 62.09% | 0.8188 [0.6944, 0.9174] |
| Temporal grounding | 145 / 45.60% | 0.6059 [0.4993, 0.7244] | 56 / 30.77% | 0.5623 [0.3949, 0.7355] |
| Aggregation | 50 / 15.72% | 0.6451 [0.4429, 0.8375] | 5 / 2.75% | 0.6000 [0.2000, 1.0000] |
| Event understanding | 9 / 2.83% | 0.8333 [0.5325, 1.0000] | 2 / 1.10% | 1.0000 [1.0000, 1.0000] |
| Complex reasoning | 12 / 3.77% | 0.7083 [0.3333, 1.0000] | 6 / 3.30% | 1.0000 [1.0000, 1.0000] |

Object recognition is the only group with more than 100 sampled rows in both
datasets. HeiCo is weighted much more heavily toward temporal grounding and
aggregation, while LapChole is dominated by object recognition. Event and
reasoning values have only 2–12 rows per dataset and should be treated as
coverage observations, not evidence of near-perfect general performance.

## All primary leaf capabilities

Every primary leaf is shown below. A dagger marks fewer than 30 questions; a
dash means the subset contains no row for that leaf.

| Code | Primary capability | HeiCo score (95% CI), n | LapChole score (95% CI), n |
| --- | --- | ---: | ---: |
| 1a | Object identification | 0.9250 [0.8000, 1.0000], 58 | 0.7928 [0.6499, 0.9167], 64 |
| 1b | Instance matching | 0.8333 [0.5000, 1.0000], 8† | 0.8571 [0.5714, 1.0000], 8† |
| 1c | Object attributes | 1.0000 [1.0000, 1.0000], 2† | 0.8000 [0.4000, 1.0000], 6† |
| 1d | Spatial localization — camera | 0.6278 [0.3257, 0.8889], 33 | 0.8704 [0.7222, 0.9815], 35 |
| 1e | Spatial localization — situs | 1.0000 [1.0000, 1.0000], 1† | —, 0 |
| 2a | Temporal localization | 0.6292 [0.5007, 0.7538], 125 | 0.5732 [0.4021, 0.7508], 55 |
| 2b | Duration estimation | 0.3833 [0.1415, 0.6500], 20† | 0.0000 [0.0000, 0.0000], 1† |
| 3a | Object aggregation | 0.6675 [0.4524, 0.8677], 36 | 0.6000 [0.2000, 1.0000], 5† |
| 3b | Event aggregation | 0.4767 [0.1000, 0.8500], 14† | —, 0 |
| 4a | FO-interaction recognition | 0.8333 [0.4958, 1.0000], 5† | —, 0 |
| 4b | FO-usage purpose | 0.5000 [0.0000, 1.0000], 2† | 1.0000 [1.0000, 1.0000], 2† |
| 4c | Temporal ordering | 1.0000 [1.0000, 1.0000], 2† | —, 0 |
| 5a | Functional reasoning | 1.0000 [1.0000, 1.0000], 1† | —, 0 |
| 5b | Causal/consequence reasoning | 1.0000 [1.0000, 1.0000], 7† | 1.0000 [1.0000, 1.0000], 6† |
| 5c | Multi-step reasoning | 0.2500 [0.0000, 0.7500], 4† | —, 0 |

Among adequately represented leaves, HeiCo object identification and
LapChole camera localization are the clearest strengths. Temporal localization
is materially lower in both datasets. Most other leaves are too sparse for
stable comparison; seven LapChole leaves are absent entirely.

The complete primary and overlapping secondary distributions are in
[`capability-leaf.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/capability-leaf.csv)
and
[`secondary-capability.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/secondary-capability.csv).
Secondary tags overlap and therefore must not be summed as disjoint rows.

## Answer-format profile

| Answer format | HeiCo score (95% CI), n | LapChole score (95% CI), n |
| --- | ---: | ---: |
| Binary | 0.7500 [0.5000, 1.0000], 18† | 0.8182 [0.5455, 1.0000], 12† |
| FO class | 0.8875 [0.7167, 1.0000], 68 | 0.8014 [0.6681, 0.9181], 62 |
| Multiple choice | 0.6278 [0.3630, 0.8778], 33 | 0.8873 [0.7451, 1.0000], 32 |
| Number | 0.6161 [0.4000, 0.8134], 45 | —, 0 |
| Open-ended | 0.8000 [0.4667, 1.0000], 14† | 0.7949 [0.5641, 0.9744], 21† |
| Percentage | 0.0000 [0.0000, 0.0000], 2† | —, 0 |
| Time | 0.5933 [0.4786, 0.7203], 138 | 0.5732 [0.4057, 0.7486], 55 |

FO-class accuracy is strong on this subset, but exact-set difficulty is more
visible on LapChole's multi-label rows. Multiple-choice results are
dataset-dependent: multi-select questions score 0.6042 video-macro on 14
HeiCo rows and 1.0000 on 14 LapChole rows, while single-select scores are
0.8095 and 0.8077. These small samples do not support a broad claim that one
dataset's multi-select task is intrinsically easier.

The full capability × answer-format distribution is in
[`capability-answer-format.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/capability-answer-format.csv).

## Cardinality and submission-adapter audit

The source epoch-6 predictions used the same weights as the submission but not
the final prompt/normalizer. Their FO instruction explicitly requested
“EXACTLY ONE” class. The packaged adapter instead requests every applicable
runtime-defined class and preserves all recognized values. Consequently, the
scores above do not estimate the gain from the packaged cardinality fix.

| Dataset / format | Eligible rows | Gold multi-value | Raw multi-value | Normalized multi-value | Raw multi-value rows lost by normalization |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo / FO class | 68 | 6 | 7 | 7 | 0 |
| LapChole / FO class | 62 | 13 | 12 | 12 | 0 |
| HeiCo / time | 138 | 2 | 1 | 0 | **1** |
| LapChole / time | 55 | 0 | 0 | 0 | 0 |
| HeiCo / multi-select MC | 14 | 5 | 5 | 5 | 0 |
| LapChole / multi-select MC | 14 | 7 | 6 | 6 | 0 |

The stored normalizer did not truncate a multi-value FO or multiple-choice
output in this subset. It did collapse one raw HeiCo multi-timestamp answer to
a single timestamp. The submission adapter directly fixes that demonstrated
loss and is also designed to prevent the same class of loss for FO and
multi-select answers. The official full-test contract audit found 256
multi-label FO rows and 360 multi-select multiple-choice rows, so preserving
cardinality is necessary even though truncation was sparse here.

FO exact-set accuracy by gold cardinality was:

| Gold cardinality | HeiCo video-macro score, n | LapChole video-macro score, n |
| --- | ---: | ---: |
| None | 1.0000, 5† | 1.0000, 2† |
| One class | 0.8590, 57 | 0.8048, 47 |
| Multiple classes | 1.0000, 6† | 0.7273, 13† |

The six perfect HeiCo multi-label rows are far too few to generalize. On the
13 LapChole multi-label rows, mean set precision was 0.9487 and recall 0.9359,
while exact question-micro accuracy was 0.6923: most failures were small set
mismatches. Per-class support, precision, and recall are available in
[`fo-class-label.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/fo-class-label.csv).

## Time and numeric diagnostics

| Dataset / scope | Time rows | Video-macro score | Single-time rows | Median absolute error | p90 absolute error | Within 5 s | Within 30 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo / all time | 138 | 0.5933 | 136 | 1.0 s | 13.5 s | 75.0% | 97.1% |
| HeiCo / temporal localization | 125 | 0.6292 | 123 | 1.0 s | 12.0 s | 78.0% | 96.7% |
| HeiCo / duration estimation | 13 | 0.1667 | 13 | 6.0 s | 20.4 s | 46.2% | 100.0% |
| LapChole / temporal localization | 55 | 0.5732 | 55 | 3.0 s | 13.6 s | 74.5% | 96.4% |

“Within 5 seconds” and “within 30 seconds” are descriptive thresholds. They
do not reproduce the official duration-dependent tolerance exactly.

HeiCo contains 45 number rows: score 0.6161, mean absolute error 0.71. Object
counts score 0.6375 on 31 rows with mean absolute error 0.39; event counts
score 0.4767 on 14 rows with mean absolute error 1.43. Both sampled percentage
questions were wrong. LapChole contributes no number or percentage row to this
subset, so no conclusion is possible there.

## Dataset slices and coverage

| Generation source | HeiCo score, n | LapChole score, n |
| --- | ---: | ---: |
| Anchor | 0.7229, 84 | 0.7941, 86 |
| Automatic | 0.6856, 220 | 0.6571, 78 |
| Manual | 0.8000, 14† | 0.8472, 18† |

Automatic questions are lower-scoring in both datasets, but this is
descriptive and confounded by capability and answer-format composition. The
subset has zero OOD and zero clinically tagged rows, so it provides no evidence
for either slice.

HeiCo per-video accuracy ranges from 0.4667 to 0.9512. LapChole ranges from
0.3333 to 1.0000, but its sampled videos contain only 2–12 questions each.
This sparse per-video coverage contributes to the wide bootstrap intervals.

## Runtime and validity

| Dataset | Rows | Mean latency | Median | p95 | Maximum | Over 5 s | Over 15 s | Errors / empty outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 318 | 0.775 s | 0.788 s | 0.952 s | 1.610 s | 0 | 0 | 0 / 0 |
| LapChole | 182 | 0.779 s | 0.792 s | 1.107 s | 2.012 s | 0 | 0 | 0 / 0 |

All 500 source responses were non-empty and evaluated, with no inference
errors or recorded timeouts. The inference frequently reached its 64-frame
cap: 228 HeiCo rows and 131 LapChole rows.

The submission bundle separately passed static validation and a nine-request
offline BF16/64-frame Apptainer smoke test on an A100. Docker-host validation
was still pending when the bundle was recorded. That smoke test validates
packaging and answer cardinality, not model quality, and must not be presented
as Docker validation.

## Interpretation and next evidence needed

1. Run all 6,254 official test rows with the exact packaged prompt and
   normalizer before using these values as a final local benchmark.
2. Retain epoch 6 for the current submission if capability-balanced pre-eval is
   the intended selection rule. Choose epoch 8 only if the equal-dataset mean
   official point estimate is explicitly preferred; the subset shows a real
   HeiCo/LapChole tradeoff.
3. Prioritize temporal localization, duration estimation, and HeiCo
   aggregation. Add targeted multi-label and multi-select evaluation because
   the current subset has only 6–14 relevant rows per dataset.
4. Add OOD and clinical coverage before making robustness claims.

## Reproducibility and artifacts

- Training method:
  [`architecture.md`](../../track-segment/lora-finetune/architecture.md)
- Exact checkpoint:
  [`checkpoint-5160/`](../../track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/checkpoint-5160/)
- Stored epoch-6 evaluation:
  [`eval_epoch_6/`](../../track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/eval_epoch_6/)
- Fixed subset IDs:
  [`eval_subset_qids.json`](../../track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/eval_subset_qids.json)
- Checkpoint history:
  [`checkpoint-selection.csv`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/checkpoint-selection.csv)
- Dataset, capability, answer-format, runtime, diagnostic, and per-video tables:
  [`artifact directory`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/)
- Machine-readable provenance and source hashes:
  [`provenance.json`](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset/provenance.json)
- Reproduction script:
  [`summarize_segment_epoch6_subset.py`](../../scripts/summarize_segment_epoch6_subset.py)
- Submission provenance:
  [`provenance.json`](../../submissions/segment/qwen3-vl-4b-lora-both-official-epoch6-20260801/provenance.json)
- Docker-machine handoff status:
  [`guidance.md`](../../submissions/segment/qwen3-vl-4b-lora-both-official-epoch6-20260801/guidance.md)
