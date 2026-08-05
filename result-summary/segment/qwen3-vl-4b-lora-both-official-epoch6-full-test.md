# Segment epoch-6 full-test analysis

> **Authoritative local full-test evaluation.** This report covers all 6,254 official Segment test questions: 4,000 HeiCo and 2,254 LapChole rows across all 38 test videos. It is an offline local result, not an online leaderboard score or a container-contract rerun.

## Outcome

| Dataset | Evaluated / full rows | Videos | Correct | Official video-macro accuracy (95% CI) | Question-micro accuracy | Pre-evaluation score |
| --- | --- | --- | --- | --- | --- | --- |
| HeiCo | 4,000 / 4,000 | 10 | 2,809 | 0.7023 [0.6420, 0.7578] | 0.7023 | 0.7182 |
| LapChole | 2,254 / 2,254 | 28 | 1,746 | 0.7745 [0.7381, 0.8133] | 0.7746 | 0.8066 |

The equal-dataset mean is **0.7384** for official video-macro accuracy and **0.7624** for the capability/OOD-bucket-balanced pre-evaluation score. Dataset values remain the primary results because HeiCo and LapChole have different question distributions.

Across the five capability groups, **temporal grounding is the weakest ability** by equal-dataset mean full-test score (0.6413); event understanding is the strongest (0.9000). Per-dataset support and uncertainty below should be used when interpreting sparse groups.

## Full test versus the 500-row selection subset

| Dataset | Subset rows | Subset official | Full rows | Full official | Full − subset |
| --- | --- | --- | --- | --- | --- |
| HeiCo | 318 | 0.6971 | 4,000 | 0.7023 | +0.0051 |
| LapChole | 182 | 0.7544 | 2,254 | 0.7745 | +0.0201 |

The fixed subset remains useful only for explaining checkpoint selection. The full-test figures above supersede it for model performance claims. Capability- and format-level deltas are in [`subset-vs-full.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/subset-vs-full.csv).

## Capability groups

| Capability group | HeiCo n | HeiCo score (95% CI) | LapChole n | LapChole score (95% CI) | Equal-dataset mean |
| --- | --- | --- | --- | --- | --- |
| object recognition | 1,383 | 0.8609 [0.8015, 0.9201] | 1,440 | 0.8222 [0.7879, 0.8586] | 0.8415 |
| temporal grounding | 1,752 | 0.6037 [0.5313, 0.6820] | 684 | 0.6789 [0.5774, 0.7712] | 0.6413 |
| aggregation | 572 | 0.6017 [0.5181, 0.6830] | 22 | 0.7056 [0.4444, 0.9167] | 0.6536 |
| event understanding | 145 | 0.8433 [0.7476, 0.9271] | 65 | 0.9567 [0.8900, 1.0000] | 0.9000 |
| complex reasoning | 148 | 0.6756 [0.5391, 0.8046] | 43 | 0.9375 [0.8281, 1.0000] | 0.8066 |

## All primary leaf capabilities

| Code | Primary capability | HeiCo n | HeiCo score (95% CI) | LapChole n | LapChole score (95% CI) |
| --- | --- | --- | --- | --- | --- |
| 1a | object identification | 713 | 0.8629 [0.7728, 0.9467] | 741 | 0.7896 [0.7266, 0.8407] |
| 1b | instance matching | 141 | 0.9359 [0.8584, 0.9938] | 141 | 0.9006 [0.8321, 0.9619] |
| 1c | object attributes | 27 | 0.8917 [0.7250, 1.0000] | 63 | 0.8690 [0.7500, 0.9585] |
| 1d | spatial localization camera | 490 | 0.8416 [0.7777, 0.8984] | 480 | 0.8425 [0.7730, 0.8981] |
| 1e | spatial localization situs | 12 | 0.5867 [0.2000, 0.9600] | 15 | 0.8636 [0.6364, 1.0000] |
| 2a | temporal localization | 1,605 | 0.6280 [0.5483, 0.7161] | 678 | 0.6817 [0.5820, 0.7747] |
| 2b | duration estimation | 147 | 0.3222 [0.2192, 0.4208] | 6 | 0.1667 [0.0000, 0.6667] |
| 3a | object aggregation | 403 | 0.6728 [0.5853, 0.7605] | 19 | 0.7091 [0.4364, 0.9636] |
| 3b | event aggregation | 169 | 0.4613 [0.3434, 0.6040] | 3 | 0.3333 [0.0000, 1.0000] |
| 4a | fo interaction recognition | 75 | 0.8258 [0.7097, 0.9389] | 5 | 0.8000 [0.4000, 1.0000] |
| 4b | fo usage purpose | 19 | 0.6667 [0.3611, 0.9444] | 56 | 0.9900 [0.9600, 1.0000] |
| 4c | temporal ordering | 51 | 0.9333 [0.8329, 1.0000] | 4 | 0.6667 [0.0000, 1.0000] |
| 5a | functional reasoning | 34 | 0.9437 [0.8375, 1.0000] | 13 | 0.7143 [0.4286, 1.0000] |
| 5b | causal consequence reasoning | 62 | 0.7449 [0.5611, 0.9077] | 29 | 0.9762 [0.9048, 1.0000] |
| 5c | multi step reasoning | 52 | 0.5438 [0.3467, 0.7571] | 1 | 1.0000 [1.0000, 1.0000] |

Secondary capability tags overlap rather than partitioning the rows. Their complete distribution is in [`secondary-capability.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/secondary-capability.csv); capability × answer-format counts are in [`capability-answer-format.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/capability-answer-format.csv).

## Answer formats

| Answer format | HeiCo n | HeiCo score (95% CI) | LapChole n | LapChole score (95% CI) |
| --- | --- | --- | --- | --- |
| binary | 287 | 0.9037 [0.8373, 0.9604] | 193 | 0.8815 [0.8063, 0.9422] |
| fo class | 851 | 0.8305 [0.7295, 0.9139] | 693 | 0.7802 [0.7266, 0.8323] |
| multiple choice | 468 | 0.8370 [0.7720, 0.8989] | 465 | 0.8411 [0.7704, 0.8985] |
| number | 481 | 0.5674 [0.4816, 0.6525] | 1 | 1.0000 [1.0000, 1.0000] |
| open ended | 201 | 0.8402 [0.7542, 0.9182] | 220 | 0.8843 [0.8241, 0.9364] |
| percentage | 22 | 0.1000 [0.0000, 0.3000] | 2 | 0.0000 [0.0000, 0.0000] |
| time | 1,690 | 0.6074 [0.5300, 0.6933] | 680 | 0.6812 [0.5743, 0.7730] |

Binary and multiple-choice subtype results are:

| Dataset | Format | Subtype | Rows | Video-macro score |
| --- | --- | --- | --- | --- |
| HeiCo | binary | no | 144 | 0.9216 |
| HeiCo | binary | yes | 143 | 0.9045 |
| HeiCo | multiple choice | multi select | 190 | 0.8167 |
| HeiCo | multiple choice | single select | 278 | 0.8295 |
| LapChole | binary | no | 97 | 0.9315 |
| LapChole | binary | yes | 96 | 0.8304 |
| LapChole | multiple choice | multi select | 170 | 0.8885 |
| LapChole | multiple choice | single select | 295 | 0.8258 |

## Cardinality audit

| Dataset | Format | Eligible | Gold multi-value | Raw multi-value | Normalized multi-value | Lost in normalization | Cardinality match rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HeiCo | fo class | 851 | 76 | 107 | 107 | 0 | 0.8825 |
| HeiCo | time | 1,690 | 23 | 34 | 34 | 0 | 0.9751 |
| HeiCo | multiple choice multi select | 190 | 103 | 104 | 104 | 0 | 0.9053 |
| LapChole | fo class | 693 | 180 | 195 | 195 | 0 | 0.8355 |
| LapChole | time | 680 | 16 | 8 | 8 | 0 | 0.9794 |
| LapChole | multiple choice multi select | 170 | 101 | 99 | 99 | 0 | 0.7765 |

The audited scopes contain **499 multi-value gold answers**. The submission-aligned normalizers discarded values from **0 raw outputs**. FO prompts request every applicable class, and time prompts request every applicable timestamp.

Per-cardinality FO exact-set scores and per-label precision/recall are in [`fo-class-cardinality.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/fo-class-cardinality.csv) and [`fo-class-label.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/fo-class-label.csv).

## Temporal and numeric diagnostics

| Dataset | Time scope | Rows | Video-macro score | Multi-time gold | Median abs. error (s) | p90 abs. error (s) | Within 5 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HeiCo | all time | 1,690 | 0.6074 | 23 | 1.0 | 22.0 | 0.7253 |
| HeiCo | temporal localization | 1,605 | 0.6280 | 23 | 1.0 | 20.0 | 0.7425 |
| HeiCo | duration estimation | 85 | 0.2322 | 0 | 8.0 | 31.2 | 0.4118 |
| LapChole | all time | 680 | 0.6812 | 16 | 1.0 | 13.8 | 0.7949 |
| LapChole | temporal localization | 678 | 0.6817 | 16 | 1.0 | 13.0 | 0.7958 |
| LapChole | duration estimation | 2 | 0.5000 | 0 | 39.5 | 69.5 | 0.5000 |

| Dataset | Format | Scope | Rows | Video-macro score | Median abs. error | Exact numeric rate |
| --- | --- | --- | --- | --- | --- | --- |
| HeiCo | number | all | 481 | 0.5674 | 0.00 | 0.5426 |
| HeiCo | number | object aggregation | 312 | 0.6393 | 0.00 | 0.6218 |
| HeiCo | number | event aggregation | 169 | 0.4613 | 1.00 | 0.3964 |
| HeiCo | percentage | all | 22 | 0.1000 | 5.42 | 0.0909 |
| HeiCo | percentage | duration estimation | 22 | 0.1000 | 5.42 | 0.0909 |
| LapChole | number | all | 1 | 1.0000 | 0.00 | 1.0000 |
| LapChole | number | object aggregation | 1 | 1.0000 | 0.00 | 1.0000 |
| LapChole | percentage | all | 2 | 0.0000 | 9.16 | 0.0000 |
| LapChole | percentage | duration estimation | 2 | 0.0000 | 9.16 | 0.0000 |

## Dataset and generation distribution

| Dataset | Generation source | Rows | Row share | Video-macro score |
| --- | --- | --- | --- | --- |
| HeiCo | anchor | 1,180 | 0.2950 | 0.7652 |
| HeiCo | automatic | 2,655 | 0.6637 | 0.6729 |
| HeiCo | manual | 165 | 0.0413 | 0.8332 |
| LapChole | anchor | 942 | 0.4179 | 0.8369 |
| LapChole | automatic | 1,107 | 0.4911 | 0.7011 |
| LapChole | manual | 205 | 0.0909 | 0.8858 |

All test rows are in-distribution and non-clinical according to the official annotation flags, so this test split cannot measure OOD or clinical-subset generalization. HeiCo contains only sigmoid-resection videos; LapChole contains only laparoscopic-cholecystectomy videos.

## Runtime and validity

| Dataset | Slurm task | Node | GPU | Mean latency (s) | p95 latency (s) | Max latency (s) | >15 s rows | 64-frame-cap rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HeiCo | 2377 | civo-h200-5 | NVIDIA H200 NVL | 1.157 | 1.383 | 187.158 | 1 | 2,807 |
| LapChole | 2373 | civo-h200-6 | NVIDIA H200 NVL | 1.158 | 1.391 | 187.388 | 1 | 1,554 |

Both H200 tasks passed strict ID, count, duplicate, empty-output, inference-error, and evaluator-coverage checks. Model-generation latency excludes model loading and may exclude video decoding/frame preparation.

## Checkpoint, scoring, and provenance

The evaluated checkpoint is epoch 6, global step 5,160, with adapter SHA-256 `92556248ad9e8a82b7f1d295bac3c32f0984e7e247ef1a74499e7134032b73cb`. It was selected using the highest equal-dataset mean pre-evaluation score on the fixed seed-42 500-row subset. The full checkpoint history and paired epoch-6 versus epoch-8 selection-subset analysis remain in [`checkpoint-selection.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/checkpoint-selection.csv) and [`checkpoint-epoch6-vs-epoch8-paired.csv`](qwen3-vl-4b-lora-both-official-epoch6-full-test/checkpoint-epoch6-vs-epoch8-paired.csv).

The official point estimate is the mean of per-video question-level correctness means. Its 95% interval is the official two-level hierarchical bootstrap (videos, then questions within video; 1,000 samples, seed 42). The pre-evaluation score is the unweighted mean over populated primary capability-group × OOD buckets.

The repository evaluation uses the packaged checkpoint, sampling limits, multi-value prompts, and answer normalization, but it invokes the repository runner rather than the built container contract. Exact source hashes, per-dataset run manifests, and artifact lineage are in [`provenance.json`](qwen3-vl-4b-lora-both-official-epoch6-full-test/provenance.json).

## Artifact index

The numerical directory contains dataset, capability group/leaf and gap, answer-format and capability-format, secondary capability, generation/flag slice, per-video, runtime, temporal, numeric, FO, categorical, cardinality, checkpoint-selection, subset-comparison, and provenance artifacts. Raw predictions and evaluator outputs remain under `track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_6`.

Report generated 2026-08-02.
