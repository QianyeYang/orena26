# Frame Model Full-Test Comparison

Test date: 2026-07-24

> **Correction (2026-08-08):** every accuracy in this document was scored
> with the pre-fix single-label `fo_class` adapter. The corrected epoch-30
> baselines are **HeiCo 0.7067 / LapChole 0.6417** — see
> [`result-summary/frame/epoch30-multilabel-rescore.md`](../result-summary/frame/epoch30-multilabel-rescore.md).
> Relative old-vs-new comparisons below remain directionally valid (both
> models were scored with the same broken adapter).

## Headline result

The new epoch-30 model is better than the old best checkpoint on both current
official test sets. Accuracy below is the FOCUS evaluator's overall metric:
the mean of per-video accuracies.

| Test set | Rows / videos | Old epoch 24 | New epoch 30 | New - old | Paired 95% CI for delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 4,000 / 10 | 0.630 [0.545, 0.710] | **0.664** [0.579, 0.738] | **+0.034** | [+0.013, +0.055] |
| LapChole | 2,252 / 28 | 0.389 [0.351, 0.428] | **0.555** [0.515, 0.595] | **+0.166** | [+0.133, +0.198] |
| Equal-dataset mean | 6,252 / 38 | 0.510 | **0.610** | **+0.100** | [+0.080, +0.119] |

Across all questions without video or dataset reweighting, the old model
answered 3,398/6,252 correctly (0.544), while the new model answered
3,907/6,252 correctly (0.625), an absolute gain of 0.081 and 509 additional
correct answers.

## Direct paired outcomes

| Test set | Both correct | New only correct | Old only correct | Both wrong |
| --- | ---: | ---: | ---: | ---: |
| HeiCo | 2,238 | 419 | 283 | 1,060 |
| LapChole | 740 | 510 | 137 | 865 |
| Total | 2,978 | **929** | 420 | 1,925 |

## Accuracy by answer format

These are also macro-averaged across videos.

| Answer format | HeiCo n | HeiCo old -> new | Delta | LapChole n | LapChole old -> new | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | 548 | 0.792 -> 0.799 | +0.008 | 176 | 0.753 -> 0.871 | +0.118 |
| Foreign-object class | 1,755 | 0.676 -> 0.708 | +0.032 | 920 | 0.338 -> 0.571 | +0.233 |
| Multiple choice | 112 | 0.906 -> 0.901 | -0.005 | 90 | 0.725 -> 0.943 | +0.218 |
| Number | 1,326 | 0.492 -> 0.536 | +0.043 | 768 | 0.291 -> 0.412 | +0.120 |
| Open ended | 259 | 0.690 -> 0.773 | +0.083 | 298 | 0.561 -> 0.716 | +0.155 |

The new model improves every LapChole answer format. On HeiCo it improves four
of five formats; the only decrease is -0.005 on multiple choice.

## Accuracy by capability group

| Capability group | HeiCo n | HeiCo old -> new | Delta | LapChole n | LapChole old -> new | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregation | 1,875 | 0.579 -> 0.611 | +0.032 | 955 | 0.375 -> 0.494 | +0.119 |
| Object recognition | 2,125 | 0.691 -> 0.728 | +0.037 | 1,296 | 0.414 -> 0.624 | +0.210 |
| Temporal grounding | 0 | - | - | 1 | 1.000 -> 0.000 | -1.000 |

The single temporal-grounding row is too small to support a general conclusion.
Every capability group with meaningful support improves.

## Secondary FOCUS `pre_evaluation` score

| Test set | Old epoch 24 | New epoch 30 | Delta |
| --- | ---: | ---: | ---: |
| HeiCo | 0.627 | **0.661** | +0.034 |
| LapChole | **0.591** | 0.364 | -0.227 |

This secondary score is not a representative ranking for the current local
test Parquets. All 6,252 rows are in-distribution, so only 2/10 expected
group-by-distribution buckets are populated for HeiCo and 3/10 for LapChole.
LapChole then gives equal bucket weight to the one-row temporal group, where
the old model is correct and the new model is wrong. The FOCUS evaluator emits
the corresponding sparse-bucket warnings.

## Why the previous result was about 0.8

The previous summary's 0.828 result came from epoch 23 on the earlier 2,000-row
HeiCo test set and the previous frame extraction. The completed old epoch log
later reached 0.829 at epoch 24, so this comparison uses that stronger
`checkpoint-6000`.

That historical 0.829 and the new model's current score are not scores on the
same inputs. When the old checkpoint is evaluated on the same current
4,000-row official HeiCo test set, it scores 0.630; the new checkpoint scores
0.664. The apparent drop from about 0.8 is therefore caused by comparing
different evaluations, not by the new model losing to the old model under an
apples-to-apples test.

## Test integrity

| Check | Result |
| --- | --- |
| Test coverage per model | 4,000 HeiCo + 2,252 LapChole = 6,252/6,252 |
| Missing or unreadable images | 0 |
| Inference errors | 0 |
| Empty or unparseable responses | 0 |
| HeiCo median latency, old / new | 0.075 s / 0.074 s |
| LapChole median latency, old / new | 0.074 s / 0.079 s |
| Maximum latency, old / new | 1.835 s / 1.836 s |
| Official frame latency limit | 5.0 s; no violations |

Both checkpoints used the original `Qwen3-VL-4B-Instruct` base, the same
official test rows and extracted frames, the same system and question prompts,
greedy decoding with 64 output tokens, a 602,112-pixel image cap, and the same
local Qwen3.5-4B FOCUS judge. The old model was
`Qwen3-VL-4B-Instruct/checkpoint-6000` (epoch 24); the new model was
`Qwen3-VL-4B-Instruct-both-official/final` (epoch 30).

Raw evaluator summaries:

- [Old HeiCo](../track-frame/lora-finetune/logs/full-test-comparison/old-epoch24/heico/eval/summary.csv)
- [New HeiCo](../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/heico/eval/summary.csv)
- [Old LapChole](../track-frame/lora-finetune/logs/full-test-comparison/old-epoch24/lapchole/eval/summary.csv)
- [New LapChole](../track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/lapchole/eval/summary.csv)
