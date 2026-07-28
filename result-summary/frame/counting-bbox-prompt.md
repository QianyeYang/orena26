# FRAME counting: direct answer versus bounding-box prompting

## Outcome

The bounding-box prompt should **not** replace direct counting globally.
Using the same fine-tuned epoch-30 model, it raised the isolated HeiCo
full-test score from 0.6643 to 0.6690, but lowered LapChole from 0.5551 to
0.5409. The paired 95% confidence interval crosses zero for both full-test
deltas.

On counting questions alone, HeiCo improved by 0.0166
(`[-0.0276, 0.0713]`), while LapChole declined by 0.0461
(`[-0.0847, -0.0068]`). The main failure is systematic undercounting when two
or more objects are present. The strongest positive result is HeiCo
"all-instance" counting (`+0.0820`, paired 95% CI
`[0.0305, 0.1359]`), but this did not transfer to LapChole.

These results were produced on 2026-07-27. No training or checkpoint update
was performed.

## Compared methods

Both methods use the unchanged fine-tuned
[`Qwen3-VL-4B-Instruct-both-official/final`](../../track-frame/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/final)
checkpoint:

| Method | Inference behavior |
| --- | --- |
| Fine-tuned direct | Ask for a single non-negative integer and evaluate that answer. |
| Fine-tuned bounding-box prompt | Ask for labeled boxes in JSON, validate them, and derive the integer deterministically from valid boxes. |

All official FRAME `number` rows are counting questions: 1,326/4,000 HeiCo
rows and 768/2,252 LapChole rows. Only those 2,094 rows were re-run. The new
answers were merged into the complete direct-prompt predictions, preserving
the original response and order for every other row.

The structured prompt includes this one-shot format example:

```json
{"objects":[{"label":"Clip","bbox_2d":[88,140,166,230]},{"label":"Clip","bbox_2d":[612,355,701,438]}]}
```

Coordinates use a normalized 0–1000 scale. The application derives:

- all-instance questions from the number of valid boxes;
- distinct-class questions from unique canonical labels on valid boxes;
- named-target questions from valid boxes with the requested canonical label.

There is deliberately no fallback to a bare number. A valid empty list becomes
zero; malformed objects are excluded and recorded for auditing.

## Primary full-test comparison

The primary comparison is isolated: it reuses baseline correctness for
untouched rows and uses the new evaluator result only on changed counting
rows. Accuracy is the official per-video macro metric. Paired intervals are
10,000 video-level bootstrap samples with seed 42.

| Dataset | Test rows / videos | Fine-tuned direct | Bounding-box isolated | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 4,000 / 10 | 0.6643 | 0.6690 | +0.0047 | [-0.0095, 0.0228] |
| LapChole | 2,252 / 28 | 0.5551 | 0.5409 | -0.0141 | [-0.0288, 0.0009] |

The full evaluator was rerun after merging. Its unpaired summary intervals
were:

| Dataset | Direct evaluator score (95% CI) | Bounding-box rerun score (95% CI) | Bounding-box isolated |
| --- | ---: | ---: | ---: |
| HeiCo | 0.6643 [0.5787, 0.7383] | 0.6685 [0.5840, 0.7508] | 0.6690 |
| LapChole | 0.5551 [0.5153, 0.5954] | 0.5409 [0.4985, 0.5856] | 0.5409 |

On HeiCo, the local judge changed two of 2,674 untouched open-ended rows from
correct in the baseline evaluation to incorrect in the rerun. LapChole had no
untouched-row mismatch. This explains the 0.0005 difference between the raw
HeiCo rerun and the isolated score; it is not an effect of the counting
prompt. The full audit is in
[`judge-rerun-audit.csv`](counting-bbox-prompt/judge-rerun-audit.csv).

## Counting-question distribution

| Dataset | Counting rows / videos | Direct | Bounding box | Delta | Paired 95% CI | Direct micro | Box micro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 1,326 / 10 | 0.5356 | 0.5521 | +0.0166 | [-0.0276, 0.0713] | 0.5249 | 0.5392 |
| LapChole | 768 / 28 | 0.4116 | 0.3655 | -0.0461 | [-0.0847, -0.0068] | 0.3958 | 0.3542 |

### By counting mode

| Dataset | Mode | Rows | Direct | Bounding box | Delta | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | All instances | 487 | 0.4619 | 0.5439 | +0.0820 | [0.0305, 0.1359] |
| HeiCo | Distinct classes | 314 | 0.7282 | 0.7390 | +0.0108 | [-0.0167, 0.0383] |
| HeiCo | Named target | 525 | 0.5059 | 0.4712 | -0.0347 | [-0.1222, 0.0740] |
| LapChole | All instances | 343 | 0.3453 | 0.3298 | -0.0155 | [-0.0732, 0.0473] |
| LapChole | Distinct classes | 122 | 0.7286 | 0.6429 | -0.0857 | [-0.1518, -0.0214] |
| LapChole | Named target | 303 | 0.3707 | 0.3028 | -0.0679 | [-0.1235, -0.0127] |

The named-target distribution is dominated by Clip questions:

| Dataset | Target | Rows | Direct | Bounding box | Delta | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | Clip | 396 | 0.3077 | 0.2675 | -0.0402 | [-0.1692, 0.1295] |
| HeiCo | Sponge | 71 | 0.9403 | 0.9273 | -0.0130 | [-0.0977, 0.0694] |
| HeiCo | External Drain | 45 | 0.8500 | 0.9938 | +0.1438 | [0.0187, 0.3104] |
| LapChole | Clip | 285 | 0.3337 | 0.2608 | -0.0729 | [-0.1323, -0.0136] |
| LapChole | Sponge | 12 | 0.8889 | 0.8889 | 0.0000 | [0.0000, 0.0000] |

External Drain is promising but has only 45 questions across eight videos.
Needle, Specimen, and Specimen Bag have at most nine questions each, so their
apparent changes are not decision-grade. All target rows are retained in
[`question-type.csv`](counting-bbox-prompt/question-type.csv).

### By ground-truth count

| Dataset | Gold count | Rows | Direct | Bounding box | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 1 | 567 | 0.8324 | 0.9292 | +0.0968 | [0.0526, 0.1393] |
| HeiCo | 2 | 373 | 0.4405 | 0.3896 | -0.0509 | [-0.1113, 0.0095] |
| HeiCo | 3 | 146 | 0.2377 | 0.1524 | -0.0853 | [-0.1582, -0.0100] |
| HeiCo | 4 | 98 | 0.2230 | 0.0832 | -0.1399 | [-0.3098, -0.0212] |
| HeiCo | 5 | 55 | 0.1235 | 0.0667 | -0.0569 | [-0.1235, 0.0000] |
| HeiCo | 6+ | 87 | 0.0000 | 0.0460 | +0.0460 | [0.0460, 0.0460] |
| LapChole | 1 | 170 | 0.7357 | 0.9554 | +0.2197 | [0.1463, 0.2971] |
| LapChole | 2 | 154 | 0.5102 | 0.3467 | -0.1635 | [-0.2776, -0.0558] |
| LapChole | 3 | 130 | 0.2986 | 0.2802 | -0.0185 | [-0.1310, 0.1042] |
| LapChole | 4 | 103 | 0.3827 | 0.1111 | -0.2716 | [-0.3951, -0.1512] |
| LapChole | 5 | 96 | 0.2880 | 0.1105 | -0.1775 | [-0.2892, -0.0728] |
| LapChole | 6+ | 115 | 0.0183 | 0.0098 | -0.0085 | [-0.0388, 0.0188] |

The prompt is very effective when the answer is one, then generally degrades
as the number of objects increases. HeiCo's `6+` bucket comes from one video,
so its degenerate interval should not be interpreted as evidence of broad
improvement.

## Capability and answer-format distribution

Only number questions were changed. In the isolated comparison, all
non-aggregation capabilities and non-number answer formats are therefore
exactly unchanged.

### Capability groups

| Dataset | Capability group | Rows | Direct | Bounding box isolated | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| HeiCo | Aggregation | 1,875 | 0.6107 | 0.6231 | +0.0124 |
| HeiCo | Object recognition | 2,125 | 0.7276 | 0.7276 | 0.0000 |
| LapChole | Aggregation | 955 | 0.4944 | 0.4571 | -0.0373 |
| LapChole | Object recognition | 1,296 | 0.6243 | 0.6243 | 0.0000 |
| LapChole | Temporal grounding | 1 | 0.0000 | 0.0000 | 0.0000 |

### Leaf capabilities

| Dataset | Leaf capability | Rows | Direct | Bounding box isolated | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| HeiCo | Object aggregation | 1,875 | 0.6107 | 0.6231 | +0.0124 |
| HeiCo | Object attributes | 21 | 0.8750 | 0.8750 | 0.0000 |
| HeiCo | Object identification | 1,591 | 0.7034 | 0.7034 | 0.0000 |
| HeiCo | Spatial localization — camera | 427 | 0.8064 | 0.8064 | 0.0000 |
| HeiCo | Spatial localization — situs | 86 | 0.6708 | 0.6708 | 0.0000 |
| LapChole | Object aggregation | 955 | 0.4944 | 0.4571 | -0.0373 |
| LapChole | Object attributes | 192 | 0.6276 | 0.6276 | 0.0000 |
| LapChole | Object identification | 866 | 0.5579 | 0.5579 | 0.0000 |
| LapChole | Spatial localization — camera | 213 | 0.8901 | 0.8901 | 0.0000 |
| LapChole | Spatial localization — situs | 25 | 0.8095 | 0.8095 | 0.0000 |
| LapChole | Temporal localization | 1 | 0.0000 | 0.0000 | 0.0000 |

### Answer formats

| Dataset | Format | Rows | Direct | Bounding box isolated | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| HeiCo | Binary | 548 | 0.7993 | 0.7993 | 0.0000 |
| HeiCo | FO class | 1,755 | 0.7082 | 0.7082 | 0.0000 |
| HeiCo | Multiple choice | 112 | 0.9009 | 0.9009 | 0.0000 |
| HeiCo | Number | 1,326 | 0.5356 | 0.5521 | +0.0166 |
| HeiCo | Open ended | 259 | 0.7732 | 0.7732 | 0.0000 |
| LapChole | Binary | 176 | 0.8707 | 0.8707 | 0.0000 |
| LapChole | FO class | 920 | 0.5711 | 0.5711 | 0.0000 |
| LapChole | Multiple choice | 90 | 0.9431 | 0.9431 | 0.0000 |
| LapChole | Number | 768 | 0.4116 | 0.3655 | -0.0461 |
| LapChole | Open ended | 298 | 0.7163 | 0.7163 | 0.0000 |

The machine-readable distributions, including evaluator confidence intervals,
are in [`capability.csv`](counting-bbox-prompt/capability.csv) and
[`answer-format.csv`](counting-bbox-prompt/answer-format.csv).

## Error behavior

### Paired outcomes on counting rows

| Dataset | Both correct | Box only correct | Direct only correct | Both wrong | Net changed correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 567 | 148 | 129 | 482 | +19 |
| LapChole | 178 | 94 | 126 | 370 | -32 |

Across both datasets, the prompt loses a net 13 correct answers out of 2,094
counting questions.

### Count bias

| Dataset | Mode | Gold mean | Direct mean | Box mean | Direct undercount | Box undercount |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HeiCo | All instances | 2.476 | 2.152 | 1.676 | 32.9% | 41.7% |
| HeiCo | Distinct classes | 1.338 | 1.223 | 1.178 | 17.5% | 18.8% |
| HeiCo | Named target | 2.830 | 2.293 | 1.610 | 34.5% | 53.7% |
| LapChole | All instances | 3.638 | 3.149 | 2.329 | 44.9% | 64.4% |
| LapChole | Distinct classes | 1.730 | 1.475 | 1.270 | 24.6% | 37.7% |
| LapChole | Named target | 3.508 | 3.066 | 2.211 | 42.9% | 66.0% |

The JSON requirement is not the limiting factor. The model usually emits
valid structure, but generates too few boxes for crowded frames. Complete
mean absolute error and overcount rates are in
[`count-bias.csv`](counting-bbox-prompt/count-bias.csv).

## Output validity and runtime

The full run completed on an NVIDIA H200 NVL with no inference errors and all
2,094 final derived answers parseable as numbers.

| Dataset | Counting rows | Parser-valid object schema | Other parse/status rows | Errors | Median latency | p95 latency | Maximum | Over 5 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeiCo | 1,326 | 1,319 (99.47%) | 7 | 0 | 0.859 s | 2.647 s | 15.844 s | 5 |
| LapChole | 768 | 757 (98.57%) | 11 | 0 | 1.363 s | 3.269 s | 23.518 s | 16 |

The 18 non-fully-valid responses were all incorrect. One HeiCo response was
recovered with the box regex; the remaining statuses were invalid JSON or
partially invalid objects. See
[`structured-output-status.csv`](counting-bbox-prompt/structured-output-status.csv).

The evaluator did not mark any response as timed out, but 21/2,094 (1.0%)
measured generations exceeded five seconds. This version is therefore not
submission-safe if five seconds is enforced externally. The historical direct
baseline ran on B200 hardware, so these H200 latencies are an operational
measurement, not a controlled direct-versus-box speed comparison.

## Recommendation

Keep the fine-tuned direct prompt as the default counting method. The
bounding-box prompt is not dataset-robust, and a global switch would reduce
the LapChole score.

A follow-up experiment could test a gated or hybrid strategy for single-object
and HeiCo all-instance cases, or generate region proposals outside the VLM so
the number of candidates is not limited by autoregressive box generation.
Ground-truth count is unavailable at inference time, however, so the strong
`count=1` result cannot itself be used as a deployable gate. The External Drain
result also needs a larger independent sample before class-conditioned routing.

## Reproducibility and artifacts

- Method and run instructions:
  [`architecture.md`](../../track-frame/counting-bbox-prompt/architecture.md)
- Shared prompt construction:
  [`src/prompts.py`](../../src/prompts.py)
- Structured parser and deterministic reducer:
  [`src/counting.py`](../../src/counting.py)
- Experiment runner and merge/evaluation scripts:
  [`track-frame/counting-bbox-prompt/`](../../track-frame/counting-bbox-prompt/)
- Completed run:
  [`bbox-json-epoch30-20260727/`](../../track-frame/counting-bbox-prompt/logs/bbox-json-epoch30-20260727/)
- Full SLURM log:
  [`slurm-1374.log`](../../track-frame/counting-bbox-prompt/logs/slurm-1374.log)
- Reproducible numerical tables and provenance:
  [`counting-bbox-prompt/`](counting-bbox-prompt/)

The complete table set includes dataset overall, paired overall, counting
overall, counting mode, named target, gold-count bucket, count bias, capability,
answer format, paired outcomes, structured-output status, runtime, and judge
rerun audit distributions.
