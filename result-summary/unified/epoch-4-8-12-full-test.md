# Unified 3-track LoRA — full official test, epochs 4 / 8 / 12

Full official test (every test question, both datasets, all three tracks) of the
single multi-task LoRA, measured against the three specialist LoRAs currently
holding each track's best full-test score.

**Verdict: procedure gains ~+0.05/+0.04 and the gain is statistically solid;
frame is a wash; segment loses ~0.02 on HeiCo. Epoch 8 is the best checkpoint.**

**Why (§6): joint training makes two trades. Frame's object data lifts
`object_recognition` in all four segment/procedure cells; procedure's coarse
timestamps drag segment's fine `temporal_grounding` down while lifting
procedure's. Procedure gains on both axes and wins; segment gains objects but
loses time, which costs it on HeiCo where 44% of questions are temporal; frame
has neither exposure and ties.**

## 1. What was run

| Item | Value |
| --- | --- |
| Run | `track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official` |
| Base model | Qwen3-VL-4B-Instruct |
| Adapter | LoRA r16 / α32, LLM + vision targets |
| Training data | all three tracks jointly, 34,367 rows/epoch (13,748 frame + 13,746 segment + 6,873 procedure) |
| Checkpoints tested | epoch 4 = `checkpoint-8592`, epoch 8 = `checkpoint-17184`, epoch 12 = `checkpoint-25776` |
| Adapter sha256 (epoch 4) | `fc5d2fdc9e1585bc4fb262c7c6e2bdc51213536afbb7ee7d0b9a3e590a228ee5` |
| Judge | local Qwen3.5-4B FOCUS judge, fixed multi-label `fo_class` normaliser |
| Test date | 2026-08-14 |
| Slurm jobs | 4538 (frame, b200), 4539 (segment, h200), 4540 (procedure, a100) — 15 array tasks, all `COMPLETED 0:0` |

Inference settings match each track's specialist exactly, and each track was
pinned to the GPU family its specialist baseline was measured on, because greedy
bf16 decoding is not bit-identical across GPU families and the epoch deltas under
test are ~0.01:

| Track | Partition | Frames | max_pixels | Stride (heico / lapchole) |
| --- | --- | ---: | ---: | --- |
| frame | b200 | — | 602,112 | — |
| segment | h200 | 64 | 262,144 | 25 / 30 |
| procedure | a100 | 96 | 131,072 | 250 / 300 |

**Validity.** 18/18 dataset cells produced a `COMPLETE` marker. Every cell passed
`validate_full_test.py`: exact expected row count, all `sample_id`s unique and
matching the test split, zero inference errors, zero empty generations.

## 2. Official metric, per dataset

The headline is `overall,MEAN` — the official **per-video macro** accuracy. Note
the effective sample size is the **video** count, not the question count, which is
why the intervals are wide: HeiCo's test split has only **10 videos**, LapChole's
has **28**.

### HeiCo (10 videos)

| Track | n | epoch 4 | epoch 8 | epoch 12 | Specialist |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame | 4000 | **0.7125** | 0.7120 | 0.7038 | 0.7067 |
| segment | 4000 | **0.6870** | 0.6795 | 0.6832 | 0.7023 |
| procedure | 2000 | 0.3450 | **0.3705** | 0.3580 | 0.3105 |

### LapChole (28 videos)

| Track | n | epoch 4 | epoch 8 | epoch 12 | Specialist |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame | 2252 | 0.6216 | 0.6394 | **0.6500** | 0.6417 |
| segment | 2254 | 0.7657 | **0.7803** | 0.7798 | 0.7745 |
| procedure | 1127 | 0.5545 | **0.5786** | 0.5740 | 0.5412 |

### Epoch-8 intervals

| Track / dataset | epoch 8 | 95% CI | Specialist |
| --- | ---: | --- | ---: |
| frame / heico | 0.7120 | [0.6349, 0.7773] | 0.7067 |
| frame / lapchole | 0.6394 | [0.5949, 0.6779] | 0.6417 |
| segment / heico | 0.6795 | [0.6213, 0.7388] | 0.7023 |
| segment / lapchole | 0.7803 | [0.7405, 0.8166] | 0.7745 |
| procedure / heico | 0.3705 | [0.3190, 0.4395] | 0.3105 |
| procedure / lapchole | 0.5786 | [0.5281, 0.6308] | 0.5412 |

These marginal intervals are **not** the right tool for comparing the two runs —
they overlap almost everywhere, including where the difference is real. Both runs
answered the *same* questions on the *same* videos, so the comparison must be
paired. See §3.

### Cross-dataset means

Reported after the per-dataset numbers above, and only as a convenience — the two
datasets differ enough that the mean hides the story.

| Track | epoch 4 | epoch 8 | epoch 12 | Specialist |
| --- | ---: | ---: | ---: | ---: |
| frame | 0.6670 | 0.6757 | **0.6769** | 0.6742 |
| segment | 0.7263 | 0.7299 | **0.7315** | 0.7384 |
| procedure | 0.4497 | **0.4745** | 0.4660 | 0.4259 |

## 3. Paired comparison against the specialists

Produced by `track-unified/lora-finetune/src/paired_vs_specialist.py`, joining on
`qID`. Two tests, because they fail in opposite directions:

- **Wilcoxon** signed-rank over per-video accuracy deltas. Respects the official
  per-video macro and the clustering of questions within a video, but on HeiCo
  it has only n=10 → badly underpowered.
- **McNemar** (exact binomial) over the individual questions that flipped. Much
  more powerful, but treats questions as independent, so it *overstates*
  significance where video-level effects dominate.

Where both agree, the effect is real. Where only McNemar fires, treat it as
suggestive.

Two provenance notes on the frame baseline. It was scored one day before the
multi-label `fo_class` fix landed (2026-07-25), so its authoritative numbers come
from `eval/rescore-results.csv` column `correctness_new`, **not** the stored
`eval/summary.csv` (which still shows the pre-fix 0.6643 / 0.5551). Segment's and
procedure's baselines both postdate the fix and need no rescore. Recomputing the
frame/lapchole macro from that rescore file gives 0.6415 against the 0.6417
published in [the rescore report](../frame/epoch30-multilabel-rescore.md); the
0.0002 gap changes no conclusion, and the Δ column below is computed against the
recomputed value.

| Track / dataset | Epoch | Δ macro | Wilcoxon p | McNemar p | uni✓ spec✗ | uni✗ spec✓ | videos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frame / heico | 4 | +0.0057 | 0.610 | 0.380 | 325 | 302 | 10 |
| frame / heico | 8 | +0.0052 | 0.507 | 0.400 | 293 | 272 | 10 |
| frame / heico | 12 | −0.0030 | 0.307 | 0.636 | 264 | 276 | 10 |
| frame / lapchole | 4 | −0.0200 | 0.085 | **0.041** | 209 | 254 | 28 |
| frame / lapchole | 8 | −0.0022 | 0.811 | 0.845 | 208 | 213 | 28 |
| frame / lapchole | 12 | +0.0084 | 0.339 | 0.384 | 223 | 204 | 28 |
| segment / heico | 4 | −0.0153 | 0.610 | **0.028** | 341 | 402 | 10 |
| segment / heico | 8 | −0.0228 | 0.307 | **0.00090** | 323 | 414 | 10 |
| segment / heico | 12 | −0.0190 | 0.169 | **0.0049** | 317 | 393 | 10 |
| segment / lapchole | 4 | −0.0088 | 0.973 | 0.336 | 206 | 227 | 28 |
| segment / lapchole | 8 | +0.0058 | 0.452 | 0.592 | 217 | 205 | 28 |
| segment / lapchole | 12 | +0.0053 | 0.561 | 0.627 | 217 | 206 | 28 |
| procedure / heico | 4 | +0.0345 | 0.103 | **0.00028** | 211 | 142 | 10 |
| procedure / heico | 8 | **+0.0600** | **0.0050** | **6.1e-11** | 229 | 109 | 10 |
| procedure / heico | 12 | +0.0475 | **0.021** | **1.4e-07** | 209 | 114 | 10 |
| procedure / lapchole | 4 | +0.0133 | 0.640 | 0.344 | 117 | 102 | 28 |
| procedure / lapchole | 8 | **+0.0374** | **0.025** | **0.0030** | 117 | 75 | 28 |
| procedure / lapchole | 12 | +0.0328 | **0.011** | **0.0094** | 115 | 78 | 28 |

Reading it:

- **Procedure is a genuine win.** At epoch 8 both tests fire on both datasets
  (HeiCo p=0.005 / 6e-11, LapChole p=0.025 / 0.003). Wilcoxon reaching p=0.005 on
  just 10 videos means nearly every video improved — that is about as strong as
  this test can get at n=10. Epoch 12 confirms it. This is the one place the
  unified model clearly beats its specialist.
- **Frame is a wash.** Nothing significant at epochs 8 or 12 on either dataset;
  the flip counts are near-symmetric (293/272, 208/213). The one significant cell
  is epoch 4 on LapChole, and it is *negative* — epoch 4 is simply undertrained
  for frame/lapchole.
- **Segment/HeiCo is a real regression.** Wilcoxon can't confirm it at n=10, but
  McNemar fires at **all three** checkpoints, always in the same direction, with
  ~80 more questions lost than won each time. Three independent checkpoints
  agreeing makes a chance finding unlikely. Segment/LapChole is a wash.

Net across all six buckets at epoch 8: **+0.0139**.

## 4. The in-loop subset eval was misleading

Training used a fixed seeded 1,500-row subset (500/track, `eval_subset_qids.json`)
for its per-epoch eval. It ranked epochs acceptably but got bucket *levels* badly
wrong, and it inflated the headline.

Epoch 8, subset vs full official test:

| Bucket | Subset | Full test | Subset error |
| --- | ---: | ---: | ---: |
| frame / heico | 0.7145 | 0.7120 | +0.002 |
| frame / lapchole | 0.7030 | 0.6394 | **+0.064** |
| segment / heico | 0.6593 | 0.6795 | −0.020 |
| segment / lapchole | 0.8155 | 0.7803 | +0.035 |
| procedure / heico | 0.3813 | 0.3705 | +0.011 |
| procedure / lapchole | 0.5504 | 0.5786 | −0.028 |
| **mean** | **0.6373** | **0.6267** | **+0.011** |

frame/lapchole is the worst offender: its 183-row subset draw was easy, reading
+0.064 high at epoch 8 and +0.079 high at epoch 12 (0.7288 subset vs 0.6500
full). The subset's headline "+0.024 over the specialists" is really **+0.014**,
and almost all of that is procedure.

**Take-away for future runs:** the in-loop subset is fine for picking an epoch,
but no subset number should ever be compared against a full-test baseline. At
n≈180–320 per bucket the per-bucket SE is ~±0.03, which is larger than every
effect we are trying to measure.

## 5. Where the unified model wins and loses

Both from §6/§7 below.

- **Procedure's gain is broad, not a single bucket.** On HeiCo, `fo_class`
  0.6464→0.7017, `complex_reasoning` 0.659→0.770, `open_ended` 0.793→0.890,
  `number` 0.272→0.309 between epochs 4 and 8. Sharing frame and segment data
  evidently teaches procedure-level questions something the 6,873-row
  procedure-only diet does not.
- **Segment's loss is entirely temporal.** In segment/heico the two temporal
  leaves both decline monotonically across epochs 4/8/12 —
  `temporal_localization` (1605 of 4000 questions, the dominant bucket)
  0.5733 → 0.5502 → 0.5441, and `duration_estimation` (147)
  0.3504 → 0.3374 → 0.3312. Nearly everything else *improves*:
  `instance_matching` 0.8541 → 0.9418, `object_aggregation` 0.6758 → 0.7008,
  `spatial_localization_camera` 0.9000 → 0.9100, `fo_usage_purpose`
  0.7963 → 0.9722. So the unified model is not broadly worse at segment; it is
  specifically worse at placing events in time, in the one bucket big enough to
  sink the track score. The mechanism is confirmed in §6: procedure shares the
  same `time` answer format at a much coarser granularity (stride 250–300 vs
  25–30), and joint training moves the shared temporal behaviour toward
  procedure's granularity.
- **`percentage` is broken everywhere** (0.00–0.075 across every bucket and
  epoch, n=17–22 per bucket). Small, but it is free score being left on the table
  and it is not a unified-model regression — worth a separate look.
- **`number` remains the weakest large format**: frame/lapchole 0.4405,
  procedure/heico 0.3086 at epoch 8. Also pre-existing.

## 6. Why: two cross-track transfers pulling in opposite directions

Bucket-level paired comparison against each specialist, from
`track-unified/lora-finetune/src/unified_vs_specialist_buckets.py` (epoch 8,
same rows, McNemar within each bucket). Bucket accuracies are macro-over-video,
matching how `summary.csv` reports them — verified against the run's own summary
rather than assumed.

The track totals in §2 are the sum of two clean, opposite effects.

### 6.1 Frame's object data lifts segment and procedure

`object_recognition` improves in **all four** segment and procedure cells, three
of them at p<0.01 — and does *not* improve for frame, which already had that
data. This is the joint training paying off.

| Cell | n | Unified | Specialist | Δ | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame / heico | 2125 | 0.7995 | 0.8034 | −0.0040 | 0.51 |
| frame / lapchole | 1296 | 0.7441 | 0.7688 | −0.0246 | 0.080 |
| segment / heico | 1383 | 0.8926 | 0.8609 | **+0.0317** | **9.2e-04** |
| segment / lapchole | 1440 | 0.8596 | 0.8222 | **+0.0374** | **2.9e-05** |
| procedure / heico | 374 | 0.7231 | 0.6585 | **+0.0646** | **0.0095** |
| procedure / lapchole | 564 | 0.8098 | 0.7584 | **+0.0514** | **0.0022** |

### 6.2 Procedure's coarse timestamps corrupt segment's fine ones

`temporal_grounding` moves in **opposite directions** on the two tracks that have
it — segment loses, procedure gains, both significantly. Segment's questions are
answered at stride 25–30 and procedure's at 250–300, so the two tracks want
different temporal resolutions out of the same weights; joint training settles
nearer procedure's.

| Cell | n | share of track | Unified | Specialist | Δ | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| segment / heico | 1752 | 44% | 0.5342 | 0.6037 | **−0.0695** | **3.4e-11** |
| segment / lapchole | 684 | 30% | 0.5855 | 0.6789 | **−0.0934** | **6.7e-04** |
| procedure / heico | 795 | 40% | 0.1629 | 0.1162 | **+0.0467** | **6.1e-08** |
| procedure / lapchole | 426 | 38% | 0.3040 | 0.2870 | +0.0169 | 0.13 |

This is why the three tracks land where they do:

- **Procedure gains on both axes** — objects *and* time — so it wins outright.
- **Segment gains objects but loses time.** On LapChole object questions are 64%
  of the split and temporal only 30%, so the two roughly cancel (net +0.006). On
  HeiCo temporal is 44%, so the loss dominates (net −0.023). Same trade, different
  question mix, opposite outcome.
- **Frame has no temporal questions to lose and no object gain to make**, so it
  ties. Its only movement is `object_aggregation` (counting) +0.020, p≈0.09–0.2 —
  suggestive, not significant.

(The bucket deltas are macro-over-video, so they do not linearly compose into the
overall macro; the shares above explain the *direction*, not an exact
decomposition.)

## 7. Recommendation

Ship **epoch 8** (`checkpoint-17184`) if a single 3-track model is wanted: it is
the best or statistically tied-best checkpoint in five of six buckets, and it is
where procedure peaks on both datasets.

Of the six track×dataset cells, only **three** differ significantly from the
specialist: procedure wins both, segment/HeiCo loses. Frame is a tie on both
datasets, so keeping the frame specialist buys nothing.

| Configuration | Models | Six-cell mean |
| --- | ---: | ---: |
| All specialists | 3 | 0.6128 |
| All unified (epoch 8) | 1 | 0.6267 |
| **Unified + segment specialist** | **2** | **0.6296** |

**Segment is the only track where keeping a specialist is worth anything**
(+0.0085 on the track mean, driven entirely by HeiCo). Going all-unified already
beats all-specialists by +0.0139; adding back just the segment specialist
recovers a further +0.0029 at the cost of a second model.

If the temporal-resolution conflict in §6.2 can be fixed — e.g. conditioning on
the track in the prompt, or per-track temporal tokenisation — a single unified
model should be able to take segment's HeiCo score back and dominate everywhere.
That is the highest-value follow-up this run points to.

## 8. Capability and answer-format breakdowns

Generated by `track-unified/lora-finetune/src/full_test_report.py`.
### 8.1 Capability groups

#### frame / heico

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 1875 | 0.6340 | 0.6309 | 0.6255 |
| object_recognition | 2125 | 0.7983 | 0.7995 | 0.7878 |

#### frame / lapchole

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 955 | 0.4819 | 0.5143 | 0.5204 |
| object_recognition | 1296 | 0.7325 | 0.7441 | 0.7522 |
| temporal_grounding | 1 | 1.0000 | 1.0000 | 1.0000 |

#### segment / heico

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 572 | 0.5764 | 0.5923 | 0.5960 |
| complex_reasoning | 148 | 0.7499 | 0.7176 | 0.7174 |
| event_understanding | 145 | 0.8945 | 0.8658 | 0.8728 |
| object_recognition | 1383 | 0.8907 | 0.8926 | 0.8987 |
| temporal_grounding | 1752 | 0.5556 | 0.5342 | 0.5289 |

#### segment / lapchole

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 22 | 0.6528 | 0.7083 | 0.5083 |
| complex_reasoning | 43 | 0.7969 | 0.8594 | 0.8542 |
| event_understanding | 65 | 0.9567 | 0.9100 | 0.9467 |
| object_recognition | 1440 | 0.8356 | 0.8596 | 0.8719 |
| temporal_grounding | 684 | 0.5816 | 0.5855 | 0.5640 |

#### procedure / heico

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 728 | 0.3134 | 0.3625 | 0.3292 |
| complex_reasoning | 51 | 0.6593 | 0.7704 | 0.7333 |
| event_understanding | 52 | 0.8267 | 0.8400 | 0.8433 |
| object_recognition | 374 | 0.7208 | 0.7231 | 0.7162 |
| temporal_grounding | 795 | 0.1474 | 0.1629 | 0.1546 |

#### procedure / lapchole

| group | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| aggregation | 95 | 0.5255 | 0.5467 | 0.5172 |
| complex_reasoning | 25 | 0.8452 | 0.8929 | 0.8810 |
| event_understanding | 17 | 0.5577 | 0.5577 | 0.5577 |
| object_recognition | 564 | 0.7754 | 0.8098 | 0.8239 |
| temporal_grounding | 426 | 0.3007 | 0.3040 | 0.2906 |

### 8.2 Leaf capabilities

#### frame / heico

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| object_aggregation | 1875 | 0.6340 | 0.6309 | 0.6255 |
| object_attributes | 21 | 0.8177 | 0.7083 | 0.5417 |
| object_identification | 1591 | 0.8011 | 0.8021 | 0.7858 |
| spatial_localization_camera | 427 | 0.8156 | 0.8289 | 0.8339 |
| spatial_localization_situs | 86 | 0.7339 | 0.7694 | 0.7504 |

#### frame / lapchole

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| object_aggregation | 955 | 0.4819 | 0.5143 | 0.5204 |
| object_attributes | 192 | 0.6088 | 0.5708 | 0.6069 |
| object_identification | 866 | 0.7444 | 0.7495 | 0.7522 |
| spatial_localization_camera | 213 | 0.7660 | 0.8327 | 0.8213 |
| spatial_localization_situs | 25 | 0.8214 | 0.9048 | 0.9524 |
| temporal_localization | 1 | 1.0000 | 1.0000 | 1.0000 |

#### segment / heico

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| causal_consequence_reasoning | 62 | 0.6704 | 0.6354 | 0.5894 |
| duration_estimation | 147 | 0.3504 | 0.3374 | 0.3312 |
| event_aggregation | 169 | 0.3633 | 0.3818 | 0.3749 |
| fo_interaction_recognition | 75 | 0.8731 | 0.8458 | 0.8472 |
| fo_usage_purpose | 19 | 0.7963 | 0.9722 | 0.9722 |
| functional_reasoning | 34 | 0.8688 | 0.8687 | 0.8750 |
| instance_matching | 141 | 0.8541 | 0.9276 | 0.9418 |
| multi_step_reasoning | 52 | 0.6962 | 0.6438 | 0.6962 |
| object_aggregation | 403 | 0.6758 | 0.6941 | 0.7008 |
| object_attributes | 27 | 0.8167 | 0.8167 | 0.8167 |
| object_identification | 713 | 0.9013 | 0.8873 | 0.8923 |
| spatial_localization_camera | 490 | 0.9000 | 0.9109 | 0.9100 |
| spatial_localization_situs | 12 | 0.6133 | 0.5133 | 0.6200 |
| temporal_localization | 1605 | 0.5733 | 0.5502 | 0.5441 |
| temporal_ordering | 51 | 0.9524 | 0.8881 | 0.9048 |

#### segment / lapchole

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| causal_consequence_reasoning | 29 | 0.8631 | 0.9048 | 0.8810 |
| duration_estimation | 6 | 0.0000 | 0.1111 | 0.1111 |
| event_aggregation | 3 | 0.6667 | 0.6667 | 0.3333 |
| fo_interaction_recognition | 5 | 0.8000 | 1.0000 | 1.0000 |
| fo_usage_purpose | 56 | 0.9900 | 0.9100 | 0.9800 |
| functional_reasoning | 13 | 0.5714 | 0.7143 | 0.7143 |
| instance_matching | 141 | 0.9405 | 0.9155 | 0.9357 |
| multi_step_reasoning | 1 | 0.0000 | 0.0000 | 1.0000 |
| object_aggregation | 19 | 0.5909 | 0.6818 | 0.5545 |
| object_attributes | 63 | 0.8869 | 0.8869 | 0.8750 |
| object_identification | 741 | 0.8264 | 0.8648 | 0.8677 |
| spatial_localization_camera | 480 | 0.8289 | 0.8323 | 0.8583 |
| spatial_localization_situs | 15 | 0.6818 | 0.8182 | 0.8182 |
| temporal_localization | 678 | 0.5843 | 0.5875 | 0.5662 |
| temporal_ordering | 4 | 0.6667 | 1.0000 | 0.3333 |

#### procedure / heico

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| duration_estimation | 94 | 0.1339 | 0.1274 | 0.1274 |
| event_aggregation | 326 | 0.1952 | 0.2554 | 0.2036 |
| fo_interaction_recognition | 23 | 0.7667 | 0.7500 | 0.7500 |
| fo_usage_purpose | 1 | 1.0000 | 0.0000 | 1.0000 |
| instance_matching | 61 | 0.8643 | 0.7917 | 0.7806 |
| multi_step_reasoning | 51 | 0.6593 | 0.7704 | 0.7333 |
| object_aggregation | 402 | 0.4246 | 0.4752 | 0.4451 |
| object_identification | 261 | 0.6871 | 0.7159 | 0.7072 |
| spatial_localization_camera | 50 | 0.8229 | 0.7729 | 0.7729 |
| spatial_localization_situs | 2 | 1.0000 | 1.0000 | 1.0000 |
| temporal_localization | 701 | 0.1503 | 0.1686 | 0.1588 |
| temporal_ordering | 28 | 0.9259 | 0.9630 | 0.9630 |

#### procedure / lapchole

| leaf | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| causal_consequence_reasoning | 17 | 0.9091 | 0.9394 | 0.9697 |
| duration_estimation | 35 | 0.1500 | 0.1812 | 0.1500 |
| event_aggregation | 18 | 0.8235 | 0.9118 | 0.9118 |
| fo_interaction_recognition | 8 | 0.4286 | 0.5714 | 0.4286 |
| fo_usage_purpose | 4 | 1.0000 | 0.7500 | 1.0000 |
| functional_reasoning | 6 | 0.5000 | 0.7000 | 0.9000 |
| instance_matching | 99 | 0.8423 | 0.8869 | 0.8690 |
| multi_step_reasoning | 2 | 1.0000 | 1.0000 | 0.5000 |
| object_aggregation | 77 | 0.4123 | 0.4123 | 0.3815 |
| object_identification | 402 | 0.7714 | 0.7959 | 0.8327 |
| spatial_localization_camera | 53 | 0.8524 | 0.8637 | 0.8190 |
| spatial_localization_situs | 10 | 0.7143 | 1.0000 | 0.8571 |
| temporal_localization | 391 | 0.3013 | 0.3039 | 0.2911 |
| temporal_ordering | 5 | 0.4000 | 0.4000 | 0.4000 |

### 8.3 Answer formats

#### frame / heico

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 548 | 0.8407 | 0.7975 | 0.7985 |
| fo_class | 1755 | 0.7992 | 0.7967 | 0.7838 |
| multiple_choice | 112 | 0.8621 | 0.9459 | 0.9237 |
| number | 1326 | 0.5512 | 0.5661 | 0.5584 |
| open_ended | 259 | 0.7742 | 0.7820 | 0.7788 |

#### frame / lapchole

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 176 | 0.8731 | 0.8472 | 0.8796 |
| fo_class | 920 | 0.7415 | 0.7493 | 0.7515 |
| multiple_choice | 90 | 0.8756 | 0.8498 | 0.8677 |
| number | 768 | 0.3973 | 0.4405 | 0.4390 |
| open_ended | 298 | 0.6537 | 0.6863 | 0.7179 |

#### segment / heico

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 287 | 0.8731 | 0.8846 | 0.8991 |
| fo_class | 851 | 0.8716 | 0.8648 | 0.8728 |
| multiple_choice | 468 | 0.8956 | 0.9099 | 0.9091 |
| number | 481 | 0.5438 | 0.5482 | 0.5516 |
| open_ended | 201 | 0.8437 | 0.8345 | 0.8291 |
| percentage | 22 | 0.0750 | 0.0500 | 0.0250 |
| time | 1690 | 0.5559 | 0.5345 | 0.5286 |

#### segment / lapchole

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 193 | 0.9107 | 0.8973 | 0.9122 |
| fo_class | 693 | 0.8244 | 0.8653 | 0.8686 |
| multiple_choice | 465 | 0.8236 | 0.8257 | 0.8540 |
| number | 1 | 0.0000 | 0.0000 | 0.0000 |
| open_ended | 220 | 0.8629 | 0.8967 | 0.8700 |
| percentage | 2 | 0.0000 | 0.0000 | 0.0000 |
| time | 680 | 0.5833 | 0.5865 | 0.5651 |

#### procedure / heico

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 129 | 0.8741 | 0.8271 | 0.8224 |
| fo_class | 380 | 0.6464 | 0.7017 | 0.6988 |
| multiple_choice | 32 | 0.5667 | 0.4333 | 0.4556 |
| number | 625 | 0.2720 | 0.3086 | 0.2677 |
| open_ended | 50 | 0.7933 | 0.8900 | 0.8483 |
| percentage | 17 | 0.0000 | 0.0000 | 0.0000 |
| time | 767 | 0.1382 | 0.1523 | 0.1454 |

#### procedure / lapchole

| answer_format | n | epoch 4 | epoch 8 | epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| binary | 146 | 0.8387 | 0.8744 | 0.8827 |
| fo_class | 371 | 0.7394 | 0.7809 | 0.8019 |
| multiple_choice | 37 | 0.6667 | 0.7000 | 0.5833 |
| number | 11 | 0.3889 | 0.5556 | 0.3889 |
| open_ended | 142 | 0.6041 | 0.6667 | 0.6534 |
| percentage | 6 | 0.0000 | 0.0000 | 0.0000 |
| time | 414 | 0.2945 | 0.2986 | 0.2819 |

