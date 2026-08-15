# Unified (multi-track) results

Experiments where a **single** model serves all three tracks, rather than one
specialist LoRA per track. Comparisons here are always against the current
best specialist for each track.

## Experiments

| Experiment | Comparison | Report |
| --- | --- | --- |
| 3-track LoRA, epochs 4/8/12 | One Qwen3-VL-4B LoRA trained on all tracks jointly vs the three specialists, full official test | [Full test, epochs 4/8/12](epoch-4-8-12-full-test.md) |

## Current standing (epoch 8, `checkpoint-17184`)

| Track | HeiCo | LapChole | vs specialist |
| --- | ---: | ---: | --- |
| frame | 0.7120 | 0.6394 | tie (n.s. both datasets) |
| segment | 0.6795 | 0.7803 | **−0.023 on HeiCo**, tie on LapChole |
| procedure | 0.3705 | 0.5786 | **+0.060 / +0.037**, significant on both |

Reports in this directory must show HeiCo and LapChole separately before any
cross-dataset aggregate, then break results down by capability and relevant
question subtype. Because a unified model is compared against a *different*
model per track, every headline claim here needs a paired test — the marginal
confidence intervals in `summary.csv` are per-video macros with an effective n of
10 (HeiCo) or 28 (LapChole) and are far too wide to settle these differences.
