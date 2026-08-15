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

Only 3 of the 6 track×dataset cells differ significantly from the specialist.
Driven by two opposite cross-track transfers: frame's object data lifts
`object_recognition` everywhere it is new (+0.03 to +0.06, p<0.01 in all four
segment/procedure cells), while procedure's coarse timestamps degrade segment's
fine `temporal_grounding` (−0.07/−0.09) and improve procedure's (+0.047).

| Deployment configuration | Models | Six-cell mean |
| --- | ---: | ---: |
| All specialists | 3 | 0.6128 |
| All unified (epoch 8) | 1 | 0.6267 |
| **Unified + segment specialist** | **2** | **0.6296** |

Reports in this directory must show HeiCo and LapChole separately before any
cross-dataset aggregate, then break results down by capability and relevant
question subtype. Because a unified model is compared against a *different*
model per track, every headline claim here needs a paired test — the marginal
confidence intervals in `summary.csv` are per-video macros with an effective n of
10 (HeiCo) or 28 (LapChole) and are far too wide to settle these differences.
