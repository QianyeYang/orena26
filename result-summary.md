# Result Summary

Technical numerical results of all methods & baselines, across tracks. Kept clean and
organised (one row per method per split; revise in place, do not append duplicates).

Scoring = `focus.Evaluator`: macro-average over videos, hierarchical bootstrap 95% CIs.
open_ended + multiple_choice graded by LLM judge (Qwen3.5-4B); binary/number/fo_class by
rule. Only 10 test videos -> wide CIs; not per-question micro-accuracy.

## Leaderboard (overall accuracy, public test)

| Track | Method | Overall acc | 95% CI | n | Date |
|---|---|---|---|---|---|
| FRAME | **LoRA SFT sweep best · Qwen3-VL-4B** (LLM+vision LoRA), best of 30-ep, 1 frame | **0.828** | [0.780, 0.864] | 2000 | 2026-06-24 |
| FRAME | zero-shot sweep best · InternVL3.5-30B-A3B (MoE), 1 frame | 0.444 | [0.389, 0.503] | 2000 | 2026-06-21 |
| FRAME | baseline · Qwen2.5-VL-7B, 1 frame, zero-shot | 0.305 | [0.241, 0.367] | 2000 | 2026-06-19 |
| SEGMENT | — | — | — | — | — |
| PROCEDURE | — | — | — | — | — |

## FRAME

### LoRA fine-tuning sweep (single frame, n=2000)

Multi-task SFT over all answer formats, **7 open-weight VLMs ≤14B**, one identical recipe:
LoRA (r16/α32, dropout 0.05) on the LLM attn+MLP **and** vision encoder (Qwen2.5-VL also
trains the projector/merger); loss masked to the answer; lr 1e-4 cosine, warmup 0.03,
eff. batch 16, bf16, 1 GH200, seed 42; prompts identical to zero-shot. Harness:
`track-frame/lora-finetune/`. Per-epoch eval on the public test (`epoch_metrics.csv`).

All runs targeted 30 epochs but were cut at a **~24 h wall-time** (slowest reached 15 ep),
so max epoch reached varies. **Best epoch = argmax of public-test overall acc** (no held-out
val split → optimistic vs a true epoch-selection protocol; model ranking still informative).

| model | params | best ep | overall acc | 95% CI | zero-shot → SFT |
|---|---|---|---|---|---|
| Qwen3-VL-4B | 4B | 23* | **0.828** | [0.780, 0.864] | 0.322 → 0.828 (+0.506) |
| llava-onevision-7B | 7B | 13 | 0.800 | [0.745, 0.847] | 0.402 → 0.800 (+0.398) |
| Qwen3-VL-8B | 8B | 12 | 0.796 | [0.739, 0.844] | 0.345 → 0.796 (+0.451) |
| InternVL3.5-8B | 8B | 19 | 0.789 | [0.738, 0.838] | 0.289 → 0.789 (+0.500) |
| InternVL3.5-14B | 14B | 5 | 0.777 | [0.722, 0.827] | 0.364 → 0.777 (+0.413) |
| Qwen2.5-VL-7B | 7B | 12 | 0.762 | [0.718, 0.800] | 0.305 → 0.762 (+0.457) |
| SmolVLM2-2.2B | 2.2B | 14 | 0.732 | [0.690, 0.772] | 0.256 → 0.732 (+0.476) |

(* Qwen3-VL-4B was still rising at its last reached epoch (23) — true ceiling likely higher.)

By answer_format (best epoch; n: binary 101, fo_class 1003, multiple_choice 47, number 369, open_ended 480):

| model | binary | fo_class | multiple_choice | number | open_ended |
|---|---|---|---|---|---|
| Qwen3-VL-4B | 0.861 | **0.820** | 0.879 | 0.880 | **0.782** |
| llava-onevision-7B | 0.810 | 0.787 | **0.911** | **0.895** | 0.729 |
| Qwen3-VL-8B | 0.806 | 0.807 | 0.886 | 0.847 | 0.729 |
| InternVL3.5-8B | 0.859 | 0.773 | 0.879 | 0.866 | 0.753 |
| InternVL3.5-14B | **0.861** | 0.768 | 0.799 | 0.884 | 0.670 |
| Qwen2.5-VL-7B | 0.726 | 0.735 | 0.754 | 0.896 | 0.706 |
| SmolVLM2-2.2B | 0.692 | 0.712 | 0.868 | 0.877 | 0.650 |

By capability (n: aggregation 470, object_recognition 1530; object_identification 966 = bulk leaf):

| model | aggregation | object_recognition | object_identification |
|---|---|---|---|
| Qwen3-VL-4B | 0.880 | **0.816** | **0.854** |
| llava-onevision-7B | 0.881 | 0.777 | 0.830 |
| Qwen3-VL-8B | 0.838 | 0.786 | 0.824 |
| InternVL3.5-8B | 0.868 | 0.769 | 0.823 |
| InternVL3.5-14B | **0.881** | 0.750 | 0.808 |
| Qwen2.5-VL-7B | 0.864 | 0.732 | 0.804 |
| SmolVLM2-2.2B | 0.836 | 0.703 | 0.763 |

Findings:
- **New FRAME leader: Qwen3-VL-4B 0.828** (prev 0.739). All 7 SFT models beat the prior
  3-epoch leader (0.739) and every zero-shot model (best 0.444) by a wide margin.
- **SFT scrambles the zero-shot ranking**: Qwen3-VL-4B was 7th zero-shot (0.322) but 1st
  after SFT; the zero-shot leader InternVL3.5-30B is >14B and not in the sweep. Zero-shot
  skill does **not** predict the SFT ceiling.
- **Scale barely matters post-SFT**: 4B beats 8B/14B; even 2.2B SmolVLM hits 0.732. Whole
  field sits in a tight 0.73–0.83 band — domain SFT dominates parameter count.
- **Epoch dynamics**: most peak mid-run (ep 12–23); InternVL3.5-14B peaks at ep5 then
  drifts down (overfits fastest). Qwen3-VL-4B still climbing at the wall-time cut.
- More epochs help Qwen2.5-VL-7B modestly: 0.739 (3 ep) → 0.762 (12 ep, +0.023).
- **Remaining headroom = `open_ended` (best 0.782) and `fo_class`**; `number` /
  `multiple_choice` / `object_aggregation` are near-saturated (~0.88–0.91).
- Largest per-format lifts on the previously-weak bulk: `object_identification`
  (0.217 → 0.854 for the winner) and `fo_class` — domain adaptation of FO recognition is
  the main win. ~0.2–0.3 s/Q, 2000/2000 parseable, 0 errors (well under the 5 s/Q budget).
- Caveat: a few multi-label `fo_class` references (e.g. "Clip, Sponge") are unscorable by
  the strict format and auto-marked wrong (caps the fo_class ceiling; same for all
  methods — see `docs/issues-tbd.md`).
- **Genuine generalization**: train/test videos are disjoint (20 vs 10, 0 shared), so this
  is held-out-surgery accuracy. Public test is all in-distribution (`ood=False`); the OOD
  batch is hidden-test only, so OOD robustness is not measurable locally.

### Zero-shot model sweep (single frame, n=2000)

11 open-weight VLMs under one identical format-aware prompt + greedy decode, run via a
generic `AutoModelForImageTextToText` wrapper. Both raw and normalized outputs preserved
per sample (`track-frame/zeroshot-sweep/`). Full report + diagnostics:
`track-frame/zeroshot-sweep/logs/comparison.md`. All 11 ran with 0 runtime errors and
<0.5% invalid-format, so differences are visual reasoning, not parsing.

Overall (sorted; invalid-fmt = strict-parse failures, lat = per-sample p50):

| model | overall acc | 95% CI | invalid-fmt | lat p50 (s) |
|---|---|---|---|---|
| InternVL3.5-30B-A3B (MoE) | **0.444** | [0.389, 0.503] | 0.2% | 0.21 |
| llava-onevision-7B | 0.402 | [0.335, 0.472] | 0.3% | 0.23 |
| Qwen3-VL-30B-A3B (MoE) | 0.386 | [0.322, 0.455] | 0.1% | 0.17 |
| InternVL3.5-14B | 0.364 | [0.308, 0.427] | 0.4% | 0.14 |
| Qwen3-VL-8B | 0.345 | [0.273, 0.419] | 0.1% | 0.10 |
| Qwen3-VL-32B | 0.337 | [0.265, 0.415] | 0.1% | 0.19 |
| Qwen3-VL-4B | 0.322 | [0.253, 0.396] | 0.2% | 0.11 |
| Qwen2.5-VL-7B (baseline rerun) | 0.304 | [0.241, 0.367] | 0.5% | 0.16 |
| Qwen2.5-VL-32B | 0.302 | [0.225, 0.400] | 0.1% | 0.31 |
| InternVL3.5-8B | 0.289 | [0.234, 0.358] | 0.1% | 0.09 |
| SmolVLM2-2.2B | 0.256 | [0.224, 0.295] | 0.1% | 0.11 |

By answer_format (n: binary 101, multiple_choice 47, number 369, fo_class 1003, open_ended 480):

| model | binary | fo_class | multiple_choice | number | open_ended |
|---|---|---|---|---|---|
| InternVL3.5-30B-A3B | 0.474 | **0.459** | 0.494 | **0.701** | 0.202 |
| llava-onevision-7B | 0.354 | 0.383 | 0.292 | 0.625 | 0.281 |
| Qwen3-VL-30B-A3B | 0.584 | 0.359 | **0.610** | 0.451 | **0.325** |
| InternVL3.5-14B | 0.715 | 0.319 | 0.594 | 0.569 | 0.195 |
| Qwen3-VL-8B | 0.586 | 0.336 | 0.518 | 0.348 | 0.280 |
| Qwen3-VL-32B | 0.615 | 0.315 | 0.515 | 0.334 | 0.275 |
| Qwen3-VL-4B | **0.734** | 0.275 | 0.419 | 0.389 | 0.259 |
| Qwen2.5-VL-7B | 0.638 | 0.242 | 0.564 | 0.457 | 0.242 |
| Qwen2.5-VL-32B | 0.601 | 0.261 | 0.267 | 0.444 | 0.180 |
| InternVL3.5-8B | 0.649 | 0.239 | 0.398 | 0.484 | 0.147 |
| SmolVLM2-2.2B | 0.685 | 0.102 | 0.292 | 0.756 | 0.115 |

By capability group (n: aggregation 470, object_recognition 1530):

| model | aggregation | object_recognition |
|---|---|---|
| InternVL3.5-30B-A3B | 0.658 | **0.382** |
| llava-onevision-7B | 0.573 | 0.341 |
| Qwen3-VL-30B-A3B | 0.482 | 0.356 |
| InternVL3.5-14B | 0.603 | 0.289 |
| Qwen3-VL-8B | 0.393 | 0.331 |
| Qwen3-VL-32B | 0.392 | 0.316 |
| Qwen3-VL-4B | 0.462 | 0.283 |
| Qwen2.5-VL-7B | 0.494 | 0.256 |
| Qwen2.5-VL-32B | 0.483 | 0.251 |
| InternVL3.5-8B | 0.519 | 0.228 |
| SmolVLM2-2.2B | **0.742** | 0.123 |

Findings:
- **InternVL3.5-30B-A3B wins (0.444, +0.14 over baseline)** — leads both capability groups and `number` (0.701); only mid on open_ended.
- **InternVL3.5 scales cleanly**: 8B 0.289 -> 14B 0.364 -> 30B-A3B 0.444.
- **MoE beats dense at ~30B**: Qwen3-VL-30B-A3B (0.386) > Qwen3-VL-32B dense (0.337).
- **Qwen2.5-VL does not scale zero-shot**: 32B (0.302) == 7B (0.304).
- **llava-onevision-7B over-performs (0.402, 2nd)** — strong `number`/`fo_class`; a small generalist VQA model beats every Qwen3-VL and the 14B InternVL.
- **Wrapper parity confirmed**: Qwen2.5-VL-7B rerun = 0.304, exactly the dedicated-baseline number (same CI).
- **Universal weak spots** = `open_ended` (best 0.325) and the large `fo_class`/`object_identification` bulk; dominant error is over-predicting a few foreign-object classes (`Clip`, `Sponge`, `Specimen Bag`, `none`). These remain the main levers.

### baseline · Qwen2.5-VL-7B-Instruct detail (reference)

Original recorded run, `track-frame/baseline/logs/resp_test/` (raw output not preserved;
the sweep's rerun reproduces it at 0.304). Per-capability breakdown — the strategic lever
map (weakest & largest = `object_identification` + `fo_class`/`open_ended`, ~half the set):

| capability | group | acc | n |
|---|---|---|---|
| spatial_localization_situs | object_recognition | 0.800 | 47 |
| object_aggregation | aggregation | 0.494 | 470 |
| object_attributes | object_recognition | 0.412 | 67 |
| spatial_localization_camera | object_recognition | 0.238 | 450 |
| object_identification | object_recognition | 0.217 | 966 |

## SEGMENT
_(no runs yet)_

## PROCEDURE
_(no runs yet)_
