# Bucket-macro evaluation — implementation plan (run on civo)

Status: plan written 2026-08-17 on SAN; to be implemented on civo by the user.
Context: [`agentic-redesign-plan.md`](agentic-redesign-plan.md) §5 step 1.
Goal: add the hidden-leaderboard-shaped **bucket macro** metric alongside the
existing micro / per-video macro, re-score every existing run under it, and re-rank
epoch choices. This becomes the ONLY selection metric going forward.

## 0. Key fact: no GPU needed

Verified 2026-08-17 (read-only): every needed run already has per-question scores at
`<run>/<dataset>/eval/results.csv` with columns
`qID, video, ood, clinical, primary, answer_format, latency, timed_out, correctness`.
Bucket eval = pure pandas regroup. The user's "re-evaluate with many GPUs" case does
NOT apply unless a new run lacks `results.csv` (fallback in §6).

Verified inventory (paths relative to `/datasets/engs2732/orena`):

| run | results.csv |
|---|---|
| unified ep4/8/12, all 3 tracks | `track-unified/lora-finetune/logs/Qwen3-VL-4B-Instruct-unified-both-official/full_test/epoch_{4,8,12}/{frame,segment,procedure}/{heico,lapchole}/eval/results.csv` |
| segment specialist ep6 | `track-segment/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_6/{heico,lapchole}/eval/results.csv` |
| procedure specialist ep8 | `track-procedure/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/full_test_epoch_8/{heico,lapchole}/eval/results.csv` |
| frame specialist ep30 (multi-label rescore) | `track-frame/lora-finetune/logs/full-test-comparison/new-epoch30/{heico,lapchole}/eval/results.csv` |
| per-epoch subset evals (all runs) | `.../eval_epoch_<N>/{heico,lapchole}/eval/results.csv` |
| frame v2 27B runs | `track-frame/v2/logs/<label>/...` (auto-discover; qIDs may be ds-prefixed) |

## 1. Metric definition

- **Group** = leaf→group mapping (exact leaf strings verified in results.csv):
  - 1 object_recognition: `object_identification, instance_matching,
    object_attributes, spatial_localization_camera, spatial_localization_situs`
  - 2 temporal_grounding: `temporal_localization, duration_estimation`
  - 3 aggregation: `object_aggregation, event_aggregation`
  - 4 event_understanding: `fo_interaction_recognition, fo_usage_purpose,
    temporal_ordering`
  - 5 complex_reasoning: `functional_reasoning, causal_consequence_reasoning,
    multi_step_reasoning`
  - Unknown leaf ⇒ **hard error** listing offenders (never silently drop).
- **Cell** = group × dataset (HeiCo/LapChole as pseudo-OOD axis — local `ood` is
  all-False; hidden metric uses capability-group × ID/OOD, so this mirrors its
  shape; both datasets are ID to a both-trained model — acknowledged limitation,
  still the closest local mirror).
- **bucket_macro** = unweighted mean over populated cells. Track masks: frame ⇒
  groups {1,3} only (4 cells, matches hidden frame metric); segment/procedure ⇒
  groups {1..5} (10 cells).
- Per cell also report: n, accuracy, Wilson 95% CI (small cells: procedure group 5
  ≈ 40 rows/ds — CIs are the honesty device).
- Continue reporting micro accuracy and per-video macro for continuity; the report
  must show all three side by side.
- `timed_out=True` counts as wrong (matches hidden forfeit≈wrong; check the
  evaluator already folded this into `correctness` — if so, note it).
- Duplicate qIDs exist across datasets (known collision id 2657592): always carry a
  `dataset` column; never merge on bare qID.

## 2. Deliverables (all new code in shared `src/`, per repo layout rules)

1. **`src/evaluation/bucket_metrics.py`** — library + CLI. In: one run dir
   containing `{heico,lapchole}/eval/results.csv` (or explicit csv paths + dataset
   labels). Out: `bucket_report.csv` (cell rows: track, dataset, group, n, acc,
   ci_lo, ci_hi) + `bucket_report.md` (matrix + bucket_macro + micro + per-video
   macro). Core ≈:

   ```python
   LEAF_TO_GROUP = {...}  # §1, assert set(df.primary) <= keys
   df["group"] = df["primary"].map(LEAF_TO_GROUP)
   cells = df.groupby(["dataset", "group"])["correctness"].agg(["mean", "size"])
   bucket_macro = cells["mean"].mean()   # unweighted over populated cells
   ```
2. **`src/evaluation/bucket_sweep.py`** — walk given roots (default: the four
   track `logs/` trees), auto-discover every `*/eval/results.csv`, pair by run,
   emit one master CSV + ranked table (`run, track, scope[full|subset], epoch,
   bucket_macro, micro, per_video_macro, worst_cell, worst_cell_acc`).
3. **Epoch re-ranking**: for each training run, rank `eval_epoch_N` subsets by
   bucket_macro; flag every run whose best epoch CHANGES vs the micro choice.
   (Subsets rank epochs only — never compare subset numbers to full-test numbers;
   known +0.064 subset bias at frame/lapchole.)
4. **Report** `result-summary/bucket-eval/README.md`: submitted checkpoints under
   the new metric; epoch-choice changes; micro-vs-bucket disagreement analysis;
   per-track worst cells (= the improvement targets for the agentic tier-1).
5. Wire-in (phase 2, optional now): add `bucket_macro` column to the in-loop
   per-epoch eval so future runs early-stop on it; align with `track-frame/v2/
   src/score.py` (already computes the 4-bucket frame view — reuse its
   dataset-as-OOD convention, don't fork a second one).

## 3. Acceptance checks (must pass before trusting the sweep)

- Frame ep30 full test reproduces the recorded four-bucket mean **0.6611**
  (HeiCo objR 0.7920 / aggr 0.6101; LapChole 0.7585 / 0.4838 —
  `docs/frame-track-v2-plan.md` §4b).
- Unified ep8 micro per cell matches `result-summary/unified/epoch-4-8-12-full-test.md`
  (e.g. segment 0.6795/0.7803, procedure 0.3705/0.5786).
- Cell n's sum to the results.csv row count per dataset (segment 4000+2254 etc.).
- Frame results contain only groups {1,3}; segment/procedure contain all 5.

## 4. Mechanics

- Env: `~/miniconda3/envs/orena/bin/python` on civo (conda env `orena` per repo
  rule). CPU-only; login node or any cpu allocation; runtime ~minutes.
- Read-only over `logs/`; writes only new files under `src/evaluation/` and
  `result-summary/bucket-eval/`.
- Workflow per machine map: implement on civo clone, commit, push; SAN pulls.

## 5. Expected first findings (hypotheses the report should confirm/refute)

- Unified-vs-specialist verdict may flip per track under bucket macro (procedure
  win was driven by big buckets; segment loss was concentrated in tempG).
- Best epochs likely shift earlier (late epochs overfit big buckets; groups 4/5
  degrade — the cplxR_ood 0/2 mechanism, now visible locally).
- Segment worst cells should be groups 4/5 + LapChole aggregation; procedure worst
  = groups 2/3 HeiCo (time errors, long-horizon counts).

## 6. Fallback — only if a needed run lacks `results.csv`

Re-judge from that run's `predictions.parquet`/`responses.json` with the existing
eval path (`track-frame/baseline/src/evaluate.py`, judge Qwen3.5-4B, GPU): a100
partition, **≤4 GPUs** (user cap), one array task per (run × dataset). Do NOT
re-run model inference — predictions already exist; only judge scoring needs GPU.
Current inventory shows this fallback is unnecessary.
