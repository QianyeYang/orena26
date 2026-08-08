# FRAME epoch-30: full-test rescore through the multi-label fo_class adapter

**Corrected official baseline: HeiCo 0.7067, LapChole 0.6417** (was 0.6643 /
0.5551). Same model, same saved raw outputs — scoring fix only.

The 2026-07-24 full test (`track-frame/lora-finetune/logs/full-test-comparison/
new-epoch30/`, model = Qwen3-VL-4B both-official epoch 30) was scored one day
before the multi-label `fo_class` adapter fix landed: the old normaliser
reduced multi-object answers to the first matched label, so raw outputs that
already named the correct label set were scored wrong. This rescore recomputes
`normalized_prediction` for fo_class rows from the saved `raw_model_output`
(`scripts/rescore_fo_class_multilabel.py` — no inference, no judge; fo_class
correctness is deterministic set-equality) and keeps every other row's stored
correctness. Artifacts: `eval/rescore-{results,summary}.csv` next to the
originals on the training cluster.

## HeiCo (4,000 rows; 1,755 fo_class; 179 flipped wrong→right, 9 right→wrong)

| bucket | old | rescored | Δ |
| --- | --- | --- | --- |
| **overall** | 0.6643 | **0.7067** | +0.0425 |
| fo_class | 0.6929 | 0.7897 | +0.0969 |
| object_identification | 0.6895 | 0.7964 | +0.1069 |
| binary / number / MC / open_ended | — | unchanged | 0 |
| other capabilities | — | unchanged | 0 |

## LapChole (2,252 rows; 920 fo_class; 199 flipped wrong→right, 4 right→wrong)

| bucket | old | rescored | Δ |
| --- | --- | --- | --- |
| **overall** | 0.5551 | **0.6417** | +0.0866 |
| fo_class | 0.5446 | 0.7565 | +0.2120 |
| object_identification | 0.5323 | 0.7575 | +0.2252 |
| binary / number / MC / open_ended | — | unchanged | 0 |
| other capabilities | — | unchanged | 0 |

## Reading

- The few right→wrong flips are rows where the old first-match reduction
  accidentally salvaged a wrong raw output; the rescored numbers are the
  honest ones.
- LapChole was hit twice as hard because 41% of its rows are fo_class and its
  multi-object share is higher — most of the apparent HeiCo↔LapChole gap in
  object_identification was a scoring artifact (0.80 vs 0.76 after, not
  0.69 vs 0.53).
- Post-rescore weakest buckets, in order: **number** (LapChole 0.3958, HeiCo
  0.5249) and **object_aggregation** (LapChole 0.4838, HeiCo 0.6101) — the
  counting workstream remains the top FRAME lever; identification is no
  longer the LapChole outlier it appeared to be.
- These are the baselines the unified-model FRAME checkpoint must beat
  (`track-unified/lora-finetune/architecture.md`).
