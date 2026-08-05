# Segment results

## Submission-checkpoint reports

| Report | Checkpoint | Evaluation scope | Main result | Status |
| --- | --- | --- | --- | --- |
| [Qwen3-VL-4B LoRA epoch 6 full test](qwen3-vl-4b-lora-both-official-epoch6-full-test.md) | Epoch 6, step 5,160 | All 6,254 official local test rows; 38 videos | HeiCo 0.7023, LapChole 0.7745 official video-macro | **Authoritative local full-test report** |
| [Qwen3-VL-4B LoRA epoch 6 selection subset](qwen3-vl-4b-lora-both-official-epoch6-seeded-test-subset.md) | Epoch 6, step 5,160 | Fixed seed-42 subset: 500 / 6,254 rows | Checkpoint-selection evidence only | Historical / superseded for performance claims |

The full-test report corresponds to the checkpoint packaged in
`submissions/segment/qwen3-vl-4b-lora-both-official-epoch6-20260801`. It
includes dataset, capability-group and leaf, answer-format, cardinality,
numeric, temporal, runtime, video, generation-source, checkpoint-selection,
and subset-versus-full distributions with exact provenance.

The 500-row report is retained to explain checkpoint selection and the earlier
single-value prompt/normalizer audit. Do not use its scores as the primary
Segment result now that full-test inference is available.

Historical baseline and provisional LoRA results remain in the
[cross-track snapshot](../../result-summary.md#segment). Do not compare that
snapshot's differently scoped values directly with the full-test report.
