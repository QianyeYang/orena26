# Procedure results

## Full-test reports

| Experiment | Scope | Main result | Report |
| --- | --- | --- | --- |
| Qwen3-VL-4B LoRA, both official datasets, epoch 8 | Complete Procedure test: 2,000 HeiCo + 1,127 LapChole rows | HeiCo 0.3105; LapChole 0.5412 | [Capability analysis](qwen3-vl-4b-lora-both-official-epoch8-full-test.md) |

The report includes official video-macro confidence intervals, every primary
capability group and leaf, secondary-capability and answer-format
distributions, temporal/counting/FO-class diagnostics, per-video variation,
runtime, coverage limitations, source hashes, and reproducible CSV tables in
[`qwen3-vl-4b-lora-both-official-epoch8-full-test/`](qwen3-vl-4b-lora-both-official-epoch8-full-test/).

## Historical snapshot

The earlier zero-shot baseline and reduced-sampling diagnostic remain in the
[cross-track snapshot](../../result-summary.md#procedure). That snapshot
predates the completed LoRA run and should not be used as the current
submission-model report.

Future Procedure reports should be added as separate files in this directory
and indexed here.
