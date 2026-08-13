# FRAME 27B: epoch selection and the deployable artefact

Which epoch of the 30-epoch Qwen3.6-27B run to ship, and what merging plus
quantizing it costs. Method: [`../../docs/large-model-deployment.md`](../../docs/large-model-deployment.md).
Handoff: [`../../docs/frame-27b-submission-handoff.md`](../../docs/frame-27b-submission-handoff.md).

All rows below are the **same 1,203 stratified test rows**
(`--limit 600 --stratified`), v1 prompts, `--prompt-strategy direct`, greedy
decoding, bf16 unless stated. Run `Qwen3.6-27B-frame-v1prompt` (job 3905,
30 epochs, 42h44m on one H200, COMPLETED 2026-08-13).

## 1. The epoch sweep is flat

| epoch | checkpoint | score | counting exact |
| ---: | --- | ---: | ---: |
| 8 | 7712 | 0.6548 | 0.4876 |
| 16 | 15424 | 0.6447 | 0.4826 |
| 24 | 23136 | **0.6597** | **0.5174** |
| 30 | 28920 | 0.6519 | 0.4925 |

**These four are not separable.** The bucket sizes are 408/250/361/183, which
puts the standard error of a four-bucket mean at roughly **±0.014** — wider than
the entire 0.015 spread between best and worst. The sequence also has no shape:
down, up, down. Epoch 24 is the nominal best and was selected on that basis, but
"epoch 24 beats epoch 8 by 0.005" is not a claim this data supports.

Two things follow:

- **Training past epoch 8 bought nothing.** Epoch 8 arrives in ~11 h of the
  42.7 h run. Worth knowing before scheduling another long one.
- **Train loss is useless for selection here** — it reaches 1e-6 by epoch 30,
  i.e. the training set is memorised. Only the benchmark discriminates, and on
  1,203 rows it barely does.

Epoch 24's one coherent feature: its whole margin sits in aggregation
(heico +0.017, lapchole +0.027) and specifically in counting (+0.030, ~12
questions of 402). Suggestive, still inside noise, and it cuts against the
[counting-head](../../track-frame/counting-head/architecture.md) finding that
counting is a perception limit.

For reference, the leaderboard leader scores 0.6235, so every epoch here clears
it on this metric.

## 2. Merging and quantizing costs 0.0026

The shipped artefact folds the epoch-24 LoRA into the base and *then* quantizes,
which is **not** the path the earlier INT8 result measured: `infer.LoRAVLM`
quantizes the base and hangs an unmerged bf16 adapter on top. Merging rounds the
LoRA deltas along with the weights, so this had to be re-measured rather than
assumed.

| | score | or_heico | or_lap | agg_heico | agg_lap |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch 24, bf16, unmerged | 0.6597 | 0.7525 | 0.7440 | 0.6233 | 0.5191 |
| epoch 24, merged + int8-mlp | **0.6571** | 0.7500 | 0.7440 | 0.6260 | 0.5082 |
| Δ | −0.0026 | −0.0025 | 0.0000 | +0.0027 | −0.0109 |

Under the 0.005 acceptance bar. The single visible move, aggregation_lapchole
−0.0109, is two questions on n=183.

## 3. What the artefact costs

| | bf16 merged | merged + int8-mlp |
| --- | ---: | ---: |
| disk | 50.97 GiB | **35.05 GiB** |
| GPU weights | 52.0 GiB | **35.0 GiB** |
| quantized Linears | — | 192 / 607 |
| warm load (H200, NFS) | 22.0 s | 15.0 s |

35.0 GiB rather than the 36.0 GiB of the earlier quantize-then-attach run,
because the 1.0 GB adapter is now folded in instead of resident as extra bf16
parameters.

Build both with:

```bash
sbatch track-frame/v2/scripts/export_merged.slurm \
  track-frame/v2/logs/Qwen3.6-27B-frame-v1prompt/checkpoint-23136 ep24
```

## 4. Open

**Cold model-ready time is unmeasured**, and it is the only number that decides
whether this is submittable. The 120 s setup budget was 99.60 s for the 8.3 GB
4B bundle; this is 35 GiB. The 15.0 s above is a warm NFS page cache on an H200
and is not evidence about the submission host.
