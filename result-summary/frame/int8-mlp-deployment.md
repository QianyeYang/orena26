# FRAME 27B deployment: INT8 on the language-model MLPs

## Outcome

Quantizing **only** the language model's MLP projections to INT8 weight-only
puts Qwen3.6-27B on a 48 GB L40S at **no measurable accuracy cost**: 0.6548
bf16 → **0.6551** int8-mlp on the same 1,203 stratified rows, with peak memory
falling 51.39 → **37.05 GiB**.

Method and reusable guidance:
[`docs/large-model-deployment.md`](../../docs/large-model-deployment.md). This
file is the numerical record.

Measured 2026-08-11 on one H200 NVL, epoch-8 LoRA adapter
(`Qwen3.6-27B-frame-v1prompt/checkpoint-7712`), v1 prompts, greedy decoding,
zero inference errors in both arms.

## Why only the MLPs

The 27B is 50.96 GiB loaded and an L40S offers ~43.5 GiB for weights after the
CUDA context and measured activations — a **16% deficit**, not a 50% one. Three
prior FP8 attempts all threw away half the model and failed:

| attempt | result |
| --- | --- |
| runtime FP8, all Linears | 0.3217 → 0.2063 zero-shot |
| runtime FP8, `modules_to_not_convert=["visual","merger","lm_head"]` | flag matched nothing (memory byte-identical); 0.1500, and non-deterministic across runs |
| `Qwen/Qwen3.6-27B-FP8` via transformers | loads, emits token salad, 18× slower |

Qwen's own FP8 config exempts **882 modules**, and the list is diagnostic: every
`linear_attn.A_log`, `.dt_bias`, `.conv1d`, `.in_proj_*` across all 48
linear-attention layers, the whole vision tower, `lm_head`, `embed_tokens` and
every norm. `A_log` is exponentiated and `dt_bias` sets an integration timestep
— rounding those changes the dynamics, not the precision.

The MLPs are plain matmuls and are 31.88 GiB of the 50.96. Converting them alone
is sufficient and touches nothing numerically delicate.

## Configuration

`--quant int8-mlp` (`track-frame/lora-finetune/src/infer.py`): torchao
`Int8WeightOnlyConfig` through `transformers.TorchAoConfig`, with these
exclusions — matched by `should_convert_module` via `re.match`, i.e. as anchored
prefixes, which was verified in the source before use rather than assumed:

```
model\.language_model\.layers\.\d+\.linear_attn
model\.language_model\.layers\.\d+\.self_attn
model\.visual
lm_head
model\.embed_tokens
```

**192 of 607 Linear modules convert** — exactly the `mlp.{gate,up,down}_proj`
set. `count_quantized_linears` logs this on every quantized load, because the
`fp8-keepvision` failure above was invisible for a whole benchmark run.

The MTP head (0.79 GiB on disk) is never instantiated by
`AutoModelForImageTextToText`, which is why the bf16 load is 50.96 GiB rather
than 51.75. No action needed.

## Results by dataset

### Bucket accuracy

| bucket | n | bf16 | int8-mlp | Δ |
| --- | ---: | ---: | ---: | ---: |
| object_recognition_heico | 408 | 0.7647 | 0.7647 | 0.0000 |
| object_recognition_lapchole | 250 | 0.7560 | 0.7600 | +0.0040 |
| aggregation_heico | 361 | 0.6066 | 0.6094 | +0.0028 |
| aggregation_lapchole | 183 | 0.4918 | 0.4863 | −0.0055 |
| **mean of 4 buckets** | 1,203 | **0.6548** | **0.6551** | **+0.0003** |

### Answer format

| format | n | bf16 | int8-mlp | Δ |
| --- | ---: | ---: | ---: | ---: |
| binary | 139 | 0.8058 | 0.8058 | 0.0000 |
| fo_class | 514 | 0.7471 | 0.7490 | +0.0019 |
| multiple_choice | 38 | 0.9474 | 0.9474 | 0.0000 |
| number | 402 | 0.4876 | 0.4876 | 0.0000 |
| open_ended | 110 | 0.7545 | 0.7545 | 0.0000 |

Three of five formats are identical to four decimal places. The deltas that do
move are within sampling noise on these row counts and point in both directions.

## Memory and runtime

| | bf16 | int8-mlp |
| --- | ---: | ---: |
| weights | 50.96 GiB | **36.04 GiB** |
| peak allocated | 51.39 GiB | **37.05 GiB** |
| load | 7.4 s | 11.4 s |
| s/question (H200) | 0.207 | 0.423 |

**Fit proof**: rerun with `--mem-fraction 0.313`, capping the CUDA allocator at
43.8 GiB so an H200 fails exactly where a 48 GB L40S would. Loaded and ran to
completion. This proves the fit rather than estimating it.

## Open risks

- **Latency is measured on the wrong hardware.** There is no L40S in the
  cluster. INT8 is 2× *slower* than bf16 on an H200 — a 4.8 TB/s card is
  compute-bound, so dequantization overhead dominates. An L40S at ~864 GB/s
  should tilt back toward INT8, but that is an argument, not a measurement.
  0.423 s/question against a 5 s budget leaves margin for a 4× slowdown.
- **Setup budget.** The 120 s window must cover loading 36 GiB. The 11.4 s here
  is off a warm page cache; cold-storage load on the submission host is
  unmeasured and is the likeliest place this breaks.
- Not yet tested against the final 30-epoch adapter (training in progress); the
  quantization is adapter-independent, but the check should be repeated.

## Reproduce

```
sbatch track-frame/v2/scripts/benchmark.slurm os-models/Qwen3.6-27B 27b-int8mlp-ep8 \
  --adapter track-frame/v2/logs/Qwen3.6-27B-frame-v1prompt/checkpoint-7712 \
  --prompt-strategy direct --quant int8-mlp --limit 600 --stratified

# fit proof
sbatch track-frame/v2/scripts/benchmark.slurm os-models/Qwen3.6-27B 27b-int8mlp-fitproof \
  --adapter <same> --prompt-strategy direct --quant int8-mlp --limit 40 --mem-fraction 0.313
```

Runs: `track-frame/v2/logs/{27b-lora-ep8,27b-int8mlp-ep8,27b-int8mlp-fitproof}/`
