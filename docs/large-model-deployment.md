# Deploying a large VLM inside the challenge budget

How to make a model that does not fit the submission GPU fit it, without losing
accuracy. Written from the Qwen3.6-27B / FRAME work of 2026-08-11, but the
method is the point — the specifics are one worked example.

Numerical results live in
[`result-summary/frame/int8-mlp-deployment.md`](../result-summary/frame/int8-mlp-deployment.md).

## 1. The budget, honestly

| | |
| --- | --- |
| submission GPU | 1x L40S, **48 GB** |
| time | 120 s setup + 20 questions x 5 s per batch |

**48 GB is not 48 GiB.** `nvidia-smi` reports ~45.0 GiB (46,068 MiB) on an L40S,
and the CUDA context plus allocator fragmentation takes ~1 GiB more. Budget:

```
45.0 GiB reported
-1.0     CUDA context + fragmentation
-0.5     measured activations (see below)
-------
~43.5 GiB for weights
```

Measure activations rather than guessing: `benchmark.py` records `weights_gib`
(after load) and `peak_gib` (`max_memory_allocated`). For the 27B at
`max_pixels=602112` the gap is **0.43 GiB**. That number matters because it
kills a whole family of bad ideas — reducing frame size cannot help a model
whose *weights* do not fit, since resolution only moves activations.

## 2. Compute the deficit before choosing a tool

This is the step that was skipped the first time, and skipping it cost three
failed attempts.

```
27B bf16 loaded:  50.96 GiB
budget:          ~43.50 GiB
deficit:           7.46 GiB  = 15%
```

**15%, not 50%.** FP8 and INT4 throw away half or three quarters of the model;
neither is required. Size the intervention to the deficit.

Get the per-component breakdown from the safetensors headers without loading
anything (`tmp/param_breakdown.py` pattern: read the 8-byte header length, parse
the JSON, price each tensor by dtype x shape). For the 27B:

| component | GiB | % |
| --- | ---: | ---: |
| LM MLP | 31.88 | 61.6 |
| LM attention | 13.49 | 26.1 |
| embed_tokens | 2.37 | 4.6 |
| lm_head | 2.37 | 4.6 |
| vision tower | 0.86 | 1.7 |
| MTP head | 0.79 | 1.5 |

The MLPs alone are more than enough. Note also that `AutoModelForImageTextToText`
never instantiates the MTP head, so 0.79 GiB is already saved — which is why the
bf16 load is 50.96 and not 51.75.

## 3. Quantize only what is numerically safe

**Rule: quantize plain matmuls, never control parameters.**

Modern hybrid models carry values that are exponentiated or that set integration
timesteps. Qwen3.6-27B has 48 linear-attention layers holding `A_log`,
`dt_bias`, `conv1d` and `in_proj_*`. Rounding those does not lose precision, it
changes the dynamics.

**Find the authoritative list in the vendor's own quantized release.**
`Qwen/Qwen3.6-27B-FP8`'s `config.json` names **882 exempt modules** — every
linear-attention control tensor, the entire vision tower, `lm_head`,
`embed_tokens` and every norm. That file is free, exact documentation of what
must not be touched.

Resulting configuration (`--quant int8-mlp`, `track-frame/lora-finetune/src/infer.py`):

- torchao `Int8WeightOnlyConfig` via `transformers.TorchAoConfig`
- converts **192 of 607** Linears — exactly `mlp.{gate,up,down}_proj`
- everything else stays bf16
- result: **36.04 GiB weights, 37.05 GiB peak, accuracy unchanged**
  (0.6548 bf16 → 0.6551)

If more headroom is ever needed the dial is granular: quantize `down_proj` only,
or a subset of layers. There is no cliff.

## 4. Verify, do not assume

Three checks, all cheap, each of which caught or would have caught a real bug.

**(a) Count what actually converted.** `infer.count_quantized_linears` logs
`quantized N/M Linear modules` on every quantized load. The `fp8-keepvision`
attempt passed an exclusion list that matched *nothing*, produced memory
byte-identical to plain FP8, and ran an entire 1,203-row benchmark before the
problem was noticed. A count makes that loud.

**(b) Read the matcher before trusting a pattern.**
`transformers.quantizers.quantizers_utils.should_convert_module` matches with
`re.match(key, full_name)` — anchored prefix or regex, plus a suffix fallback.
That is why five compact patterns cover all 415 non-MLP Linears. A different
quantizer may match differently; check the source, do not infer from the
argument name.

**(c) Prove the fit, do not estimate it.**

```bash
--mem-fraction 0.313      # 43.8 / 139.8 GiB on an H200
```

`torch.cuda.set_per_process_memory_fraction` makes a large GPU's allocator fail
exactly where the small one would. If the run completes, it fits.

## 5. What transfers across hardware and what does not

| property | transfers? |
| --- | --- |
| weight memory | **yes** — weights are weights |
| accuracy | **yes** — same rows, same prompts, deterministic decoding |
| **latency** | **no** |

There is no L40S in the cluster, so every timing is an H200 or A100 number.
INT8 weight-only is **2x slower than bf16 on an H200** (0.423 vs 0.207
s/question): at 4.8 TB/s the model is compute-bound, so dequantization overhead
dominates. An L40S at ~864 GB/s should tilt back toward INT8, but that is an
argument, not a measurement. Budget for the pessimistic case: 0.423 s/question
tolerates a 4x slowdown against the 5 s limit.

**The setup window is the tighter constraint.** 120 s must cover loading 36 GiB.
The 11.4 s measured here is off a warm page cache; a cold read on the submission
host is unmeasured and is the likeliest place this breaks. The shipped 4B bundle
recorded 99.60 s model-ready time for a 8.3 GB model — that is the calibration
point to worry about.

## 6. Approaches that do not work

Recorded so they are not retried.

| approach | outcome |
| --- | --- |
| Reduce frame size / `max_pixels` | Cannot help. Activations are 0.43 GiB against 51 GiB of weights. |
| transformers runtime FP8, all Linears | 0.3217 → 0.2063. Quantizes the linear-attention control values. |
| transformers FP8 + `modules_to_not_convert` prefixes | Matched nothing; 0.1500, and **non-deterministic** — 0.2063 vs 0.1500 on identical inputs. |
| `Qwen/Qwen3.6-27B-FP8` via transformers | Loads at 33.62 GiB, emits multilingual token salad, 18x slower. The checkpoint is fine; the `kernels-community/finegrained-fp8` Triton path is not. |
| CPU offload of the overflow | ~5 GiB moved per forward pass, every generated token. Dead against a 5 s budget. |

**Do not use transformers' FP8 path on this model.** If FP8 is ever needed,
serve the official checkpoint through vLLM, which is what Qwen builds and tests
those releases against.

## 7. Checklist for a new deployment

1. Measure `weights_gib` and `peak_gib` in bf16; compute the deficit against
   ~43.5 GiB.
2. Price components from the safetensors headers.
3. Fetch the vendor's quantized `config.json` for the exemption list.
4. Convert the smallest sufficient set of plain matmuls; verify the count.
5. Merge the LoRA before quantizing, so what is measured is what ships.
6. Benchmark on the same stratified rows as the bf16 baseline; require the drop
   to be under 0.005.
7. Prove the fit with `--mem-fraction`.
8. Record s/question and load time, and state plainly that they are from the
   wrong GPU.
