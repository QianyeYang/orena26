# FRAME 27B submission handoff

Instructions for building the FRAME submission bundle around the Qwen3.6-27B
epoch-24 model, written for whoever adapts the existing 4B bundle. It states
only what **differs** from
`submissions/frame/qwen3-vl-4b-lora-both-official-epoch30-20260725`; everything
not mentioned here stays exactly as it is.

Method background: [`large-model-deployment.md`](large-model-deployment.md).
Numbers: [`../result-summary/frame/int8-mlp-deployment.md`](../result-summary/frame/int8-mlp-deployment.md).

## 1. The one-line summary

The bundle stops carrying **base + LoRA adapter** and starts carrying **one
already-merged, already-quantized model directory**. PEFT disappears from the
runtime, torchao becomes a hard dependency, and GPU memory goes from 9.6 GiB to
~35 GiB — which invalidates the V100 smoke test.

## 2. The model

```
/datasets/engs2732/orena/os-models/Qwen3.6-27B-frame-ep24-int8mlp
```

| | |
| --- | --- |
| size on disk | **35.05 GiB**, a single `model.safetensors` |
| weights on GPU | **35.0 GiB** (measured, `bench-4414.log`) |
| base | Qwen3.6-27B |
| adapter | FRAME LoRA **epoch 24** (`Qwen3.6-27B-frame-v1prompt/checkpoint-23136`), **already merged** |
| quantization | INT8 weight-only on the language-model MLP projections, **192 of 607** Linears; everything else bf16 |
| load | plain `from_pretrained` — `config.json` carries `quantization_config` (`quant_method: torchao`) |
| accuracy | **0.6571** on the 1,203 stratified rows (bf16 unmerged epoch 24: 0.6597, so **−0.0026**) |

A bf16 merged copy also exists at `Qwen3.6-27B-frame-ep24-merged` (50.97 GiB).
It is **not** what ships; it is the intermediate the quantized model was built
from, and is useful only for A/B checks.

Rebuild either from scratch with:

```bash
sbatch track-frame/v2/scripts/export_merged.slurm \
  track-frame/v2/logs/Qwen3.6-27B-frame-v1prompt/checkpoint-23136 ep24
```

## 3. What changes in `inference.py`

**Delete** the PEFT import and the two-path model load:

```python
from peft import PeftModel                      # DELETE

BASE_PATH    = APP_PATH / "resources" / "base_model"    # DELETE
ADAPTER_PATH = APP_PATH / "resources" / "adapter"       # DELETE

base = AutoModelForImageTextToText.from_pretrained(BASE_PATH, ...)   # DELETE
self.model = PeftModel.from_pretrained(base, ADAPTER_PATH, ...)      # DELETE
```

**Replace** with a single load. The processor comes from the same directory —
the exporter copied the full processor set (`chat_template.jinja`,
`preprocessor_config.json`, `video_preprocessor_config.json`,
`processor_config.json`, `tokenizer*`) beside the weights:

```python
MODEL_PATH = APP_PATH / "resources" / "model"

self.processor = AutoProcessor.from_pretrained(
    MODEL_PATH, max_pixels=MAX_PIXELS,
    local_files_only=True, trust_remote_code=False,
)
self.processor.tokenizer.padding_side = "left"
if self.processor.tokenizer.pad_token_id is None:
    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

self.model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,          # see §4 — not overridable any more
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    local_files_only=True,
    trust_remote_code=False,
).eval()
```

Do **not** pass a `quantization_config`. The checkpoint is already quantized;
passing one would try to quantize it twice.

`resources/` accordingly becomes `model/` instead of `base_model/` + `adapter/`.
Regenerate `weights.sha256` over the new tree.

## 4. `FOCUS_MODEL_DTYPE` / the FP16 override is dead

The 4B bundle exposes `FOCUS_MODEL_DTYPE` and the old handoff used
`FOCUS_TEST_MODEL_DTYPE=float16` for the V100 smoke test. **Both must go.**

INT8 weight-only weights are torchao tensor subclasses. A dtype flag does not
re-cast them; at best it is ignored, at worst it silently dequantizes and the
memory saving vanishes. The model is bf16-or-nothing. Remove the env var rather
than leaving a knob that appears to work.

## 5. The V100 host cannot run this

The old procedure smoke-tested the image on the 32 GB V100 Docker host. That is
no longer possible — **35 GiB of weights does not fit in 32 GB at any dtype**,
and §4 removes the FP16 escape hatch. The 4B fitted because it peaked at
9.62 GiB.

Options, in order of preference:

1. Smoke-test with Apptainer **on civo**, where an h200 (141 GB) or the a100 box
   is available, and prove the 48 GB target fits with
   `torch.cuda.set_per_process_memory_fraction(0.313)` (= 43.8 GiB of 139.8 GiB).
   That makes a large GPU's allocator fail exactly where an L40S would.
2. If the Docker host must produce the evidence, it needs a ≥48 GB card. A CPU-
   only run proves packaging and I/O but **not** the GPU path — label it as such
   and do not record it as a GPU test.

Do not mark the bundle GPU-tested on a V100 result.

## 5b. POST-MORTEM: the first submission failed on mmap, 2026-08-15

The bundle uploaded on 2026-08-13 failed on Grand Challenge with *"The algorithm
failed on one or more cases"*. The container died before answering a single
question:

```
RuntimeError: unable to mmap 37613882728 bytes from
  </opt/app/resources/model/model.safetensors>: Cannot allocate memory (12)
```

**Cause.** `export_merged.py` called `save_pretrained` without `max_shard_size`.
transformers' default is ~50 GB, so a 35.03 GiB quantized model was written as
**one** `model.safetensors` — while the 51 GiB *merged* model, being over the
default, was correctly split into two. transformers mmaps a checkpoint file
whole (`quantizer_torchao.set_metadata` → `safe_open`), so loading needed 35 GiB
of address space in a single allocation. A cluster node has the host RAM for
that; a memory-capped submission container does not.

**Why the smoke test missed it.** Apptainer job 4446 ran on an A100 node with no
meaningful memory cap and passed. Nothing about the model, the GPU, the
quantization or the latency was ever wrong — the bundle simply could not be
*opened* under a RAM limit. A GPU-memory-clean, accuracy-clean artifact can
still be unloadable.

**Fix.** `export_merged.py` now passes `max_shard_size="4GB"` on every
`save_pretrained`. Re-exporting via `scripts/requantize.slurm` (quantize-only,
reusing the merged model already on disk) gives **10 shards, largest 3.95 GiB**,
same 35.03 GiB total, same 192/607 quantized Linears. Shards are opened one at a
time, so peak mapped bytes drops ~10x.

**Second defect found and fixed at the same time.** `inference.py` had no
warmup, so the first `generate()` paid CUDA kernel autotune on a *scored*
question: 5.55 s against 0.90 s steady, versus a stated FRAME budget of **5 s
per question**. It now runs one discarded warmup generate immediately after load.
This was not what broke the submission, but it would have broken the next one.

**Still unproven.** Sharding is verified to load under Apptainer, not under
Grand Challenge's specific memory cap. If GC's limit is tighter than one ~4 GiB
mapping plus overhead, shrink `MAX_SHARD_SIZE` further. Use a GC *try-out* run to
confirm before spending one of the 10 submission attempts.

## 5c. The reshard is confirmed under an actual memory cap, 2026-08-15

The rebuilt image was produced on the `pt8` Docker host and the mmap failure was
reproduced-in-reverse there: **a memory-capped container is exactly what
Apptainer could not give us**, and Docker's `--memory` supplies it.

| container limit | what ran | result |
| --- | --- | --- |
| `--memory=6g` | `safe_open` every shard, take all 1,568 tensor handles | 10/10 mapped, peak RSS **0.50 GiB**, exit 0 |
| `--memory=4g` | additionally materialize every tensor | **35.03 GiB** read, largest single tensor 2.368 GiB, peak RSS **5.23 GiB**, no OOM kill, exit 0 |

The monolith needed one 37.6 GB mapping and died at the syscall. The sharded
build completes the same path under a **4 GiB** cgroup limit. Peak RSS may exceed
the cap without an OOM kill because the excess is clean file-backed page cache,
which the kernel reclaims on demand — a single 37.6 GB `mmap` never gets that
far. This is the strongest available evidence short of GC itself.

Also verified in-image: all **22** checksums re-checked inside the container
(byte-identical to the civo bundle), 10 shards and no `model.safetensors`,
`quant_method: torchao`, the three env defaults intact, and no dataset, cache,
credential, or source-cluster path in any layer.

**A Docker host with 32 GB Volta GPUs cannot GPU-test any image on this base.**
`torch.cuda.get_arch_list()` for the pinned `pytorch/pytorch:2.11.0-cuda12.8`
base is `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`; a GV100/V100 is
`sm_70`. A 64x64 matmul fails with `CUDA error: no kernel image is available for
execution on the device` in **both** bf16 and fp16. So the "test-only FP16 across
both 32 GB GPUs" escape hatch is dead twice over — once for the torchao reason in
§4, and once because no kernel exists at all. Do not plan a GPU smoke on that
host for any future large-model submission; use civo.

## 6. The setup budget is the real risk

The submission allows **120 s of setup**, and this is the likeliest place the
whole thing fails.

| | 4B (shipped) | 27B (this) |
| --- | ---: | ---: |
| model bytes | 8.3 GB | **35.05 GiB** |
| model-ready time | 99.60 s | **355.81 s cold, over civo NFS** |
| warm load (cluster) | — | 15.0 s |

The 4B used 99.6 s of a 120 s budget for a *quarter* of the bytes. The 15.0 s
figure is off a warm NFS page cache on an h200 and is not evidence about the
submission host.

**The cold load has now been measured and it is 355.81 s — roughly 3x the stated
budget.** Decomposed against civo's storage:

| component | seconds |
| --- | ---: |
| vendor extract + imports | ~19 |
| irreducible bytes (35 GiB at 355 MB/s sequential) | ~106 |
| mmap demand-fault overhead over NFS | ~215 |
| deserialize + host-to-device | ~15 |

The effective read rate through mmap was 110 MB/s against 355 MB/s for a plain
sequential read of the same files — a **3.2x penalty that is a property of
demand-faulting over NFS, not of the model**. Grand Challenge unpacks the image
to the node's local disk, so the 215 s term should largely disappear there and
the floor is the ~106 s of bytes plus overhead. That is still uncomfortably close
to 120 s and it is *inferred*, not measured on GC.

Do not pre-warm the page cache to hide this: on a cgroup-limited container the
page cache counts toward the memory limit, which is how we got the original
`ENOMEM`. If the try-out run shows a setup timeout, the honest levers are a
smaller model, a faster storage path, or asking the organisers what the setup
budget actually enforces.

## 7. Batch size

`MICRO_BATCH_SIZE` defaults to 8 (`FOCUS_BATCH_SIZE`), tuned for a 9.6 GiB 4B on
a 48 GB card. With 35 GiB of weights there is ~8 GiB of headroom, not ~38 GiB.
Start at **1–2** and raise only against a measured `max_memory_allocated`.
Activations were 0.43 GiB per question at `max_pixels=602112`.

## 8. Dependencies

- **add `torchao==0.18.0`** — without it `from_pretrained` cannot read
  `quant_method: torchao` and the load fails outright.
- `peft` is no longer needed at runtime (the adapter is merged). Keep it only if
  `validate_bundle.py` asserts on it.
- Unchanged: `torch 2.13.0+cu130`, `transformers 5.14.1`.

## 9. Do NOT touch these

The model was fine-tuned on **v1 prompts** and evaluated with
`--prompt-strategy direct`. That strategy resolves to the *byte-identical*
`SYSTEM_PROMPT` the 4B bundle already hardcodes, so the prompt path needs no
change — and must not be "improved". The v2 prompts in `src/prompts.py` belong
to a different, untrained-for evaluation path and would degrade this model.

Also unchanged: `MAX_PIXELS = 602_112`, `MAX_NEW_TOKENS = 64`,
`MAX_TEXT_LENGTH = 300`, greedy decoding, left padding, and the answer
normalisation in `src/adapter.py`.

## 10. Acceptance checks before shipping

0. **The model directory is sharded** — many `model-000NN-of-000NN.safetensors`
   plus `model.safetensors.index.json`, no single file over ~4 GiB. One
   monolithic `model.safetensors` is what killed the 2026-08-13 submission (§5b).
   Check with `ls -la resources/model/` before anything else; it is the cheapest
   check here and the one that actually failed.
1. `count_quantized_linears` reports **192/607**. An exclusion list that matches
   nothing once ran a whole 1,203-row benchmark unnoticed.
2. Weights on GPU ≈ **35 GiB**; peak under 48 GB with `--mem-fraction 0.313`.
3. **Cold** model-ready time under 120 s (§6). **Currently failing on civo:
   355.81 s.** Most of that is an NFS demand-fault penalty that should not exist
   on GC's local disk, but this is the one acceptance item still unmet on
   measured evidence, so treat the try-out run as testing *this* as much as the
   mmap fix.
4. Smoke fixture returns 3/3 structurally valid answers, no traceback.
5. Accuracy on the 1,203 stratified rows reproduces **0.6571** (already verified
   once, `bench-4414.log` / `logs/27b-ep24-int8mlp-merged`).
6. **The model loads under a memory cap** — run the image with `--memory=4g` and
   materialize every tensor (§5c). Apptainer cannot test this; Docker can, and it
   is the check that would have caught the 2026-08-13 failure in minutes.
