# FRAME — bounding-box prompting for counting

## Question

Can the current fine-tuned FRAME model answer aggregation questions more
accurately if it must localize the requested objects before the application
derives the final count?

This is an inference-only intervention. It uses the evaluated epoch-30
`Qwen3-VL-4B-Instruct-both-official/final` LoRA adapter unchanged.

## Intervention

All official FRAME `number` rows are counting questions. The direct baseline
asks for digits only. This experiment replaces that instruction with a
one-shot JSON prompt:

```json
{"objects":[{"label":"Clip","bbox_2d":[88,140,166,230]}]}
```

Coordinates are normalized to 0–1000. The parser requires a label and a valid
`[x_min, y_min, x_max, y_max]` box for each object, then derives the answer:

- instance count: number of valid boxes;
- distinct-class count: number of unique canonical labels on valid boxes;
- named-object count: number of valid boxes carrying the requested label.

A valid empty `objects` list becomes zero. A malformed response is recorded
and becomes zero; the code deliberately does not extract a bare number as a
fallback because that would contaminate the localize-then-count experiment.

The runner performs one unscored generation on the first selected image before
timing responses. This absorbs lazy-kernel/compiler startup so the response
latencies measure the prompt intervention rather than a one-time hardware
warm-up.

## Evaluation design

Only the 2,094 affected test rows are re-run (HeiCo 1,326; LapChole 768).
Their responses replace the corresponding rows in the completed direct-prompt
epoch-30 run. All other predictions remain byte-for-byte unchanged. The merged
6,252 responses are then scored with the same FOCUS evaluator and local
Qwen3.5-4B judge used for the direct baseline.

This paired design isolates the prompt and deterministic box reduction while
avoiding redundant inference on the 4,158 unaffected questions.

## Layout

```text
src/merge_outputs.py       Merge counting replacements into the full baseline
scripts/run_experiment.slurm
logs/<label>/<dataset>/
  counting-only/           Raw JSON, parsed boxes, and derived counts
  predictions.parquet      Full merged predictions
  responses.json           Full merged challenge responses
  eval/                    Full FOCUS evaluation
```

Shared classification, parsing, and reduction live in `src/counting.py`.
Prompt construction lives in `src/prompts.py`; the normal direct prompt remains
the default for training and every existing runner.

## Run

First inspect GPU availability as required by the repository instructions.
Then run a small smoke test and inspect the raw structured outputs before the
full paired evaluation:

```bash
~/check_gpus.sh
sbatch track-frame/counting-bbox-prompt/scripts/run_experiment.slurm \
  smoke-bbox-json \
  track-frame/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/final \
  8

sbatch track-frame/counting-bbox-prompt/scripts/run_experiment.slurm \
  bbox-json-epoch30 \
  track-frame/lora-finetune/logs/Qwen3-VL-4B-Instruct-both-official/final
```

The limited run performs inference and a partial merge but intentionally skips
the full evaluator. A no-limit run evaluates both complete test sets and writes
`COMPLETE` only after both pass.

## Result

The completed 2026-07-27 run does not support replacing direct counting
globally. HeiCo counting improved from 0.5356 to 0.5521, while LapChole
declined from 0.4116 to 0.3655. See the
[full dataset, capability, count-mode, and count-distribution report](../../result-summary/frame/counting-bbox-prompt.md).
