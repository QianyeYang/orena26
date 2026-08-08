# Frame results

## Experiments

| Experiment | Comparison | Report |
| --- | --- | --- |
| Epoch-30 both-dataset fine-tune | Epoch 24 versus epoch 30 on current official tests | [Full-test comparison](../../docs/frame-model-full-test-comparison.md) |
| Epoch-30 multi-label rescore | Same run rescored through the fixed fo_class adapter — corrected baseline 0.7067/0.6417 | [Multi-label rescore](epoch30-multilabel-rescore.md) |
| Bounding-box counting prompt | Direct count versus localize-then-count with the same epoch-30 model | [Counting prompt comparison](counting-bbox-prompt.md) |

Reports in this directory must show HeiCo and LapChole separately before any
cross-dataset aggregate, then break results down by capability and relevant
question subtype.
