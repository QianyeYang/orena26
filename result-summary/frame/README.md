# Frame results

## Experiments

| Experiment | Comparison | Report |
| --- | --- | --- |
| Epoch-30 both-dataset fine-tune | Epoch 24 versus epoch 30 on current official tests | [Full-test comparison](../../docs/frame-model-full-test-comparison.md) |
| Epoch-30 multi-label rescore | Same run rescored through the fixed fo_class adapter — corrected baseline 0.7067/0.6417 | [Multi-label rescore](epoch30-multilabel-rescore.md) |
| Bounding-box counting prompt | Direct count versus localize-then-count with the same epoch-30 model | [Counting prompt comparison](counting-bbox-prompt.md) |
| 27B INT8 deployment | bf16 versus INT8-on-MLPs — fits a 48 GB L40S at 0.6548 → 0.6551 | [INT8 MLP deployment](int8-mlp-deployment.md) |
| 27B counting head | Ordinal count head versus token generation — negative, +0.0053 | [Counting head](../../track-frame/counting-head/architecture.md) |
| 27B epoch sweep + shipped artefact | Epochs 8/16/24/30 are inseparable; epoch 24 merged+INT8 = 0.6571 at 35 GiB | [27B epoch sweep](27b-epoch-sweep.md) |

Reports in this directory must show HeiCo and LapChole separately before any
cross-dataset aggregate, then break results down by capability and relevant
question subtype.
