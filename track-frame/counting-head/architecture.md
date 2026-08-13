# Counting head

A template-conditioned count classifier on top of a frozen fine-tuned VLM,
replacing token generation for `number` questions only.

## Why

Counting is a third of the FRAME test set and both the shipped Qwen3-VL-4B and
the new Qwen3.6-27B sit at **0.48 exact** on it, while every other answer format
is 0.74–0.92. Scale does not touch it: 4B 0.4826 vs 27B 0.4801.

Four interventions have already failed to move it — native resolution
(+0.7 pts, McNemar p=0.76), bounding-box prompting (LapChole **−0.046**),
oversampling high counts in training (6+ still ≈ 0), and 7× more parameters.

What has *not* been tried is changing the objective. The VLM is trained with
cross-entropy over answer *tokens*, so predicting 4 when the truth is 5 costs
exactly as much as predicting 11. The model is **exact 48% but within ±1 82%**
of the time — a large share of its errors are near-misses that an ordinal
objective can pull onto the right integer. It also never emits a number above 7
though ground truth reaches 11, which is a token-distribution artefact rather
than a perceptual limit.

This is explicitly **not** a fix for perception. If the model cannot see five
clips in a cluster, a better-calibrated head returns a confident wrong number.
The expected prize is the near-miss band, so roughly +0.03–0.08 on counting,
i.e. +0.01–0.03 overall.

## Design

### Which object the question refers to

FRAME asks exactly **ten** distinct counting questions across all 6,356 `number`
rows of both datasets and both splits:

```
How many different foreign object instances appear in this frame?   -> id 0
How many different foreign object classes  appear in this frame?    -> id 1
How many {Sponges|Clips|External drains|Silicone loops|Specimen bags|
          Needles|Specimens|Gallstones} appear in this frame?       -> id 2+i
```

(Mesh and Absorbable Hemostatic Agent never get counting questions.) So this is
a lookup, not a language problem: `templates.template_id` reuses the already
tested `src.counting.classify_counting_question` parser and returns a dense id,
which the head consumes as a learned embedding.

A class-agnostic head predicting a 10-vector of per-class counts was considered
and **rejected**: most frames carry only one question (11,962 of ~15,200 frames
have exactly one), so each frame would contribute a single scalar constraint on
a 10-dimensional vector. Conditioning on the template instead means every row
supervises exactly the output it names.

### How it is activated

The official `Request` exposes only `qID`, `videoID`, `start_time`, `end_time`,
`procedure_type`, `question` — **there is no `answer_format`**. Routing
therefore has to come from the question text, which is safe here because the ten
templates are fixed strings; `template_id` returns `None` for anything else and
the caller falls through to normal generation.

### How the answer is formed

The head predicts the answer for the template it was asked about, so there is no
aggregation step. What remains is the bounds ground truth is known to obey:

- **every count ≥ 1** — FRAME counts are never zero across all 6,356 rows, both
  datasets, both splits (segment and procedure *do* contain zeros; this is
  FRAME-specific);
- **"different classes" ≤ 4** — holds on all 1,233 such rows.

Both are applied to the head *and* to the baseline it is compared against, so
the reported delta isolates the head rather than the free clamping.

### The head

`pooled state ‖ template embedding → LayerNorm → MLP → logits over counts 1..16`

Classification, not regression: the metric is exact match, so the optimal point
prediction is the conditional **mode**, which `argmax` gives directly — an L1
regressor would target the conditional median. Neighbour mass in the target
distribution (`--smooth`, default 0.2 split onto ±1) is what makes the
classifier ordinal rather than nominal.

The pooled state is the **last layer at the final prompt position** — the state
the model conditions on when emitting its first answer token — taken from
`generate(output_hidden_states=True)` so the head's input and the baseline
number it must beat come from one call on identical weights and inputs.

### Training protocol

The backbone is **frozen**. Features are extracted once and the head is fitted
offline in seconds, which makes hyperparameter sweeps free and answers the
first-order question cleanly: *is the count information already in the
representation?* If it is not, no amount of head capacity helps and the next
step is a perception change, not a bigger MLP.

Validation splits by **video**, not by row: frames from one video are heavily
correlated and the official metric is a per-video macro, so a random row split
would report a number the leaderboard will not reproduce.

## The null test

`src/calibrate.py` makes the same bet in its cheapest form — learn
`P(true | predicted, template)` from saved predictions and emit the mode. Zero
GPU, cross-fitted by video. If a lookup table recovers nothing from the scalar,
the head is unlikely to find much more in the representation, and this is the
cheap way to learn that. Run it before committing GPU time.

## Layout

```
src/templates.py    ten-template id map + the verified answer bounds
src/features.py     frozen-VLM feature extraction + paired baseline generation
src/train_head.py   head, ordinal loss, video-grouped validation, evaluation
src/calibrate.py    zero-GPU null test on existing predictions
scripts/counting_head.slurm   extract then fit, one job
```

## Result — negative (2026-08-11)

Frozen Qwen3-VL-4B + epoch-30 frame adapter, 2,094 test counting rows, head
cross-fitted by video over the test split:

| | exact | within ±1 | MAE |
| --- | ---: | ---: | ---: |
| VLM baseline | 0.4766 | 0.7985 | 0.860 |
| count head | **0.4819** | 0.7889 | 0.896 |

**+0.0053 on counting, so +0.0018 overall.** Nothing. Note the head is *worse*
on within-±1 and MAE — it hits the exact integer marginally more often without
being any closer on average.

Per template, the verdict is in the largest and worst one: **Clip (681 rows)
moves 0.3040 → 0.3025**, i.e. not at all. The gains are confined to templates
that were already near-solved (Sponge 0.928 → 0.940) or too small to read
(External Drain n=45, Specimen n=6).

Together with the zero-GPU recalibration null test — which was flat on HeiCo and
**−0.0156** on LapChole, with the fitted maps pointing in *opposite directions*
on the two datasets for the same template — this closes the objective/calibration
hypothesis. The near-miss band is noise, not recoverable bias, and the
representation carries no count information beyond the number the model already
emits. Counting is a perception limit.

### The protocol bug that nearly hid this

The first run fitted the head on the **train** split and reported 100%
validation accuracy. The cause: the 4B answers **4,262 of 4,262** of its own
training counting questions correctly (versus 47.7% on test), so on those rows
the final-position hidden state encodes a *memorised* answer and the head simply
learns to decode it. That function is trivial on train and worthless on test —
the head reproduced the baseline byte-for-byte on 6 of 8 templates.

**Any probe fitted on a converged backbone's own training data measures recall,
not representation.** `--protocol test-cv` is the default for this reason;
`holdout` is retained only to reproduce the finding.
