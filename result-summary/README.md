# Result summaries

This directory is the project-level home for experiment results. New results
are organized by track and experiment instead of extending one all-purpose
Markdown file.

## Index

| Track | Summary index |
| --- | --- |
| Frame | [Frame results](frame/README.md) |
| Segment | [Segment results](segment/README.md) |
| Procedure | [Procedure results](procedure/README.md) |

The existing [`result-summary.md`](../result-summary.md) is retained as a
historical cross-track snapshot so its prior results and uncommitted edits are
not lost. New experiment reports belong here.

## Reporting contract

Every comparison should record:

- exact model/checkpoint and inference intervention;
- test dataset, row count, video count, and date;
- official overall metric and uncertainty, clearly distinguished from
  per-question micro accuracy;
- performance by dataset, capability group and leaf capability, and answer
  format;
- finer question-type distributions when the intervention targets a subset;
- paired outcomes or a paired uncertainty estimate when both methods answer
  the same rows;
- output validity, runtime/errors, and links to raw artifacts.

Smoke tests and incomplete runs must be labeled as such and kept out of
headline comparisons.
