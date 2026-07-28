# Frame Annotation Tool

Local bounding-box labeling for 4,000 priority-ranked FOCUS Frame training
cases. The pool contains 2,000 HeiCo and 2,000 LapChole frames and groups every
QA row for the same frame into one case.

## Launch over SSH

On the cluster:

```bash
cd /datasets/engs2732/orena
./frame-annotation-tool.sh
```

On your own computer, keep a second terminal open:

```bash
ssh -N -L 8770:127.0.0.1:8770 <user>@<cluster-host>
```

Open `http://localhost:8770`. Set `FRAME_ANNOTATION_PORT` before launching if
port 8770 is already in use.

## Labeling workflow

1. Work through priority batch 1 before moving to later batches.
2. Draw one tight box per visible foreign-object instance.
3. Review the class guide and all QA shown beside the image.
4. Complete all four checks. A case cannot be marked complete when counts or
   classes disagree with applicable QA.
5. If the image and QA genuinely conflict, or a class is uncertain, save it as
   **Needs review**. Use **Skip** only with a written reason.

Drafts autosave. Case records are atomic and revision checked, and the Progress
page exposes every unstarted, draft, review, complete, and skipped case.

The Progress page exports complete labels as canonical JSONL and COCO JSON
under `data/annotations/frame-annotation-tool/exports/`.
