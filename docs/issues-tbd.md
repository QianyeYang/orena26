# Issues TBD (to revisit later)

Deferred decisions from the FRAME baseline plan review (2026-06-19).
Each: context · current decision · what to revisit.

## 4. Resolved: multi-label fo_class answers
- The current official `FOClass` parser accepts one or more comma-separated
  canonical names and compares them as an order- and duplicate-insensitive set.
- `src.adapter.normalize_fo_class()` preserves every recognized class and masks
  overlapping names such as `Specimen Bag`/`Specimen`.
- Adapter contract tests cover multiple, overlapping, duplicate, and
  runtime-defined class names. Do not reintroduce first-match reduction.

## 7. Score breakdown granularity
- `focus` `Evaluator` reports accuracy by capability AND by answer_format *separately*,
  not the capability x answer_format cross-tab the plan wanted.
- Its accuracy is a macro-average over videos (mean of per-video means) with
  hierarchical bootstrap CIs, NOT a per-question micro-average. Only 10 test videos
  -> wide CIs.
- **Current decision:** go with the built-in Evaluator output as-is for now.
- **Revisit:** if needed, compute the capability x answer_format cross-tab (and/or
  micro-accuracy) ourselves from `results.csv`.

## SEC. Rotate the leaked HF token (action: user, on huggingface.co)
- The leaked HF token (value redacted from this doc; rotate at huggingface.co) was hardcoded plaintext in
  `scripts/download_data.slurm`, in a **world-readable** file on shared Lustre. Never
  committed to git / no remote, so NOT public — but readable by other cluster users +
  passed through an AI session. Cannot be assumed secret anymore.
- Repo side already fixed: token removed; script now sources gitignored `.env`
  (mode 600) and fails-closed if `HF_TOKEN` unset; `.env.example` template added.
- **TODO (only the user can):** 1) revoke that token at
  https://huggingface.co/settings/tokens; 2) create a new one (read scope is enough);
  3) paste into `.env` (`HF_TOKEN=hf_...`). Data already downloaded, so nothing blocked.
- Urgency: hygiene, not emergency. Definitely rotate if it was write-scoped.

## 9. Foreign-object class list
- Use `FOType.names()` for current local evaluation.
- Submission inference must load the runtime definitions because hidden batches
  may introduce additional types.
