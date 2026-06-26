# Issues TBD (to revisit later)

Deferred decisions from the FRAME baseline plan review (2026-06-19).
Each: context · current decision · what to revisit.

## 4. Multi-label fo_class answers are unscorable by the stock scorer
- Some `fo_class` answers list >1 object, e.g. `"Clip, Sponge"` (27 train / 5 test rows).
- The official `FOClass` format accepts only ONE class name or `"none"`. On a comma
  answer it raises; the `Evaluator` reads the *reference* through the format too, so
  even the correct reference fails parsing and the question is auto-marked wrong.
- Impact: these questions are unwinnable with the stock scorer; small (~0.3% of test)
  but caps the fo_class ceiling.
- **Current decision:** leave as-is (challenge server uses the stock format anyway).
- **Revisit:** confirm how the official leaderboard scores multi-label FO; add a
  multi-label FOClass variant only if it matches server behaviour.

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

## 9. Foreign-object class list (7 vs 9 vs more)
- `focus` defines 9 canonical FO classes (adds Gallstone, Mesh); train data shows only
  7; the package notes the test phase may introduce more (provided via metadata).
- **Current decision:** use the full 9 classes (`FOType.names()`) in prompts now.
- **Revisit:** adjust the class list when test-phase metadata arrives; check if the
  hidden test adds new types.
