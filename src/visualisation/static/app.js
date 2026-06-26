"use strict";

const S = {
  allRuns: [],          // every RunInfo from /api/runs
  track: null,
  split: null,
  runsForSplit: [],     // RunInfo for current split, in display order
  selected: new Set(),  // run ids shown as columns
  focus: "",            // run id for "focused model wrong" filter
  questions: [],        // /api/questions rows
  filtered: [],
  view: "matrix",
  caseIndex: 0,
};

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (a) => (a == null ? "—" : (a * 100).toFixed(1) + "%");
const norm = (s) => String(s ?? "").trim().toLowerCase();

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

// --------------------------------------------------------------------------- //
// init
// --------------------------------------------------------------------------- //
async function init() {
  S.allRuns = await getJSON("/api/runs");
  const splits = [...new Set(S.allRuns.map((r) => r.track + "|" + r.split))];
  const sel = $("splitSel");
  sel.innerHTML = splits.map((s) => `<option value="${s}">${s.replace("|", "/")}</option>`).join("");
  const preferred = splits.includes("frame|test") ? "frame|test" : splits[0];
  if (!preferred) {
    $("matrixWrap").innerHTML = `<div class="empty">No runs found on disk.</div>`;
    return;
  }
  sel.value = preferred;
  wireControls();
  await loadSplit(preferred);
}

async function loadSplit(value) {
  [S.track, S.split] = value.split("|");
  S.runsForSplit = S.allRuns.filter((r) => r.track === S.track && r.split === S.split);
  S.selected = new Set(S.runsForSplit.map((r) => r.id));
  S.focus = "";
  const data = await getJSON(`/api/questions?track=${S.track}&split=${S.split}`);
  S.questions = data.questions;
  S.caseIndex = 0;
  buildFilterOptions();
  buildModelChips();
  buildFocusOptions();
  applyAndRender();
}

// --------------------------------------------------------------------------- //
// control wiring
// --------------------------------------------------------------------------- //
function wireControls() {
  $("splitSel").onchange = (e) => loadSplit(e.target.value);
  ["search", "fmtSel", "capSel", "vidSel", "oodSel", "clinSel", "agreeSel", "focusSel"].forEach((id) => {
    const el = $(id);
    el.oninput = el.onchange = () => {
      if (id === "focusSel") S.focus = el.value;
      applyAndRender();
    };
  });
  $("modelAll").onclick = () => setAllModels(true);
  $("modelNone").onclick = () => setAllModels(false);
  $("viewMatrix").onclick = () => setView("matrix");
  $("viewCase").onclick = () => setView("case");
  $("prevCase").onclick = () => stepCase(-1);
  $("nextCase").onclick = () => stepCase(1);
  document.addEventListener("keydown", (e) => {
    if (S.view !== "case") return;
    if (e.key === "ArrowLeft") stepCase(-1);
    if (e.key === "ArrowRight") stepCase(1);
  });
}

function buildFilterOptions() {
  const uniq = (key) => [...new Set(S.questions.map((q) => q[key]).filter((v) => v != null))].sort();
  fillSelect("fmtSel", uniq("answer_format"));
  fillSelect("capSel", uniq("primary_capability"));
  fillSelect("vidSel", uniq("video"));
}
function fillSelect(id, vals) {
  $(id).innerHTML = `<option value="">all</option>` + vals.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
}

function buildModelChips() {
  $("modelChips").innerHTML = S.runsForSplit
    .map((r) => {
      const on = S.selected.has(r.id);
      return (
        `<label class="chip ${on ? "on" : ""}" data-id="${esc(r.id)}">` +
        `<input type="checkbox" ${on ? "checked" : ""} />` +
        `<span class="name">${esc(r.model_name)}</span>` +
        `<span class="acc">${pct(r.accuracy)}</span></label>`
      );
    })
    .join("");
  // Bind the checkbox's change event (fires once per toggle whether the user
  // clicks the box or the surrounding label) — no manual double-toggle.
  $("modelChips").querySelectorAll(".chip").forEach((chip) => {
    const cb = chip.querySelector("input");
    cb.onchange = () => {
      cb.checked ? S.selected.add(chip.dataset.id) : S.selected.delete(chip.dataset.id);
      chip.classList.toggle("on", cb.checked);
      applyAndRender();
    };
  });
}

function setAllModels(on) {
  S.selected = on ? new Set(S.runsForSplit.map((r) => r.id)) : new Set();
  buildModelChips();
  applyAndRender();
}

function buildFocusOptions() {
  $("focusSel").innerHTML =
    `<option value="">—</option>` +
    S.runsForSplit.map((r) => `<option value="${esc(r.id)}">${esc(r.model_name)}</option>`).join("");
}

// --------------------------------------------------------------------------- //
// filtering
// --------------------------------------------------------------------------- //
function selectedRunIds() {
  // Honour an explicitly empty selection (the "none" button) — no fall back to all.
  return S.runsForSplit.map((r) => r.id).filter((id) => S.selected.has(id));
}

function applyFilters() {
  const term = norm($("search").value);
  const fmt = $("fmtSel").value, cap = $("capSel").value, vid = $("vidSel").value;
  const ood = $("oodSel").value, clin = $("clinSel").value, agree = $("agreeSel").value;
  const ids = selectedRunIds();

  return S.questions.filter((q) => {
    if (term && !norm(q.question).includes(term)) return false;
    if (fmt && q.answer_format !== fmt) return false;
    if (cap && q.primary_capability !== cap) return false;
    if (vid && q.video !== vid) return false;
    if (ood && String(q.ood ? 1 : 0) !== ood) return false;
    if (clin && String(q.clinical ? 1 : 0) !== clin) return false;
    if (agree) {
      const corr = ids.map((id) => q.preds[id]).filter(Boolean).map((p) => p.correct);
      const known = corr.filter((c) => c != null);
      if (agree === "all_correct" && !(known.length && known.every((c) => c))) return false;
      if (agree === "all_wrong" && !(known.length && known.every((c) => !c))) return false;
      if (agree === "disagree") {
        const preds = ids.map((id) => q.preds[id]).filter(Boolean).map((p) => norm(p.prediction));
        if (new Set(preds).size <= 1) return false;
      }
      if (agree === "focus_wrong") {
        const p = S.focus && q.preds[S.focus];
        if (!p || p.correct !== false) return false;
      }
    }
    return true;
  });
}

function applyAndRender() {
  S.filtered = applyFilters();
  $("counts").textContent = `${S.filtered.length} / ${S.questions.length} cases · ${selectedRunIds().length} models`;
  if (S.view === "matrix") renderMatrix();
  else renderCase();
}

// --------------------------------------------------------------------------- //
// matrix view
// --------------------------------------------------------------------------- //
function renderMatrix() {
  const ids = selectedRunIds();
  const runById = Object.fromEntries(S.runsForSplit.map((r) => [r.id, r]));
  if (!ids.length) {
    $("matrixWrap").innerHTML = `<div class="empty">No models selected — tick a model above.</div>`;
    return;
  }
  if (!S.filtered.length) {
    $("matrixWrap").innerHTML = `<div class="empty">No cases match the current filters.</div>`;
    return;
  }
  const head =
    `<thead><tr><th>Frame</th><th>Question</th><th>GT</th><th>Fmt</th><th>Cap</th>` +
    ids.map((id) => {
      const r = runById[id];
      return `<th class="modelcol">${esc(r.model_name)}<br><span class="acc">${pct(r.accuracy)}</span></th>`;
    }).join("") +
    `</tr></thead>`;

  const rows = S.filtered.map((q, i) => {
    const img = q.has_image
      ? `<img class="thumb" loading="lazy" src="/img?track=${S.track}&split=${S.split}&qid=${q.qID}" />`
      : `<div class="thumb"></div>`;
    const cells = ids.map((id) => {
      const p = q.preds[id];
      if (!p) return `<td><div class="cellpred na">—</div></td>`;
      const cls = p.correct == null ? "na" : p.correct ? "ok" : "bad";
      const mark = p.correct == null ? "" : `<span class="mark ${cls}">${p.correct ? "✓" : "✗"}</span>`;
      return `<td><div class="cellpred ${cls}">${mark}${esc(p.prediction)}</div></td>`;
    }).join("");
    return (
      `<tr data-i="${i}"><td>${img}</td>` +
      `<td class="qcell"><div class="qtext">${esc(q.question)}</div></td>` +
      `<td><span class="gt">${esc(q.gt)}</span></td>` +
      `<td><span class="tag">${esc(q.answer_format)}</span></td>` +
      `<td><span class="tag">${esc(q.primary_capability)}</span></td>` +
      cells + `</tr>`
    );
  }).join("");

  const tbl = `<table class="matrix">${head}<tbody>${rows}</tbody></table>`;
  $("matrixWrap").innerHTML = tbl;
  $("matrixWrap").querySelectorAll("tbody tr").forEach((tr) => {
    tr.onclick = () => { S.caseIndex = +tr.dataset.i; setView("case"); };
  });
}

// --------------------------------------------------------------------------- //
// case view
// --------------------------------------------------------------------------- //
function setView(v) {
  S.view = v;
  $("viewMatrix").classList.toggle("active", v === "matrix");
  $("viewCase").classList.toggle("active", v === "case");
  $("matrixView").classList.toggle("hidden", v !== "matrix");
  $("caseView").classList.toggle("hidden", v !== "case");
  if (v === "matrix") renderMatrix();
  else renderCase();
}

function stepCase(d) {
  if (!S.filtered.length) return;
  S.caseIndex = Math.max(0, Math.min(S.filtered.length - 1, S.caseIndex + d));
  renderCase();
}

async function renderCase() {
  if (!S.filtered.length) {
    $("caseBody").innerHTML = `<div class="empty">No cases match the current filters.</div>`;
    $("casePos").textContent = "";
    return;
  }
  S.caseIndex = Math.min(S.caseIndex, S.filtered.length - 1);
  const q = S.filtered[S.caseIndex];
  $("casePos").textContent = `${S.caseIndex + 1} / ${S.filtered.length} · qID ${q.qID}`;
  const d = await getJSON(`/api/question/${q.qID}?track=${S.track}&split=${S.split}`);
  const ids = new Set(selectedRunIds());
  const runs = d.runs.filter((r) => ids.has(r.run_id));

  // System prompt: show once if all shown runs share it, else per-row.
  const sysSet = new Set(runs.map((r) => r.system_prompt || ""));
  const sharedSys = sysSet.size === 1 ? [...sysSet][0] : null;

  const img = d.has_image
    ? `<img class="case-img" src="/img?track=${S.track}&split=${S.split}&qid=${d.qID}" />`
    : `<div class="empty">no frame image</div>`;

  const tags = [
    ["video", d.video], ["time", `${d.ts_start ?? ""}${d.ts_end && d.ts_end !== d.ts_start ? "–" + d.ts_end : ""}`],
    ["format", d.answer_format], ["capability", d.primary_capability],
    ["secondary", (d.secondaries || []).join(", ") || "—"],
    ["ood", d.ood ? "yes" : "no"], ["clinical", d.clinical ? "yes" : "no"],
  ].map(([k, v]) => `<span class="tag">${esc(k)}: ${esc(v)}</span>`).join("");

  const cmpRows = runs.map((r) => {
    const cls = r.correct == null ? "" : r.correct ? "ok" : "bad";
    const mark = r.correct == null ? "" : `<span class="mark ${cls}">${r.correct ? "✓" : "✗"}</span>`;
    const sys = !sharedSys && r.system_prompt
      ? `<details class="sys"><summary>system prompt</summary><pre>${esc(r.system_prompt)}</pre></details>` : "";
    const err = r.error ? `<div class="tag" style="color:#ef7a7a">error: ${esc(r.error)}</div>` : "";
    return (
      `<tr class="${cls}"><td class="modelname">${esc(r.label)}${sys}</td>` +
      `<td>${mark}<strong>${esc(r.prediction)}</strong></td>` +
      `<td class="raw">${esc(r.raw_output)}${err}</td>` +
      `<td class="lat">${r.latency == null ? "" : r.latency.toFixed(2) + "s"}</td></tr>`
    );
  }).join("");

  $("caseBody").innerHTML = `
    <div class="case-grid">
      <div>${img}</div>
      <div>
        <div class="panel">
          <div class="kv">${tags}</div>
          <div class="qbig">${esc(d.question)}</div>
          <div class="gtbig">Ground truth: <span class="gt">${esc(d.gt)}</span></div>
        </div>
        <div class="panel">
          <h3>Model input</h3>
          ${sharedSys ? `<details class="sys" open><summary>system prompt</summary><pre>${esc(sharedSys)}</pre></details>` : ""}
          <pre class="prompt">${esc(d.prompt)}</pre>
        </div>
        <div class="panel">
          <h3>Model outputs (${runs.length})</h3>
          <table class="cmp">
            <thead><tr><th>Model</th><th>Prediction</th><th>Raw output</th><th>Latency</th></tr></thead>
            <tbody>${cmpRows}</tbody>
          </table>
        </div>
      </div>
    </div>`;
}

init().catch((e) => {
  document.getElementById("matrixWrap").innerHTML = `<div class="empty">Failed to load: ${esc(e.message)}</div>`;
});
