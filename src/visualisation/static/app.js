"use strict";

const S = {
  allRuns: [],          // every RunInfo from /api/runs
  capabilityData: { source: "", groups: [] },
  track: null,
  split: null,
  dataset: null,
  runsForSplit: [],     // RunInfo for current split, in display order
  selected: new Set(),  // run ids shown as columns
  focus: "",            // run id for "focused model wrong" filter
  questions: [],        // /api/questions rows
  filtered: [],
  view: "matrix",
  caseIndex: 0,
  caseQID: "",
  statsModel: "",
  statsScope: "primary",
  drilldown: null,
  matrixPosition: { top: 0, left: 0, qID: "" },
};

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (a) => (a == null ? "—" : (a * 100).toFixed(1) + "%");
const norm = (s) => String(s ?? "").trim().toLowerCase();
const datasetLabel = (dataset) =>
  dataset === "heico" ? "HeiCo" : dataset === "lapchole" ? "LapChole" : dataset;

function capabilityItems() {
  return S.capabilityData.groups.flatMap((group) => [group, ...(group.children || [])]);
}

function capabilityInfo(code) {
  return capabilityItems().find((item) => item.code === code);
}

function capabilityLabel(code, includeCode = true) {
  const item = capabilityInfo(code);
  if (!item) return String(code ?? "—");
  return `${includeCode ? item.code + " · " : ""}${item.name}`;
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

// --------------------------------------------------------------------------- //
// init
// --------------------------------------------------------------------------- //
async function init() {
  [S.allRuns, S.capabilityData] = await Promise.all([
    getJSON("/api/runs"),
    getJSON("/api/capabilities"),
  ]);
  const groups = [
    ...new Map(
      S.allRuns.map((r) => [
        [r.track, r.split, r.dataset].join("|"),
        r,
      ])
    ).entries(),
  ];
  const sel = $("splitSel");
  sel.innerHTML = groups
    .map(([value, run]) => {
      const dataset = datasetLabel(run.dataset);
      return `<option value="${esc(value)}">${esc(dataset)} · ${run.count} cases</option>`;
    })
    .join("");
  const preferred = groups.find(([value]) => value === "frame|test|heico")?.[0] || groups[0]?.[0];
  if (!preferred) {
    $("matrixWrap").innerHTML = `<div class="empty">No runs found on disk.</div>`;
    return;
  }
  sel.value = preferred;
  wireControls();
  await loadSplit(preferred);
}

async function loadSplit(value) {
  [S.track, S.split, S.dataset] = value.split("|");
  S.runsForSplit = S.allRuns.filter(
    (r) => r.track === S.track && r.split === S.split && r.dataset === S.dataset
  );
  S.selected = new Set(S.runsForSplit.map((r) => r.id));
  S.focus = "";
  S.statsModel = S.runsForSplit[0]?.id || "";
  S.drilldown = null;
  S.matrixPosition = { top: 0, left: 0, qID: "" };
  S.caseQID = "";
  const data = await getJSON(
    `/api/questions?track=${encodeURIComponent(S.track)}` +
    `&split=${encodeURIComponent(S.split)}` +
    `&dataset=${encodeURIComponent(S.dataset)}`
  );
  S.questions = data.questions;
  S.caseIndex = 0;
  buildFilterOptions();
  buildModelChips();
  buildFocusOptions();
  buildStatsControls();
  setView("matrix");
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
  $("viewStats").onclick = () => setView("stats");
  $("viewCase").onclick = () => setView("case");
  $("backMatrix").onclick = () => setView("matrix");
  $("prevCase").onclick = () => stepCase(-1);
  $("nextCase").onclick = () => stepCase(1);
  $("statsModelSel").onchange = (e) => {
    S.statsModel = e.target.value;
    if (S.view === "stats") renderStats();
  };
  $("statsScopeSel").onchange = (e) => {
    S.statsScope = e.target.value;
    if (S.view === "stats") renderStats();
  };
  document.addEventListener("keydown", (e) => {
    if (S.view !== "case") return;
    if (e.key === "ArrowLeft") stepCase(-1);
    if (e.key === "ArrowRight") stepCase(1);
    if (e.key === "Escape") setView("matrix");
  });
}

function buildFilterOptions() {
  const uniq = (key) => [...new Set(S.questions.map((q) => q[key]).filter((v) => v != null))].sort();
  fillSelect("fmtSel", uniq("answer_format"));
  fillSelect("capSel", uniq("primary_capability"), (v) => capabilityLabel(v));
  fillSelect("vidSel", uniq("video"));
}
function fillSelect(id, vals, label = (v) => v) {
  $(id).innerHTML = `<option value="">all</option>` +
    vals.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
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
  if (!S.runsForSplit.some((r) => r.id === S.focus)) {
    S.focus = S.runsForSplit[0]?.id || "";
  }
  $("focusSel").innerHTML =
    `<option value="">—</option>` +
    S.runsForSplit.map((r) => `<option value="${esc(r.id)}">${esc(r.model_name)}</option>`).join("");
  $("focusSel").value = S.focus;
}

function buildStatsControls() {
  $("statsModelSel").innerHTML = S.runsForSplit
    .map((r) => `<option value="${esc(r.id)}">${esc(r.model_name)}</option>`)
    .join("");
  if (!S.runsForSplit.some((r) => r.id === S.statsModel)) {
    S.statsModel = S.runsForSplit[0]?.id || "";
  }
  $("statsModelSel").value = S.statsModel;
  $("statsScopeSel").value = S.statsScope;
}

// --------------------------------------------------------------------------- //
// filtering
// --------------------------------------------------------------------------- //
function selectedRunIds() {
  // Honour an explicitly empty selection (the "none" button) — no fall back to all.
  return S.runsForSplit.map((r) => r.id).filter((id) => S.selected.has(id));
}

function capabilityCodes(q, scope) {
  const codes = [q.primary_capability];
  if (scope === "tagged") codes.push(...(q.secondaries || []));
  return [...new Set(codes.filter(Boolean).map(String))];
}

function matchesCapability(q, code, scope) {
  const codes = capabilityCodes(q, scope);
  return code.length === 1
    ? codes.some((value) => value.startsWith(code))
    : codes.includes(code);
}

function applyFilters() {
  if (S.drilldown) {
    const { code, scope, modelId } = S.drilldown;
    return S.questions.filter((q) => {
      if (!matchesCapability(q, code, scope)) return false;
      return q.preds[modelId]?.correct === false;
    });
  }

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
  if (S.caseQID) {
    const current = S.filtered.findIndex((q) => q.qID === S.caseQID);
    if (current >= 0) S.caseIndex = current;
  }
  $("counts").textContent =
    `${S.filtered.length} / ${S.questions.length} cases · ${selectedRunIds().length} models` +
    (S.drilldown ? " · wrong-answer drill-down" : "");
  if (S.view === "matrix") renderMatrix();
  else if (S.view === "case") renderCase();
  else renderStats();
}

// --------------------------------------------------------------------------- //
// matrix view
// --------------------------------------------------------------------------- //
function rememberMatrixPosition(qID = S.matrixPosition.qID) {
  const wrap = $("matrixWrap");
  S.matrixPosition = {
    top: wrap.scrollTop,
    left: wrap.scrollLeft,
    qID: qID || "",
  };
}

function positionMatrixAtCase(qID) {
  const wrap = $("matrixWrap");
  const row = [...wrap.querySelectorAll("tbody tr")]
    .find((candidate) => candidate.dataset.qid === qID);
  S.matrixPosition.qID = qID;
  if (row) {
    S.matrixPosition.top = Math.max(
      0,
      row.offsetTop - (wrap.clientHeight - row.offsetHeight) / 2
    );
  }
}

function renderDrilldownBar() {
  const bar = $("drilldownBar");
  if (!S.drilldown) {
    bar.classList.add("hidden");
    bar.innerHTML = "";
    return;
  }
  const { code, scope, modelId } = S.drilldown;
  const run = S.runsForSplit.find((r) => r.id === modelId);
  bar.classList.remove("hidden");
  bar.innerHTML = `
    <div>
      <strong>Wrong-answer drill-down:</strong>
      ${esc(capabilityLabel(code))} ·
      ${scope === "primary" ? "primary assignment" : "primary or secondary tag"} ·
      ${esc(run?.model_name || modelId)}
      <span class="muted">Matrix filters are paused.</span>
    </div>
    <div class="drill-actions">
      <button id="drillStats">← capability statistics</button>
      <button id="drillClear">clear drill-down</button>
    </div>`;
  $("drillStats").onclick = () => setView("stats");
  $("drillClear").onclick = () => {
    const saved = S.drilldown.returnPosition;
    S.drilldown = null;
    if (saved) S.matrixPosition = saved;
    applyAndRender();
  };
}

function renderMatrix() {
  const wrap = $("matrixWrap");
  const position = { ...S.matrixPosition };
  const ids = selectedRunIds();
  const runById = Object.fromEntries(S.runsForSplit.map((r) => [r.id, r]));
  renderDrilldownBar();
  if (!ids.length) {
    wrap.innerHTML = `<div class="empty">No models selected — tick a model above.</div>`;
    return;
  }
  if (!S.filtered.length) {
    wrap.innerHTML = `<div class="empty">No cases match the current filters.</div>`;
    return;
  }
  const head =
    `<thead><tr><th>Frame</th><th>Question</th><th>GT</th><th>Fmt</th><th>Primary capability</th>` +
    ids.map((id) => {
      const r = runById[id];
      return `<th class="modelcol">${esc(r.model_name)}<br><span class="acc">${pct(r.accuracy)}</span></th>`;
    }).join("") +
    `</tr></thead>`;

  const rows = S.filtered.map((q, i) => {
    const img = q.has_image
      ? `<img class="thumb" loading="lazy" src="/img?track=${encodeURIComponent(S.track)}&split=${encodeURIComponent(S.split)}&dataset=${encodeURIComponent(S.dataset)}&qid=${encodeURIComponent(q.qID)}" />`
      : `<div class="thumb"></div>`;
    const cells = ids.map((id) => {
      const p = q.preds[id];
      if (!p) return `<td><div class="cellpred na">—</div></td>`;
      const cls = p.correct == null ? "na" : p.correct ? "ok" : "bad";
      const mark = p.correct == null ? "" : `<span class="mark ${cls}">${p.correct ? "✓" : "✗"}</span>`;
      return `<td><div class="cellpred ${cls}">${mark}${esc(p.prediction)}</div></td>`;
    }).join("");
    return (
      `<tr data-i="${i}" data-qid="${esc(q.qID)}" class="${q.qID === position.qID ? "selected-case" : ""}"><td>${img}</td>` +
      `<td class="qcell"><div class="qtext">${esc(q.question)}</div></td>` +
      `<td><span class="gt">${esc(q.gt)}</span></td>` +
      `<td><span class="tag">${esc(q.answer_format)}</span></td>` +
      `<td><span class="tag" title="${esc(capabilityInfo(q.primary_capability)?.definition || "")}">` +
      `${esc(capabilityLabel(q.primary_capability))}</span></td>` +
      cells + `</tr>`
    );
  }).join("");

  const tbl = `<table class="matrix">${head}<tbody>${rows}</tbody></table>`;
  wrap.innerHTML = tbl;
  wrap.querySelectorAll("tbody tr").forEach((tr) => {
    tr.onclick = () => {
      S.caseIndex = +tr.dataset.i;
      S.caseQID = S.filtered[S.caseIndex].qID;
      rememberMatrixPosition(S.caseQID);
      setView("case");
    };
  });
  requestAnimationFrame(() => {
    wrap.scrollTop = position.top;
    wrap.scrollLeft = position.left;
  });
}

function setView(v) {
  if (S.view === "matrix" && v !== "matrix") rememberMatrixPosition();
  S.view = v;
  $("viewMatrix").classList.toggle("active", v === "matrix");
  $("viewStats").classList.toggle("active", v === "stats");
  $("viewCase").classList.toggle("active", v === "case");
  $("matrixView").classList.toggle("hidden", v !== "matrix");
  $("statsView").classList.toggle("hidden", v !== "stats");
  $("caseView").classList.toggle("hidden", v !== "case");
  applyAndRender();
}

// --------------------------------------------------------------------------- //
// capability statistics
// --------------------------------------------------------------------------- //
function capabilityMetrics(code, scope, modelId) {
  const assigned = S.questions.filter((q) => matchesCapability(q, code, scope));
  const evaluated = assigned.filter((q) => q.preds[modelId]?.correct != null);
  const correct = evaluated.filter((q) => q.preds[modelId].correct === true).length;
  return {
    assigned: assigned.length,
    evaluated: evaluated.length,
    correct,
    wrong: evaluated.length - correct,
    accuracy: evaluated.length ? correct / evaluated.length : null,
  };
}

function statsRow(item, metrics, isGroup = false) {
  const width = metrics.accuracy == null ? 0 : metrics.accuracy * 100;
  const unknown = metrics.assigned - metrics.evaluated;
  return `
    <tr class="${isGroup ? "cap-group" : "cap-child"} ${metrics.assigned ? "" : "no-data"}">
      <td class="cap-code">${esc(item.code)}</td>
      <td>
        <div class="cap-name">${esc(item.name)}</div>
        <div class="cap-definition">${esc(item.definition)}</div>
      </td>
      <td class="num">${metrics.assigned.toLocaleString()}</td>
      <td class="num">${metrics.correct.toLocaleString()} / ${metrics.evaluated.toLocaleString()}` +
        `${unknown ? `<div class="unknown">${unknown} unscored</div>` : ""}</td>
      <td class="accuracy-cell">
        <div class="accuracy-value">${pct(metrics.accuracy)}</div>
        <div class="accuracy-track"><span style="width:${width.toFixed(1)}%"></span></div>
      </td>
      <td class="num wrong-count">${metrics.wrong.toLocaleString()}</td>
      <td><button class="review-wrong" data-code="${esc(item.code)}" ${metrics.wrong ? "" : "disabled"}>` +
        `review wrong</button></td>
    </tr>`;
}

function renderStats() {
  const run = S.runsForSplit.find((r) => r.id === S.statsModel) || S.runsForSplit[0];
  if (!run) {
    $("statsBody").innerHTML = `<div class="empty">No model is available for this dataset.</div>`;
    return;
  }
  S.statsModel = run.id;
  $("statsModelSel").value = run.id;
  $("statsScopeSel").value = S.statsScope;

  const overall = (() => {
    const evaluated = S.questions.filter((q) => q.preds[run.id]?.correct != null);
    const correct = evaluated.filter((q) => q.preds[run.id].correct === true).length;
    return {
      evaluated: evaluated.length,
      correct,
      wrong: evaluated.length - correct,
      accuracy: evaluated.length ? correct / evaluated.length : null,
    };
  })();

  const rows = S.capabilityData.groups.map((group) => {
    const groupRow = statsRow(
      group,
      capabilityMetrics(group.code, S.statsScope, run.id),
      true
    );
    const children = (group.children || [])
      .map((item) => statsRow(item, capabilityMetrics(item.code, S.statsScope, run.id)))
      .join("");
    return groupRow + children;
  }).join("");

  $("statsBody").innerHTML = `
    <div class="stats-summary">
      <div class="metric"><span>Overall accuracy</span><strong>${pct(overall.accuracy)}</strong></div>
      <div class="metric"><span>Evaluated cases</span><strong>${overall.evaluated.toLocaleString()}</strong></div>
      <div class="metric bad-metric"><span>Wrong answers</span><strong>${overall.wrong.toLocaleString()}</strong></div>
      <div class="stats-context">
        ${esc(datasetLabel(S.dataset))} · ${esc(S.split)} · ${esc(run.model_name)}
      </div>
    </div>
    <div class="stats-note">
      ${S.statsScope === "primary"
        ? "Primary assignment: every question contributes to exactly one sub-capability."
        : "Tagged assignment: a question contributes to its primary capability and every secondary tag."}
      Group rows aggregate the sub-capability codes beneath them.
      <a href="${esc(S.capabilityData.source)}" target="_blank" rel="noreferrer">Official taxonomy ↗</a>
    </div>
    <div class="cap-table-wrap">
      <table class="cap-table">
        <thead><tr>
          <th>Code</th><th>Official capability and definition</th><th>Cases</th>
          <th>Correct / scored</th><th>Accuracy</th><th>Wrong</th><th></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  $("statsBody").querySelectorAll(".review-wrong:not(:disabled)").forEach((button) => {
    button.onclick = () => activateCapabilityDrilldown(button.dataset.code);
  });
}

function activateCapabilityDrilldown(code) {
  const returnPosition = S.drilldown?.returnPosition || { ...S.matrixPosition };
  S.drilldown = {
    code,
    scope: S.statsScope,
    modelId: S.statsModel,
    returnPosition,
  };
  S.focus = S.statsModel;
  S.selected.add(S.statsModel);
  buildModelChips();
  $("focusSel").value = S.focus;
  S.matrixPosition = { top: 0, left: 0, qID: "" };
  S.caseIndex = 0;
  S.caseQID = "";
  setView("matrix");
}

// --------------------------------------------------------------------------- //
// case view
// --------------------------------------------------------------------------- //
function stepCase(d) {
  if (!S.filtered.length) return;
  S.caseIndex = Math.max(0, Math.min(S.filtered.length - 1, S.caseIndex + d));
  S.caseQID = S.filtered[S.caseIndex].qID;
  positionMatrixAtCase(S.caseQID);
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
  S.caseQID = q.qID;
  $("casePos").textContent = `${S.caseIndex + 1} / ${S.filtered.length} · qID ${q.qID}`;
  const d = await getJSON(
    `/api/question/${encodeURIComponent(q.qID)}?track=${encodeURIComponent(S.track)}` +
    `&split=${encodeURIComponent(S.split)}` +
    `&dataset=${encodeURIComponent(S.dataset)}`
  );
  if (S.view !== "case" || S.caseQID !== q.qID) return;
  const ids = new Set(selectedRunIds());
  const runs = d.runs.filter((r) => ids.has(r.run_id));

  // System prompt: show once if all shown runs share it, else per-row.
  const sysSet = new Set(runs.map((r) => r.system_prompt || ""));
  const sharedSys = sysSet.size === 1 ? [...sysSet][0] : null;

  const img = d.has_image
    ? `<img class="case-img" src="/img?track=${encodeURIComponent(S.track)}&split=${encodeURIComponent(S.split)}&dataset=${encodeURIComponent(S.dataset)}&qid=${encodeURIComponent(d.qID)}" />`
    : `<div class="empty">no frame image</div>`;

  const tags = [
    ["video", d.video], ["time", `${d.ts_start ?? ""}${d.ts_end && d.ts_end !== d.ts_start ? "–" + d.ts_end : ""}`],
    ["format", d.answer_format], ["capability", capabilityLabel(d.primary_capability)],
    ["secondary", (d.secondaries || []).map((code) => capabilityLabel(code)).join(", ") || "—"],
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
