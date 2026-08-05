"use strict";

const S = {
  allRuns: [],          // every RunInfo from /api/runs
  config: { initial_track: "frame", available_tracks: [] },
  groups: [],           // unique (track, split, dataset) contexts
  trackSelection: {},   // last selected context for each track
  loadToken: 0,
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

const TRACKS = {
  frame: {
    title: "Frame predictions",
    description: "Inspect single-frame visual questions and model outputs.",
    media: "frame image",
  },
  segment: {
    title: "Segment predictions",
    description: "Review bounded video clips from their annotated start and end times.",
    media: "video segment",
  },
  procedure: {
    title: "Procedure predictions",
    description: "Follow long-range procedural evidence through the source video.",
    media: "procedure video",
  },
};

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (a) => (a == null ? "—" : (a * 100).toFixed(1) + "%");
const norm = (s) => String(s ?? "").trim().toLowerCase();
const datasetLabel = (dataset) =>
  dataset === "heico" ? "HeiCo" : dataset === "lapchole" ? "LapChole" : dataset;
const trackLabel = (track) =>
  track ? track.charAt(0).toUpperCase() + track.slice(1) : "Prediction";

function timestampSeconds(value) {
  const parts = String(value ?? "").split(":").map(Number);
  if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v))) return 0;
  return parts[0] * 3600 + parts[1] * 60 + parts[2];
}

function videoURL(qid) {
  return (
    `/video?track=${encodeURIComponent(S.track)}` +
    `&split=${encodeURIComponent(S.split)}` +
    `&dataset=${encodeURIComponent(S.dataset)}` +
    `&qid=${encodeURIComponent(qid)}`
  );
}

function closeVideoModal() {
  const modal = $("videoModal");
  const video = $("trackVideo");
  if (!modal || modal.classList.contains("hidden")) return;
  video.pause();
  video.removeAttribute("src");
  video.load(); // cancel any in-flight range request and release the media resource
  for (const key of ["targetSeconds", "rangeStart", "rangeEnd", "rangePlayback", "track"]) {
    delete video.dataset[key];
  }
  $("openVideoDirect").setAttribute("href", "#");
  $("videoStatus").textContent = "Video closed; no source data is being transferred.";
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

function seekVideo(seconds) {
  const video = $("trackVideo");
  if (!Number.isFinite(seconds) || !Number.isFinite(video.duration)) return;
  video.currentTime = Math.max(0, Math.min(seconds, Math.max(video.duration - 0.05, 0)));
}

function openVideoModal(detail) {
  const modal = $("videoModal");
  const video = $("trackVideo");
  const url = videoURL(detail.qID);
  const rangeStart = timestampSeconds(detail.ts_start);
  const rangeEnd = timestampSeconds(detail.ts_end);
  const isSegment = S.track === "segment";
  const target = isSegment ? rangeStart : rangeEnd;
  const meta = TRACKS[S.track] || TRACKS.procedure;

  $("videoTitle").textContent = detail.video || meta.media;
  $("videoSubtitle").textContent = isSegment
    ? `${datasetLabel(S.dataset)} · annotated segment ${detail.ts_start || "?"}–${detail.ts_end || "?"}`
    : `${datasetLabel(S.dataset)} · evidence window ${detail.ts_start || "?"}–${detail.ts_end || "?"}`;
  $("videoStatus").textContent = "Loading video metadata…";
  $("openVideoDirect").setAttribute("href", url);
  video.dataset.targetSeconds = String(target);
  video.dataset.rangeStart = String(rangeStart);
  video.dataset.rangeEnd = String(rangeEnd);
  video.dataset.track = S.track;
  $("seekRangeStart").textContent = isSegment ? "segment start" : "evidence start";
  $("seekRangeEnd").textContent = isSegment ? "segment end" : "question time";
  $("playSegment").classList.toggle("hidden", !isSegment);
  $("playSegment").disabled = true;

  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");

  // Assigning src here (and nowhere in page rendering) guarantees that opening
  // the overlay is the first action capable of requesting source-video bytes.
  video.src = url;
  video.load();
  video.focus();
}

function playSegmentRange() {
  const video = $("trackVideo");
  const start = Number(video.dataset.rangeStart);
  const end = Number(video.dataset.rangeEnd);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
  video.dataset.rangePlayback = "1";
  seekVideo(start);
  const playback = video.play();
  if (playback?.catch) {
    playback.catch(() => {
      delete video.dataset.rangePlayback;
      $("videoStatus").textContent = "Playback could not start automatically; press play in the video controls.";
    });
  }
}

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
  [S.allRuns, S.capabilityData, S.config] = await Promise.all([
    getJSON("/api/runs"),
    getJSON("/api/capabilities"),
    getJSON("/api/config"),
  ]);
  const grouped = new Map();
  for (const run of S.allRuns) {
    const value = [run.track, run.split, run.dataset].join("|");
    const group = grouped.get(value) || {
      value,
      track: run.track,
      split: run.split,
      dataset: run.dataset,
      count: 0,
      models: 0,
    };
    group.count = Math.max(group.count, run.count || 0);
    group.models += 1;
    grouped.set(value, group);
  }
  S.groups = [...grouped.values()];
  if (!S.groups.length) {
    $("matrixWrap").innerHTML = `<div class="empty">No runs found on disk.</div>`;
    return;
  }
  wireControls();
  configureTrackNav();
  const requestedTrack = new URLSearchParams(window.location.search).get("track");
  const initialTrack = S.config.available_tracks.includes(requestedTrack)
    ? requestedTrack
    : S.config.initial_track;
  await activateTrack(initialTrack || S.groups[0].track);
}

function configureTrackNav() {
  const available = new Set(S.groups.map((group) => group.track));
  document.querySelectorAll(".track-tab").forEach((button) => {
    const enabled = available.has(button.dataset.track);
    button.disabled = !enabled;
    button.title = enabled ? `Open the ${trackLabel(button.dataset.track)} module` : "No loaded results";
  });
}

function resetCrossTrackFilters() {
  for (const id of ["search", "oodSel", "clinSel", "agreeSel", "focusSel"]) {
    $(id).value = "";
  }
}

async function activateTrack(track) {
  const groups = S.groups.filter((group) => group.track === track);
  if (!groups.length) return;
  const changingTrack = Boolean(S.track && S.track !== track);
  if (changingTrack) resetCrossTrackFilters();
  closeVideoModal();

  document.body.dataset.track = track;
  document.querySelectorAll(".track-tab").forEach((button) => {
    const active = button.dataset.track === track;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  const selector = $("splitSel");
  selector.innerHTML = groups.map((group) => {
    const modelWord = group.models === 1 ? "model" : "models";
    const label = `${datasetLabel(group.dataset)} · ${group.split} · ${group.count.toLocaleString()} cases · ${group.models} ${modelWord}`;
    return `<option value="${esc(group.value)}">${esc(label)}</option>`;
  }).join("");
  const remembered = S.trackSelection[track];
  const preferred = groups.find((group) => group.value === remembered)
    || groups.find((group) => group.split === "test" && group.dataset === "heico")
    || groups[0];
  selector.value = preferred.value;
  await loadSplit(preferred.value);
}

async function loadSplit(value) {
  const token = ++S.loadToken;
  closeVideoModal();
  [S.track, S.split, S.dataset] = value.split("|");
  S.trackSelection[S.track] = value;
  const trackMeta = TRACKS[S.track] || TRACKS.frame;
  document.body.dataset.track = S.track;
  $("trackEyebrow").textContent = `${trackLabel(S.track)} track`;
  $("trackTitle").textContent = trackMeta.title;
  $("trackDescription").textContent = trackMeta.description;
  document.title = `ORena FOCUS — ${trackMeta.title}`;
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
  if (token !== S.loadToken) return;
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
  document.querySelectorAll(".track-tab").forEach((button) => {
    button.onclick = () => activateTrack(button.dataset.track);
  });
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
  $("closeVideo").onclick = closeVideoModal;
  $("videoBackdrop").onclick = closeVideoModal;
  $("seekVideoStart").onclick = () => seekVideo(0);
  $("seekRangeStart").onclick = () => seekVideo(Number($("trackVideo").dataset.rangeStart || 0));
  $("seekRangeEnd").onclick = () => seekVideo(Number($("trackVideo").dataset.rangeEnd || 0));
  $("playSegment").onclick = playSegmentRange;
  document.querySelectorAll("[data-seek-relative]").forEach((button) => {
    button.onclick = () => {
      const video = $("trackVideo");
      seekVideo(video.currentTime + Number(button.dataset.seekRelative));
    };
  });
  const trackVideo = $("trackVideo");
  trackVideo.addEventListener("loadedmetadata", () => {
    seekVideo(Number(trackVideo.dataset.targetSeconds || 0));
    $("playSegment").disabled = trackVideo.dataset.track !== "segment";
    $("videoStatus").textContent =
      trackVideo.dataset.track === "segment"
        ? "Ready at the annotated segment start. Play the segment or use the full timeline."
        : "Ready at the question time. Use the native timeline or the nearby seek buttons.";
  });
  trackVideo.addEventListener("waiting", () => {
    $("videoStatus").textContent = "Waiting for source-video bytes…";
  });
  trackVideo.addEventListener("playing", () => {
    $("videoStatus").textContent = "Streaming video directly; only requested byte ranges are transferred.";
  });
  trackVideo.addEventListener("timeupdate", () => {
    if (trackVideo.dataset.rangePlayback !== "1") return;
    const end = Number(trackVideo.dataset.rangeEnd);
    if (Number.isFinite(end) && trackVideo.currentTime >= end) {
      trackVideo.pause();
      delete trackVideo.dataset.rangePlayback;
      $("videoStatus").textContent = "Segment playback reached the annotated end time.";
    }
  });
  trackVideo.addEventListener("error", () => {
    $("playSegment").disabled = true;
    $("videoStatus").textContent =
      "The browser could not load or decode this source video or its proxy.";
  });
  $("statsModelSel").onchange = (e) => {
    S.statsModel = e.target.value;
    if (S.view === "stats") renderStats();
  };
  $("statsScopeSel").onchange = (e) => {
    S.statsScope = e.target.value;
    if (S.view === "stats") renderStats();
  };
  document.addEventListener("keydown", (e) => {
    if (!$("videoModal").classList.contains("hidden")) {
      if (e.key === "Escape") closeVideoModal();
      return;
    }
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
  const mediaHeading = S.track === "frame" ? "Frame" : "Video preview";
  const head =
    `<thead><tr><th>${mediaHeading}</th><th>Question</th><th>GT</th><th>Fmt</th><th>Primary capability</th>` +
    ids.map((id) => {
      const r = runById[id];
      return `<th class="modelcol">${esc(r.model_name)}<br><span class="acc">${pct(r.accuracy)}</span></th>`;
    }).join("") +
    `</tr></thead>`;

  const rows = S.filtered.map((q, i) => {
    const img = q.has_image
      ? `<img class="thumb" loading="lazy" alt="Representative ${esc(TRACKS[S.track]?.media || "frame")}" src="/img?track=${encodeURIComponent(S.track)}&split=${encodeURIComponent(S.split)}&dataset=${encodeURIComponent(S.dataset)}&qid=${encodeURIComponent(q.qID)}" />`
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
  if (S.view === "case" && v !== "case") closeVideoModal();
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
  closeVideoModal();
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
    ? `<img class="case-img" alt="Representative ${esc(TRACKS[S.track]?.media || "frame")}" src="/img?track=${encodeURIComponent(S.track)}&split=${encodeURIComponent(S.split)}&dataset=${encodeURIComponent(S.dataset)}&qid=${encodeURIComponent(d.qID)}" />`
    : `<div class="empty">no frame image</div>`;
  const videoAction = S.track === "segment" ? "stream annotated segment" : "stream procedure video";
  const videoNote = S.track === "segment"
    ? "Opens at the annotated segment start · play only the bounded segment or inspect the full timeline"
    : "Opens at the question time · browser-compatible 480p proxy preferred when available";
  const media = d.has_video
    ? `<div class="case-media">${img}` +
      `<button id="openCaseVideo" class="video-launch" type="button">` +
      `<span>▶ ${videoAction}</span></button></div>` +
      `<div class="video-note">${videoNote}<br>Video bytes load only after click.</div>`
    : img;

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
      <div>${media}</div>
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
  const openVideo = $("openCaseVideo");
  if (openVideo) openVideo.onclick = () => openVideoModal(d);
}

init().catch((e) => {
  document.getElementById("matrixWrap").innerHTML = `<div class="empty">Failed to load: ${esc(e.message)}</div>`;
});
