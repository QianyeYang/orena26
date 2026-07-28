"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const CHECK_KEYS = [
  "full_frame_scanned",
  "every_instance_boxed",
  "classes_reviewed",
  "qa_compared",
];

const state = {
  meta: null,
  cases: [],
  caseMap: new Map(),
  currentCase: null,
  annotation: null,
  selectedBoxId: null,
  activeClass: "Clip",
  dirty: false,
  editVersion: 0,
  saving: false,
  savePromise: null,
  autosaveTimer: null,
  queueLimit: 300,
  filteredCases: [],
  zoom: 1,
  interaction: null,
  serverBlockers: [],
  progress: null,
  readOnly: false,
};

const canvas = $("#box-canvas");
const ctx = canvas.getContext("2d");
const caseImage = $("#case-image");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function classInfo(name) {
  return state.meta.classes.find((item) => item.name === name) || {
    name,
    color: "#ffffff",
  };
}

function toast(message, type = "") {
  const item = document.createElement("div");
  item.className = `toast ${type}`.trim();
  item.textContent = message;
  $("#toast-region").append(item);
  setTimeout(() => item.remove(), 4300);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function statusLabel(status) {
  return {
    unstarted: "Unstarted",
    in_progress: "In progress",
    needs_review: "Needs review",
    complete: "Complete",
    skipped: "Skipped",
  }[status] || status;
}

function setSaveState(label, style = "") {
  const element = $("#save-state");
  element.textContent = label;
  element.className = `save-state ${style}`.trim();
}

function switchView(view) {
  $$(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  $$(".view").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  });
  if (view === "progress") refreshProgress();
}

function populateStaticControls() {
  $("#class-select").innerHTML = state.meta.classes
    .map(
      (item) =>
        `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`,
    )
    .join("");
  $("#class-select").value = state.activeClass;

  const batches = state.meta.pool.summary.batches;
  $("#batch-filter").innerHTML = [
    `<option value="all">All 4,000 cases</option>`,
    ...Array.from(
      { length: batches },
      (_, index) =>
        `<option value="${index + 1}">Batch ${index + 1} · ranks ${
          index * 100 + 1
        }–${(index + 1) * 100}</option>`,
    ),
  ].join("");
  $("#batch-filter").value = "1";
}

function caseMatches(caseItem) {
  const batch = $("#batch-filter").value;
  const dataset = $("#dataset-filter").value;
  const status = $("#status-filter").value;
  const search = $("#queue-search").value.trim().toLowerCase();
  if (batch !== "all" && Number(batch) !== caseItem.batch) return false;
  if (dataset !== "all" && dataset !== caseItem.dataset) return false;
  if (status !== "all" && status !== caseItem.status) return false;
  if (!search) return true;
  const haystack = [
    caseItem.rank,
    caseItem.id,
    caseItem.video,
    caseItem.timestamp,
    caseItem.dataset_name,
    caseItem.procedure_type,
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(search);
}

function renderQueue() {
  state.filteredCases = state.cases.filter(caseMatches);
  const shown = state.filteredCases.slice(0, state.queueLimit);
  const list = $("#case-list");
  if (!shown.length) {
    list.innerHTML =
      `<div class="empty-small" style="padding:18px">No cases match these filters.</div>`;
  } else {
    list.innerHTML = shown
      .map((caseItem) => {
        const active = state.currentCase?.id === caseItem.id ? " active" : "";
        return `
          <button class="case-item${active}" data-case-id="${caseItem.id}" role="listitem">
            <div class="case-top">
              <span class="status-dot status-${caseItem.status}" title="${escapeHtml(
                statusLabel(caseItem.status),
              )}"></span>
              <span class="rank">#${caseItem.rank}</span>
              <span class="tier tier-${caseItem.priority_tier}">${escapeHtml(
                caseItem.priority_tier,
              )}</span>
              <span class="case-score">${caseItem.priority_score} pts</span>
            </div>
            <div class="case-source">${escapeHtml(caseItem.dataset_name)} · ${escapeHtml(
              caseItem.video,
            )}</div>
            <div class="case-detail">${escapeHtml(caseItem.timestamp)} · ${
              caseItem.qa_count
            } QA · ${escapeHtml(statusLabel(caseItem.status))}</div>
          </button>`;
      })
      .join("");
  }
  $("#queue-summary").innerHTML =
    `<span><strong>${state.filteredCases.length.toLocaleString()}</strong> cases</span>` +
    `<span>${shown.length < state.filteredCases.length ? `showing ${shown.length}` : "all shown"}</span>`;
  $("#show-more").classList.toggle(
    "hidden",
    shown.length >= state.filteredCases.length,
  );
  requestAnimationFrame(() => {
    list.querySelector(".case-item.active")?.scrollIntoView({ block: "nearest" });
  });
}

function updateCaseLocalStatus(annotation) {
  const queueCase = state.caseMap.get(annotation.case_id);
  if (!queueCase) return;
  queueCase.status = annotation.status;
  queueCase.revision = annotation.revision;
  queueCase.box_count = annotation.boxes.length;
  queueCase.updated_at = annotation.updated_at;
}

async function refreshPoolStatuses() {
  const payload = await fetchJson("/api/pool");
  const currentId = state.currentCase?.id;
  state.cases = payload.cases;
  state.caseMap = new Map(state.cases.map((item) => [item.id, item]));
  if (currentId && state.caseMap.has(currentId)) {
    const summary = state.caseMap.get(currentId);
    if (!state.dirty && state.annotation?.revision !== summary.revision) {
      await loadCase(currentId, { skipSave: true });
      return;
    }
  }
  renderQueue();
  await refreshProgress(false);
  toast("Queue statuses refreshed.");
}

function currentBox() {
  return (
    state.annotation?.boxes.find((box) => box.id === state.selectedBoxId) || null
  );
}

function buildCaseHeading() {
  const caseItem = state.currentCase;
  if (!caseItem) return;
  const reasons = caseItem.priority_reasons
    .slice(0, 4)
    .map((reason) => `<span class="reason-tag" title="${escapeHtml(reason)}">${escapeHtml(reason)}</span>`)
    .join("");
  $("#case-heading").classList.remove("skeleton");
  $("#case-heading").innerHTML = `
    <div class="case-heading-main">
      <div class="eyebrow">${escapeHtml(caseItem.dataset_name)} · ${escapeHtml(
        caseItem.procedure_type,
      )} · Batch ${caseItem.batch}</div>
      <h2 title="${escapeHtml(caseItem.video)}">${escapeHtml(caseItem.video)}</h2>
      <div class="case-tags">${reasons}</div>
    </div>
    <div class="rank-large">#${caseItem.rank}<small>${caseItem.priority_score} priority points</small></div>`;
  $("#image-meta").textContent =
    `${caseItem.image_width}×${caseItem.image_height} · ${caseItem.timestamp} · ` +
    `${caseItem.id}`;
}

function renderQuestions(mismatchIds = new Set()) {
  const questions = state.currentCase?.questions || [];
  $("#qa-count").textContent = `${questions.length} QA`;
  $("#qa-list").innerHTML =
    questions
      .map(
        (question) => `
      <article class="qa-card ${mismatchIds.has(String(question.id)) ? "mismatch" : ""}">
        <div class="qa-meta">
          <span>#${escapeHtml(question.id)}</span>
          <span>${escapeHtml(question.primary_capability)}</span>
          <span>${escapeHtml(question.answer_format)}</span>
          <span>${escapeHtml(question.kind.replaceAll("_", " "))}</span>
        </div>
        <div class="qa-question">${escapeHtml(question.question)}</div>
        <div class="qa-answer">${escapeHtml(question.answer)}</div>
      </article>`,
      )
      .join("") || `<div class="empty-small">No QA records.</div>`;
}

function countBoxes() {
  const counts = new Map();
  for (const box of state.annotation?.boxes || []) {
    counts.set(box.class_name, (counts.get(box.class_name) || 0) + 1);
  }
  return counts;
}

function liveValidation() {
  if (!state.annotation || !state.currentCase) {
    return {
      structural: [],
      mismatches: [],
      inventory: new Map(),
      uncertain: 0,
      canComplete: false,
      canReview: false,
    };
  }
  const annotation = state.annotation;
  const boxes = annotation.boxes;
  const inventory = countBoxes();
  const observedClasses = new Set(inventory.keys());
  const structural = [];
  if (!annotation.no_foreign_objects && !boxes.length) {
    structural.push("Draw at least one box or choose “No foreign objects”.");
  }
  const checkLabels = {
    full_frame_scanned: "Confirm that you scanned the entire frame.",
    every_instance_boxed: "Confirm that every visible instance is boxed.",
    classes_reviewed: "Confirm that every class identity was reviewed.",
    qa_compared: "Confirm that boxes were compared with all QA.",
  };
  for (const key of CHECK_KEYS) {
    if (!annotation.checks[key]) structural.push(checkLabels[key]);
  }

  const mismatches = [];
  const addMismatch = (question, message) =>
    mismatches.push({ questionId: String(question.id), message });

  for (const question of state.currentCase.questions) {
    const expectedNumber = question.numeric_answer;
    if (
      question.kind === "total_instance_count" &&
      Number.isInteger(expectedNumber) &&
      boxes.length !== expectedNumber
    ) {
      addMismatch(
        question,
        `QA #${question.id}: expected ${expectedNumber} total instance(s), boxes show ${boxes.length}.`,
      );
    } else if (
      question.kind === "distinct_class_count" &&
      Number.isInteger(expectedNumber) &&
      observedClasses.size !== expectedNumber
    ) {
      addMismatch(
        question,
        `QA #${question.id}: expected ${expectedNumber} distinct class(es), boxes show ${observedClasses.size}.`,
      );
    } else if (question.kind === "class_count" && question.target_class) {
      const observed = inventory.get(question.target_class) || 0;
      if (Number.isInteger(expectedNumber) && observed !== expectedNumber) {
        addMismatch(
          question,
          `QA #${question.id}: expected ${expectedNumber} ${question.target_class} instance(s), boxes show ${observed}.`,
        );
      }
    } else if (question.kind === "class_inventory") {
      const answerNone = ["none", "no", "no foreign object", "no foreign objects"].includes(
        String(question.answer).trim().toLowerCase(),
      );
      const expected = new Set(answerNone ? [] : question.answer_classes || []);
      const equal =
        expected.size === observedClasses.size &&
        [...expected].every((name) => observedClasses.has(name));
      if (!equal) {
        addMismatch(
          question,
          `QA #${question.id}: class inventory differs (QA: ${
            [...expected].join(", ") || "none"
          }; boxes: ${[...observedClasses].join(", ") || "none"}).`,
        );
      }
    } else if (question.kind === "single_object_identity") {
      const expected = new Set(question.answer_classes || []);
      const equal =
        boxes.length === 1 &&
        expected.size === observedClasses.size &&
        [...expected].every((name) => observedClasses.has(name));
      if (!equal) {
        addMismatch(
          question,
          `QA #${question.id}: single-object identity disagrees with the boxes.`,
        );
      }
    } else if (
      question.kind === "closest_to_center" &&
      question.answer_classes?.length &&
      boxes.length
    ) {
      const centerX = state.currentCase.image_width / 2;
      const centerY = state.currentCase.image_height / 2;
      const closest = [...boxes].sort((a, b) => {
        const da =
          (a.x + a.width / 2 - centerX) ** 2 +
          (a.y + a.height / 2 - centerY) ** 2;
        const db =
          (b.x + b.width / 2 - centerX) ** 2 +
          (b.y + b.height / 2 - centerY) ** 2;
        return da - db;
      })[0];
      if (!question.answer_classes.includes(closest.class_name)) {
        addMismatch(
          question,
          `QA #${question.id}: the box nearest frame centre is ${closest.class_name}, not ${question.answer_classes.join(
            " / ",
          )}.`,
        );
      }
    }
  }
  const uncertain = boxes.filter((box) => box.uncertain).length;
  const canReview = structural.length === 0;
  return {
    structural,
    mismatches,
    inventory,
    uncertain,
    canReview,
    canComplete: canReview && mismatches.length === 0 && uncertain === 0,
  };
}

function renderBoxList() {
  const boxes = state.annotation?.boxes || [];
  if (!boxes.length) {
    $("#box-list").innerHTML = `<div class="empty-small">No boxes yet.</div>`;
  } else {
    $("#box-list").innerHTML = boxes
      .map((box, index) => {
        const info = classInfo(box.class_name);
        const flags = [
          box.difficult ? `<span class="flag">difficult</span>` : "",
          box.uncertain ? `<span class="flag">uncertain</span>` : "",
        ].join("");
        return `
        <button class="box-row ${box.id === state.selectedBoxId ? "selected" : ""}"
                data-box-id="${escapeHtml(box.id)}">
          <span class="color-chip" style="background:${escapeHtml(info.color)}"></span>
          <span>
            <span class="box-name">${index + 1}. ${escapeHtml(box.class_name)}${flags}</span>
            <span class="box-coords">x ${Math.round(box.x)}, y ${Math.round(
              box.y,
            )}, ${Math.round(box.width)}×${Math.round(box.height)}</span>
          </span>
          <span aria-hidden="true">›</span>
        </button>`;
      })
      .join("");
  }
  const selected = currentBox();
  $("#delete-box").disabled = !selected || state.readOnly;
  $("#box-difficult").disabled = !selected || state.readOnly;
  $("#box-uncertain").disabled = !selected || state.readOnly;
  $("#box-difficult").checked = Boolean(selected?.difficult);
  $("#box-uncertain").checked = Boolean(selected?.uncertain);
  if (selected) {
    $("#class-select").value = selected.class_name;
  } else {
    $("#class-select").value = state.activeClass;
  }
}

function renderCompleteness() {
  const report = liveValidation();
  const chips = [...report.inventory.entries()]
    .map(([name, count]) => {
      const color = classInfo(name).color;
      return `<span class="inventory-chip" style="background:${escapeHtml(
        color,
      )}55;border:1px solid ${escapeHtml(color)}">${escapeHtml(name)} ×${count}</span>`;
    })
    .join("");
  $("#derived-summary").innerHTML = `
    <strong>Derived from boxes:</strong> ${state.annotation?.boxes.length || 0}
    instance(s), ${report.inventory.size} class(es)
    <div>${chips || `<span class="muted">no boxed classes</span>`}</div>`;

  const messages = [
    ...report.structural.map((message) => ({ message, type: "structural" })),
    ...report.mismatches.map((item) => ({ message: item.message, type: "mismatch" })),
    ...(report.uncertain
      ? [
          {
            message: `${report.uncertain} uncertain box(es) require review.`,
            type: "uncertain",
          },
        ]
      : []),
    ...state.serverBlockers.map((item) => ({
      message: item.message || String(item),
      type: "server",
    })),
  ];
  $("#validation-list").innerHTML = messages.length
    ? messages
        .map(
          (item) =>
            `<div class="validation-message">${escapeHtml(item.message)}</div>`,
        )
        .join("")
    : `<div class="validation-message validation-ok">All completion checks pass.</div>`;
  const mismatchIds = new Set(report.mismatches.map((item) => item.questionId));
  renderQuestions(mismatchIds);

  $("#mark-complete").disabled =
    !report.canComplete || state.readOnly || state.saving;
  $("#needs-review").disabled =
    !report.canReview || state.readOnly || state.saving;
}

function renderAnnotation() {
  if (!state.annotation) return;
  $("#no-objects").checked = state.annotation.no_foreign_objects;
  $("#no-objects").disabled = state.readOnly;
  $("#class-select").disabled = state.readOnly;
  $("#case-notes").value = state.annotation.notes || "";
  $("#case-notes").disabled = state.readOnly;
  $$(".completion-checks input").forEach((input) => {
    input.checked = Boolean(state.annotation.checks[input.dataset.check]);
    input.disabled = state.readOnly;
  });
  $("#save-draft").disabled = state.readOnly || state.saving;
  $("#skip-case").disabled = state.readOnly || state.saving;
  renderBoxList();
  renderCompleteness();
  renderCanvas();
}

function setZoom(scale) {
  if (!state.currentCase) return;
  state.zoom = Math.max(0.25, Math.min(2.5, scale));
  const width = state.currentCase.image_width * state.zoom;
  const height = state.currentCase.image_height * state.zoom;
  const surface = $("#image-surface");
  surface.style.width = `${width}px`;
  surface.style.height = `${height}px`;
  $("#zoom-range").value = String(Math.round(state.zoom * 100));
  $("#zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
  renderCanvas();
}

function fitImage() {
  if (!state.currentCase) return;
  const viewport = $("#canvas-viewport");
  const availableWidth = Math.max(100, viewport.clientWidth - 40);
  const availableHeight = Math.max(100, viewport.clientHeight - 40);
  setZoom(
    Math.min(
      availableWidth / state.currentCase.image_width,
      availableHeight / state.currentCase.image_height,
      1.5,
    ),
  );
  viewport.scrollTo({ top: 0, left: 0 });
}

function loadImage() {
  $("#canvas-empty").classList.add("hidden");
  $("#image-surface").classList.remove("hidden");
  canvas.width = state.currentCase.image_width;
  canvas.height = state.currentCase.image_height;
  caseImage.onload = () => {
    fitImage();
    renderCanvas();
  };
  caseImage.onerror = () => {
    toast("The selected JPEG could not be loaded.", "error");
    $("#canvas-empty").classList.remove("hidden");
    $("#canvas-empty p").textContent = "Image loading failed.";
  };
  caseImage.src = `/image/${encodeURIComponent(state.currentCase.id)}`;
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(
      0,
      Math.min(
        state.currentCase.image_width,
        ((event.clientX - rect.left) / rect.width) * canvas.width,
      ),
    ),
    y: Math.max(
      0,
      Math.min(
        state.currentCase.image_height,
        ((event.clientY - rect.top) / rect.height) * canvas.height,
      ),
    ),
  };
}

function boxContains(box, point) {
  return (
    point.x >= box.x &&
    point.x <= box.x + box.width &&
    point.y >= box.y &&
    point.y <= box.y + box.height
  );
}

function handlePoints(box) {
  const x1 = box.x;
  const x2 = box.x + box.width;
  const y1 = box.y;
  const y2 = box.y + box.height;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return {
    nw: [x1, y1],
    n: [mx, y1],
    ne: [x2, y1],
    e: [x2, my],
    se: [x2, y2],
    s: [mx, y2],
    sw: [x1, y2],
    w: [x1, my],
  };
}

function hitHandle(box, point) {
  const threshold = 8 / state.zoom;
  for (const [name, [x, y]] of Object.entries(handlePoints(box))) {
    if (Math.abs(point.x - x) <= threshold && Math.abs(point.y - y) <= threshold) {
      return name;
    }
  }
  return null;
}

function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.annotation || !state.currentCase) return;
  const inverse = 1 / state.zoom;
  for (const [index, box] of state.annotation.boxes.entries()) {
    const selected = box.id === state.selectedBoxId;
    const color = classInfo(box.class_name).color;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = `${color}${selected ? "2b" : "16"}`;
    ctx.lineWidth = (selected ? 3 : 2) * inverse;
    if (box.uncertain) ctx.setLineDash([7 * inverse, 5 * inverse]);
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.setLineDash([]);

    const fontSize = 11 * inverse;
    ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
    const label = `${index + 1} · ${box.class_name}${box.uncertain ? " ?" : ""}`;
    const textWidth = ctx.measureText(label).width;
    const labelHeight = 18 * inverse;
    const labelY = Math.max(0, box.y - labelHeight);
    ctx.fillStyle = color;
    ctx.fillRect(box.x, labelY, textWidth + 9 * inverse, labelHeight);
    ctx.fillStyle = color.toLowerCase() === "#f2f2f2" ? "#17211f" : "#ffffff";
    ctx.fillText(label, box.x + 4 * inverse, labelY + 13 * inverse);

    if (selected) {
      const size = 7 * inverse;
      for (const [x, y] of Object.values(handlePoints(box))) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 * inverse;
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
        ctx.strokeRect(x - size / 2, y - size / 2, size, size);
      }
    }
    ctx.restore();
  }
}

function newBoxId() {
  if (globalThis.crypto?.randomUUID) {
    return `box-${crypto.randomUUID().replaceAll("-", "").slice(0, 16)}`;
  }
  return `box-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function selectBox(boxId) {
  state.selectedBoxId = boxId;
  const box = currentBox();
  if (box) state.activeClass = box.class_name;
  renderBoxList();
  renderCanvas();
}

function markDirty() {
  if (!state.annotation || state.readOnly) return;
  state.dirty = true;
  state.editVersion += 1;
  state.serverBlockers = [];
  if (state.annotation.status !== "in_progress") {
    state.annotation.status = "in_progress";
  }
  setSaveState("Unsaved changes");
  renderCompleteness();
  scheduleAutosave();
}

function scheduleAutosave() {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => {
    if (state.dirty && !state.interaction) saveAnnotation("in_progress");
  }, 1100);
}

function resizeBox(original, handle, dx, dy) {
  let left = original.x;
  let right = original.x + original.width;
  let top = original.y;
  let bottom = original.y + original.height;
  if (handle.includes("w")) left += dx;
  if (handle.includes("e")) right += dx;
  if (handle.includes("n")) top += dy;
  if (handle.includes("s")) bottom += dy;
  left = Math.max(0, Math.min(left, right - 2));
  right = Math.min(state.currentCase.image_width, Math.max(right, left + 2));
  top = Math.max(0, Math.min(top, bottom - 2));
  bottom = Math.min(state.currentCase.image_height, Math.max(bottom, top + 2));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.annotation || state.readOnly || state.annotation.no_foreign_objects) return;
  event.preventDefault();
  const point = pointFromEvent(event);
  const selected = currentBox();
  const handle = selected ? hitHandle(selected, point) : null;
  if (handle) {
    state.interaction = {
      type: "resize",
      handle,
      start: point,
      boxId: selected.id,
      original: deepCopy(selected),
    };
  } else {
    const hit = [...state.annotation.boxes].reverse().find((box) => boxContains(box, point));
    if (hit) {
      selectBox(hit.id);
      state.interaction = {
        type: "move",
        start: point,
        boxId: hit.id,
        original: deepCopy(hit),
      };
    } else {
      const box = {
        id: newBoxId(),
        class_name: state.activeClass,
        x: point.x,
        y: point.y,
        width: 0,
        height: 0,
        difficult: false,
        uncertain: false,
      };
      state.annotation.boxes.push(box);
      state.selectedBoxId = box.id;
      state.interaction = {
        type: "draw",
        start: point,
        boxId: box.id,
      };
    }
  }
  canvas.setPointerCapture(event.pointerId);
  renderCanvas();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.interaction) return;
  const point = pointFromEvent(event);
  const box = state.annotation.boxes.find(
    (item) => item.id === state.interaction.boxId,
  );
  if (!box) return;
  const dx = point.x - state.interaction.start.x;
  const dy = point.y - state.interaction.start.y;
  if (state.interaction.type === "draw") {
    box.x = Math.min(point.x, state.interaction.start.x);
    box.y = Math.min(point.y, state.interaction.start.y);
    box.width = Math.abs(point.x - state.interaction.start.x);
    box.height = Math.abs(point.y - state.interaction.start.y);
  } else if (state.interaction.type === "move") {
    const original = state.interaction.original;
    box.x = Math.max(
      0,
      Math.min(state.currentCase.image_width - original.width, original.x + dx),
    );
    box.y = Math.max(
      0,
      Math.min(state.currentCase.image_height - original.height, original.y + dy),
    );
  } else if (state.interaction.type === "resize") {
    Object.assign(
      box,
      resizeBox(state.interaction.original, state.interaction.handle, dx, dy),
    );
  }
  renderCanvas();
});

function finishInteraction(event) {
  if (!state.interaction) return;
  const interaction = state.interaction;
  const box = state.annotation.boxes.find((item) => item.id === interaction.boxId);
  if (interaction.type === "draw" && box && (box.width <= 2 || box.height <= 2)) {
    state.annotation.boxes = state.annotation.boxes.filter(
      (item) => item.id !== box.id,
    );
    state.selectedBoxId = null;
  } else if (box) {
    markDirty();
  }
  state.interaction = null;
  if (event && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  renderAnnotation();
}

canvas.addEventListener("pointerup", finishInteraction);
canvas.addEventListener("pointercancel", finishInteraction);

async function saveAnnotation(status = "in_progress", { quiet = false } = {}) {
  if (!state.annotation || state.readOnly) return false;
  clearTimeout(state.autosaveTimer);
  if (state.savePromise) await state.savePromise;
  const caseId = state.currentCase.id;
  const expectedRevision = state.annotation.revision;
  const outgoing = deepCopy(state.annotation);
  outgoing.status = status;
  const sentEditVersion = state.editVersion;
  state.saving = true;
  setSaveState("Saving…", "saving");
  renderCompleteness();

  state.savePromise = (async () => {
    try {
      const payload = await fetchJson(
        `/api/annotation/${encodeURIComponent(caseId)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            expected_revision: expectedRevision,
            annotation: outgoing,
          }),
        },
      );
      if (state.currentCase?.id !== caseId) return true;
      const fullySaved = state.editVersion === sentEditVersion;
      if (fullySaved) {
        state.annotation = payload.annotation;
        state.dirty = false;
      } else {
        // Preserve edits made while the request was in flight, but advance the
        // optimistic revision so the follow-up autosave cannot conflict with
        // the just-completed write.
        state.annotation.revision = payload.annotation.revision;
        state.annotation.created_at = payload.annotation.created_at;
        state.annotation.updated_at = payload.annotation.updated_at;
        state.annotation.status = "in_progress";
        state.dirty = true;
        scheduleAutosave();
      }
      state.serverBlockers = [];
      updateCaseLocalStatus(payload.annotation);
      setSaveState(
        fullySaved
          ? `${statusLabel(payload.annotation.status)} · saved`
          : "Unsaved changes",
        fullySaved ? "saved" : "",
      );
      renderQueue();
      renderAnnotation();
      updateHeaderProgressLocally();
      if (!quiet) toast(`${statusLabel(payload.annotation.status)} saved.`);
      return true;
    } catch (error) {
      if (state.currentCase?.id !== caseId) return false;
      state.serverBlockers = error.payload?.details || [];
      setSaveState("Save failed", "error");
      if (error.status === 409) {
        const reload = window.confirm(
          "This case was changed in another browser tab. Reload the saved version? " +
            "Cancel keeps your unsaved local version.",
        );
        if (reload) await loadCase(caseId, { skipSave: true });
      } else {
        toast(error.message, error.status === 422 ? "warning" : "error");
        renderCompleteness();
      }
      return false;
    } finally {
      state.saving = false;
      state.savePromise = null;
      renderCompleteness();
    }
  })();
  return state.savePromise;
}

async function flushDraft() {
  clearTimeout(state.autosaveTimer);
  if (state.savePromise) await state.savePromise;
  if (state.dirty) return saveAnnotation("in_progress", { quiet: true });
  return true;
}

async function loadCase(caseId, { skipSave = false } = {}) {
  if (!skipSave && state.currentCase?.id !== caseId) {
    const saved = await flushDraft();
    if (!saved) return;
  }
  const summary = state.caseMap.get(caseId);
  if (!summary && !skipSave) return;
  setSaveState("Loading…", "saving");
  try {
    const payload = await fetchJson(`/api/case/${encodeURIComponent(caseId)}`);
    state.currentCase = payload.case;
    state.annotation = payload.annotation;
    state.editVersion = 0;
    state.readOnly = payload.read_only;
    state.selectedBoxId = state.annotation.boxes[0]?.id || null;
    state.activeClass = state.annotation.boxes[0]?.class_name || state.activeClass;
    state.dirty = false;
    state.serverBlockers = [];
    state.interaction = null;
    buildCaseHeading();
    loadImage();
    renderAnnotation();
    setSaveState(
      state.readOnly
        ? "Guide example · read only"
        : `${statusLabel(state.annotation.status)} · revision ${state.annotation.revision}`,
      state.annotation.status === "complete" ? "saved" : "",
    );
    renderQueue();
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Load failed", "error");
  }
}

function findNextActionable() {
  if (!state.cases.length) return null;
  const currentRank = state.currentCase?.rank || 0;
  const ordered = [
    ...state.cases.filter((item) => item.rank > currentRank),
    ...state.cases.filter((item) => item.rank <= currentRank),
  ];
  return (
    ordered.find((item) => ["unstarted", "in_progress"].includes(item.status)) ||
    ordered.find((item) => item.status === "needs_review") ||
    null
  );
}

async function nextActionable() {
  const next = findNextActionable();
  if (!next) {
    toast("No incomplete cases remain.");
    return;
  }
  $("#batch-filter").value = String(next.batch);
  $("#status-filter").value = "all";
  $("#dataset-filter").value = "all";
  $("#queue-search").value = "";
  state.queueLimit = 300;
  renderQueue();
  await loadCase(next.id);
}

function deleteSelectedBox() {
  if (!state.selectedBoxId || state.readOnly) return;
  state.annotation.boxes = state.annotation.boxes.filter(
    (box) => box.id !== state.selectedBoxId,
  );
  state.selectedBoxId = state.annotation.boxes[0]?.id || null;
  markDirty();
  renderAnnotation();
}

function updateHeaderProgressLocally() {
  const complete = state.cases.filter((item) => item.status === "complete").length;
  const review = state.cases.filter((item) => item.status === "needs_review").length;
  const total = state.cases.length || 4000;
  $("#header-progress").innerHTML =
    `<strong>${complete.toLocaleString()}</strong> / ${total.toLocaleString()} complete` +
    (review ? `<br><span class="muted">${review} awaiting review</span>` : "");
}

function renderGuide() {
  const examples = state.meta.guide_examples;
  $("#class-guide").innerHTML = state.meta.classes
    .map((item) => {
      const ids = examples[item.name] || [];
      const exampleMarkup = ids.length
        ? `<div class="example-grid">${ids
            .map(
              (id) => `
              <button class="example-button" data-guide-case="${escapeHtml(id)}"
                      title="Open released-data example and QA">
                <img src="/image/${encodeURIComponent(id)}" loading="lazy"
                     alt="Released training example for ${escapeHtml(item.name)}">
              </button>`,
            )
            .join("")}</div>`
        : `<div class="no-examples">No verified example exists in the released Frame
             training QA for this class. Do not infer appearance from an unrelated
             surgical material; use the clinical reference and mark uncertainty.</div>`;
      const external = item.external_reference
        ? `<p><a class="external-link" href="${escapeHtml(
            item.external_reference.url,
          )}" target="_blank" rel="noopener noreferrer">${escapeHtml(
            item.external_reference.label,
          )} ↗</a></p>`
        : "";
      const remoteImage = item.external_image
        ? `<a class="example-button" style="display:block;max-width:180px;margin-top:8px"
              href="${escapeHtml(item.external_reference.url)}" target="_blank"
              rel="noopener noreferrer" title="External clinical product reference">
             <img src="${escapeHtml(item.external_image)}"
                  alt="External clinical appearance reference for ${escapeHtml(item.name)}">
           </a>`
        : "";
      return `
        <article class="guide-card" style="--class-color:${escapeHtml(item.color)}">
          <div class="guide-accent"></div>
          <div class="guide-card-body">
            <div class="guide-title">
              <h3>${escapeHtml(item.name)}</h3>
              <span class="color-chip" style="background:${escapeHtml(item.color)}"></span>
            </div>
            <p>${escapeHtml(item.description)}</p>
            <div class="guide-facts">
              <div class="guide-fact"><strong>Look for</strong>${escapeHtml(
                item.look_for,
              )}</div>
              <div class="guide-fact"><strong>Do not count / confuse</strong>${escapeHtml(
                item.do_not_count,
              )}</div>
            </div>
            <div class="confusion-line"><strong>Common confusions:</strong>
              ${escapeHtml((item.confusions || []).join(" · ") || "—")}</div>
            ${exampleMarkup}
            ${external}
            ${remoteImage}
          </div>
        </article>`;
    })
    .join("");
}

async function openGuideCase(caseId) {
  try {
    const payload = await fetchJson(`/api/case/${encodeURIComponent(caseId)}`);
    const caseItem = payload.case;
    $("#dialog-content").innerHTML = `
      <img class="dialog-image" src="/image/${encodeURIComponent(caseId)}"
           alt="Released training example">
      <div class="dialog-info">
        <div class="eyebrow">${escapeHtml(caseItem.dataset_name)} · released training data</div>
        <h3>${escapeHtml(caseItem.video)} · ${escapeHtml(caseItem.timestamp)}</h3>
        ${caseItem.questions
          .map(
            (question) => `<div class="dialog-qa">
              <strong>${escapeHtml(question.answer)}</strong><br>
              ${escapeHtml(question.question)}
            </div>`,
          )
          .join("")}
      </div>`;
    $("#guide-dialog").showModal();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderProgress() {
  const progress = state.progress;
  if (!progress) return;
  const statuses = progress.statuses;
  const values = [
    ["Complete", statuses.complete, "var(--success)"],
    ["Needs review", statuses.needs_review, "var(--warning)"],
    ["In progress", statuses.in_progress, "#377ba8"],
    ["Unstarted", statuses.unstarted, "#7d8985"],
    ["Skipped", statuses.skipped, "#806a85"],
  ];
  $("#progress-cards").innerHTML = values
    .map(
      ([label, value, color]) => `<article class="progress-card">
        <div class="number" style="color:${color}">${Number(value).toLocaleString()}</div>
        <div class="label">${label}</div>
      </article>`,
    )
    .join("");
  $("#batch-progress").innerHTML = progress.batches
    .map((batch) => {
      const percent = Math.round((batch.complete / batch.size) * 100);
      const start = (batch.batch - 1) * 100 + 1;
      return `<tr data-progress-batch="${batch.batch}">
        <td><strong>Batch ${batch.batch}</strong></td>
        <td>${start}–${start + batch.size - 1}</td>
        <td>${batch.complete}</td>
        <td>${batch.needs_review}</td>
        <td>${batch.in_progress}</td>
        <td>${batch.unstarted}</td>
        <td><div style="display:flex;align-items:center;gap:8px">
          <div class="progress-bar"><span style="width:${percent}%"></span></div>
          <span>${percent}%</span></div></td>
      </tr>`;
    })
    .join("");
  $("#dataset-progress").innerHTML = Object.entries(progress.datasets)
    .map(([dataset, item]) => {
      const size =
        item.complete +
        item.needs_review +
        item.in_progress +
        item.unstarted +
        item.skipped;
      const percent = Math.round((item.complete / size) * 100);
      return `<article class="dataset-card">
        <div class="section-title-row"><strong>${
          dataset === "heico" ? "HeiCo" : "LapChole"
        }</strong><span>${item.complete}/${size} complete</span></div>
        <div class="progress-bar" style="width:100%;margin-top:9px">
          <span style="width:${percent}%"></span></div>
        <div class="muted" style="font-size:10px;margin-top:7px">
          ${item.needs_review} review · ${item.in_progress} draft ·
          ${item.unstarted} unstarted · ${item.skipped} skipped
        </div>
      </article>`;
    })
    .join("");
}

async function refreshProgress(showError = true) {
  try {
    state.progress = await fetchJson("/api/progress");
    renderProgress();
    updateHeaderProgressLocally();
  } catch (error) {
    if (showError) toast(error.message, "error");
  }
}

async function createExport() {
  const button = $("#create-export");
  button.disabled = true;
  button.textContent = "Exporting…";
  try {
    const result = await fetchJson("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const panel = $("#export-result");
    panel.classList.remove("hidden");
    panel.innerHTML = `<strong>Export created.</strong><br>
      ${result.complete_case_count.toLocaleString()} complete cases and
      ${result.complete_box_count.toLocaleString()} boxes<br>
      <span class="mono">${escapeHtml(result.directory)}</span>`;
    toast("Training export created.");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Create training export";
  }
}

function bindEvents() {
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => switchView(tab.dataset.view)),
  );
  for (const id of ["batch-filter", "dataset-filter", "status-filter"]) {
    $(`#${id}`).addEventListener("change", () => {
      state.queueLimit = 300;
      renderQueue();
    });
  }
  $("#queue-search").addEventListener("input", () => {
    state.queueLimit = 300;
    renderQueue();
  });
  $("#case-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-case-id]");
    if (button) loadCase(button.dataset.caseId);
  });
  $("#show-more").addEventListener("click", () => {
    state.queueLimit += 300;
    renderQueue();
  });
  $("#queue-refresh").addEventListener("click", refreshPoolStatuses);
  $("#box-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-box-id]");
    if (button) selectBox(button.dataset.boxId);
  });
  $("#class-select").addEventListener("change", (event) => {
    state.activeClass = event.target.value;
    const box = currentBox();
    if (box) {
      box.class_name = event.target.value;
      markDirty();
      renderAnnotation();
    }
  });
  $("#box-difficult").addEventListener("change", (event) => {
    const box = currentBox();
    if (!box) return;
    box.difficult = event.target.checked;
    markDirty();
    renderAnnotation();
  });
  $("#box-uncertain").addEventListener("change", (event) => {
    const box = currentBox();
    if (!box) return;
    box.uncertain = event.target.checked;
    markDirty();
    renderAnnotation();
  });
  $("#delete-box").addEventListener("click", deleteSelectedBox);
  $("#no-objects").addEventListener("change", (event) => {
    if (event.target.checked && state.annotation.boxes.length) {
      const confirmed = window.confirm(
        "Choosing “No foreign objects” will remove every box in this case. Continue?",
      );
      if (!confirmed) {
        event.target.checked = false;
        return;
      }
      state.annotation.boxes = [];
      state.selectedBoxId = null;
    }
    state.annotation.no_foreign_objects = event.target.checked;
    markDirty();
    renderAnnotation();
  });
  $$(".completion-checks input").forEach((input) =>
    input.addEventListener("change", () => {
      state.annotation.checks[input.dataset.check] = input.checked;
      markDirty();
      renderCompleteness();
    }),
  );
  $("#case-notes").addEventListener("input", (event) => {
    state.annotation.notes = event.target.value;
    markDirty();
  });
  $("#save-draft").addEventListener("click", () =>
    saveAnnotation("in_progress"),
  );
  $("#needs-review").addEventListener("click", async () => {
    await saveAnnotation("needs_review");
  });
  $("#mark-complete").addEventListener("click", async () => {
    const saved = await saveAnnotation("complete");
    if (saved) await nextActionable();
  });
  $("#skip-case").addEventListener("click", async () => {
    if (!state.annotation.notes.trim()) {
      toast("Add a skip reason in Notes first.", "warning");
      $("#case-notes").focus();
      return;
    }
    const saved = await saveAnnotation("skipped");
    if (saved) await nextActionable();
  });
  $("#zoom-range").addEventListener("input", (event) =>
    setZoom(Number(event.target.value) / 100),
  );
  $("#zoom-in").addEventListener("click", () => setZoom(state.zoom + 0.1));
  $("#zoom-out").addEventListener("click", () => setZoom(state.zoom - 0.1));
  $("#zoom-fit").addEventListener("click", fitImage);
  $("#class-guide").addEventListener("click", (event) => {
    const button = event.target.closest("[data-guide-case]");
    if (button) openGuideCase(button.dataset.guideCase);
  });
  $("#close-dialog").addEventListener("click", () => $("#guide-dialog").close());
  $("#guide-dialog").addEventListener("click", (event) => {
    if (event.target === $("#guide-dialog")) $("#guide-dialog").close();
  });
  $("#batch-progress").addEventListener("click", (event) => {
    const row = event.target.closest("[data-progress-batch]");
    if (!row) return;
    $("#batch-filter").value = row.dataset.progressBatch;
    $("#dataset-filter").value = "all";
    $("#status-filter").value = "all";
    $("#queue-search").value = "";
    state.queueLimit = 300;
    renderQueue();
    switchView("annotate");
    const first = state.filteredCases.find((item) => item.status !== "complete");
    if (first) loadCase(first.id);
  });
  $("#create-export").addEventListener("click", createExport);
  window.addEventListener("resize", () => {
    if (state.currentCase && state.zoom <= 1.5) fitImage();
  });
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty || state.saving) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  document.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(
      document.activeElement?.tagName,
    );
    if ((event.key === "Delete" || event.key === "Backspace") && !typing) {
      event.preventDefault();
      deleteSelectedBox();
    } else if (event.key.toLowerCase() === "n" && !typing) {
      event.preventDefault();
      nextActionable();
    } else if (event.key.toLowerCase() === "s" && !typing) {
      event.preventDefault();
      saveAnnotation("in_progress");
    } else if (event.key === "Escape") {
      selectBox(null);
    }
  });
}

async function init() {
  try {
    const [meta, pool, progress] = await Promise.all([
      fetchJson("/api/meta"),
      fetchJson("/api/pool"),
      fetchJson("/api/progress"),
    ]);
    state.meta = meta;
    state.cases = pool.cases;
    state.caseMap = new Map(state.cases.map((item) => [item.id, item]));
    state.progress = progress;
    populateStaticControls();
    bindEvents();
    renderGuide();
    renderProgress();
    renderQueue();
    updateHeaderProgressLocally();
    const first =
      state.cases.find(
        (item) => item.batch === 1 && ["unstarted", "in_progress"].includes(item.status),
      ) || state.cases[0];
    if (first) await loadCase(first.id, { skipSave: true });
  } catch (error) {
    console.error(error);
    toast(`Cannot start annotation tool: ${error.message}`, "error");
    $("#case-heading").innerHTML = `
      <div><div class="eyebrow">Startup error</div>
      <h2>${escapeHtml(error.message)}</h2></div>`;
  }
}

init();
