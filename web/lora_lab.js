import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "LoRALens.V7";
const SIDEBAR_ID = "lora-lab-v5";
const DIRECT_OPEN_KEY = "loralab.directOpen";
const PROMPT_SUITES_KEY = "loralab.promptSuites.v1";

const OBJECTIVES = [
  { id: "best_checkpoint", name: "Best checkpoint", icon: "pi-chart-line", description: "Coarse matched screening across saved checkpoints.", requirement: "Select one or more checkpoints" },
  { id: "best_strength", name: "Best strength", icon: "pi-sliders-h", description: "One checkpoint tested across matched LoRA strengths.", requirement: "Select exactly one checkpoint" },
  { id: "raw_vs_turbo", name: "Raw vs Turbo", icon: "pi-bolt", description: "Raw and Raw+Turbo variants inside one paired run.", requirement: "Select exactly one identity LoRA" },
  { id: "enhancer", name: "Enhancer", icon: "pi-sparkles", description: "Enhancer Off, Standard and Advanced in one run.", requirement: "Select exactly one identity LoRA" },
  { id: "overfit", name: "Diagnose overfit", icon: "pi-exclamation-triangle", description: "Broader pose/style coverage across checkpoint curve.", requirement: "Select several checkpoints" },
  { id: "final", name: "Final validation", icon: "pi-verified", description: "Native 1024 Raw verification with three matched seeds.", requirement: "Select top 2–3 finalists" },
];

const state = {
  boot: null,
  bootPromise: null,
  overlay: null,
  page: "setup",
  busy: "",
  error: "",
  selected: new Set(),
  search: "",
  group: "",
  everyN: 1,
  stepMin: "",
  stepMax: "",
  currentRunId: "",
  runData: null,
  prompts: [],
  promptMode: "preset",
  promptImportText: "",
  promptSuiteName: "",
  promptSuites: {},
  reuseRunId: "",
  showAdvancedStack: false,
  objective: "best_checkpoint",
  viewer: null,
  blind: true,
  matrixScenario: "all",
  matrixCandidate: "all",
  matrixCategory: "all",
  revealAutomatic: false,
  directOpen: localStorage.getItem(DIRECT_OPEN_KEY) !== "false",
  modelCategory: "all",
  workflowAdapter: "native",
  apiWorkflow: "",
  apiOutputNodeId: "",
  apiOutputIndex: 0,
  form: {
    profile: "krea2_turbo",
    preset: "quick",
    mode: "compare",
    trigger: "",
    subjectClass: "",
    includeBaseline: true,
    commonStrength: 1.0,
    strengths: "0.65, 0.80, 0.95, 1.10, 1.25",
    seeds: "20260710",
    width: 768,
    height: 768,
    referenceFolder: "lora_reference",
    negativePrompt: "deformed face, duplicate person, malformed hands",
    gridMode: "off",
    outputPrefix: "LoRA_Lens",
    modelName: "",
    clipName: "",
    clipName2: "",
    vaeName: "",
    turboLora: false,
    turboLoraName: "",
    turboLoraStrength: 1.0,
    auxLoras: [],
    steps: 8,
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    negativeMode: "zero",
    enhancer: "off",
    enhancerStrength: 1.0,
    enhancerTextScale: 1.5,
    customPatches: "[]",
  },
};

let pollTimer = null;

function ensureCss() {
  if (document.querySelector("link[data-loralab-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./lora_lab.css", import.meta.url).href;
  link.dataset.loralabCss = "1";
  document.head.appendChild(link);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function help(text) {
  return `<span class="ll-help" data-tip="${esc(text)}">?</span>`;
}

function formatSeconds(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  seconds = Math.max(0, Math.round(Number(seconds)));
  if (seconds < 60) return `${seconds}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function shortFile(filename, max = 42) {
  const text = String(filename || "").replace(/\.safetensors$/i, "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function aliasFor(index, baseline = false) {
  if (baseline) return "Control";
  let number = index;
  let label = "";
  do {
    label = String.fromCharCode(65 + (number % 26)) + label;
    number = Math.floor(number / 26) - 1;
  } while (number >= 0);
  return `Candidate ${label}`;
}

async function jsonFetch(path, options = {}) {
  const init = { ...options };
  init.headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const response = await api.fetchApi(path, init);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status}: invalid JSON response`);
  }
  if (!response.ok || data?.ok === false) throw new Error(data?.error || `HTTP ${response.status}`);
  return data;
}

function toast(summary, detail = "", severity = "info") {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: severity === "error" ? 7000 : 3500 });
  } catch {
    console[severity === "error" ? "error" : "log"](`[LoRA Lab] ${summary}: ${detail}`);
  }
}

async function confirmAction(title, message) {
  try {
    return await app.extensionManager.dialog.confirm({ title, message });
  } catch {
    return window.confirm(`${title}\n\n${message}`);
  }
}

function currentProfile() {
  return state.boot?.profiles?.find((item) => item.id === state.form.profile) || null;
}

function parseNumbers(text, fallback = []) {
  const values = String(text || "")
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite);
  return values.length ? [...new Set(values)] : fallback;
}

function applyProfile(profileId) {
  const profile = state.boot?.profiles?.find((item) => item.id === profileId);
  if (!profile) return;
  state.form.profile = profileId;
  state.form.steps = profile.steps;
  state.form.cfg = profile.cfg;
  state.form.sampler = profile.sampler;
  state.form.scheduler = profile.scheduler;
  state.form.negativeMode = profile.negative_mode;
  if (profile.installed_model) state.form.modelName = profile.installed_model;
  state.form.clipName = profile.installed_clip || state.form.clipName || state.boot?.text_encoders?.[0] || "";
  state.form.clipName2 = profile.installed_clip_2 || "";
  state.form.vaeName = profile.installed_vae || state.form.vaeName || state.boot?.vaes?.[0] || "";
  if (!profile.supports_acceleration_lora) state.form.turboLora = false;
  if (profile.family !== "Krea 2") {
    state.form.enhancer = "off";
    if (["raw_vs_turbo", "enhancer"].includes(state.objective)) { state.objective = "best_checkpoint"; state.form.mode = "compare"; }
  }
  if (state.form.turboLora) applyTurboDefaults();
}

function profileCategories() {
  return [...new Set((state.boot?.profiles || []).map((item) => item.category || "Other"))].sort();
}

function profileOptions() {
  const profiles = (state.boot?.profiles || []).filter((item) => state.modelCategory === "all" || item.category === state.modelCategory);
  const families = [...new Set(profiles.map((item) => item.family))].sort();
  return families.map((family) => `<optgroup label="${esc(family)}">${profiles.filter((item) => item.family === family).map((item) => `<option value="${esc(item.id)}" ${item.id === state.form.profile ? "selected" : ""}>${esc(item.variant)} · ${esc(item.acceleration)}${item.available ? "" : " · prerequisites missing"}</option>`).join("")}</optgroup>`).join("");
}

function applyTurboDefaults() {
  state.form.steps = 8;
  state.form.cfg = 1;
  state.form.sampler = "euler";
  state.form.scheduler = "beta";
  state.form.negativeMode = "zero";
}

function applyPreset(presetId, resetPrompts = true) {
  const preset = state.boot?.presets?.find((item) => item.id === presetId);
  if (!preset) return;
  state.form.preset = presetId;
  state.form.seeds = preset.seeds.join(", ");
  state.form.width = preset.width;
  state.form.height = preset.height;
  if (resetPrompts && state.promptMode === "preset") {
    state.prompts = state.boot.default_prompts
      .slice(0, preset.prompt_count)
      .map((item) => ({ ...item, enabled: true }));
  }
}

function applyObjective(objectiveId) {
  state.objective = objectiveId;
  if (objectiveId === "best_checkpoint") {
    state.form.mode = "compare"; applyPreset("quick", true);
  } else if (objectiveId === "best_strength") {
    state.form.mode = "strength"; applyPreset("standard", true);
  } else if (objectiveId === "raw_vs_turbo") {
    state.form.mode = "stack_compare"; applyProfile("krea2_raw"); applyPreset("quick", true);
    state.form.modelName = state.boot.diffusion_models.find((name) => /krea2_raw_fp8_scaled/i.test(name)) || state.boot.diffusion_models.find((name) => /krea2.*raw/i.test(name)) || state.form.modelName;
    applyTurboDefaults(); state.form.turboLora = false; state.form.enhancer = "off";
  } else if (objectiveId === "enhancer") {
    state.form.mode = "enhancer_compare"; applyProfile("krea2_raw"); applyPreset("quick", true); state.form.turboLora = false; state.form.enhancer = "off";
  } else if (objectiveId === "overfit") {
    state.form.mode = "compare"; applyPreset("standard", true);
  } else if (objectiveId === "final") {
    state.form.mode = "compare"; applyPreset("deep", true);
  }
}

function visibleObjectives() {
  const family = currentProfile()?.family;
  return OBJECTIVES.filter((item) => !["raw_vs_turbo", "enhancer"].includes(item.id) || family === "Krea 2");
}

async function loadBootstrap(force = false) {
  if (state.boot && !force) return state.boot;
  if (state.bootPromise && !force) return state.bootPromise;
  state.bootPromise = jsonFetch("/loralab/v1/bootstrap")
    .then((data) => {
      state.boot = data;
      try { state.promptSuites = JSON.parse(localStorage.getItem(PROMPT_SUITES_KEY) || "{}"); } catch { state.promptSuites = {}; }
      if (!state.form.turboLoraName) state.form.turboLoraName = data.defaults?.turbo_lora || "";
      if (!state.prompts.length) {
        state.form.trigger = data.defaults?.trigger || state.form.trigger;
        state.form.subjectClass = data.defaults?.subject_class || "";
        state.form.referenceFolder = data.defaults?.reference_folder || state.form.referenceFolder;
        state.form.turboLoraName = data.defaults?.turbo_lora || "";
        applyProfile(data.defaults?.profile || "krea2_turbo");
        applyPreset("quick", true);
      }
      return data;
    })
    .finally(() => { state.bootPromise = null; });
  return state.bootPromise;
}

function headerStatus() {
  const boot = state.boot;
  if (!boot) return "";
  const analyzer = boot.analyzer || {};
  const analyzerClass = analyzer.state === "ready" ? "good" : analyzer.state === "error" ? "bad" : "warn";
  const modelReady = (boot.diffusion_models || []).includes(state.form.modelName);
  return `
    <span class="ll-badge"><span class="ll-dot"></span>${esc(boot.hardware.gpu)} · ${esc(boot.hardware.vram_gb)} GB</span>
    <span class="ll-badge ${modelReady ? "good" : "bad"}">${modelReady ? "Model ready" : "Model missing"}</span>
    <span class="ll-badge ${analyzerClass}">Analyzer ${esc(analyzer.state || "unknown")}</span>
  `;
}

function renderShell() {
  if (!state.overlay) return;
  const pages = [
    ["setup", "pi pi-sliders-h", "Test setup"],
    ["run", "pi pi-bolt", "Run monitor"],
    ["results", "pi pi-chart-line", "Results"],
    ["history", "pi pi-history", "History"],
  ];
  state.overlay.innerHTML = `
    <div class="ll-shell" role="dialog" aria-modal="true" aria-label="LoRA Lab">
      <header class="ll-header">
        <div class="ll-mark">L</div>
        <div class="ll-titlebox"><div class="ll-title">LoRA Lens</div><div class="ll-subtitle">Model-aware LoRA checkpoint evaluation · v${esc(state.boot?.version || "7")}</div></div>
        <div class="ll-header-spacer"></div>
        ${headerStatus()}
        <button class="ll-icon-btn" id="ll-refresh" title="Refresh catalog and runs"><i class="pi pi-refresh"></i></button>
        <button class="ll-icon-btn" id="ll-close" title="Close LoRA Lab"><i class="pi pi-times"></i></button>
      </header>
      <nav class="ll-tabs">
        ${pages.map(([id, icon, label]) => `<button class="ll-tab ${state.page === id ? "active" : ""}" data-page="${id}"><i class="${icon}"></i>${label}</button>`).join("")}
      </nav>
      <main class="ll-main" id="ll-content">${renderPage()}</main>
    </div>
  `;
  bindShell();
  bindPage();
}

function renderPage() {
  if (state.error) {
    const error = `<div class="ll-error">${esc(state.error)}</div>`;
    state.error = "";
    return `<div class="ll-page narrow">${error}${renderPageBody()}</div>`;
  }
  return renderPageBody();
}

function renderPageBody() {
  if (!state.boot) return `<div class="ll-empty"><i class="pi pi-spin pi-spinner"></i><br>Loading LoRA catalog…</div>`;
  if (state.busy) return `<div class="ll-empty"><i class="pi pi-spin pi-spinner"></i><br>${esc(state.busy)}</div>`;
  if (state.page === "run") return renderMonitor();
  if (state.page === "results") return renderResults();
  if (state.page === "history") return renderHistory();
  return renderSetup();
}

function selectedCandidates() {
  return (state.boot?.loras || []).filter((item) => state.selected.has(item.filename));
}

function visibleCandidates() {
  const query = state.search.trim().toLowerCase();
  return (state.boot?.loras || []).filter((item) => {
    if (state.group && item.group !== state.group) return false;
    if (!query) return true;
    return `${item.filename} ${item.group} ${item.step ?? ""}`.toLowerCase().includes(query);
  });
}

function estimate() {
  const profile = currentProfile();
  const promptCount = state.prompts.filter((item) => item.enabled !== false && item.text.trim()).length;
  const seedCount = parseNumbers(state.form.seeds, [20260710]).length;
  let candidateCount;
  if (state.form.mode === "strength") candidateCount = state.selected.size === 1 ? parseNumbers(state.form.strengths, [1]).length : 0;
  else if (state.form.mode === "stack_compare") candidateCount = state.selected.size === 1 ? 2 + (state.form.includeBaseline ? 2 : 0) : 0;
  else if (state.form.mode === "enhancer_compare") candidateCount = state.selected.size === 1 ? 3 + (state.form.includeBaseline ? 1 : 0) : 0;
  else candidateCount = state.selected.size;
  if (!["stack_compare","enhancer_compare"].includes(state.form.mode) && state.form.includeBaseline) candidateCount += 1;
  const jobs = promptCount * seedCount * candidateCount;
  const scale = Number(state.form.width) * Number(state.form.height) / (1024 * 1024);
  const defaultSteps = Number(profile?.steps || state.form.steps || 1);
  const perCell = (state.form.turboLora || state.form.mode === "stack_compare") && Number(state.form.steps) === 8
    ? 28
    : Number(profile?.seconds_per_cell_4090 || 35) * Number(state.form.steps || 1) / Math.max(1, defaultSteps);
  const seconds = jobs * perCell * scale;
  const storage = Math.ceil(jobs * Number(state.form.width) * Number(state.form.height) * 3 / 1024 / 1024 * .42);
  return { jobs, seconds, storage, promptCount, seedCount, candidateCount };
}

function groupOptions() {
  const counts = new Map();
  for (const item of state.boot.loras || []) counts.set(item.group, (counts.get(item.group) || 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function renderCandidates() {
  const rows = visibleCandidates();
  if (!rows.length) return `<div class="ll-empty">No LoRA matches current filters.</div>`;
  return rows.map((item) => `
    <label class="ll-candidate ${state.selected.has(item.filename) ? "selected" : ""}" title="${esc(item.filename)}">
      <input type="checkbox" class="ll-candidate-check" data-file="${esc(item.filename)}" ${state.selected.has(item.filename) ? "checked" : ""}>
      <span class="ll-step">${item.step == null ? "—" : `S${item.step}${item.epoch == null ? "" : ` · E${item.epoch}`}`}</span>
      <span class="ll-file">${esc(item.filename)}</span>
      <span class="ll-size">${item.size_mb == null ? "" : `${item.size_mb} MB`}</span>
    </label>
  `).join("");
}

function renderPromptList() {
  return state.prompts.map((item, index) => `
    <div class="ll-prompt-item ${item.enabled === false ? "disabled" : ""}" data-prompt-index="${index}">
      <div class="ll-prompt-top">
        <input type="checkbox" class="ll-prompt-enabled" data-index="${index}" ${item.enabled === false ? "" : "checked"} title="Enable this prompt">
        <span class="ll-prompt-index">P${String(index + 1).padStart(2, "0")}</span>
        <input class="ll-prompt-label" data-prompt-label="${index}" value="${esc(item.label)}" aria-label="Prompt label">
        <span class="ll-badge">${esc(item.category)}</span>
        <button class="ll-icon-btn ll-move-prompt" data-index="${index}" data-direction="-1" title="Move up" ${index === 0 ? "disabled" : ""} style="width:27px;height:27px;font-size:12px"><i class="pi pi-angle-up"></i></button>
        <button class="ll-icon-btn ll-move-prompt" data-index="${index}" data-direction="1" title="Move down" ${index === state.prompts.length - 1 ? "disabled" : ""} style="width:27px;height:27px;font-size:12px"><i class="pi pi-angle-down"></i></button>
        <button class="ll-icon-btn ll-duplicate-prompt" data-index="${index}" title="Duplicate prompt" style="width:27px;height:27px;font-size:12px"><i class="pi pi-copy"></i></button>
        <button class="ll-icon-btn ll-remove-prompt" data-index="${index}" title="Remove prompt" style="width:27px;height:27px;font-size:12px"><i class="pi pi-times"></i></button>
      </div>
      <textarea class="ll-textarea ll-prompt-text" data-prompt-text="${index}">${esc(item.text)}</textarea>
      <div class="ll-prompt-resolved"><span>Final positive</span><code>${esc(resolvePromptText(item.text))}</code></div>
    </div>
  `).join("");
}

function modelStackProblems() {
  const problems = [];
  if (state.workflowAdapter === "api_template") {
    try {
      const parsed = JSON.parse(state.apiWorkflow || "{}");
      if (!Object.keys(parsed).length) problems.push("Import a ComfyUI API workflow.");
      else if (!Object.values(parsed).some((node) => node?.class_type === "LoRALabIdentityLoader")) problems.push("API workflow needs a LoRA Lab Identity / Baseline Loader node.");
    } catch { problems.push("Imported API workflow JSON is invalid."); }
    return problems;
  }
  const model = String(state.form.modelName || "").toLowerCase();
  if (!model) problems.push("Select a diffusion model.");
  if (currentProfile()?.missing?.length) problems.push(`Missing profile prerequisites: ${currentProfile().missing.join(" · ")}`);
  if (state.form.turboLora && model.includes("turbo")) problems.push("Turbo model + Turbo LoRA would apply acceleration twice. Use a Raw model.");
  if (state.form.turboLora && state.selected.has(state.form.turboLoraName)) problems.push("Turbo LoRA is also selected as a checkpoint candidate. Remove duplicate.");
  const auxFiles = state.form.auxLoras.map((item) => item.filename).filter(Boolean);
  if (state.form.auxLoras.some((item) => !item.filename)) problems.push("Select a file for every always-on auxiliary LoRA.");
  if (new Set(auxFiles).size !== auxFiles.length) problems.push("The always-on auxiliary stack contains a duplicate LoRA.");
  if (state.form.turboLora && auxFiles.includes(state.form.turboLoraName)) problems.push("Turbo LoRA is duplicated in the auxiliary stack.");
  const duplicateCandidate = auxFiles.find((filename) => state.selected.has(filename));
  if (duplicateCandidate) problems.push(`Auxiliary LoRA is also selected as a checkpoint candidate: ${shortFile(duplicateCandidate)}`);
  if (["stack_compare","enhancer_compare"].includes(state.form.mode) && state.selected.size !== 1) problems.push("This objective requires exactly one selected identity LoRA.");
  return problems;
}

function renderAuxLoras() {
  const rows = state.form.auxLoras.map((item, index) => `
    <div class="ll-aux-row" data-aux-row="${index}">
      <span class="ll-step">#${index + 1}</span>
      <select class="ll-select ll-aux-file" data-aux-file="${index}">
        <option value="">Select an installed LoRA…</option>
        ${(state.boot.loras || []).map((entry) => `<option value="${esc(entry.filename)}" ${entry.filename === item.filename ? "selected" : ""}>${esc(entry.filename)}</option>`).join("")}
      </select>
      <input class="ll-input ll-aux-strength" data-aux-strength="${index}" type="number" min="-2" max="2" step="0.05" value="${esc(item.strength)}" title="Model strength">
      <button class="ll-icon-btn ll-move-aux" data-index="${index}" data-direction="-1" title="Move earlier" ${index === 0 ? "disabled" : ""}><i class="pi pi-angle-up"></i></button>
      <button class="ll-icon-btn ll-move-aux" data-index="${index}" data-direction="1" title="Move later" ${index === state.form.auxLoras.length - 1 ? "disabled" : ""}><i class="pi pi-angle-down"></i></button>
      <button class="ll-icon-btn ll-remove-aux" data-index="${index}" title="Remove auxiliary LoRA"><i class="pi pi-times"></i></button>
    </div>
  `).join("");
  return `
    <div class="ll-field ll-aux-editor">
      <label class="ll-label">Always-on auxiliary LoRAs ${help("Applied in this exact order before every candidate checkpoint. The control column keeps this stack but removes the candidate, making comparisons fair.")}</label>
      ${rows || `<div class="ll-hint">No auxiliary LoRAs. Add acceleration, style, detail, or compatibility LoRAs that every generated cell must share.</div>`}
      <div class="ll-toolbar"><button class="ll-btn small" id="ll-add-aux" ${state.form.auxLoras.length >= 8 ? "disabled" : ""}><i class="pi pi-plus"></i> Add auxiliary LoRA</button><span class="ll-hint">${state.form.auxLoras.length}/8 · order matters</span></div>
    </div>
  `;
}

function renderModelStack() {
  const candidateText = state.form.mode === "strength" ? (state.selected.size ? "1 LoRA · strength sweep" : "Select one LoRA") : `${state.selected.size} checkpoint LoRA${state.selected.size === 1 ? "" : "s"}`;
  const nodes = [
    ["Base", shortFile(state.form.modelName, 31)],
    ["Turbo", state.form.mode === "stack_compare" ? "Compared Off / On" : state.form.turboLora ? `On · ${shortFile(state.form.turboLoraName, 25)}` : "Off"],
    ["Auxiliary", state.form.auxLoras.length ? `${state.form.auxLoras.length} always on` : "None"],
    ["Identity", candidateText],
    ["Enhancer", state.form.mode === "enhancer_compare" ? "Off / Standard / Advanced" : state.form.enhancer === "off" ? "Off" : `${state.form.enhancer} · ${state.form.enhancerStrength}`],
    ["Sampler", `${state.form.steps} · CFG ${state.form.cfg} · ${state.form.sampler} / ${state.form.scheduler}`],
  ];
  const problems = modelStackProblems();
  return `<div class="ll-stack"><div class="ll-stack-title">Execution order</div><div class="ll-stack-flow">${nodes.map(([title,value], index) => `${index ? `<i class="pi pi-angle-right"></i>` : ""}<div class="ll-stack-node"><span>${esc(title)}</span><strong title="${esc(value)}">${esc(value)}</strong></div>`).join("")}</div>${problems.map((problem) => `<div class="ll-warning">${esc(problem)}</div>`).join("")}</div>`;
}

function resolvePromptText(text) {
  const trigger = String(state.form.trigger || "").trim();
  const subjectClass = String(state.form.subjectClass || "").trim();
  const subject = [trigger, subjectClass].filter(Boolean).join(" ");
  return String(text || "").replaceAll("{trigger}", trigger).replaceAll("{class}", subjectClass).replaceAll("{subject}", subject).trim();
}

function renderSetup() {
  const estimateData = estimate();
  const groups = groupOptions();
  const profile = currentProfile();
  const preset = state.boot.presets.find((item) => item.id === state.form.preset);
  const modeCompare = state.form.mode === "compare";
  const modeStrength = state.form.mode === "strength";
  const stackProblems = modelStackProblems();
  const objective = OBJECTIVES.find((item) => item.id === state.objective) || OBJECTIVES[0];
  const groupWatcher = (state.boot.watchers || []).find((item) => item.group === state.group);
  return `
    <div class="ll-page">
      <h1 class="ll-page-title">Build controlled test</h1>
      <p class="ll-page-lead">Use identical prompts and seeds across checkpoints. Baseline shows actual identity gain. Start coarse, then retest finalists with more seeds.</p>
      <label class="ll-check ll-direct-open"><input type="checkbox" id="ll-direct-open" ${state.directOpen ? "checked" : ""}> Open full dashboard directly from ComfyUI icon ${help("Enabled by default. Disable only if you prefer the compact sidebar first.")}</label>
      <article class="ll-card ll-objective-wizard">
        <div class="ll-card-head"><div class="ll-card-title">Test-plan wizard</div><div class="ll-card-sub">Choose decision, then edit any generated setting</div></div>
        <div class="ll-card-body">
          <div class="ll-objective-grid">${visibleObjectives().map((item) => `<button class="ll-objective ${item.id === state.objective ? "active" : ""}" data-objective="${item.id}"><i class="pi ${item.icon}"></i><span><strong>${esc(item.name)}</strong><small>${esc(item.description)}</small></span></button>`).join("")}</div>
          <div class="ll-objective-footer"><span><strong>${esc(objective.name)}:</strong> ${esc(objective.requirement)}. Wizard changes mode, profile, prompt coverage, seeds and resolution; controls remain editable.</span><button class="ll-btn primary" id="ll-apply-objective">Apply plan</button></div>
        </div>
      </article>
      <div class="ll-kpis">
        <div class="ll-kpi"><div class="ll-kpi-label">GPU</div><div class="ll-kpi-value">${esc(state.boot.hardware.gpu)}</div><div class="ll-kpi-note">${esc(state.boot.hardware.vram_gb)} GB VRAM</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Installed LoRAs</div><div class="ll-kpi-value">${state.boot.loras.length}</div><div class="ll-kpi-note">Grouped training checkpoints</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Selected</div><div class="ll-kpi-value">${state.selected.size}</div><div class="ll-kpi-note">${modeCompare ? "checkpoint candidates" : "one LoRA required"}</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Planned jobs</div><div class="ll-kpi-value">${estimateData.jobs}</div><div class="ll-kpi-note">${estimateData.promptCount} prompts × ${estimateData.seedCount} seeds × ${estimateData.candidateCount}</div></div>
      </div>

      <div class="ll-grid setup">
        <section class="ll-grid">
          <article class="ll-card">
            <div class="ll-card-head"><div class="ll-card-title">1 · Model and preset</div><div class="ll-card-sub">Preset fills values; every value remains editable</div></div>
            <div class="ll-card-body">
              <div class="ll-field">
                <label class="ll-label">Workflow adapter ${help("Native adapters provide a polished setup for supported families. API workflow mode preserves an arbitrary working ComfyUI graph and replaces only declared test values.")}</label>
                <div class="ll-segment">
                  <button data-workflow-adapter="native" class="${state.workflowAdapter === "native" ? "active" : ""}">Native model adapter</button>
                  <button data-workflow-adapter="api_template" class="${state.workflowAdapter === "api_template" ? "active" : ""}">Import any API workflow</button>
                </div>
              </div>
              ${state.workflowAdapter === "api_template" ? `<div class="ll-import-box">
                <div class="ll-note"><strong>Universal compatibility:</strong> export a known-working graph with <em>Save (API Format)</em>. Add the <strong>LoRA Lab · Identity / Baseline Loader</strong> node, and use exact placeholders such as <code>{{PROMPT}}</code>, <code>{{SEED}}</code>, <code>{{STEPS}}</code>, <code>{{CFG}}</code>, <code>{{WIDTH}}</code>, and <code>{{HEIGHT}}</code>. Custom nodes and enhancer chains are preserved.</div>
                <textarea class="ll-textarea" id="ll-api-workflow" rows="9" placeholder="Paste ComfyUI API workflow JSON…">${esc(state.apiWorkflow)}</textarea>
                <div class="ll-toolbar"><input type="file" id="ll-api-workflow-file" accept=".json,application/json"><input class="ll-input" id="ll-api-output-node" placeholder="Optional IMAGE output node ID" value="${esc(state.apiOutputNodeId)}"><input class="ll-input" id="ll-api-output-index" type="number" min="0" value="${esc(state.apiOutputIndex)}" style="max-width:90px"></div>
                <div class="ll-hint">If output ID is blank, LoRA Lens connects to the image input of the first PreviewImage or SaveImage node.</div>
              </div>` : ""}
              <div class="ll-field">
                <label class="ll-label">Model category ${help("Categories keep the growing model catalog readable. They only filter the profile list.")}</label>
                <select class="ll-select" id="ll-model-category"><option value="all">All categories</option>${profileCategories().map((name) => `<option value="${esc(name)}" ${name === state.modelCategory ? "selected" : ""}>${esc(name)}</option>`).join("")}</select>
              </div>
              <div class="ll-field">
                <label class="ll-label">Model family and variant ${help("A native adapter selects the correct loader, encoders, latent type and sampling nodes. Every sampling value remains editable.")}</label>
                <select class="ll-select" id="ll-profile">
                  ${profileOptions()}
                </select>
                <div class="ll-hint"><strong>${esc(profile?.name || "")}</strong> · ${esc(profile?.description || "")} · Source: ${esc(profile?.source || "built in")}</div>
              </div>
              <div class="ll-field">
                <label class="ll-label">Diffusion model ${help("Independent from the LoRA choice. Select any installed file compatible with the active native adapter.")}</label>
                <select class="ll-select" id="ll-model-name">${(state.boot.diffusion_models || []).map((name) => `<option value="${esc(name)}" ${name === state.form.modelName ? "selected" : ""}>${esc(name)}</option>`).join("")}</select>
              </div>
              ${state.workflowAdapter === "native" && profile?.supports_acceleration_lora ? `<label class="ll-turbo-toggle ${state.form.turboLora ? "active" : ""}">
                <input type="checkbox" id="ll-turbo-lora" ${state.form.turboLora ? "checked" : ""} ${state.boot.turbo_lora?.available && state.form.mode !== "stack_compare" ? "" : "disabled"}>
                <span><strong>Apply Krea 2 Turbo LoRA</strong><small>${state.form.mode === "stack_compare" ? "Managed per candidate by Raw vs Turbo objective." : state.boot.turbo_lora?.available ? (state.form.turboLora ? "Active · 8 steps · CFG 1 · Euler / beta · zero negative" : "One toggle. Applied before every checkpoint LoRA.") : "Turbo LoRA not installed"}</small></span>
              </label>` : `<div class="ll-note">${state.workflowAdapter === "api_template" ? "Acceleration and custom-node behavior come from the imported workflow." : `Acceleration mode: <strong>${esc(profile?.acceleration || "None")}</strong>. Choose another variant above for a native Turbo/Schnell workflow.`}</div>`}
              ${renderAuxLoras()}
              ${renderModelStack()}
              <div class="ll-field">
                <label class="ll-label">Evaluation mode ${help("Compare ranks different checkpoints at one strength. Strength sweep applies one checkpoint at several strengths after checkpoint selection.")}</label>
                <div class="ll-segment">
                  <button data-mode="compare" class="${modeCompare ? "active" : ""}">Compare checkpoints</button>
                  <button data-mode="strength" class="${modeStrength ? "active" : ""}">Sweep strength</button>
                </div>
                ${state.form.mode === "stack_compare" ? `<div class="ll-note" style="margin-top:7px">Wizard mode: paired Raw vs Raw+Turbo variants.</div>` : state.form.mode === "enhancer_compare" ? `<div class="ll-note" style="margin-top:7px">Wizard mode: paired enhancer Off / Standard / Advanced variants.</div>` : ""}
              </div>
              <div class="ll-field-row">
                <div class="ll-field">
                  <label class="ll-label">Test preset ${help("Quick screens all checkpoints cheaply. Standard and Deep increase prompt/seed coverage for finalists.")}</label>
                  <select class="ll-select" id="ll-preset">${state.boot.presets.map((item) => `<option value="${item.id}" ${item.id === state.form.preset ? "selected" : ""}>${esc(item.name)}</option>`).join("")}</select>
                  <div class="ll-hint">${esc(preset?.description || "")}</div>
                </div>
                <div class="ll-field">
                  <label class="ll-label">${state.form.turboLora || state.form.auxLoras.length ? "Always-on stack control" : "No-LoRA control"} ${help(state.form.turboLora || state.form.auxLoras.length ? "Control keeps Turbo and auxiliary LoRAs but removes the candidate checkpoint." : "Generates the same prompt and seed without a candidate LoRA.")}</label>
                  <label class="ll-check" style="min-height:35px"><input type="checkbox" id="ll-baseline" ${state.form.includeBaseline ? "checked" : ""}> Include control column</label>
                </div>
              </div>
              <div class="ll-field-row">
                <div class="ll-field ${modeStrength ? "ll-hidden" : ""}">
                  <label class="ll-label">Common strength ${help("All checkpoints must use identical strength for fair comparison. Use a separate strength sweep on the winner.")}</label>
                  <input class="ll-input" id="ll-common-strength" type="number" min="-2" max="2" step="0.05" value="${esc(state.form.commonStrength)}">
                </div>
                <div class="ll-field ${modeStrength ? "" : "ll-hidden"}">
                  <label class="ll-label">Sweep values ${help("Comma-separated LoRA strengths. Each value becomes a candidate column with matched prompt and seed.")}</label>
                  <input class="ll-input" id="ll-strengths" value="${esc(state.form.strengths)}">
                </div>
                <div class="ll-field">
                  <label class="ll-label">Trigger token or phrase · exact ${help("Inserted exactly as written. LoRA Lab adds no hidden class word.")}</label>
                  <input class="ll-input" id="ll-trigger" value="${esc(state.form.trigger)}">
                </div>
              </div>
              <div class="ll-field">
                <label class="ll-label">Optional subject/class ${help("Blank by default. {subject} resolves to trigger plus this value; {trigger} and {class} remain independently available.")}</label>
                <input class="ll-input" id="ll-subject-class" placeholder="Example: woman · leave blank for trigger only" value="${esc(state.form.subjectClass)}">
                <div class="ll-hint">Placeholders: <code>{subject}</code> = “<span id="ll-subject-resolution">${esc([state.form.trigger.trim(), state.form.subjectClass.trim()].filter(Boolean).join(" "))}</span>” · <code>{trigger}</code> = exact trigger · <code>{class}</code> = optional class. Nothing else is inserted.</div>
              </div>
            </div>
          </article>

          <article class="ll-card">
            <div class="ll-card-head"><div class="ll-card-title">2 · Candidates</div><div class="ll-card-sub">${state.selected.size} selected</div></div>
            <div class="ll-card-body">
              <div class="ll-toolbar">
                <input class="ll-input" id="ll-search" placeholder="Search filename, group, step…" value="${esc(state.search)}">
                <select class="ll-select" id="ll-group"><option value="">All groups</option>${groups.map(([name, count]) => `<option value="${esc(name)}" ${name === state.group ? "selected" : ""}>${esc(name)} (${count})</option>`).join("")}</select>
              </div>
              <div class="ll-toolbar">
                <button class="ll-btn small" id="ll-select-visible">Select visible</button>
                <button class="ll-btn small" id="ll-select-latest">Latest 4</button>
                <button class="ll-btn small" id="ll-select-every">Every Nth</button>
                <input class="ll-input" id="ll-every-n" type="number" min="1" max="20" value="${state.everyN}" style="max-width:64px;min-width:64px">
                <button class="ll-btn small ghost" id="ll-clear-selection">Clear</button>
              </div>
              <div class="ll-toolbar ll-range-tools">
                <input class="ll-input" id="ll-step-min" type="number" min="0" placeholder="Min step" value="${esc(state.stepMin)}">
                <input class="ll-input" id="ll-step-max" type="number" min="0" placeholder="Max step" value="${esc(state.stepMax)}">
                <button class="ll-btn small" id="ll-select-range">Select step range</button>
                <button class="ll-btn small ghost" id="ll-continue-screening">From latest run</button>
              </div>
              <div class="ll-candidates" id="ll-candidate-list">${renderCandidates()}</div>
              <div class="ll-selected-line"><span>${visibleCandidates().length} visible</span><span>${state.selected.size} selected</span></div>
              <div class="ll-watch-box ${groupWatcher?.active ? "active" : ""}">
                <div><strong>OneTrainer checkpoint watcher</strong><span>${state.group ? groupWatcher?.active ? `Watching ${esc(state.group)} · ${groupWatcher.known_count} known · ${groupWatcher.run_ids.length} automatic runs` : `Watch new stable files added to ${esc(state.group)}. Existing files are ignored.` : "Choose one checkpoint group above."}</span></div>
                ${groupWatcher?.active ? `<button class="ll-btn small danger" id="ll-stop-watch" data-watcher-id="${esc(groupWatcher.watcher_id)}">Stop watcher</button>` : `<button class="ll-btn small" id="ll-start-watch" ${state.group ? "" : "disabled"}>Start watcher</button>`}
              </div>
            </div>
          </article>
        </section>

        <section class="ll-grid">
          <article class="ll-card">
            <div class="ll-card-head"><div class="ll-card-title">3 · Prompt suite</div><div class="ll-card-sub">Matched across every candidate</div></div>
            <div class="ll-card-body">
              <div class="ll-prompt-modes">
                ${[["preset","Preset"],["custom","Custom"],["import","Import"],["reuse","Reuse run"]].map(([id,label]) => `<button class="ll-btn small ${state.promptMode === id ? "active" : "ghost"}" data-prompt-mode="${id}">${label}</button>`).join("")}
              </div>
              ${state.promptMode === "preset" ? `<div class="ll-note" style="margin-bottom:10px">Preset owns prompt count. Changing test preset refreshes this list. Switch to Custom before editing if you want prompts preserved.</div>` : ""}
              ${state.promptMode === "custom" ? `<div class="ll-note" style="margin-bottom:10px">Manual mode. Test preset may change seeds or resolution but never resets these prompts.</div>` : ""}
              ${state.promptMode === "import" ? `<div class="ll-import-box"><textarea class="ll-textarea" id="ll-prompt-import" placeholder="Paste one prompt per line, or a JSON array…">${esc(state.promptImportText)}</textarea><div class="ll-toolbar"><input type="file" id="ll-prompt-file" accept=".txt,.json,text/plain,application/json"><button class="ll-btn small" id="ll-import-prompts">Import text</button></div></div>` : ""}
              ${state.promptMode === "reuse" ? `<div class="ll-toolbar"><select class="ll-select" id="ll-reuse-run"><option value="">Select previous run…</option>${(state.boot.runs || []).map((run) => `<option value="${esc(run.run_id)}" ${state.reuseRunId === run.run_id ? "selected" : ""}>${esc(run.name || run.run_id)} · ${esc(run.created_at || "")}</option>`).join("")}</select><button class="ll-btn small" id="ll-load-run-prompts">Load prompts</button></div>` : ""}
              <div class="ll-suite-tools">
                <input class="ll-input" id="ll-suite-name" placeholder="Named prompt suite" value="${esc(state.promptSuiteName)}">
                <button class="ll-btn small" id="ll-save-suite">Save suite</button>
                <select class="ll-select" id="ll-saved-suite"><option value="">Saved suites…</option>${Object.keys(state.promptSuites).sort().map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("")}</select>
                <button class="ll-btn small" id="ll-load-suite">Load</button>
                <button class="ll-btn small ghost" id="ll-delete-suite">Delete</button>
              </div>
              <div class="ll-prompt-list" id="ll-prompt-list">${renderPromptList()}</div>
              <div class="ll-toolbar" style="margin-top:9px;margin-bottom:0"><button class="ll-btn small" id="ll-add-prompt"><i class="pi pi-plus"></i> Add prompt</button>${state.promptMode === "preset" ? `<button class="ll-btn small ghost" id="ll-reset-prompts">Reset preset prompts</button>` : ""}<span class="ll-hint">${state.prompts.filter((item) => item.enabled !== false).length}/${state.prompts.length} enabled</span></div>
            </div>
          </article>

          <article class="ll-card">
            <div class="ll-card-head"><div class="ll-card-title">4 · Sampling and references</div><div class="ll-card-sub">Reproducible settings</div></div>
            <div class="ll-card-body">
              <div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Matched seeds ${help("Every candidate receives the same seeds for each prompt. Add seeds only after coarse checkpoint screening.")}</label><input class="ll-input" id="ll-seeds" value="${esc(state.form.seeds)}"></div>
                <div class="ll-field"><label class="ll-label">Reference faces ${help("Folder is relative to ComfyUI/input. Use 8–20 varied, high-quality single-face holdouts not used for training.")}</label><input class="ll-input" id="ll-reference" list="ll-reference-list" value="${esc(state.form.referenceFolder)}"><datalist id="ll-reference-list">${state.boot.references.map((item) => `<option value="${esc(item.folder)}">${item.count} images</option>`).join("")}</datalist></div>
              </div>
              <div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Width ${help("Quick uses 768 to screen cheaply. Final decisions should be verified at Krea's native 1024 resolution.")}</label><input class="ll-input" id="ll-width" type="number" min="512" max="2048" step="8" value="${state.form.width}"></div>
                <div class="ll-field"><label class="ll-label">Height ${help("Keep the same resolution across candidates. Aspect ratio is a test variable only when deliberately planned.")}</label><input class="ll-input" id="ll-height" type="number" min="512" max="2048" step="8" value="${state.form.height}"></div>
              </div>
              <button class="ll-btn small ghost ll-advanced-toggle" id="ll-advanced-stack-toggle"><i class="pi ${state.showAdvancedStack ? "pi-angle-up" : "pi-angle-down"}"></i> ${state.showAdvancedStack ? "Hide" : "Show"} advanced model stack</button>
              <div class="ll-field-row ${state.showAdvancedStack ? "" : "ll-hidden"}">
                <div class="ll-field"><label class="ll-label">Text encoder 1 ${help("The profile selects its preferred installed encoder; you can deliberately override it.")}</label><select class="ll-select" id="ll-clip-name">${(state.boot.text_encoders || []).map((name) => `<option value="${esc(name)}" ${name === state.form.clipName ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div>
                ${profile?.adapter === "split_dual" ? `<div class="ll-field"><label class="ll-label">Text encoder 2 ${help("FLUX adapters require both CLIP-L and T5XXL.")}</label><select class="ll-select" id="ll-clip-name-2">${(state.boot.text_encoders || []).map((name) => `<option value="${esc(name)}" ${name === state.form.clipName2 ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div>` : ""}
                <div class="ll-field"><label class="ll-label">VAE ${help("The profile selects the official VAE when it is installed; this remains editable.")}</label><select class="ll-select" id="ll-vae-name">${(state.boot.vaes || []).map((name) => `<option value="${esc(name)}" ${name === state.form.vaeName ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div>
              </div>
              <div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Negative conditioning ${help("Zero ignores negative text, typical Turbo behavior. Encode uses the negative prompt with real CFG. Choice is independent from model preset.")}</label><select class="ll-select" id="ll-negative-mode"><option value="zero" ${state.form.negativeMode === "zero" ? "selected" : ""}>Zero conditioning</option><option value="encode" ${state.form.negativeMode === "encode" ? "selected" : ""}>Encode negative prompt</option></select></div>
                <div class="ll-field"><label class="ll-label">Negative prompt</label><input class="ll-input" id="ll-negative" value="${esc(state.form.negativePrompt)}"></div>
              </div>
              <div class="ll-conditioning-preview"><span>Final negative conditioning</span><code id="ll-negative-preview">${state.form.negativeMode === "zero" ? "ZERO CONDITIONING — negative text is not encoded" : esc(state.form.negativePrompt)}</code></div>
              <div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Output prefix ${help("Creates a unique UUID run folder. Existing results are never overwritten.")}</label><input class="ll-input" id="ll-prefix" value="${esc(state.form.outputPrefix)}"></div>
                <div class="ll-field"><label class="ll-label">Legacy grid export ${help("Interactive Results replaces giant PNG grids. Keep Off to avoid large duplicate files; enable only when a static grid is required.")}</label><select class="ll-select" id="ll-grid-mode"><option value="off" ${state.form.gridMode === "off" ? "selected" : ""}>Off · recommended</option><option value="master_only" ${state.form.gridMode === "master_only" ? "selected" : ""}>Master grid</option><option value="per_prompt_only" ${state.form.gridMode === "per_prompt_only" ? "selected" : ""}>Per-prompt grids</option></select></div>
              </div>
              <div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Steps ${help("Always editable. Preset only supplies starting value.")}</label><input class="ll-input" id="ll-steps" type="number" min="1" max="1000" value="${state.form.steps}"></div>
                <div class="ll-field"><label class="ll-label">CFG ${help("Always editable. Keep matched across candidates within one run.")}</label><input class="ll-input" id="ll-cfg" type="number" min="0" max="100" step="0.1" value="${state.form.cfg}"></div>
                <div class="ll-field"><label class="ll-label">Sampler</label><select class="ll-select" id="ll-sampler">${(state.boot.samplers || []).map((name) => `<option value="${esc(name)}" ${name === state.form.sampler ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div>
                <div class="ll-field"><label class="ll-label">Scheduler</label><select class="ll-select" id="ll-scheduler">${(state.boot.schedulers || []).map((name) => `<option value="${esc(name)}" ${name === state.form.scheduler ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div>
              </div>
              ${profile?.family === "Krea 2" ? `<div class="ll-field-row">
                <div class="ll-field"><label class="ll-label">Krea2T enhancer ${help("Optional MODEL patch inserted after LoRA and before sampler. Standard and Advanced nodes come from ComfyUI-Krea2T-Enhancer.")}</label><select class="ll-select" id="ll-enhancer" ${state.form.mode === "enhancer_compare" ? "disabled" : ""}><option value="off" ${state.form.enhancer === "off" ? "selected" : ""}>Off</option><option value="standard" ${state.form.enhancer === "standard" ? "selected" : ""}>Standard</option><option value="advanced" ${state.form.enhancer === "advanced" ? "selected" : ""}>Advanced</option></select></div>
                <div class="ll-field"><label class="ll-label">Enhancer strength</label><input class="ll-input" id="ll-enhancer-strength" type="number" min="0" max="2" step="0.05" value="${state.form.enhancerStrength}"></div>
                <div class="ll-field"><label class="ll-label">Advanced text scale</label><input class="ll-input" id="ll-enhancer-text-scale" type="number" min="0.25" max="4" step="0.05" value="${state.form.enhancerTextScale}" ${state.form.enhancer === "advanced" ? "" : "disabled"}></div>
              </div>` : ""}
              <div class="ll-field ${state.showAdvancedStack ? "" : "ll-hidden"}">
                <label class="ll-label">Additional MODEL patch chain · JSON ${help("No graph editing. Add any installed MODEL-to-MODEL Comfy node. LoRA Lab wires model input automatically. Example: [{\"class_type\":\"NodeName\",\"inputs\":{\"strength\":1}}].")}</label>
                <textarea class="ll-textarea" id="ll-custom-patches" list="ll-patch-node-list">${esc(state.form.customPatches)}</textarea>
                <datalist id="ll-patch-node-list">${(state.boot.model_patch_nodes || []).map((item) => `<option value="${esc(item.class_type)}"></option>`).join("")}</datalist>
                <div class="ll-hint">${(state.boot.model_patch_nodes || []).length} compatible installed MODEL patch nodes detected. Empty chain: []</div>
              </div>
            </div>
          </article>

          <article class="ll-card">
            <div class="ll-card-head"><div class="ll-card-title">Run budget</div><div class="ll-card-sub">Exact before queueing</div></div>
            <div class="ll-card-body">
              <div class="ll-estimate">
                <div><strong>${estimateData.jobs}</strong><span>generated cells</span></div>
                <div><strong>${formatSeconds(estimateData.seconds)}</strong><span>4090 estimate</span></div>
                <div><strong>~${estimateData.storage} MB</strong><span>cell storage</span></div>
              </div>
              ${state.form.mode === "strength" && state.selected.size !== 1 ? `<div class="ll-warning" style="margin-top:9px">Strength sweep needs exactly one selected LoRA.</div>` : ""}
              ${estimateData.jobs > 220 ? `<div class="ll-warning" style="margin-top:9px">Large run. Screen checkpoints with Quick first; retest only top finalists.</div>` : ""}
              <div class="ll-actions"><button class="ll-btn primary" id="ll-create-run" ${estimateData.jobs < 1 || stackProblems.length || (state.form.mode === "strength" && state.selected.size !== 1) ? "disabled" : ""}><i class="pi pi-bolt"></i> Create and start run</button></div>
            </div>
          </article>
        </section>
      </div>
    </div>
  `;
}

function renderMonitor() {
  const data = state.runData;
  if (!state.currentRunId) return `<div class="ll-page narrow"><div class="ll-empty">No active run. Create one in Test setup or select one from History.</div></div>`;
  if (!data) return `<div class="ll-empty"><i class="pi pi-spin pi-spinner"></i><br>Loading run…</div>`;
  const run = data.run;
  const progress = data.progress;
  const percent = clamp(Number(progress.percent || 0), 0, 100);
  const circumference = 2 * Math.PI * 52;
  const offset = circumference * (1 - percent / 100);
  const latest = [...(progress.cells || [])].slice(-16).reverse();
  const statusLabel = String(progress.status || "unknown").replaceAll("_", " ");
  const errors = run.queue_errors || [];
  const canAnalyze = progress.completed >= progress.total;
  return `
    <div class="ll-page ll-progress-wrap">
      <h1 class="ll-page-title">${esc(run.name || run.run_id)}</h1>
      <p class="ll-page-lead">${esc(run.profile.name)} · ${run.width}×${run.height} · ${run.candidate_count} columns · ${run.scenario_count} matched scenarios</p>
      <div class="ll-grid cols-2">
        <article class="ll-card ll-progress-hero">
          <div class="ll-progress-ring">
            <svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52"></circle><circle class="value" cx="60" cy="60" r="52" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"></circle></svg>
            <div class="ll-progress-number">${percent.toFixed(0)}%</div>
          </div>
          <div class="ll-progress-state">${esc(statusLabel)}</div>
          <div class="ll-progress-note">${progress.completed} completed · ${progress.submitted} submitted · ${progress.total} total</div>
          <div class="ll-progress-note">${progress.remaining_seconds == null ? `Planned ${formatSeconds(run.estimate?.seconds)}` : `About ${formatSeconds(progress.remaining_seconds)} remaining`}</div>
          <div class="ll-progress-controls">
            ${progress.status === "paused" ? `<button class="ll-btn primary" data-run-action="resume"><i class="pi pi-play"></i> Resume</button>` : `<button class="ll-btn" data-run-action="pause"><i class="pi pi-pause"></i> Pause submissions</button>`}
            <button class="ll-btn" data-run-action="retry"><i class="pi pi-refresh"></i> Retry missing</button>
            <button class="ll-btn danger" data-run-action="stop"><i class="pi pi-stop"></i> Stop run now</button>
            <button class="ll-btn" data-run-action="free"><i class="pi pi-trash"></i> Release VRAM</button>
            <button class="ll-btn primary" id="ll-analyze" ${canAnalyze ? "" : "disabled"}><i class="pi pi-chart-line"></i> Analyze completed run</button>
          </div>
        </article>
        <article class="ll-card">
          <div class="ll-card-head"><div class="ll-card-title">Run specification</div><div class="ll-card-sub">${esc(run.run_id)}</div></div>
          <div class="ll-card-body">
            <div class="ll-table-wrap"><table class="ll-table"><tbody>
              <tr><th>Objective</th><td>${esc(run.objective || "custom")}</td></tr>
              <tr><th>Base model</th><td>${esc(run.model_name)}</td></tr>
              <tr><th>Turbo LoRA</th><td>${run.turbo_lora?.enabled ? `${esc(run.turbo_lora.filename)} · ${Number(run.turbo_lora.strength).toFixed(2)}` : "Off"}</td></tr>
              <tr><th>Auxiliary LoRAs</th><td>${run.aux_loras?.length ? run.aux_loras.map((item) => `${esc(item.filename)} · ${Number(item.strength).toFixed(2)}`).join("<br>") : "None"}</td></tr>
              <tr><th>Sampling</th><td>${run.profile.steps} steps · CFG ${run.profile.cfg} · ${esc(run.profile.sampler)} / ${esc(run.profile.scheduler)}</td></tr>
              <tr><th>Seeds</th><td>${run.seeds.map(esc).join(", ")}</td></tr>
              <tr><th>Reference</th><td>${esc(run.reference_folder)}</td></tr>
              <tr><th>Trigger</th><td>${esc(run.trigger)}</td></tr>
              <tr><th>Subject/class</th><td>${esc(run.subject_class || "—")}</td></tr>
              <tr><th>Created</th><td>${esc(run.created_at)}</td></tr>
            </tbody></table></div>
            ${errors.length ? `<div class="ll-error" style="margin-top:10px">${errors.slice(-3).map((item) => `${item.job}: ${item.error}`).join("\n")}</div>` : `<div class="ll-note" style="margin-top:10px">Stop cancels this run's submitter, pending prompts and active prompt, then requests model unload and VRAM cleanup. Closing this window alone does not stop backend submission.</div>`}
          </div>
        </article>
      </div>
      <article class="ll-card" style="margin-top:14px">
        <div class="ll-card-head"><div class="ll-card-title">Latest completed cells</div><div class="ll-card-sub">Live preview</div></div>
        <div class="ll-card-body">${latest.length ? `<div class="ll-thumbs">${latest.map((cell) => `<div class="ll-thumb ll-zoomable" data-src="${esc(cell.asset_url)}"><img src="${esc(cell.asset_url)}" loading="lazy"><span>${esc(cell.candidate.label)} · ${esc(cell.scenario.label)}</span></div>`).join("")}</div>` : `<div class="ll-empty">Waiting for first image…</div>`}</div>
      </article>
    </div>
  `;
}

function analysisEntryMap(analysis) {
  const map = new Map();
  for (const entry of analysis?.entries || []) map.set(entry.key, entry);
  return map;
}

function renderRanking(analysis) {
  return `
    <div class="ll-table-wrap"><table class="ll-table">
      <thead><tr>
        <th>#</th><th>Candidate</th><th>Step</th><th>Strength</th>
        <th>Final ${help("Automatic paired identity score. Human blind ratings blend in only after enough cells are rated.")}</th>
        <th>Ensemble ${help("Calibrated identity score: 65% KP-RPE AdaFace and 35% AntelopeV2. Quality never adds identity points.")}</th>
        <th>KP-RPE ${help("Raw CVLFace ViT-KP-RPE AdaFace cosine similarity before calibration.")}</th>
        <th>Antelope ${help("Raw InsightFace AntelopeV2 cosine similarity before calibration.")}</th>
        <th>95% CI ${help("Bootstrap interval across matched prompt/seed scenarios. Overlapping top candidates are not forced into a winner.")}</th>
        <th>Gain vs base ${help("Mean identity-index change versus no-LoRA control using the same prompt and seed.")}</th>
        <th>Faces</th><th>Best probability</th><th>Human</th>
      </tr></thead>
      <tbody>${analysis.ranking.map((item, index) => {
        const candidate = item.candidate;
        return `<tr class="${index === 0 ? "winner" : ""}">
          <td class="ll-rank">${item.rank}</td>
          <td title="${esc(candidate.filename)}">${esc(candidate.label)}</td>
          <td class="ll-num">${candidate.step ?? "—"}</td>
          <td class="ll-num">${Number(candidate.strength).toFixed(2)}</td>
          <td class="ll-num">${Number(item.final_score).toFixed(2)}</td>
          <td class="ll-num">${Number(item.mean_similarity).toFixed(3)}</td>
          <td class="ll-num">${item.mean_kprpe_similarity == null ? "—" : Number(item.mean_kprpe_similarity).toFixed(3)}</td>
          <td class="ll-num">${item.mean_antelope_similarity == null ? "—" : Number(item.mean_antelope_similarity).toFixed(3)}</td>
          <td class="ll-ci ll-num">${Number(item.ci_low).toFixed(3)}…${Number(item.ci_high).toFixed(3)}</td>
          <td class="ll-num">${item.identity_gain_vs_baseline == null ? "—" : `${item.identity_gain_vs_baseline >= 0 ? "+" : ""}${Number(item.identity_gain_vs_baseline).toFixed(3)}`}</td>
          <td class="ll-num">${Number(item.detection_rate).toFixed(0)}%${item.missing_faces ? ` · ${item.missing_faces} miss` : ""}</td>
          <td class="ll-num">${item.probability_best == null ? "—" : `${(item.probability_best * 100).toFixed(1)}%`}</td>
          <td class="ll-num">${item.human_score == null ? `— (${item.human_rating_count})` : `${item.human_score.toFixed(1)} (${item.human_rating_count})`}</td>
        </tr>`;
      }).join("")}</tbody>
    </table></div>
  `;
}

function renderCurve(analysis) {
  const points = analysis.ranking
    .filter((item) => Number.isFinite(Number(item.candidate.step)))
    .sort((a, b) => Number(a.candidate.step) - Number(b.candidate.step));
  if (points.length < 2) return `<div class="ll-empty">Checkpoint curve appears when at least two candidates contain parseable training steps.</div>`;
  const width = 800, height = 230, left = 48, right = 18, top = 18, bottom = 34;
  const xs = points.map((item) => Number(item.candidate.step));
  const ys = points.map((item) => Number(item.final_score));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys) - 1, maxY = Math.max(...ys) + 1;
  const x = (value) => left + (value - minX) / Math.max(1, maxX - minX) * (width - left - right);
  const y = (value) => top + (maxY - value) / Math.max(.001, maxY - minY) * (height - top - bottom);
  const path = points.map((item, index) => `${index ? "L" : "M"}${x(item.candidate.step).toFixed(1)},${y(item.final_score).toFixed(1)}`).join(" ");
  const grid = [0, .25, .5, .75, 1].map((ratio) => {
    const value = minY + (maxY - minY) * ratio;
    const py = y(value);
    return `<line class="gridline" x1="${left}" y1="${py}" x2="${width - right}" y2="${py}"></line><text x="5" y="${py + 3}">${value.toFixed(1)}</text>`;
  }).join("");
  return `<div class="ll-chart"><svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    ${grid}<line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line>
    <path class="line" d="${path}"></path>
    ${points.map((item) => `<circle class="point" cx="${x(item.candidate.step)}" cy="${y(item.final_score)}" r="5"><title>step ${item.candidate.step}: ${item.final_score.toFixed(2)}</title></circle><text x="${x(item.candidate.step)}" y="${height - 12}" text-anchor="middle">${item.candidate.step}</text>`).join("")}
  </svg></div>`;
}

function ratingOptions(value) {
  return `<option value="">—</option>${[1,2,3,4,5].map((number) => `<option value="${number}" ${Number(value) === number ? "selected" : ""}>${number}</option>`).join("")}`;
}

function renderMatrix(data) {
  const run = data.run;
  const analysis = data.analysis;
  const entries = analysisEntryMap(analysis);
  const cellMap = new Map((data.progress.cells || []).map((cell) => [cell.key, cell]));
  const ratingMap = data.ratings?.ratings || {};
  const categories = [...new Set(run.scenarios.map((item) => item.category))];
  let scenarios = run.scenarios;
  if (state.matrixCategory !== "all") scenarios = scenarios.filter((item) => item.category === state.matrixCategory);
  if (state.matrixScenario !== "all") scenarios = scenarios.filter((item) => String(item.scenario_index) === state.matrixScenario);
  let candidates = run.candidates.map((item, index) => ({ ...item, index }));
  if (state.matrixCandidate !== "all") candidates = candidates.filter((item) => String(item.index) === state.matrixCandidate);
  return `
    <div class="ll-matrix-controls">
      <select class="ll-select" id="ll-matrix-category"><option value="all">All prompt categories</option>${categories.map((value) => `<option value="${esc(value)}" ${state.matrixCategory === value ? "selected" : ""}>${esc(value)}</option>`).join("")}</select>
      <select class="ll-select" id="ll-matrix-scenario"><option value="all">All prompt / seed rows</option>${run.scenarios.map((item) => `<option value="${item.scenario_index}" ${state.matrixScenario === String(item.scenario_index) ? "selected" : ""}>${esc(item.label)} · seed ${item.seed}</option>`).join("")}</select>
      <select class="ll-select" id="ll-matrix-candidate"><option value="all">All candidates</option>${run.candidates.map((item, index) => `<option value="${index}" ${state.matrixCandidate === String(index) ? "selected" : ""}>${esc(item.label)}</option>`).join("")}</select>
      <label class="ll-check"><input type="checkbox" id="ll-blind" ${state.blind ? "checked" : ""}> Blind names ${help("Hides checkpoint names during visual scoring. Candidate aliases remain stable within this run.")}</label>
      <button class="ll-btn small" id="ll-reanalyze"><i class="pi pi-refresh"></i> Recompute after ratings</button>
    </div>
    <div class="ll-note" style="margin-bottom:10px">Rate identity, prompt adherence, and visual quality separately. Automatic sharpness never decides winner. Missing faces count as failures.</div>
    <div class="ll-matrix">${scenarios.map((scenario) => `
      <div class="ll-matrix-row">
        <div class="ll-matrix-row-head"><strong>${esc(scenario.category)}</strong>${esc(scenario.label)} · seed ${scenario.seed}<br><span class="ll-hint">${esc(scenario.text)}</span></div>
        <div class="ll-matrix-cells">${candidates.map((candidate) => {
          const key = `p${String(scenario.scenario_index).padStart(3,"0")}_l${String(candidate.index).padStart(3,"0")}`;
          const cell = cellMap.get(key);
          const entry = entries.get(key);
          const rating = ratingMap[key] || entry?.rating || {};
          const display = state.blind ? aliasFor(candidate.index, candidate.baseline) : candidate.label;
          return `<div class="ll-cell" data-p="${scenario.scenario_index}" data-l="${candidate.index}">
            <div class="ll-cell-title"><span class="ll-cell-name" title="${esc(display)}">${esc(display)}</span><span class="ll-cell-score" title="${entry ? `Ensemble ${entry.identity_similarity?.toFixed(3) ?? "missing"} · KP-RPE ${entry.kprpe_similarity?.toFixed(3) ?? "missing"} · Antelope ${entry.antelope_similarity?.toFixed(3) ?? "missing"}` : ""}">${entry ? (entry.face_detected ? entry.identity_similarity.toFixed(3) : "face missing") : ""}</span></div>
            <div class="ll-cell-image ${cell ? "ll-zoomable" : ""}" ${cell ? `data-src="${esc(cell.asset_url)}"` : ""}>${cell ? `<img src="${esc(cell.asset_url)}" loading="lazy">` : `<div class="ll-cell-missing">Missing cell</div>`}</div>
            <div class="ll-ratings">
              <div class="ll-rating"><label>Identity<select data-rating="identity">${ratingOptions(rating.identity)}</select></label></div>
              <div class="ll-rating"><label>Adherence<select data-rating="adherence">${ratingOptions(rating.adherence)}</select></label></div>
              <div class="ll-rating"><label>Quality<select data-rating="quality">${ratingOptions(rating.quality)}</select></label></div>
              <label class="ll-check ll-artifact"><input type="checkbox" data-rating="artifact" ${rating.artifact ? "checked" : ""}> Visible artifact</label>
            </div>
          </div>`;
        }).join("")}</div>
      </div>
    `).join("")}</div>
  `;
}

function renderTournament(data) {
  const tournament = data.tournament;
  if (!tournament) {
    return `
      <article class="ll-card ll-tournament">
        <div class="ll-card-head"><div class="ll-card-title">Blind 1-vs-1 tournament</div><div class="ll-card-sub">Human judgement first; AI result stays hidden</div></div>
        <div class="ll-card-body">
          <p>Each duel uses same prompt and seed. Winner advances through that scenario, then next scenario begins. Candidate names and AI scores remain hidden until completion.</p>
          <label class="ll-check"><input type="checkbox" id="ll-tournament-baseline" checked> Include no-LoRA control</label>
          <div class="ll-actions"><button class="ll-btn ghost" id="ll-reveal-ai">Skip blind test · reveal AI</button><button class="ll-btn primary" id="ll-start-tournament"><i class="pi pi-eye-slash"></i> Start blind tournament</button></div>
        </div>
      </article>`;
  }
  if (tournament.status === "active") {
    const match = tournament.next_match;
    if (!match) return `<article class="ll-card"><div class="ll-card-body ll-empty">Preparing next blind duel…</div></article>`;
    return `
      <article class="ll-card ll-tournament">
        <div class="ll-card-head"><div class="ll-card-title">Blind duel ${tournament.completed + 1}/${tournament.total}</div><div class="ll-card-sub">Scenario ${match.scenario_number}/${match.scenario_total} · round ${match.round}</div></div>
        <div class="ll-card-body">
          <div class="ll-duel-prompt"><strong>${esc(match.prompt_label)}</strong><span>seed ${match.seed}</span><p>${esc(match.prompt_text)}</p></div>
          <div class="ll-duel">
            <div class="ll-duel-side"><button class="ll-duel-image ll-zoomable" data-src="${esc(match.left.asset_url)}" aria-label="Enlarge image A"><span>Image A</span><img src="${esc(match.left.asset_url)}"></button><div class="ll-duel-flags"><label><input type="checkbox" id="ll-artifact-left"> Artifact</label><label><input type="checkbox" id="ll-identity-failure-left"> Identity failure</label></div></div>
            <div class="ll-versus">VS</div>
            <div class="ll-duel-side"><button class="ll-duel-image ll-zoomable" data-src="${esc(match.right.asset_url)}" aria-label="Enlarge image B"><span>Image B</span><img src="${esc(match.right.asset_url)}"></button><div class="ll-duel-flags"><label><input type="checkbox" id="ll-artifact-right"> Artifact</label><label><input type="checkbox" id="ll-identity-failure-right"> Identity failure</label></div></div>
          </div>
          <div class="ll-duel-question"><strong>1. Identity only</strong><span>Which face looks more like the trained person? This decision advances the identity bracket.</span></div>
          <div class="ll-duel-actions">
            <button class="ll-btn primary" data-duel-choice="left" data-match-id="${esc(match.match_id)}">A has better identity</button>
            <button class="ll-btn" data-duel-choice="tie" data-match-id="${esc(match.match_id)}">Identity tie</button>
            <button class="ll-btn primary" data-duel-choice="right" data-match-id="${esc(match.match_id)}">B has better identity</button>
          </div>
          <div class="ll-duel-question"><strong>2. Overall preference · optional</strong><span>Choose separately so beauty, realism and composition do not contaminate identity validation.</span></div>
          <div class="ll-overall-choice"><label><input type="radio" name="ll-overall-choice" value="left"> Prefer A overall</label><label><input type="radio" name="ll-overall-choice" value="tie"> Overall tie</label><label><input type="radio" name="ll-overall-choice" value="right"> Prefer B overall</label><label><input type="radio" name="ll-overall-choice" value="skip" checked> No overall vote</label></div>
          <div class="ll-duel-actions secondary"><button class="ll-btn danger" data-duel-choice="left_broken" data-match-id="${esc(match.match_id)}">A broken</button><button class="ll-btn ghost" data-duel-choice="skip" data-match-id="${esc(match.match_id)}">Skip pair</button><button class="ll-btn danger" data-duel-choice="right_broken" data-match-id="${esc(match.match_id)}">B broken</button></div>
          <div class="ll-mini-progress"><span style="width:${tournament.percent}%"></span></div>
          <div class="ll-hint" style="margin-top:6px">AI agreement is measured only against your identity decision. Overall preference remains a separate human signal.</div>
          ${tournament.can_undo ? `<div class="ll-actions"><button class="ll-btn small ghost" id="ll-undo-tournament">Undo previous vote</button></div>` : ""}
        </div>
      </article>`;
  }
  const agreement = Number(tournament.agreement_rate || 0) * 100;
  return `
    <article class="ll-card ll-tournament">
      <div class="ll-card-head"><div class="ll-card-title">Blind tournament complete</div><div class="ll-card-sub">Human vs AI exposed only after final vote</div></div>
      <div class="ll-card-body">
        <div class="ll-grid cols-4" style="margin-bottom:12px">
          <div class="ll-kpi"><div class="ll-kpi-label">Human identity winner</div><div class="ll-kpi-value small">${esc(tournament.human_winner)}</div></div>
          <div class="ll-kpi"><div class="ll-kpi-label">Human overall winner</div><div class="ll-kpi-value small">${esc(tournament.overall_winner || "No overall votes")}</div></div>
          <div class="ll-kpi"><div class="ll-kpi-label">AI winner</div><div class="ll-kpi-value small">${esc(tournament.automatic_winner)}</div></div>
          <div class="ll-kpi"><div class="ll-kpi-label">Pair agreement</div><div class="ll-kpi-value">${agreement.toFixed(1)}%</div><div class="ll-kpi-note">${tournament.agreement_count}/${tournament.agreement_total} comparable decisions</div></div>
        </div>
        <div class="ll-field" style="max-width:520px">
          <label class="ll-label">Human weight in combined rank: <span id="ll-human-weight-label">${Math.round(tournament.human_weight * 100)}%</span> ${help("AI and human rankings remain visible separately. Slider changes only combined score.")}</label>
          <input id="ll-human-weight" type="range" min="0" max="100" step="5" value="${Math.round(tournament.human_weight * 100)}">
        </div>
        <div class="ll-table-wrap"><table class="ll-table"><thead><tr><th>#</th><th>Candidate</th><th>Identity Elo</th><th>Identity BT</th><th>Overall Elo</th><th>Overall BT</th><th>AI normalized</th><th>A/B sides</th><th>Issues</th><th>Scenario wins</th><th>Combined</th></tr></thead><tbody>
          ${(tournament.standings || []).map((row) => `<tr class="${row.rank === 1 ? "winner" : ""}"><td class="ll-rank">${row.rank}</td><td>${esc(row.candidate.label)}</td><td class="ll-num">${row.human_elo.toFixed(1)}</td><td class="ll-num">${row.bradley_terry_score.toFixed(1)}</td><td class="ll-num">${row.overall_elo?.toFixed(1) ?? "—"}</td><td class="ll-num">${row.overall_bradley_terry_score?.toFixed(1) ?? "—"}</td><td class="ll-num">${row.automatic_score.toFixed(1)}</td><td class="ll-num">${row.left_count}/${row.right_count}</td><td class="ll-num">${row.artifact_count}A · ${row.identity_failure_count}I</td><td class="ll-num">${row.scenario_wins}</td><td class="ll-num">${row.combined_score.toFixed(1)}</td></tr>`).join("")}
        </tbody></table></div>
        ${tournament.analyzer_agreement?.ensemble?.total ? `<div class="ll-category-agreement"><span><strong>Human identity ↔ ensemble</strong>${(tournament.analyzer_agreement.ensemble.rate*100).toFixed(0)}% · ${tournament.analyzer_agreement.ensemble.same}/${tournament.analyzer_agreement.ensemble.total}</span><span><strong>Human identity ↔ KP-RPE</strong>${(tournament.analyzer_agreement.kprpe.rate*100).toFixed(0)}% · ${tournament.analyzer_agreement.kprpe.same}/${tournament.analyzer_agreement.kprpe.total}</span><span><strong>Human identity ↔ Antelope</strong>${(tournament.analyzer_agreement.antelopev2.rate*100).toFixed(0)}% · ${tournament.analyzer_agreement.antelopev2.same}/${tournament.analyzer_agreement.antelopev2.total}</span></div>` : `<div class="ll-note">Reset this legacy tournament to collect separate identity-only and overall-preference validation.</div>`}
        ${(tournament.category_agreement || []).length ? `<div class="ll-category-agreement">${tournament.category_agreement.map((item) => `<span><strong>${esc(item.category)}</strong>${(item.rate*100).toFixed(0)}% · ${item.same}/${item.total}</span>`).join("")}</div>` : ""}
        <div class="ll-actions">${tournament.can_undo ? `<button class="ll-btn ghost" id="ll-undo-tournament">Undo previous vote</button>` : ""}<button class="ll-btn ghost" id="ll-reset-tournament">Reset blind tournament</button></div>
      </div>
    </article>`;
}

function renderResults() {
  const data = state.runData;
  if (!state.currentRunId) return `<div class="ll-page narrow"><div class="ll-empty">Select a completed run from History.</div></div>`;
  if (!data) return `<div class="ll-empty"><i class="pi pi-spin pi-spinner"></i><br>Loading results…</div>`;
  if (data.progress.completed < data.progress.total) return `<div class="ll-page narrow"><div class="ll-warning">Run has ${data.progress.completed}/${data.progress.total} cells. Finish or retry missing cells before final analysis.</div><div class="ll-actions"><button class="ll-btn" data-go="run">Open monitor</button></div></div>`;
  if (!data.analysis) return `<div class="ll-page narrow"><article class="ll-card"><div class="ll-card-body ll-empty"><i class="pi pi-chart-line" style="font-size:32px;color:var(--ll-cyan)"></i><h2>Images complete</h2><p>Run paired identity analysis and confidence estimation.</p><button class="ll-btn primary" id="ll-analyze"><i class="pi pi-bolt"></i> Analyze now</button></div></article></div>`;
  const analysis = data.analysis;
  const tournament = renderTournament(data);
  if (data.tournament?.status === "active" || (!data.tournament && !state.revealAutomatic)) {
    return `<div class="ll-page"><div class="ll-blind-lock"><i class="pi pi-eye-slash"></i> Automatic ranking hidden to protect blind judgement.</div>${tournament}</div>`;
  }
  return `
    <div class="ll-page">
      ${tournament}
      <div class="ll-result-banner ${analysis.decisive ? "" : "indecisive"}">
        <div class="ll-result-icon"><i class="pi ${analysis.decisive ? "pi-check" : "pi-exclamation-triangle"}"></i></div>
        <div><div class="ll-result-title">${esc(analysis.winner)}</div><div class="ll-result-note">${esc(analysis.confidence)} · ${esc(analysis.recommendation)}</div></div>
        <div class="ll-header-spacer"></div>
        <span class="ll-badge ${analysis.decisive ? "good" : "warn"}">${analysis.decisive ? "Decisive" : "Finalists overlap"}</span>
      </div>
      <div class="ll-actions ll-result-actions"><a class="ll-btn primary" href="/loralab/v1/export?run_id=${encodeURIComponent(state.currentRunId)}" download><i class="pi pi-download"></i> Export evidence ZIP</a><button class="ll-btn ghost" id="ll-reanalyze-top"><i class="pi pi-refresh"></i> Reanalyse with current models</button></div>
      <div class="ll-grid cols-4" style="margin-bottom:14px">
        <div class="ll-kpi"><div class="ll-kpi-label">Reference inliers</div><div class="ll-kpi-value">${analysis.reference.inliers}/${analysis.reference.total}</div><div class="ll-kpi-note">cohesion ${Number(analysis.reference.cohesion).toFixed(3)}</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Model agreement</div><div class="ll-kpi-value">${analysis.analyzer_status?.model_agreement_rate == null ? "—" : `${(analysis.analyzer_status.model_agreement_rate*100).toFixed(0)}%`}</div><div class="ll-kpi-note">${analysis.analyzer_status?.model_agreement_count || 0}/${analysis.analyzer_status?.model_agreement_total || 0} prompt winners match</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Top vs second</div><div class="ll-kpi-value">${analysis.probability_top_beats_second == null ? "—" : `${(analysis.probability_top_beats_second*100).toFixed(1)}%`}</div><div class="ll-kpi-note">paired bootstrap probability</div></div>
        <div class="ll-kpi"><div class="ll-kpi-label">Baseline identity</div><div class="ll-kpi-value">${analysis.baseline ? Number(analysis.baseline.mean_similarity).toFixed(3) : "not run"}</div><div class="ll-kpi-note">same prompts and seeds</div></div>
      </div>
      ${(analysis.analyzer_status?.category_agreement || []).length ? `<article class="ll-card ll-diagnostics"><div class="ll-card-head"><div class="ll-card-title">Model disagreement diagnostics</div><div class="ll-card-sub">Where KP-RPE and Antelope select the same prompt winner</div></div><div class="ll-card-body"><div class="ll-category-agreement">${analysis.analyzer_status.category_agreement.map((item) => `<span><strong>${esc(item.category)}</strong>${(item.rate*100).toFixed(0)}% · ${item.same}/${item.total}</span>`).join("")}</div><div class="ll-note">Low agreement is not hidden. Use the identity-only blind tournament to learn which analyser follows your judgement on this subject.</div></div></article>` : ""}
      <div class="ll-grid cols-2">
        <article class="ll-card" style="grid-column:1/-1"><div class="ll-card-head"><div class="ll-card-title">Paired ranking</div><div class="ll-card-sub">No forced winner when uncertainty overlaps</div></div><div class="ll-card-body">${renderRanking(analysis)}</div></article>
        <article class="ll-card"><div class="ll-card-head"><div class="ll-card-title">Checkpoint curve</div><div class="ll-card-sub">score by parsed save step</div></div><div class="ll-card-body">${renderCurve(analysis)}</div></article>
        <article class="ll-card"><div class="ll-card-head"><div class="ll-card-title">How to read result</div></div><div class="ll-card-body">
          <div class="ll-note">${esc(analysis.metric_notes.confidence)}</div>
          <p><strong>Identity index:</strong> ${esc(analysis.metric_notes.identity)}</p>
          <p><strong>Failure handling:</strong> ${esc(analysis.metric_notes.missing_face)}</p>
          <p><strong>Quality:</strong> ${esc(analysis.metric_notes.quality)}</p>
          <p><strong>Analyser:</strong> ${esc(analysis.analyzer)}</p>
          ${analysis.analyzer_status?.cvlface_error ? `<div class="ll-warning"><strong>CVLFace fallback:</strong> ${esc(analysis.analyzer_status.cvlface_error)}</div>` : ""}
          ${analysis.paired_top_two_ci ? `<p><strong>Top-two paired difference CI:</strong> ${Number(analysis.paired_top_two_ci[0]).toFixed(4)}…${Number(analysis.paired_top_two_ci[1]).toFixed(4)}</p>` : ""}
        </div></article>
      </div>
      <article class="ll-card" style="margin-top:14px"><div class="ll-card-head"><div class="ll-card-title">Blind review matrix</div><div class="ll-card-sub">Human signal stays separate and explicit</div></div><div class="ll-card-body">${renderMatrix(data)}</div></article>
    </div>
  `;
}

function renderHistory() {
  const runs = state.boot.runs || [];
  return `
    <div class="ll-page narrow">
      <h1 class="ll-page-title">Run history</h1>
      <p class="ll-page-lead">Persistent run specifications, progress, ratings, and analysis. Select any row to reopen it.</p>
      <div class="ll-history">${runs.length ? runs.map((item) => `
        <div class="ll-history-row" data-open-run="${esc(item.run_id)}">
          <div class="ll-history-name"><strong>${esc(item.name || item.run_id)}</strong><span>${esc(item.created_at)} · ${esc(item.run_id)}</span></div>
          <div class="ll-history-meta">${esc(item.profile || "")}</div>
          <div class="ll-history-meta">${item.candidate_count} cols × ${item.scenario_count} rows</div>
          <div><div class="ll-mini-progress"><span style="width:${item.progress.percent}%"></span></div><div class="ll-history-meta">${item.progress.completed}/${item.progress.total}</div></div>
          <div>${item.winner ? `<span class="ll-badge ${item.decisive ? "good" : "warn"}">${esc(item.winner)}</span>` : `<span class="ll-badge">${esc(item.progress.status)}</span>`}</div>
        </div>
      `).join("") : `<div class="ll-empty">No LoRA Lab runs yet.</div>`}</div>
    </div>
  `;
}

function importedPrompts(text) {
  const source = String(text || "").trim();
  if (!source) throw new Error("Paste prompts or choose a text/JSON file first.");
  let rows;
  if (source.startsWith("[") || source.startsWith("{")) {
    const parsed = JSON.parse(source);
    rows = Array.isArray(parsed) ? parsed : parsed.prompts;
    if (!Array.isArray(rows)) throw new Error("JSON must be an array or contain a prompts array.");
  } else {
    rows = source.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  }
  if (rows.length > 24) throw new Error("Maximum 24 prompts per suite.");
  const prompts = rows.map((item, index) => {
    const object = typeof item === "string" ? { text: item } : item;
    if (!object || typeof object !== "object" || !String(object.text || "").trim()) return null;
    const label = String(object.label || `Imported ${index + 1}`).trim();
    return {
      label,
      category: String(object.category || label).toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || `imported_${index + 1}`,
      text: String(object.text).trim(),
      enabled: object.enabled !== false,
    };
  }).filter(Boolean);
  if (!prompts.length) throw new Error("No valid prompts found.");
  return prompts;
}

function persistPromptSuites() {
  localStorage.setItem(PROMPT_SUITES_KEY, JSON.stringify(state.promptSuites));
}

function refreshConditioningPreviews() {
  const subject = [state.form.trigger.trim(), state.form.subjectClass.trim()].filter(Boolean).join(" ");
  const subjectResolution = document.getElementById("ll-subject-resolution");
  if (subjectResolution) subjectResolution.textContent = subject;
  document.querySelectorAll(".ll-prompt-item").forEach((item) => {
    const textarea = item.querySelector("[data-prompt-text]");
    const preview = item.querySelector(".ll-prompt-resolved code");
    if (textarea && preview) preview.textContent = resolvePromptText(textarea.value);
  });
  const negative = document.getElementById("ll-negative-preview");
  if (negative) negative.textContent = state.form.negativeMode === "zero" ? "ZERO CONDITIONING — negative text is not encoded" : state.form.negativePrompt;
}

function syncSetupInputs() {
  const get = (id) => document.getElementById(id);
  if (get("ll-api-workflow")) state.apiWorkflow = get("ll-api-workflow").value;
  if (get("ll-api-output-node")) state.apiOutputNodeId = get("ll-api-output-node").value;
  if (get("ll-api-output-index")) state.apiOutputIndex = Number(get("ll-api-output-index").value) || 0;
  if (get("ll-trigger")) state.form.trigger = get("ll-trigger").value;
  if (get("ll-subject-class")) state.form.subjectClass = get("ll-subject-class").value;
  if (get("ll-common-strength")) state.form.commonStrength = Number(get("ll-common-strength").value);
  if (get("ll-strengths")) state.form.strengths = get("ll-strengths").value;
  if (get("ll-seeds")) state.form.seeds = get("ll-seeds").value;
  if (get("ll-width")) state.form.width = Number(get("ll-width").value);
  if (get("ll-height")) state.form.height = Number(get("ll-height").value);
  if (get("ll-reference")) state.form.referenceFolder = get("ll-reference").value;
  if (get("ll-negative")) state.form.negativePrompt = get("ll-negative").value;
  if (get("ll-prefix")) state.form.outputPrefix = get("ll-prefix").value;
  if (get("ll-grid-mode")) state.form.gridMode = get("ll-grid-mode").value;
  if (get("ll-model-name")) state.form.modelName = get("ll-model-name").value;
  if (get("ll-clip-name")) state.form.clipName = get("ll-clip-name").value;
  if (get("ll-clip-name-2")) state.form.clipName2 = get("ll-clip-name-2").value;
  if (get("ll-vae-name")) state.form.vaeName = get("ll-vae-name").value;
  if (get("ll-steps")) state.form.steps = Number(get("ll-steps").value);
  if (get("ll-cfg")) state.form.cfg = Number(get("ll-cfg").value);
  if (get("ll-sampler")) state.form.sampler = get("ll-sampler").value;
  if (get("ll-scheduler")) state.form.scheduler = get("ll-scheduler").value;
  if (get("ll-negative-mode")) state.form.negativeMode = get("ll-negative-mode").value;
  if (get("ll-enhancer")) state.form.enhancer = get("ll-enhancer").value;
  if (get("ll-enhancer-strength")) state.form.enhancerStrength = Number(get("ll-enhancer-strength").value);
  if (get("ll-enhancer-text-scale")) state.form.enhancerTextScale = Number(get("ll-enhancer-text-scale").value);
  if (get("ll-custom-patches")) state.form.customPatches = get("ll-custom-patches").value;
  document.querySelectorAll("[data-aux-file]").forEach((element) => {
    const index = Number(element.dataset.auxFile);
    if (state.form.auxLoras[index]) state.form.auxLoras[index].filename = element.value;
  });
  document.querySelectorAll("[data-aux-strength]").forEach((element) => {
    const index = Number(element.dataset.auxStrength);
    if (state.form.auxLoras[index]) state.form.auxLoras[index].strength = Number(element.value);
  });
  if (get("ll-prompt-import")) state.promptImportText = get("ll-prompt-import").value;
  if (get("ll-suite-name")) state.promptSuiteName = get("ll-suite-name").value;
  if (get("ll-reuse-run")) state.reuseRunId = get("ll-reuse-run").value;
  if (get("ll-step-min")) state.stepMin = get("ll-step-min").value;
  if (get("ll-step-max")) state.stepMax = get("ll-step-max").value;
  document.querySelectorAll("[data-prompt-label]").forEach((element) => { const index = Number(element.dataset.promptLabel); if (state.prompts[index]) state.prompts[index].label = element.value; });
  document.querySelectorAll("[data-prompt-text]").forEach((element) => { const index = Number(element.dataset.promptText); if (state.prompts[index]) state.prompts[index].text = element.value; });
  document.querySelectorAll(".ll-prompt-enabled").forEach((element) => { const index = Number(element.dataset.index); if (state.prompts[index]) state.prompts[index].enabled = element.checked; });
}

function bindShell() {
  document.getElementById("ll-close")?.addEventListener("click", closeLab);
  document.getElementById("ll-refresh")?.addEventListener("click", async () => {
    state.busy = "Refreshing catalog…";
    renderShell();
    try {
      await loadBootstrap(true);
      if (state.currentRunId) await refreshRun(false);
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = "";
      renderShell();
    }
  });
  document.querySelectorAll(".ll-tab").forEach((button) => button.addEventListener("click", async () => {
    if (state.page === "setup") syncSetupInputs();
    state.page = button.dataset.page;
    if (["run", "results"].includes(state.page) && state.currentRunId) await refreshRun(false);
    if (state.page === "history") {
      try { state.boot.runs = (await jsonFetch("/loralab/v1/runs")).runs; } catch {}
    }
    renderShell();
  }));
}

function viewerItemsFor(element) {
  const tournament = state.runData?.tournament;
  if (tournament?.status === "active" && tournament.next_match) {
    const match = tournament.next_match;
    return {
      blind: true,
      items: [
        { src: match.left.asset_url, display: "Image A", promptLabel: match.prompt_label, promptText: match.prompt_text, seed: match.seed },
        { src: match.right.asset_url, display: "Image B", promptLabel: match.prompt_label, promptText: match.prompt_text, seed: match.seed },
      ],
    };
  }
  const entries = state.runData?.analysis?.entries || [];
  if (entries.length) {
    const ratingMap = state.runData?.ratings?.ratings || {};
    return {
      blind: Boolean(state.blind && element.closest(".ll-matrix")),
      items: entries.map((entry) => ({
        src: entry.asset_url,
        display: entry.candidate?.label || `Candidate ${entry.lora_index + 1}`,
        baseline: Boolean(entry.candidate?.baseline),
        promptLabel: entry.scenario?.label || `Prompt ${entry.prompt_index + 1}`,
        promptText: entry.scenario?.text || "",
        seed: entry.scenario?.seed,
        score: entry.identity_similarity,
        kprpeScore: entry.kprpe_similarity,
        antelopeScore: entry.antelope_similarity,
        promptIndex: Number(entry.prompt_index),
        candidateIndex: Number(entry.lora_index),
        rating: ratingMap[entry.key] || entry.rating || {},
      })),
    };
  }
  const items = [...document.querySelectorAll(".ll-zoomable[data-src]")].map((node, index) => ({ src: node.dataset.src, display: node.dataset.title || `Image ${index + 1}`, promptLabel: node.dataset.prompt || "Preview", promptText: "" }));
  return { blind: false, items };
}

function openViewer(element) {
  const collection = viewerItemsFor(element);
  const index = Math.max(0, collection.items.findIndex((item) => item.src === element.dataset.src));
  state.viewer = { ...collection, index, scale: 1, x: 0, y: 0 };
  renderViewer();
}

function closeViewer() {
  document.getElementById("ll-viewer")?.remove();
  state.viewer = null;
}

function navigateViewer(delta) {
  if (!state.viewer?.items.length) return;
  state.viewer.index = (state.viewer.index + delta + state.viewer.items.length) % state.viewer.items.length;
  state.viewer.scale = 1; state.viewer.x = 0; state.viewer.y = 0;
  renderViewer();
}

function jumpViewer(kind, direction) {
  if (!state.viewer) return;
  const current = state.viewer.items[state.viewer.index];
  for (let offset = 1; offset < state.viewer.items.length; offset++) {
    const index = (state.viewer.index + direction * offset + state.viewer.items.length) % state.viewer.items.length;
    const item = state.viewer.items[index];
    const match = kind === "candidate" ? item.promptIndex === current.promptIndex && item.candidateIndex !== current.candidateIndex : item.candidateIndex === current.candidateIndex && item.promptIndex !== current.promptIndex;
    if (match) { state.viewer.index = index; state.viewer.scale = 1; state.viewer.x = 0; state.viewer.y = 0; renderViewer(); return; }
  }
}

function updateViewerTransform() {
  const image = document.querySelector("#ll-viewer .ll-viewer-image");
  if (image && state.viewer) image.style.transform = `translate(${state.viewer.x}px, ${state.viewer.y}px) scale(${state.viewer.scale})`;
}

async function saveViewerRating() {
  const item = state.viewer?.items[state.viewer.index];
  if (!item || item.promptIndex == null || item.candidateIndex == null) return;
  const payload = { run_id: state.currentRunId, prompt_index: item.promptIndex, lora_index: item.candidateIndex };
  for (const key of ["identity","adherence","quality"]) {
    const value = document.querySelector(`#ll-viewer [data-viewer-rating='${key}']`)?.value;
    payload[key] = value ? Number(value) : null;
  }
  payload.artifact = Boolean(document.querySelector("#ll-viewer [data-viewer-rating='artifact']")?.checked);
  try {
    await jsonFetch("/loralab/v1/rating", { method: "POST", body: payload });
    const key = `p${String(item.promptIndex).padStart(3,"0")}_l${String(item.candidateIndex).padStart(3,"0")}`;
    if (!state.runData.ratings) state.runData.ratings = { ratings: {} };
    state.runData.ratings.ratings[key] = payload; item.rating = payload;
  } catch (error) { toast("Rating not saved", error.message, "error"); }
}

function renderViewer() {
  if (!state.viewer?.items.length) return;
  document.getElementById("ll-viewer")?.remove();
  const item = state.viewer.items[state.viewer.index];
  const display = state.viewer.blind && item.candidateIndex != null ? aliasFor(item.candidateIndex, item.baseline) : item.display;
  const canRate = item.promptIndex != null && item.candidateIndex != null;
  const rating = item.rating || {};
  const viewer = document.createElement("div");
  viewer.id = "ll-viewer"; viewer.className = "ll-viewer";
  viewer.innerHTML = `<header class="ll-viewer-head"><div><strong>${esc(display)}</strong><span>${esc(item.promptLabel || "")}${item.seed == null ? "" : ` · seed ${esc(item.seed)}`}${state.viewer.blind || item.score == null ? "" : ` · ensemble ${Number(item.score).toFixed(3)} · KP-RPE ${item.kprpeScore == null ? "—" : Number(item.kprpeScore).toFixed(3)} · Antelope ${item.antelopeScore == null ? "—" : Number(item.antelopeScore).toFixed(3)}`}</span></div><div class="ll-viewer-tools"><button data-viewer-zoom="out" title="Zoom out">−</button><button data-viewer-zoom="fit" title="Fit image">Fit</button><button data-viewer-zoom="in" title="Zoom in">+</button><button id="ll-viewer-close" title="Close">×</button></div></header>
    <main class="ll-viewer-stage"><button class="ll-viewer-arrow left" data-viewer-nav="-1" aria-label="Previous image"><i class="pi pi-angle-left"></i></button><img class="ll-viewer-image" src="${esc(item.src)}" draggable="false"><button class="ll-viewer-arrow right" data-viewer-nav="1" aria-label="Next image"><i class="pi pi-angle-right"></i></button></main>
    <footer class="ll-viewer-foot"><div class="ll-viewer-meta"><span>${state.viewer.index + 1}/${state.viewer.items.length}</span><p>${esc(item.promptText || "")}</p><div class="ll-viewer-jumps"><button data-viewer-jump="candidate:-1">Previous candidate</button><button data-viewer-jump="candidate:1">Next candidate</button><button data-viewer-jump="prompt:-1">Previous prompt</button><button data-viewer-jump="prompt:1">Next prompt</button></div></div>${canRate ? `<div class="ll-viewer-ratings"><label>Identity<select data-viewer-rating="identity">${ratingOptions(rating.identity)}</select></label><label>Adherence<select data-viewer-rating="adherence">${ratingOptions(rating.adherence)}</select></label><label>Quality<select data-viewer-rating="quality">${ratingOptions(rating.quality)}</select></label><label class="ll-check"><input type="checkbox" data-viewer-rating="artifact" ${rating.artifact ? "checked" : ""}> Artifact</label></div>` : ""}</footer>`;
  document.body.appendChild(viewer);
  bindViewer(); updateViewerTransform();
  for (const offset of [-1, 1]) { const preload = new Image(); preload.src = state.viewer.items[(state.viewer.index + offset + state.viewer.items.length) % state.viewer.items.length].src; }
}

function bindViewer() {
  const viewer = document.getElementById("ll-viewer"); if (!viewer) return;
  viewer.querySelector("#ll-viewer-close")?.addEventListener("click", closeViewer);
  viewer.querySelectorAll("[data-viewer-nav]").forEach((button) => button.addEventListener("click", () => navigateViewer(Number(button.dataset.viewerNav))));
  viewer.querySelectorAll("[data-viewer-jump]").forEach((button) => button.addEventListener("click", () => { const [kind,direction] = button.dataset.viewerJump.split(":"); jumpViewer(kind, Number(direction)); }));
  viewer.querySelectorAll("[data-viewer-zoom]").forEach((button) => button.addEventListener("click", () => { const action = button.dataset.viewerZoom; if (action === "fit") { state.viewer.scale = 1; state.viewer.x = 0; state.viewer.y = 0; } else state.viewer.scale = clamp(state.viewer.scale * (action === "in" ? 1.25 : .8), .25, 8); updateViewerTransform(); }));
  viewer.querySelectorAll("[data-viewer-rating]").forEach((control) => control.addEventListener("change", saveViewerRating));
  const stage = viewer.querySelector(".ll-viewer-stage");
  stage?.addEventListener("wheel", (event) => { event.preventDefault(); state.viewer.scale = clamp(state.viewer.scale * (event.deltaY < 0 ? 1.12 : .89), .25, 8); updateViewerTransform(); }, { passive: false });
  stage?.addEventListener("mousedown", (event) => { if (event.target.closest("button")) return; const start = { x: event.clientX, y: event.clientY, ox: state.viewer.x, oy: state.viewer.y }; const move = (moveEvent) => { state.viewer.x = start.ox + moveEvent.clientX - start.x; state.viewer.y = start.oy + moveEvent.clientY - start.y; updateViewerTransform(); }; const up = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); }; document.addEventListener("mousemove", move); document.addEventListener("mouseup", up); });
}

function bindZoomables() {
  document.querySelectorAll(".ll-zoomable[data-src]").forEach((element) => element.addEventListener("click", (event) => { if (event.target.closest("select,input,label")) return; openViewer(element); }));
}

function bindSetup() {
  document.getElementById("ll-direct-open")?.addEventListener("change", (event) => {
    state.directOpen = event.target.checked;
    localStorage.setItem(DIRECT_OPEN_KEY, String(state.directOpen));
  });
  document.getElementById("ll-profile")?.addEventListener("change", (event) => { syncSetupInputs(); applyProfile(event.target.value); renderShell(); });
  document.querySelectorAll("[data-workflow-adapter]").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); state.workflowAdapter = button.dataset.workflowAdapter; if (state.workflowAdapter !== "native") state.form.turboLora = false; renderShell(); }));
  document.getElementById("ll-api-workflow-file")?.addEventListener("change", async (event) => {
    const file = event.target.files?.[0]; if (!file) return;
    state.apiWorkflow = await file.text(); renderShell();
  });
  document.getElementById("ll-model-category")?.addEventListener("change", (event) => {
    syncSetupInputs();
    state.modelCategory = event.target.value;
    const visible = (state.boot.profiles || []).filter((item) => state.modelCategory === "all" || item.category === state.modelCategory);
    if (!visible.some((item) => item.id === state.form.profile) && visible[0]) applyProfile(visible[0].id);
    renderShell();
  });
  document.querySelectorAll("[data-objective]").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); state.objective = button.dataset.objective; renderShell(); }));
  document.getElementById("ll-apply-objective")?.addEventListener("click", () => { syncSetupInputs(); applyObjective(state.objective); renderShell(); });
  document.getElementById("ll-turbo-lora")?.addEventListener("change", (event) => {
    syncSetupInputs();
    state.form.turboLora = event.target.checked;
    if (state.form.turboLora) {
      if (state.form.modelName.toLowerCase().includes("turbo")) {
        state.form.modelName = state.boot.diffusion_models.find((name) => /krea2_raw_fp8_scaled/i.test(name)) || state.boot.diffusion_models.find((name) => /krea2.*raw/i.test(name)) || state.form.modelName;
      }
      applyTurboDefaults();
    }
    renderShell();
  });
  document.getElementById("ll-add-aux")?.addEventListener("click", () => {
    syncSetupInputs();
    if (state.form.auxLoras.length < 8) state.form.auxLoras.push({ filename: "", strength: 1.0 });
    renderShell();
  });
  document.querySelectorAll(".ll-remove-aux").forEach((button) => button.addEventListener("click", () => {
    syncSetupInputs();
    state.form.auxLoras.splice(Number(button.dataset.index), 1);
    renderShell();
  }));
  document.querySelectorAll(".ll-move-aux").forEach((button) => button.addEventListener("click", () => {
    syncSetupInputs();
    const index = Number(button.dataset.index);
    const target = index + Number(button.dataset.direction);
    if (target >= 0 && target < state.form.auxLoras.length) [state.form.auxLoras[index], state.form.auxLoras[target]] = [state.form.auxLoras[target], state.form.auxLoras[index]];
    renderShell();
  }));
  document.querySelectorAll(".ll-aux-file,.ll-aux-strength").forEach((control) => control.addEventListener("change", () => { syncSetupInputs(); renderShell(); }));
  document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); state.form.mode = button.dataset.mode; state.objective = state.form.mode === "strength" ? "best_strength" : "best_checkpoint"; if (state.form.mode === "strength" && state.selected.size > 1) state.selected = new Set([...state.selected].slice(-1)); renderShell(); }));
  document.getElementById("ll-preset")?.addEventListener("change", (event) => { syncSetupInputs(); applyPreset(event.target.value, true); renderShell(); });
  document.getElementById("ll-baseline")?.addEventListener("change", (event) => { state.form.includeBaseline = event.target.checked; renderShell(); });
  document.getElementById("ll-search")?.addEventListener("input", (event) => { state.search = event.target.value; const list = document.getElementById("ll-candidate-list"); if (list) list.innerHTML = renderCandidates(); bindCandidateChecks(); });
  document.getElementById("ll-group")?.addEventListener("change", (event) => { state.group = event.target.value; const list = document.getElementById("ll-candidate-list"); if (list) list.innerHTML = renderCandidates(); bindCandidateChecks(); });
  document.getElementById("ll-every-n")?.addEventListener("change", (event) => { state.everyN = Math.max(1, Number(event.target.value) || 1); });
  document.getElementById("ll-select-visible")?.addEventListener("click", () => { for (const item of visibleCandidates()) state.selected.add(item.filename); if (["strength","stack_compare","enhancer_compare"].includes(state.form.mode) && state.selected.size > 1) state.selected = new Set([[...state.selected].at(-1)]); renderShell(); });
  document.getElementById("ll-select-latest")?.addEventListener("click", () => {
    const candidates = visibleCandidates().filter((item) => item.step != null).sort((a,b) => b.step - a.step).slice(0, ["strength","stack_compare","enhancer_compare"].includes(state.form.mode) ? 1 : 4);
    state.selected = new Set(candidates.map((item) => item.filename));
    renderShell();
  });
  document.getElementById("ll-select-every")?.addEventListener("click", () => {
    const every = Math.max(1, Number(document.getElementById("ll-every-n")?.value) || 1);
    const candidates = visibleCandidates().filter((item) => item.step != null).sort((a,b) => a.step - b.step).filter((_, index) => index % every === 0);
    state.selected = new Set((["strength","stack_compare","enhancer_compare"].includes(state.form.mode) ? candidates.slice(-1) : candidates).map((item) => item.filename));
    renderShell();
  });
  document.getElementById("ll-select-range")?.addEventListener("click", () => {
    syncSetupInputs();
    const low = state.stepMin === "" ? -Infinity : Number(state.stepMin);
    const high = state.stepMax === "" ? Infinity : Number(state.stepMax);
    const matches = visibleCandidates().filter((item) => item.step != null && item.step >= low && item.step <= high).sort((a,b) => a.step - b.step);
    const selected = ["strength","stack_compare","enhancer_compare"].includes(state.form.mode) ? matches.slice(-1) : matches;
    state.selected = new Set(selected.map((item) => item.filename)); renderShell();
  });
  document.getElementById("ll-continue-screening")?.addEventListener("click", async () => {
    try {
      const catalog = new Set((state.boot.loras || []).map((item) => item.filename));
      for (const summary of (state.boot.runs || []).slice(0, 20)) {
        const data = await jsonFetch(`/loralab/v1/run?run_id=${encodeURIComponent(summary.run_id)}`);
        const files = [...new Set((data.run.candidates || []).filter((item) => !item.baseline && catalog.has(item.filename)).map((item) => item.filename))];
        if (!files.length) continue;
        state.selected = new Set(["strength","stack_compare","enhancer_compare"].includes(state.form.mode) ? files.slice(-1) : files);
        const first = state.boot.loras.find((item) => state.selected.has(item.filename)); if (first) state.group = first.group;
        renderShell(); toast("Previous screening restored", `${state.selected.size} checkpoints`, "success"); return;
      }
      toast("No reusable screening found", "Recent runs contain no currently installed checkpoint files.", "error");
    } catch (error) { toast("Could not restore screening", error.message, "error"); }
  });
  document.getElementById("ll-start-watch")?.addEventListener("click", async () => {
    if (!state.group) return;
    try {
      syncSetupInputs(); const template = buildPlanPayload([]); template.mode = "compare"; template.objective = "best_checkpoint"; template.include_baseline = true;
      const data = await jsonFetch("/loralab/v1/watch", { method: "POST", body: { action: "start", group: state.group, interval_seconds: 15, template } });
      state.boot.watchers = data.watchers; renderShell(); toast("Checkpoint watcher started", state.group, "success");
    } catch (error) { toast("Watcher could not start", error.message, "error"); }
  });
  document.getElementById("ll-stop-watch")?.addEventListener("click", async (event) => {
    try { const data = await jsonFetch("/loralab/v1/watch", { method: "POST", body: { action: "stop", watcher_id: event.currentTarget.dataset.watcherId } }); state.boot.watchers = data.watchers; renderShell(); toast("Checkpoint watcher stopped", state.group, "info"); }
    catch (error) { toast("Watcher could not stop", error.message, "error"); }
  });
  document.getElementById("ll-clear-selection")?.addEventListener("click", () => { state.selected.clear(); renderShell(); });
  bindCandidateChecks();
  document.querySelectorAll("[data-prompt-mode]").forEach((button) => button.addEventListener("click", () => {
    syncSetupInputs();
    state.promptMode = button.dataset.promptMode;
    if (state.promptMode === "preset") applyPreset(state.form.preset, true);
    renderShell();
  }));
  document.getElementById("ll-add-prompt")?.addEventListener("click", () => { syncSetupInputs(); state.promptMode = "custom"; state.prompts.push({ label: `Custom ${state.prompts.length + 1}`, category: `custom_${state.prompts.length + 1}`, text: "{subject}, ", enabled: true }); renderShell(); });
  document.getElementById("ll-reset-prompts")?.addEventListener("click", () => { applyPreset(state.form.preset, true); renderShell(); });
  document.querySelectorAll(".ll-remove-prompt").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); state.promptMode = "custom"; state.prompts.splice(Number(button.dataset.index), 1); renderShell(); }));
  document.querySelectorAll(".ll-duplicate-prompt").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); const index = Number(button.dataset.index); const copy = { ...state.prompts[index], label: `${state.prompts[index].label} copy`, category: `${state.prompts[index].category}_copy` }; state.prompts.splice(index + 1, 0, copy); state.promptMode = "custom"; renderShell(); }));
  document.querySelectorAll(".ll-move-prompt").forEach((button) => button.addEventListener("click", () => { syncSetupInputs(); const index = Number(button.dataset.index); const target = index + Number(button.dataset.direction); if (target >= 0 && target < state.prompts.length) [state.prompts[index], state.prompts[target]] = [state.prompts[target], state.prompts[index]]; state.promptMode = "custom"; renderShell(); }));
  document.querySelectorAll(".ll-prompt-enabled").forEach((checkbox) => checkbox.addEventListener("change", () => { syncSetupInputs(); state.promptMode = "custom"; renderShell(); }));
  document.querySelectorAll("[data-prompt-label],[data-prompt-text]").forEach((control) => control.addEventListener("change", () => { syncSetupInputs(); state.promptMode = "custom"; renderShell(); }));
  document.querySelectorAll("[data-prompt-text]").forEach((control) => control.addEventListener("input", () => { syncSetupInputs(); state.promptMode = "custom"; refreshConditioningPreviews(); }));
  document.getElementById("ll-import-prompts")?.addEventListener("click", () => {
    try { syncSetupInputs(); state.prompts = importedPrompts(state.promptImportText); state.promptMode = "custom"; renderShell(); toast("Prompts imported", `${state.prompts.length} prompts`, "success"); }
    catch (error) { toast("Prompt import failed", error.message, "error"); }
  });
  document.getElementById("ll-prompt-file")?.addEventListener("change", async (event) => {
    try { const file = event.target.files?.[0]; if (!file) return; state.promptImportText = await file.text(); state.prompts = importedPrompts(state.promptImportText); state.promptMode = "custom"; renderShell(); toast("Prompt file imported", `${state.prompts.length} prompts`, "success"); }
    catch (error) { toast("Prompt file failed", error.message, "error"); }
  });
  document.getElementById("ll-save-suite")?.addEventListener("click", () => {
    syncSetupInputs(); const name = state.promptSuiteName.trim(); if (!name) return toast("Suite name required", "Enter a name before saving.", "error");
    state.promptSuites[name] = state.prompts.map((item) => ({ ...item })); persistPromptSuites(); renderShell(); toast("Prompt suite saved", name, "success");
  });
  document.getElementById("ll-load-suite")?.addEventListener("click", () => {
    const name = document.getElementById("ll-saved-suite")?.value; if (!name || !state.promptSuites[name]) return;
    state.prompts = state.promptSuites[name].map((item) => ({ ...item })); state.promptSuiteName = name; state.promptMode = "custom"; renderShell();
  });
  document.getElementById("ll-delete-suite")?.addEventListener("click", () => {
    const name = document.getElementById("ll-saved-suite")?.value; if (!name || !state.promptSuites[name]) return;
    delete state.promptSuites[name]; persistPromptSuites(); if (state.promptSuiteName === name) state.promptSuiteName = ""; renderShell();
  });
  document.getElementById("ll-load-run-prompts")?.addEventListener("click", async () => {
    syncSetupInputs(); if (!state.reuseRunId) return;
    try { const data = await jsonFetch(`/loralab/v1/run?run_id=${encodeURIComponent(state.reuseRunId)}`); state.prompts = (data.run.prompts || []).map((item) => ({ label: item.label, category: item.category, text: item.text, enabled: true })); state.promptMode = "custom"; renderShell(); toast("Run prompts loaded", `${state.prompts.length} prompts`, "success"); }
    catch (error) { toast("Could not load run prompts", error.message, "error"); }
  });
  document.getElementById("ll-enhancer")?.addEventListener("change", (event) => { syncSetupInputs(); state.form.enhancer = event.target.value; renderShell(); });
  document.getElementById("ll-advanced-stack-toggle")?.addEventListener("click", () => { syncSetupInputs(); state.showAdvancedStack = !state.showAdvancedStack; renderShell(); });
  document.getElementById("ll-create-run")?.addEventListener("click", createAndStartRun);
  ["ll-common-strength","ll-strengths","ll-seeds","ll-width","ll-height","ll-reference","ll-prefix","ll-grid-mode","ll-clip-name","ll-clip-name-2","ll-vae-name","ll-custom-patches"].forEach((id) => document.getElementById(id)?.addEventListener("change", () => { syncSetupInputs(); }));
  ["ll-model-name","ll-steps","ll-cfg","ll-sampler","ll-scheduler","ll-enhancer-strength","ll-enhancer-text-scale"].forEach((id) => document.getElementById(id)?.addEventListener("change", () => { syncSetupInputs(); renderShell(); }));
  ["ll-trigger","ll-subject-class","ll-negative"].forEach((id) => document.getElementById(id)?.addEventListener("input", () => { syncSetupInputs(); refreshConditioningPreviews(); }));
  document.getElementById("ll-negative-mode")?.addEventListener("change", () => { syncSetupInputs(); renderShell(); });
}

function bindCandidateChecks() {
  document.querySelectorAll(".ll-candidate-check").forEach((checkbox) => checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      if (["strength","stack_compare","enhancer_compare"].includes(state.form.mode)) state.selected.clear();
      state.selected.add(checkbox.dataset.file);
    } else {
      state.selected.delete(checkbox.dataset.file);
    }
    renderShell();
  }));
}

function buildPlanPayload(selectedOverride = null) {
  let modelPatches;
  try {
    modelPatches = JSON.parse(state.form.customPatches || "[]");
  } catch (error) {
    throw new Error(`Additional MODEL patch JSON is invalid: ${error.message}`);
  }
  if (!Array.isArray(modelPatches)) throw new Error("Additional MODEL patch chain must be a JSON array.");
  if (currentProfile()?.family === "Krea 2" && state.form.mode !== "enhancer_compare" && state.form.enhancer === "standard") {
    modelPatches.unshift({ class_type: "ComfyUI-Krea2T-Enhancer", inputs: { enabled: true, strength: state.form.enhancerStrength, debug: false } });
  } else if (currentProfile()?.family === "Krea 2" && state.form.mode !== "enhancer_compare" && state.form.enhancer === "advanced") {
    modelPatches.unshift({ class_type: "Krea2T-Enhancer-Advanced", inputs: { enabled: true, strength: state.form.enhancerStrength, text_scale: state.form.enhancerTextScale, debug: false } });
  }
  let apiWorkflow = null;
  if (state.workflowAdapter === "api_template") {
    try { apiWorkflow = JSON.parse(state.apiWorkflow); } catch (error) { throw new Error(`API workflow JSON is invalid: ${error.message}`); }
  }
  return {
      profile: state.form.profile,
      workflow_adapter: state.workflowAdapter,
      api_workflow: apiWorkflow,
      api_output_node_id: state.apiOutputNodeId,
      api_output_index: state.apiOutputIndex,
      objective: state.objective,
      model_name: state.form.modelName,
      clip_name: state.form.clipName,
      clip_name_2: state.form.clipName2,
      vae_name: state.form.vaeName,
      turbo_lora: { enabled: state.form.turboLora, filename: state.form.turboLoraName, strength: state.form.turboLoraStrength },
      aux_loras: state.form.auxLoras.map((item) => ({ enabled: true, filename: item.filename, strength: item.strength })),
      mode: state.form.mode,
      selected_loras: selectedOverride == null ? [...state.selected] : selectedOverride,
      trigger: state.form.trigger,
      subject_class: state.form.subjectClass,
      include_baseline: state.form.includeBaseline,
      common_strength: state.form.commonStrength,
      strengths: parseNumbers(state.form.strengths, [1]),
      prompts: state.prompts.filter((item) => item.enabled !== false && item.text.trim()),
      seeds: parseNumbers(state.form.seeds, [20260710]),
      width: state.form.width,
      height: state.form.height,
      reference_folder: state.form.referenceFolder,
      negative_prompt: state.form.negativePrompt,
      grid_mode: state.form.gridMode,
      output_prefix: state.form.outputPrefix,
      advanced: { steps: state.form.steps, cfg: state.form.cfg, sampler: state.form.sampler, scheduler: state.form.scheduler, negative_mode: state.form.negativeMode },
      model_patches: modelPatches,
  };
}

async function createAndStartRun() {
  syncSetupInputs();
  state.busy = "Validating and creating controlled run…";
  state.error = "";
  renderShell();
  try {
    const payload = buildPlanPayload();
    const planned = await jsonFetch("/loralab/v1/plan", { method: "POST", body: payload });
    state.currentRunId = planned.run.run_id;
    await jsonFetch("/loralab/v1/start", { method: "POST", body: { run_id: state.currentRunId, client_id: api.clientId } });
    state.page = "run";
    toast("LoRA Lab run started", `${planned.run.expected_cells} controlled cells`, "success");
    await refreshRun(false);
  } catch (error) {
    state.error = error.message;
    toast("Run could not start", error.message, "error");
  } finally {
    state.busy = "";
    renderShell();
  }
}

async function refreshRun(render = true) {
  if (!state.currentRunId) return;
  try {
    state.runData = await jsonFetch(`/loralab/v1/run?run_id=${encodeURIComponent(state.currentRunId)}`);
  } catch (error) {
    state.error = error.message;
  }
  if (render && state.overlay) renderShell();
}

async function runAction(action) {
  if (!state.currentRunId) return;
  if (action === "stop") {
    const confirmed = await confirmAction("Stop LoRA Lens run", "Stop submission, remove this run's pending prompts, interrupt its active prompt, and release model memory? Completed images remain reusable.");
    if (!confirmed) return;
  }
  try {
    const result = await jsonFetch("/loralab/v1/status", { method: "POST", body: { run_id: state.currentRunId, action, client_id: api.clientId } });
    if (action === "stop") {
      const remaining = result.stop_result?.remaining_prompt_ids?.length || 0;
      toast(remaining ? "Stop still settling" : "Run fully stopped", remaining ? `${remaining} prompt(s) still exiting; no new work will be submitted.` : "Queue cleared and VRAM cleanup requested.", remaining ? "warning" : "success");
    } else if (action === "free") {
      toast("VRAM cleanup requested", "ComfyUI will unload models and clear its execution cache.", "success");
    } else {
      toast("Run updated", action, "info");
    }
    await refreshRun(true);
  } catch (error) {
    state.error = error.message;
    renderShell();
  }
}

async function analyzeRun() {
  if (!state.currentRunId) return;
  state.busy = "Analyzing faces and paired uncertainty…";
  renderShell();
  try {
    await jsonFetch("/loralab/v1/analyze", { method: "POST", body: { run_id: state.currentRunId, reference_folder: state.runData?.run?.reference_folder } });
    await refreshRun(false);
    state.page = "results";
    toast("Analysis complete", state.runData?.analysis?.winner || "Results ready", "success");
  } catch (error) {
    state.error = error.message;
    toast("Analysis failed", error.message, "error");
  } finally {
    state.busy = "";
    renderShell();
  }
}

async function saveCellRating(cell) {
  const promptIndex = Number(cell.dataset.p);
  const loraIndex = Number(cell.dataset.l);
  const payload = { run_id: state.currentRunId, prompt_index: promptIndex, lora_index: loraIndex };
  cell.querySelectorAll("[data-rating]").forEach((control) => {
    const key = control.dataset.rating;
    payload[key] = key === "artifact" ? control.checked : (control.value ? Number(control.value) : null);
  });
  try {
    await jsonFetch("/loralab/v1/rating", { method: "POST", body: payload });
    if (!state.runData.ratings) state.runData.ratings = { ratings: {} };
    const key = `p${String(promptIndex).padStart(3,"0")}_l${String(loraIndex).padStart(3,"0")}`;
    state.runData.ratings.ratings[key] = payload;
  } catch (error) {
    toast("Rating not saved", error.message, "error");
  }
}

async function tournamentAction(action, extra = {}) {
  if (!state.currentRunId) return;
  state.busy = action === "vote" ? "Recording blind vote…" : "Preparing blind tournament…";
  renderShell();
  try {
    await jsonFetch("/loralab/v1/tournament", { method: "POST", body: { run_id: state.currentRunId, action, ...extra } });
    await refreshRun(false);
    if (state.runData?.tournament?.status === "complete") state.revealAutomatic = true;
  } catch (error) {
    state.error = error.message;
    toast("Tournament update failed", error.message, "error");
  } finally {
    state.busy = "";
    renderShell();
  }
}

function bindMonitor() {
  document.querySelectorAll("[data-run-action]").forEach((button) => button.addEventListener("click", () => runAction(button.dataset.runAction)));
  document.getElementById("ll-analyze")?.addEventListener("click", analyzeRun);
  bindZoomables();
}

function bindResults() {
  document.getElementById("ll-analyze")?.addEventListener("click", analyzeRun);
  document.querySelector("[data-go='run']")?.addEventListener("click", () => { state.page = "run"; renderShell(); });
  document.getElementById("ll-matrix-category")?.addEventListener("change", (event) => { state.matrixCategory = event.target.value; renderShell(); });
  document.getElementById("ll-matrix-scenario")?.addEventListener("change", (event) => { state.matrixScenario = event.target.value; renderShell(); });
  document.getElementById("ll-matrix-candidate")?.addEventListener("change", (event) => { state.matrixCandidate = event.target.value; renderShell(); });
  document.getElementById("ll-blind")?.addEventListener("change", (event) => { state.blind = event.target.checked; renderShell(); });
  document.getElementById("ll-reanalyze")?.addEventListener("click", analyzeRun);
  document.getElementById("ll-reanalyze-top")?.addEventListener("click", analyzeRun);
  document.getElementById("ll-reveal-ai")?.addEventListener("click", () => { state.revealAutomatic = true; renderShell(); });
  document.getElementById("ll-start-tournament")?.addEventListener("click", () => tournamentAction("start", { include_baseline: document.getElementById("ll-tournament-baseline")?.checked !== false, human_weight: 0.5 }));
  document.getElementById("ll-reset-tournament")?.addEventListener("click", async () => {
    const confirmed = await confirmAction("Reset blind tournament", "Delete current duel votes and reshuffle this run?");
    if (confirmed) { state.revealAutomatic = false; tournamentAction("reset", { include_baseline: state.runData?.tournament?.include_baseline !== false, human_weight: state.runData?.tournament?.human_weight ?? 0.5 }); }
  });
  document.querySelectorAll("[data-duel-choice]").forEach((button) => button.addEventListener("click", () => tournamentAction("vote", {
    choice: button.dataset.duelChoice,
    match_id: button.dataset.matchId,
    artifact_left: document.getElementById("ll-artifact-left")?.checked || false,
    artifact_right: document.getElementById("ll-artifact-right")?.checked || false,
    identity_failure_left: document.getElementById("ll-identity-failure-left")?.checked || false,
    identity_failure_right: document.getElementById("ll-identity-failure-right")?.checked || false,
    overall_choice: document.querySelector("input[name='ll-overall-choice']:checked")?.value || null,
  })));
  document.getElementById("ll-undo-tournament")?.addEventListener("click", () => tournamentAction("undo"));
  document.getElementById("ll-human-weight")?.addEventListener("input", (event) => { const label = document.getElementById("ll-human-weight-label"); if (label) label.textContent = `${event.target.value}%`; });
  document.getElementById("ll-human-weight")?.addEventListener("change", (event) => tournamentAction("weight", { human_weight: Number(event.target.value) / 100 }));
  document.querySelectorAll(".ll-cell [data-rating]").forEach((control) => control.addEventListener("change", () => saveCellRating(control.closest(".ll-cell"))));
  bindZoomables();
}

function bindHistory() {
  document.querySelectorAll("[data-open-run]").forEach((row) => row.addEventListener("click", async () => {
    state.currentRunId = row.dataset.openRun;
    state.runData = null;
    state.page = "run";
    renderShell();
    await refreshRun(true);
    if (state.runData?.analysis) { state.page = "results"; renderShell(); }
  }));
}

function bindPage() {
  if (state.page === "setup") bindSetup();
  else if (state.page === "run") bindMonitor();
  else if (state.page === "results") bindResults();
  else if (state.page === "history") bindHistory();
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!state.overlay || !state.currentRunId || state.busy) return;
    if (state.page !== "run") return;
    await refreshRun(true);
  }, 2500);
}

async function openLab() {
  ensureCss();
  if (state.overlay) return;
  const overlay = document.createElement("div");
  overlay.className = "ll-overlay";
  overlay.addEventListener("mousedown", (event) => { if (event.target === overlay) closeLab(); });
  state.overlay = overlay;
  document.body.appendChild(overlay);
  renderShell();
  try {
    await loadBootstrap(false);
    if (!state.currentRunId && state.boot.runs?.length) state.currentRunId = state.boot.runs[0].run_id;
    if (state.currentRunId) await refreshRun(false);
  } catch (error) {
    state.error = error.message;
  }
  renderShell();
  startPolling();
}

function closeLab() {
  closeViewer();
  state.overlay?.remove();
  state.overlay = null;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function renderSidebar(element) {
  ensureCss();
  element.innerHTML = `<div class="ll-side"><div class="ll-side-title">LoRA Lab</div><div class="ll-side-status" id="ll-side-status">Loading run state…</div><button class="ll-btn primary" id="ll-side-open"><i class="pi pi-chart-line"></i> Open full dashboard</button></div>`;
  element.querySelector("#ll-side-open")?.addEventListener("click", openLab);
  loadBootstrap(false).then((boot) => {
    const latest = boot.runs?.[0];
    const status = element.querySelector("#ll-side-status");
    if (!status) return;
    status.innerHTML = latest
      ? `<strong>${esc(latest.name || latest.run_id)}</strong><br><span style="color:var(--ll-muted)">${latest.progress.completed}/${latest.progress.total} cells · ${esc(latest.progress.status)}</span>`
      : `No LoRA Lab runs yet.<br><span style="color:var(--ll-muted)">${boot.loras.length} LoRAs available</span>`;
  }).catch((error) => {
    const status = element.querySelector("#ll-side-status");
    if (status) status.textContent = error.message;
  });
}

function installDirectIconHandler() {
  if (document.documentElement.dataset.loralabDirectIconHandler === "1") return;
  document.documentElement.dataset.loralabDirectIconHandler = "1";
  document.addEventListener("click", (event) => {
    if (!state.directOpen) return;
    const button = event.target?.closest?.("button");
    if (!button) return;
    const descriptor = [
      button.getAttribute("title"),
      button.getAttribute("aria-label"),
      button.getAttribute("data-tooltip"),
    ].filter(Boolean).join(" ");
    if (descriptor !== "Controlled LoRA checkpoint testing") return;
    event.preventDefault();
    event.stopImmediatePropagation();
    openLab();
  }, true);
}

document.addEventListener("keydown", (event) => {
  if (state.viewer) {
    if (event.key === "Escape") { event.preventDefault(); closeViewer(); return; }
    if (event.target?.matches?.("input,select,textarea")) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); navigateViewer(-1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); navigateViewer(1); }
    else if (event.key === "+" || event.key === "=") { event.preventDefault(); state.viewer.scale = clamp(state.viewer.scale * 1.25, .25, 8); updateViewerTransform(); }
    else if (event.key === "-") { event.preventDefault(); state.viewer.scale = clamp(state.viewer.scale * .8, .25, 8); updateViewerTransform(); }
    else if (event.key === "0") { event.preventDefault(); state.viewer.scale = 1; state.viewer.x = 0; state.viewer.y = 0; updateViewerTransform(); }
    return;
  }
  if (event.key === "Escape" && state.overlay) closeLab();
});

api.addEventListener("loralab.progress", (event) => {
  if (event.detail?.run_id === state.currentRunId && state.overlay && state.page === "run") refreshRun(true);
});
api.addEventListener("lorapromptqueue.matrix_complete", (event) => {
  if (event.detail?.run_id === state.currentRunId && state.overlay) refreshRun(true);
});

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: "loralab.open",
      label: "Open LoRA Lab",
      icon: "pi pi-chart-line",
      function: openLab,
    },
  ],
  menuCommands: [
    {
      path: ["Extensions", "LoRA Lab"],
      commands: ["loralab.open"],
    },
  ],
  async setup() {
    ensureCss();
    installDirectIconHandler();
    try {
      app.extensionManager.registerSidebarTab({
        id: SIDEBAR_ID,
        icon: "pi pi-chart-line",
        title: "LoRA Lab",
        tooltip: "Controlled LoRA checkpoint testing",
        type: "custom",
        render: renderSidebar,
      });
    } catch (error) {
      console.warn("[LoRA Lab] Sidebar registration failed:", error);
    }
  },
});
