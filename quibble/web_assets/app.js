"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="quibble-token"]').content;
let planId = null, previewed = false, reportText = "", polling = null;
let existingReview = null;
let activeExperiment = null, startingExperiment = false;
let deviceState = null;
let usingSavedDataset = false;
let dataPath = "existing", readyDataset = "";
let lastWorkflowView = "data";
let variantOriginal = null;
let variantSpecification = null;
let thresholdDefaults = null, savedThresholds = null;
function effectiveThresholds() {
  if (!thresholdDefaults) throw new Error("Threshold defaults have not loaded. Refresh before running.");
  if ($("threshold-mode").value === "defaults") return {...thresholdDefaults};
  const overrides = JSON.parse($("thresholds").value);
  if (!overrides || Array.isArray(overrides) || typeof overrides !== "object" || !Object.keys(overrides).length) {
    throw new Error("Custom thresholds must be a non-empty JSON object.");
  }
  for (const key of Object.keys(overrides)) {
    if (!Object.hasOwn(thresholdDefaults, key)) throw new Error(`Unknown threshold key: ${key}`);
    const value = overrides[key];
    if (key !== "primary" && (typeof value !== "number" || !Number.isFinite(value))) {
      throw new Error(`${key} must be a finite number.`);
    }
  }
  return {...thresholdDefaults, ...overrides};
}
function updateThresholds() {
  const smoke = $("mode").value === "smoke", custom = $("threshold-mode").value === "custom";
  $("threshold-controls").disabled = smoke;
  $("thresholds").readOnly = !custom;
  $("threshold-file").disabled = !custom;
  try {
    $("effective-thresholds").textContent = JSON.stringify(effectiveThresholds(), null, 2);
    $("threshold-status").textContent = smoke ? "Inactive: smoke runs do not perform swap checks." :
      custom ? "Custom policy. The server validates all values before training; omitted keys retain defaults." :
        "Using packaged defaults. Review these limits against your application's costs.";
    $("threshold-status").classList.remove("error");
  } catch (error) {
    $("effective-thresholds").textContent = "";
    $("threshold-status").textContent = error.message;
    $("threshold-status").classList.add("error");
  }
}
function downloadJSON(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"],
    {type: "application/json;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("mode").addEventListener("change", updateThresholds);
$("threshold-mode").onchange = () => {
  if ($("threshold-mode").value === "defaults") {
    $("thresholds").value = JSON.stringify(thresholdDefaults, null, 2);
    $("threshold-file").value = "";
  }
  updateThresholds();
};
$("thresholds").oninput = updateThresholds;
$("reset-thresholds").onclick = () => {
  $("threshold-mode").value = "defaults";
  $("thresholds").value = JSON.stringify(thresholdDefaults, null, 2);
  $("threshold-file").value = "";
  updateThresholds();
};
$("threshold-file").onchange = () => busy($("download-thresholds"), async () => {
  const file = $("threshold-file").files[0];
  if (!file) return;
  if (file.size > 16000) throw new Error("Threshold JSON must be no larger than 16,000 bytes.");
  $("thresholds").value = await file.text();
  updateThresholds();
  effectiveThresholds();
});
$("download-thresholds").onclick = () => busy($("download-thresholds"), async () =>
  downloadJSON(effectiveThresholds(), "quibble-limits.json"));
$("download-job-thresholds").onclick = () => {
  if (savedThresholds) downloadJSON(savedThresholds, "quibble-saved-limits.json");
};
function showWorkflow(view) {
  if (view !== "history") lastWorkflowView = view;
  $("workflow-steps").hidden = view === "history";
  $("nav-workflow").setAttribute("aria-pressed", String(view !== "history"));
  $("nav-history").setAttribute("aria-pressed", String(view === "history"));
  $("data-choice").hidden = view !== "data";
  $("existing-path").hidden = dataPath !== "existing";
  $("new-path").hidden = view !== "data" || dataPath !== "new";
  $("review-path").hidden = view !== "data" || dataPath !== "new";
  $("run-path").hidden = view !== "run";
  $("results-path").hidden = !["results", "history"].includes(view);
  for (const [id, value] of [["nav-data", "data"], ["nav-run", "run"],
    ["nav-results", "results"], ["nav-history", "history"]]) {
    if (value === view) $(id).setAttribute("aria-current", "step");
    else $(id).removeAttribute("aria-current");
  }
  $("results-heading").textContent = view === "history" ? "Past experiments" : "Experiment results";
}
function clearReadyDataset() {
  readyDataset = "";
  $("nav-run").disabled = true;
}
function chooseDataPath(path) {
  const changed = dataPath !== path;
  dataPath = path;
  clearReadyDataset();
  if (changed && path === "new") {
    variantOriginal = null;
    variantSpecification = null;
    $("discard-variant").hidden = true;
    planId = null;
    invalidate();
    $("scenario").value = "";
    $("spec").value = "";
    $("dataset-name").value = "";
    $("assumptions").replaceChildren();
    $("saved").replaceChildren();
  }
  $("choose-existing").setAttribute("aria-pressed", String(path === "existing"));
  $("choose-new").setAttribute("aria-pressed", String(path === "new"));
  setSavedDatasetMode(path === "existing");
  showWorkflow("data");
}
function continueToExperiment(name) {
  readyDataset = name;
  $("dataset").value = name;
  $("chosen-dataset").textContent = `Selected dataset: ${name}. The saved rows will be used unchanged.`;
  $("nav-run").disabled = false;
  showWorkflow("run");
}
$("choose-existing").onclick = () => chooseDataPath("existing");
$("choose-new").onclick = () => chooseDataPath("new");
$("nav-data").onclick = $("change-dataset").onclick = () => showWorkflow("data");
$("nav-run").onclick = () => { if (readyDataset) showWorkflow("run"); };
$("nav-results").onclick = () => showWorkflow("results");
$("nav-history").onclick = () => showWorkflow("history");
$("nav-workflow").onclick = () => showWorkflow(lastWorkflowView);
function updatePreviewButton() {
  let unchangedVariant = false;
  if (variantSpecification) {
    try { unchangedVariant = JSON.stringify(JSON.parse($("spec").value)) === variantSpecification; }
    catch { unchangedVariant = true; }
  }
  $("preview").disabled = usingSavedDataset || !planId || previewed || unchangedVariant;
}
function setSavedDatasetMode(saved) {
  usingSavedDataset = saved;
  for (const id of ["scenario", "rows", "fields", "seed", "consent", "plan", "spec"]) {
    $(id).disabled = saved;
  }
  $("dataset-name").disabled = saved || !planId;
  updatePreviewButton();
  $("reviewed").disabled = saved || !previewed;
  $("save").disabled = saved || !previewed || !$("reviewed").checked;
  $("design-mode").textContent = saved ?
    "Saved dataset. Choose Design new dataset for a new scenario, or Advanced in its review to create a recipe variant." :
    "Describe a new dataset, preview and save it, then continue to configure the experiment.";
}
function updateDeviceMethods() {
  if (!deviceState) return;
  const requested = $("device").value;
  const resolved = requested === "auto" ? (deviceState.cuda_available ? "cuda" : "cpu") : requested;
  const excluded = resolved === "cuda" ? deviceState.cuda_exclusions : {};
  for (const input of $("methods").querySelectorAll("input")) {
    input.disabled = Boolean(excluded[input.value]);
    if (input.disabled) input.checked = false;
  }
  $("device").addEventListener("change", updateDeviceMethods);
  $("device-status").textContent = `Resolved model device: ${resolved.toUpperCase()}. ` +
    (requested === "cuda" && !deviceState.cuda_available ? "CUDA is unavailable; the run will be rejected. " : "") +
    Object.entries(excluded).filter(([name]) => name !== "weak").map(([name, reason]) =>
      `Excluded ${name}: ${reason}`).join(" ") +
    " Data preparation, metrics, and reporting still use CPU. Choose at least one enabled method.";
}
function updateRunning(job) {
  activeExperiment = job || null;
  $("run").disabled = startingExperiment || Boolean(activeExperiment);
  $("run-progress").hidden = !activeExperiment && !startingExperiment;
  $("run-progress-text").textContent = startingExperiment ? "Starting experiment. Please wait..." :
    activeExperiment ? `Still working: ${activeExperiment.dataset} | ${activeExperiment.mode} | experiment ${activeExperiment.id}. Checking status every 2.5 seconds. You can close the browser, but keep the Quibble server running.` : "";
}
function bookmarkJob(identifier) {
  const url = new URL(location.href);
  url.hash = new URLSearchParams({job: identifier}).toString();
  history.replaceState(null, "", url);
  return url.href;
}

const fieldHelp = {
  scenario: "Describe the decision, what one row represents, what is known before predicting, and the outcome. Only this text and generation controls are sent when you approve provider planning.",
  rows: "Total synthetic examples to generate, including rows later used for training and evaluation. Allowed: 50 to 50,000, subject to the 2,000,000 predictor-cell limit.",
  fields: "Number of input columns used to predict the outcome. The target is an additional column: 8 predictors means 9 columns in total. Allowed: 1 to 100.",
  seed: "Controls random data generation. The same specification, seed, and software environment reproduce the same table. This is separate from training and split randomness.",
  task: "Classification answers a yes/no question using labels 0 and 1. Regression predicts a continuous synthetic score, which can be negative. The selected type overrides yes/no scenario wording; review the planner's reinterpretation and target meaning in assumptions.",
  "sample-task": "Filter all available datasets by classification (binary or multiclass) or regression. Each dataset is labeled with its task and grouped by source; all are usable existing data.",
  consent: "Allows a planning request to your configured provider, which may charge for it. Existing dataset rows and local documents are not sent. The local example needs no consent or provider request.",
  "dataset-name": "Unique name for the new dataset folder. Use a portable lowercase identifier, such as grocery_noise_v2. Existing datasets are never overwritten; changing the name requires another preview.",
  spec: "The complete generation recipe: field distributions, effects, interactions, missingness, outcome noise, and assumptions. It is data, not executable code. Every edit requires another preview before saving.",
  reviewed: "Confirm that you checked the recipe and preview. Saving writes the complete generated table, not just the 20 displayed rows, together with its specification and provenance.",
  dataset: "Choose a built-in or saved local dataset. Review selected dataset shows its actual saved rows and metadata. Use as-is reuses it without regenerating or changing rows.",
  device: "CPU pins all selected models to CPU. CUDA requires CUDA and visibly excludes methods that cannot meet GPU-only model execution. Auto chooses CUDA if available, otherwise CPU, once per experiment. No CPU fallback after GPU failure; data preparation and reporting still use CPU.",
  mode: "Smoke checks setup using at most 128 training and 32 test rows, without swap approval. Full uses the available training splits and configured comparison checks; it can take much longer.",
  cv: "Full runs only. Holdout uses one 70/30 train/test split. Three folds trains on two thirds and tests on the remaining third; two repeats uses six fits per method. Methods share evaluation rows.",
  "existing-agreed": "Accept the inspected saved dataset unchanged. Its source fingerprint is checked again on confirmation; if the source changed, review it again.",
  "full-confirmed": "Approve the selected full comparison, including possible model-weight downloads and long execution. This does not certify deployment suitability or enable cancellation.",
  jobs: "Reopen current or previous experiments from this results directory. Completed means the benchmark exited successfully; failed jobs may still have useful logs or partial reports."
};
const methodHelp = {
  histgb: "Scikit-learn histogram gradient-boosted trees. A CPU-only baseline trained on this dataset; no pretrained-weight download.",
  lightgbm: "LightGBM gradient-boosted trees, trained on this dataset. This adapter uses CPU and is excluded from CUDA runs. GPU support requires a compatible build and adapter; LightGBM's CUDA backend is not supported on Windows.",
  catboost: "CatBoost gradient-boosted trees, trained on this dataset, with categorical-feature support. Supports CPU or CUDA; GPU execution depends on your installation and resources.",
  tabpfn: "TabPFN is a pretrained tabular predictor that conditions on training examples. First use may download weights and require authentication. Quibble enforces row, feature, and class limits; full inference remains unverified here.",
  tabfm: "Google Research TabFM via its PyTorch package. A pretrained tabular predictor using CPU or CUDA. First use downloads weights; repeated inference and larger tables can be expensive.",
  kumo_tabular: "NVIDIA KumoTabular via structured-data-models. A pretrained tabular predictor using CPU or CUDA. First use downloads weights; model size and ensembles affect memory and runtime.",
  autogluon: "AutoGluon Tabular trains and combines multiple methods using a preset and time budget. Excluded from GPU-only runs because a GPU budget does not prevent CPU components. Available for CPU comparisons."
};
const helpTooltip = document.createElement("div");
helpTooltip.id = "field-help-tooltip";
helpTooltip.className = "help-tooltip";
helpTooltip.role = "tooltip";
helpTooltip.hidden = true;
document.body.append(helpTooltip);
let activeHelp = null, helpPinned = false, helpTimer = null;
function positionHelp() {
  if (!activeHelp) return;
  const width = document.documentElement.clientWidth, height = document.documentElement.clientHeight;
  helpTooltip.style.maxWidth = `${Math.max(0, width - 16)}px`;
  const anchor = activeHelp.getBoundingClientRect(), box = helpTooltip.getBoundingClientRect();
  helpTooltip.style.left = `${Math.max(8, Math.min(anchor.left, width - box.width - 8))}px`;
  helpTooltip.style.top = `${Math.max(8, anchor.bottom + box.height + 8 <= height ?
    anchor.bottom + 6 : anchor.top - box.height - 6)}px`;
}
function closeHelp() {
  clearTimeout(helpTimer);
  if (activeHelp) activeHelp.removeAttribute("aria-describedby");
  activeHelp = null; helpPinned = false; helpTooltip.hidden = true;
}
function deferCloseHelp() {
  clearTimeout(helpTimer);
  helpTimer = setTimeout(() => {
    if (!helpPinned && !helpTooltip.matches(":hover") &&
        activeHelp && !activeHelp.matches(":hover, :focus")) closeHelp();
  }, 150);
}
function helpButton(label, text) {
  const button = document.createElement("button");
  button.type = "button"; button.className = "help-button"; button.textContent = "i";
  button.setAttribute("aria-label", `Help: ${label}`);
  const show = () => {
    clearTimeout(helpTimer);
    if (activeHelp !== button) closeHelp();
    activeHelp = button;
    button.setAttribute("aria-describedby", helpTooltip.id);
    helpTooltip.textContent = text; helpTooltip.hidden = false; positionHelp();
  };
  button.addEventListener("mouseenter", show);
  button.addEventListener("focus", show);
  button.addEventListener("mouseleave", deferCloseHelp);
  button.addEventListener("blur", deferCloseHelp);
  button.addEventListener("click", () => {
    if (activeHelp === button && helpPinned) closeHelp();
    else { show(); helpPinned = true; }
  });
  return button;
}
helpTooltip.addEventListener("mouseenter", () => clearTimeout(helpTimer));
helpTooltip.addEventListener("mouseleave", deferCloseHelp);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeHelp(); });
document.addEventListener("pointerdown", event => {
  if (activeHelp && !activeHelp.contains(event.target) && !helpTooltip.contains(event.target)) closeHelp();
});
window.addEventListener("resize", positionHelp);
document.addEventListener("scroll", positionHelp, true);
for (const [id, text] of Object.entries(fieldHelp)) {
  const input = $(id), label = input.labels[0], wrapper = document.createElement("div");
  const labelText = [...label.childNodes].filter(node => node.nodeType === Node.TEXT_NODE)
    .map(node => node.textContent).join(" ").trim();
  const button = helpButton(labelText, text);
  if (input.type === "checkbox") {
    wrapper.className = "check-help";
    label.before(wrapper); wrapper.append(label, button);
  } else {
    const field = document.createElement("div");
    field.className = "help-field"; wrapper.className = "field-heading";
    label.before(field);
    label.htmlFor = id;
    field.append(wrapper); wrapper.append(label, button); field.append(input);
  }
}

function decorateReport(root) {
  const verdicts = {PASS: "pass", FAIL: "fail", INCONCLUSIVE: "inconclusive"};
  for (const node of root.querySelectorAll("strong, td")) {
    const word = node.textContent.trim();
    if (!Object.hasOwn(verdicts, word) || node.children.length) continue;
    const badge = document.createElement("span");
    badge.className = `verdict verdict-${verdicts[word]}`;
    badge.textContent = word;
    node.replaceChildren(badge);
  }
}
function message(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Quibble-Token": token},
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function controls() {
  return {scenario: $("scenario").value, rows: Number($("rows").value),
    fields: Number($("fields").value), seed: Number($("seed").value),
    task: $("task").value, consent: $("consent").checked};
}
function invalidate() {
  clearReadyDataset();
  previewed = false;
  $("reviewed").checked = false;
  $("reviewed").disabled = true;
  $("save").disabled = true;
  $("summary").replaceChildren();
  $("preview-table").replaceChildren();
  $("dataset-explanation").hidden = true;
  updatePreviewButton();
}
async function busy(button, action) {
  button.disabled = true;
  try { await action(); } catch (error) { message(error.message, true); }
  finally {
    button.disabled = false;
    $("save").disabled = !previewed || !$("reviewed").checked;
    $("use-existing").disabled = !existingReview || !$("existing-agreed").checked;
    $("run").disabled = startingExperiment || Boolean(activeExperiment);
    setSavedDatasetMode(usingSavedDataset);
  }
}
function option(value, text = value) {
  const node = document.createElement("option"); node.value = value; node.textContent = text; return node;
}
function showSample(container, result) {
  const table = document.createElement("table"), head = document.createElement("thead");
  const headings = document.createElement("tr"), body = document.createElement("tbody");
  for (const name of result.columns) {
    const th = document.createElement("th"); th.textContent = name; headings.append(th);
  }
  head.append(headings); table.append(head);
  for (const row of result.sample) {
    const line = document.createElement("tr");
    for (const name of result.columns) {
      const cell = document.createElement("td");
      cell.textContent = row[name] === null ? "(missing)" : typeof row[name] === "number" ?
        String(Number(row[name].toFixed(4))) : String(row[name]);
      line.append(cell);
    }
    body.append(line);
  }
  table.append(body); container.replaceChildren(table);
}
function explainDataset(spec, id, stage) {
  const content = $(`${id}-content`);
  content.replaceChildren();
  const paragraphs = [
    `${stage}: ${spec.name}. ${spec.rows} independent synthetic rows, ${spec.features.length} predictors, and target "${spec.target}".`,
    `Requested scenario: ${spec.scenario}`,
    spec.task === "regression" ?
      `Regression predicts "${spec.target}" as a continuous synthetic score. It can be negative; it is not a yes/no answer, calibrated probability, or automatically a quantity in real-world units. Review the assumptions for the intended interpretation.` :
      `Binary classification predicts "${spec.target}" using labels 0 and 1, sampled from a logistic outcome rule. Review the assumptions for which label represents which outcome; this is not a deterministic coverage calculation.`,
    `Inputs: ${spec.features.map(feature => `${feature.name} (${feature.description})`).join("; ")}.`,
    "Predictors are sampled independently. Outcomes combine configured effects, interactions, and noise; missingness is applied afterward. This tests the recipe's assumptions, not validated retail operations."
  ];
  for (const text of paragraphs) {
    const p = document.createElement("p"); p.textContent = text; content.append(p);
  }
  const list = document.createElement("ul");
  for (const text of spec.assumptions) {
    const li = document.createElement("li"); li.textContent = text; list.append(li);
  }
  content.append(list);
  $(id).hidden = false;
}
function loadSpecification(result) {
  planId = result.plan_id;
  $("spec").value = JSON.stringify(result.spec, null, 2);
  $("dataset-name").value = result.spec.name;
  $("dataset-name").disabled = false;
  $("preview").disabled = false;
  invalidate();
  explainDataset(result.spec, "dataset-explanation", "Proposed design, not yet saved");
  $("assumptions").replaceChildren(...result.spec.assumptions.map(text => {
    const li = document.createElement("li"); li.textContent = text; return li;
  }));
}
async function refreshState(selectedDataset, selectedJob) {
  const state = await api("/api/state");
  if (!state.cuda_exclusions || typeof state.cuda_available !== "boolean") {
    throw new Error("Restart the Quibble web server and reload this page to enable the updated GPU-only device policy.");
  }
  deviceState = state;
  if (!state.threshold_defaults) throw new Error("Restart the web server to load packaged swap thresholds.");
  if (!thresholdDefaults) {
    thresholdDefaults = state.threshold_defaults;
    $("thresholds").value = JSON.stringify(thresholdDefaults, null, 2);
  }
  updateThresholds();
  updateRunning(state.jobs.find(j => ["queued", "running"].includes(j.status)));
  $("provider").textContent = state.provider.configured ?
    `Planner: ${state.provider.name} | Model: ${state.provider.model} | Authentication: ${state.provider.authentication}${state.provider.authentication === "aad" ? ` (${state.provider.credential})` : ""} | ${state.provider.endpoint}` :
    `Planner setup needed: ${state.provider.error} You can still use the local example.`;
  $("limits").textContent = `${state.limits}. Predictor count excludes the target column.`;
  const dataset = selectedDataset || $("dataset").value;
  if (existingReview && existingReview.name !== dataset) clearExistingReview();
  if (!state.dataset_catalog) throw new Error("Restart the web server to enable the unified dataset list.");
  if (selectedDataset) {
    const selected = state.dataset_catalog.find(item => item.name === selectedDataset);
    if (selected) $("sample-task").value = selected.task;
  }
  populateDatasets(state.dataset_catalog, dataset);
  if (!$("methods").children.length) {
    for (const name of state.backends) {
      const label = document.createElement("label");
      const input = document.createElement("input"); input.type = "checkbox"; input.value = name;
      input.checked = ["histgb", "lightgbm"].includes(name);
      const choice = document.createElement("div"); choice.className = "method-choice";
      label.append(input, document.createTextNode(name)); choice.append(label);
      choice.append(helpButton(name, methodHelp[name] || "A registered prediction method. Check its documented task, device, and resource limits before running."));
      $("methods").append(choice);
    }
  }
  updateDeviceMethods();
  const job = selectedJob || $("jobs").value || new URLSearchParams(location.hash.slice(1)).get("job");
  $("jobs").replaceChildren(option("", "Select an experiment to view its report"),
    ...state.jobs.map(j => option(j.id, `${j.dataset} | ${j.mode} | ${j.status} | ${j.id}`)));
  if (state.jobs.some(j => j.id === job)) $("jobs").value = job;
  else if (job) {
    try {
      const saved = await api(`/api/jobs/${encodeURIComponent(job)}`);
      $("jobs").append(option(saved.id, `${saved.dataset} | ${saved.mode} | ${saved.status} | ${saved.id}`));
      $("jobs").value = saved.id;
    } catch (error) {
      message(`Could not reopen experiment ${job}: ${error.message}`, true);
      throw error;
    }
  }
}
async function propose(local) {
  if (!local && !$("scenario").value.trim()) throw new Error("Describe the dataset you want to design first.");
  planId = null;
  invalidate();
  $("spec").value = "";
  $("dataset-name").value = "";
  $("dataset-name").disabled = true;
  $("preview").disabled = true;
  $("assumptions").replaceChildren();
  $("saved").replaceChildren();
  message(local ? "Preparing local recommendation..." : "Requesting a specification from your configured provider...");
  const result = await api(local ? "/api/recommend" : "/api/plan", controls());
  loadSpecification(result);
  message(local ? "Local grocery example loaded, not a design from your scenario. Review before previewing." :
    "Your scenario design is ready. Review assumptions, field definitions, and the outcome rule before previewing.");
}
function populateDatasets(catalog, selected = "") {
  $("dataset").replaceChildren(option("", "Choose a dataset"));
  for (const group of [...new Set(catalog.map(item => item.group))]) {
    const items = catalog.filter(item => item.group === group && item.task === $("sample-task").value);
    if (!items.length) continue;
    const node = document.createElement("optgroup"); node.label = group;
    node.append(...items.map(item => option(item.name, `${item.name} (${item.kind})`)));
    $("dataset").append(node);
  }
  if ([...$("dataset").options].some(item => item.value === selected)) $("dataset").value = selected;
}
function updateTaskExample() {
  $("task-example").textContent = $("task").value === "regression" ?
    "Regression example: how high is inventory pressure before delivery? Predict a continuous synthetic score; higher means greater shortage pressure, and negative values are allowed." :
    "Binary example: will inventory fall short before the next delivery? Predict 0 (covers demand) or 1 (shortfall).";
}
$("sample-task").addEventListener("change", () => {
  clearReadyDataset();
  clearExistingReview();
  populateDatasets(deviceState.dataset_catalog);
});
$("plan").onclick = () => busy($("plan"), () => propose(false));
$("task").addEventListener("change", updateTaskExample);
$("spec").addEventListener("input", () => {
  invalidate();
  try { $("dataset-name").value = JSON.parse($("spec").value).name || ""; }
  catch { $("dataset-name").value = ""; }
});
$("dataset-name").addEventListener("input", () => {
  invalidate();
  try {
    const spec = JSON.parse($("spec").value);
    spec.name = $("dataset-name").value;
    $("spec").value = JSON.stringify(spec, null, 2);
    updatePreviewButton();
  } catch { message("Fix the specification JSON before changing its name.", true); }
});
$("reviewed").onchange = () => { $("save").disabled = !previewed || !$("reviewed").checked; };
$("preview").onclick = () => busy($("preview"), async () => {
  invalidate();
  message("Generating the full table locally and checking the preview...");
  const spec = JSON.parse($("spec").value);
  const result = await api("/api/preview", {spec, plan_id: planId});
  $("assumptions").replaceChildren(...spec.assumptions.map(text => {
    const li = document.createElement("li"); li.textContent = text; return li;
  }));
  $("summary").textContent = `${result.warning}\n${result.rows} rows, ${result.predictors} predictors + target. Outcome summary: ${JSON.stringify(result.target_summary)}. Missing counts: ${JSON.stringify(result.missing)}`;
  explainDataset(spec, "dataset-explanation", "Generated preview, not yet saved");
  showSample($("preview-table"), result);
  previewed = true; $("reviewed").disabled = false;
  updatePreviewButton();
  message("Preview validated. Confirm review to save the complete dataset. Any specification edit requires a new preview.");
});
$("save").onclick = () => busy($("save"), async () => {
  const result = await api("/api/save", {spec: JSON.parse($("spec").value), plan_id: planId, reviewed: $("reviewed").checked});
  explainDataset(JSON.parse($("spec").value), "dataset-explanation", "Saved dataset");
  $("saved").textContent = `Saved ${result.rows} rows and ${result.predictors} predictors to ${result.directory}. SHA-256: ${result.sha256} `;
  const link = document.createElement("a"); link.href = `/api/datasets/${encodeURIComponent(result.name)}`; link.download = `${result.name}.csv`; link.textContent = "Download CSV";
  $("saved").append(link);
  await refreshState(result.name);
  $("reviewed").checked = false; previewed = false;
  variantOriginal = null;
  variantSpecification = null;
  $("discard-variant").hidden = true;
  setSavedDatasetMode(true);
  continueToExperiment(result.name);
  message("Dataset saved. Choose methods and start the experiment.");
});
function clearExistingReview() {
  existingReview = null;
  $("existing-review").hidden = true;
  $("existing-explanation").hidden = true;
  $("existing-agreed").checked = false;
  $("use-existing").disabled = true;
  $("variant-options").open = false;
  $("variant-options").hidden = true;
  $("edit-existing").hidden = true;
}
$("dataset").onchange = () => {
  clearReadyDataset();
  clearExistingReview();
  setSavedDatasetMode(Boolean($("dataset").value));
};
$("inspect-existing").onclick = () => busy($("inspect-existing"), async () => {
  clearExistingReview();
  const name = $("dataset").value;
  const result = await api("/api/inspect", {dataset: name});
  if ($("dataset").value !== name) return;
  showExistingReview(result);
  message("Review the saved rows, confirm, then continue to experiment.");
});
function showExistingReview(result) {
  existingReview = result;
  if (result.generation) explainDataset(result.generation.specification, "existing-explanation", "Saved dataset");
  $("existing-summary").textContent = `${result.name}: ${result.rows} saved rows, ${result.predictors} predictors; ${result.task}; target: ${result.target}. Outcome: ${JSON.stringify(result.target_summary)}. Missing counts: ${JSON.stringify(result.missing)}.`;
  showSample($("existing-table"), result);
  const metadata = {};
  if (result.source) metadata.source = result.source;
  if (result.generation) metadata.generation = result.generation;
  const hasMetadata = Object.keys(metadata).length > 0;
  $("existing-metadata").textContent = hasMetadata ? JSON.stringify(metadata, null, 2) : "";
  $("existing-metadata-details").hidden = !hasMetadata;
  $("existing-metadata-details").open = false;
  $("existing-metadata-title").textContent = result.generation ?
    "Saved source and generation specification" : "Saved source details";
  $("existing-source-note").hidden = hasMetadata;
  $("existing-source-note").textContent = hasMetadata ? "" :
    "Built-in benchmark dataset. No synthetic generation recipe applies; its saved rows are ready to use.";
  $("edit-existing").hidden = !result.generation;
  $("variant-options").hidden = !result.generation;
  $("existing-review").hidden = false;
}
$("existing-agreed").onchange = () => {
  $("use-existing").disabled = !existingReview || !$("existing-agreed").checked;
};
$("use-existing").onclick = () => busy($("use-existing"), async () => {
  if (!existingReview || existingReview.name !== $("dataset").value) throw new Error("Review the selected dataset first.");
  const result = await api("/api/use-existing", {dataset: existingReview.name,
    fingerprint: existingReview.fingerprint, reviewed: $("existing-agreed").checked});
  message(`Using ${result.name} as-is. No rows were regenerated or changed. Choose methods and start the experiment.`);
  continueToExperiment(result.name);
});
$("edit-existing").onclick = () => busy($("edit-existing"), async () => {
  if (!existingReview || existingReview.name !== $("dataset").value) throw new Error("Review the selected dataset first.");
  const original = existingReview.name;
  const result = await api("/api/edit-existing", {dataset: original});
  chooseDataPath("new");
  variantOriginal = original;
  variantSpecification = JSON.stringify(result.spec);
  $("discard-variant").hidden = false;
  setSavedDatasetMode(false);
  loadSpecification(result);
  $("dataset-name").focus();
  $("dataset-name").scrollIntoView({behavior: "smooth", block: "center"});
  message("Loaded the saved recipe. Change its name or recipe to enable preview, or discard changes to use the existing data unchanged.");
});
$("discard-variant").onclick = () => busy($("discard-variant"), async () => {
  if (!variantOriginal) throw new Error("No original dataset is associated with this variant.");
  const name = variantOriginal;
  const review = await api("/api/inspect", {dataset: name});
  await api("/api/use-existing", {dataset: name, fingerprint: review.fingerprint, reviewed: true});
  planId = null;
  invalidate();
  $("spec").value = "";
  $("scenario").value = "";
  $("dataset-name").value = "";
  $("assumptions").replaceChildren();
  $("saved").replaceChildren();
  variantOriginal = null;
  variantSpecification = null;
  $("discard-variant").hidden = true;
  chooseDataPath("existing");
  await refreshState(name);
  clearExistingReview();
  showExistingReview(review);
  $("existing-agreed").checked = true;
  $("use-existing").disabled = false;
  continueToExperiment(name);
  message(`Unsaved variant discarded. Using ${name} unchanged; no saved files were modified.`);
});
$("run").onclick = () => busy($("run"), async () => {
  if (!readyDataset) throw new Error("Choose and review data before configuring an experiment.");
  let thresholds = null;
  if ($("mode").value === "full" && $("threshold-mode").value === "custom") {
    effectiveThresholds();
    thresholds = JSON.parse($("thresholds").value);
  }
  startingExperiment = true; updateRunning(activeExperiment);
  try {
  const result = await api("/api/run", {dataset: readyDataset,
    backends: [...$("methods").querySelectorAll("input:checked:not(:disabled)")].map(e => e.value),
    mode: $("mode").value, device: $("device").value, cv: $("cv").value,
    thresholds,
    full_confirmed: $("full-confirmed").checked});
  await refreshState(undefined, result.id);
  showWorkflow("results");
  message("Experiment started. You can inspect progress and any partial report below.");
  await watch();
  } finally {
    startingExperiment = false; updateRunning(activeExperiment);
  }
});
async function refreshJob() {
  if (!$("jobs").value) {
    $("job-status").textContent = "";
    $("job-return").replaceChildren();
    $("log").textContent = "Choose an experiment.";
    $("report").replaceChildren();
    $("job-policy").hidden = true;
    savedThresholds = null;
    reportText = "";
    $("download-report").disabled = true;
    const url = new URL(location.href);
    url.hash = "";
    history.replaceState(null, "", url);
    return null;
  }
  const job = await api(`/api/jobs/${encodeURIComponent($("jobs").value)}`);
  savedThresholds = job.thresholds || null;
  $("job-policy").hidden = !savedThresholds;
  $("job-thresholds").textContent = savedThresholds ? JSON.stringify(savedThresholds, null, 2) : "";
  $("job-policy-source").textContent = job.thresholds_default ?
    "Packaged defaults at run time are placeholders. Review for your application; PASS is not a production-safety guarantee." :
    job.threshold_provenance?.kind === "partial" ?
      `Partial custom policy. Inherited packaged defaults: ${job.threshold_provenance.inherited.join(", ")}.` :
      job.threshold_provenance?.kind === "explicit" ?
        "Explicit custom policy used for every setting. PASS is not a production-safety guarantee." :
        "Custom policy used; setting-origin metadata is unavailable for this older experiment.";
  const link = document.createElement("a");
  link.href = bookmarkJob(job.id); link.textContent = `Bookmark experiment ${job.id}`;
  $("job-return").replaceChildren(link, document.createTextNode(
    ` | ${job.backends?.join(", ") || "Selected methods"} | Saved files: ${job.directory}. Return using this link and the same Quibble server address. Reports persist after restart when using the same output directory; interrupted jobs do not resume.`));
  const exclusions = Object.entries(job.excluded_backends || {}).map(([name, reason]) =>
    `Excluded ${name}: ${reason}`).join(" ");
  const status = document.createElement("span"), metadata = document.createElement("span");
  status.className = "job-state";
  status.dataset.state = job.status;
  status.textContent = job.status.toUpperCase();
  metadata.textContent = `${job.dataset} | ${job.mode} | requested device: ${job.device}${job.device_resolved ? ` | resolved: ${job.device_resolved}` : ""}${job.exit_code === null ? "" : ` | exit ${job.exit_code}`}${job.error ? ` | ${job.error}` : ""}${exclusions ? ` | ${exclusions}` : ""}`;
  $("job-status").replaceChildren(status, metadata);
  $("log").textContent = job.log;
  reportText = job.report || "";
  if (job.report_html) {
    // Only the server's HTML-disabled, sanitized Markdown renderer supplies this field.
    $("report").innerHTML = job.report_html;
    decorateReport($("report"));
  } else if (reportText) {
    const notice = document.createElement("p"), source = document.createElement("pre");
    notice.textContent = "Your report is available. Restart the Quibble web server and reload this page to enable formatted Markdown. The original report is shown below and can be downloaded.";
    source.textContent = reportText;
    $("report").replaceChildren(notice, source);
  } else if (job.status === "completed") {
    $("report").textContent = "This experiment is marked completed, but its report could not be found. Check the saved-files location and execution log.";
  } else {
    $("report").textContent = "No report yet. Inspect the execution log for progress or failures.";
  }
  $("download-report").disabled = !reportText;
  return job;
}
async function watch() {
  clearInterval(polling);
  const job = await refreshJob();
  if (activeExperiment || job && ["queued", "running"].includes(job.status)) {
    polling = setInterval(async () => {
      try {
        await refreshState();
        const update = await refreshJob();
        if (!activeExperiment && (!update || !["queued", "running"].includes(update.status))) {
          clearInterval(polling); await refreshState();
          message(update?.status === "completed" ? "Experiment completed. Read the report and its limitations." :
            "Experiment did not complete successfully. Read the log and partial report.", update?.status !== "completed");
        }
      } catch (error) {
        clearInterval(polling);
        $("run-progress").hidden = false;
        $("run-progress-text").textContent = "Status connection lost. This does not mean the experiment stopped. Keep the server running and use Refresh or reopen your experiment link to reconnect.";
        message(error.message, true);
      }
    }, 2500);
  }
}
$("jobs").onchange = () => watch().catch(error => message(error.message, true));
$("refresh").onclick = () => busy($("refresh"), async () => { await refreshState(); await watch(); });
$("download-report").onclick = () => {
  const url = URL.createObjectURL(new Blob([reportText], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = "quibble-report.md"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.querySelector("#methods").previousElementSibling.append(
  helpButton("Prediction methods", "Choose one to four methods to compare on identical evaluation rows. Installed packages, first-use downloads, authentication, and resource limits vary by method. These are prediction methods, not the language model used to draft a scenario."));
showWorkflow("data");
refreshState().then(() => {
  message("Choose Use existing dataset or Design new dataset to begin.");
  if (new URLSearchParams(location.hash.slice(1)).get("job")) showWorkflow("results");
  return watch();
})
  .catch(error => message(error.message, true));
