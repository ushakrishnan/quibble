"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="quibble-token"]').content;
let planId = null, previewed = false, reportText = "", polling = null;
let existingReview = null;
let activeExperiment = null, startingExperiment = false, startUncertain = false;
let deviceState = null;
let usingSavedDataset = false;
let dataPath = "existing", readyDataset = "";
let routeState = {page: "home"}, stateReady = false, selectedJobId = "";
let lastStartedJob = null, watchPromise = null, renderedReport = null, failedViewedJob = null;
const busyActions = new Set(), actionHandlers = {};
let variantOriginal = null;
let variantSpecification = null;
let thresholdDefaults = null, savedThresholds = null;
let uploadId = null, uploadReview = null, readyFingerprint = null;
let relationalStage = null, relationalReview = null, relationalSignature = null;
let verificationJob = null, verificationArtifact = null;
let verificationPending = false;
let environmentResult = null, comparisonResult = null;
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
  clearFinalApproval();
  $("threshold-mode").value = "defaults";
  $("thresholds").value = JSON.stringify(thresholdDefaults, null, 2);
  $("threshold-file").value = "";
  updateThresholds();
  renderActions();
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
function parseWorkbenchRoute(hash) {
  const fragment = hash.replace(/^#/, "");
  if (!fragment) return {page: "home"};
  if (!fragment.includes("=")) return null;
  const params = new URLSearchParams(fragment);
  const pages = ["home", "data", "run", "results", "history", "backends", "settings", "verify", "compare"];
  for (const key of params.keys()) {
    if (!["page", "step", "source", "job"].includes(key) || params.getAll(key).length !== 1) {
      throw new Error("Unknown or repeated workbench route parameter.");
    }
  }
  const page = params.get("page") || (params.has("job") ? "results" : "home");
  if (!pages.includes(page)) throw new Error("Unknown workbench page.");
  const route = {page};
  if (page === "run") {
    route.step = params.get("step") || "data";
    if (!["data", "methods", "start"].includes(route.step)) throw new Error("Unknown experiment step.");
  } else if (params.has("step")) throw new Error("This page has no experiment step.");
  if (params.has("source")) {
    route.source = params.get("source");
    if (!(page === "data" || page === "run" && route.step === "data") ||
        !["existing", "new", "upload"].includes(route.source)) throw new Error("Unknown data source route.");
  }
  if (params.has("job")) {
    route.job = params.get("job");
    if (page !== "results" || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$/.test(route.job)) {
      throw new Error("Invalid report route.");
    }
  }
  if (page === "results" && !route.job) throw new Error("Choose an experiment from history to open its report.");
  return route;
}
function workbenchHash(route) {
  const params = new URLSearchParams({page: route.page});
  for (const key of ["step", "source", "job"]) if (route[key]) params.set(key, route[key]);
  return `#${params}`;
}
function clearFinalApproval() {
  $("full-confirmed").checked = false;
  if ($("tuning-confirmed")) $("tuning-confirmed").checked = false;
  lastStartedJob = null;
}
function updateTuning() {
  const tuned = $("tier").value === "tuned";
  $("tuning-controls").hidden = !tuned;
  $("tuning-approval").hidden = !tuned;
  const trials = Number($("tune-trials").value), folds = Number($("tune-folds").value);
  $("tuning-cost").textContent = `Equal trials, not equal compute: ${trials} trials × ${folds} inner folds plus refit = ${trials * folds + 1} fits per tree per outer split when all succeed. Only outer-training rows are used. The optional cooperative deadline cannot interrupt a fit; the outer worker timeout is authoritative.`;
}
function runRequest(approval = false) {
  if (!readyDataset) throw new Error("Choose and review data first.");
  const backends = [...$("methods").querySelectorAll("input:checked:not(:disabled)")].map(e => e.value);
  if (backends.length < 1 || backends.length > 4) throw new Error("Choose one to four enabled prediction methods.");
  const mode = $("mode").value, device = $("device").value, cv = $("cv").value;
  if (!["smoke", "full"].includes(mode) || !["cpu", "auto", "cuda"].includes(device) ||
      !["holdout", "3x1", "3x2"].includes(cv)) throw new Error("Choose valid experiment settings.");
  if (device === "cuda" && !deviceState?.cuda_available) throw new Error("CUDA is unavailable. Choose CPU or Auto.");
  let thresholds = null;
  if (mode === "full") {
    effectiveThresholds();
    if ($("threshold-mode").value === "custom") thresholds = JSON.parse($("thresholds").value);
    if (approval && !$("full-confirmed").checked) throw new Error("Approve this exact full-run configuration before starting.");
  }
  const tier = $("tier")?.value || "defaults";
  if (!["defaults", "tuned"].includes(tier)) throw new Error("Choose a valid comparison tier.");
  const tuning = tier === "defaults" ? {tune_trials: 20, tune_folds: 3, tune_seed: 0,
    tune_metric: "auto", tune_timeout_s: 0} : {tune_trials: Number($("tune-trials")?.value ?? 20),
    tune_folds: Number($("tune-folds")?.value ?? 3), tune_seed: Number($("tune-seed")?.value ?? 0),
    tune_metric: $("tune-metric")?.value || "auto",
    tune_timeout_s: Number($("tune-timeout-s")?.value ?? 0)};
  if (tier === "tuned") {
    for (const [key, low, high] of [["tune_trials", 1, 1000], ["tune_folds", 2, 20],
      ["tune_seed", 0, 4294967295]]) {
      if (!Number.isInteger(tuning[key]) || tuning[key] < low || tuning[key] > high) {
        throw new Error(`${key} must be an integer in ${low}..${high}.`);
      }
    }
    if (!Number.isFinite(tuning.tune_timeout_s) || tuning.tune_timeout_s < 0) throw new Error("Invalid cooperative deadline.");
    if (!["auto", "log_loss", "rmse"].includes(tuning.tune_metric)) throw new Error("Invalid selection metric.");
    if (!backends.some(name => ["histgb", "lightgbm", "catboost"].includes(name))) throw new Error("Select at least one supported tree for tuning.");
    if (approval && !$("tuning-confirmed").checked) throw new Error("Approve this exact tuning configuration before starting.");
  }
  return {dataset: readyDataset, fingerprint: readyFingerprint, backends, mode, device, cv,
    tier, ...tuning, tuning_confirmed: $("tuning-confirmed")?.checked === true,
    thresholds, full_confirmed: $("full-confirmed").checked};
}
function workflowAction() {
  const data = routeState.page === "data" || routeState.page === "run" && routeState.step === "data";
  if (data) {
    let id, label, reason;
    if (dataPath === "existing") {
      [id, label, reason] = existingReview ?
        ["use-existing", "Continue to methods", "Confirm that you reviewed the current saved sample."] :
        ["inspect-existing", "Review selected dataset", "Choose a dataset to inspect."];
    } else if (dataPath === "new") {
      [id, label, reason] = !planId ? ["plan", "Propose design", "Enter a scenario and approve sending it to the configured provider."] :
        !previewed ? ["preview", "Generate and inspect preview", "Change an unchanged variant, then generate a fresh preview."] :
          ["save", "Save and continue", "Confirm review of the current specification and sample."];
    } else {
      [id, label, reason] = !uploadId ? ["upload-send", "Upload and inspect locally", "Choose a CSV or Parquet file."] :
        !uploadReview ? ["upload-validate", "Validate roles and splits", "Review the target, roles and evaluation design."] :
          ["upload-save", "Save and continue", "Confirm review of the validated upload."];
    }
    let disabled = $(id).disabled;
    if (id === "inspect-existing") disabled ||= !$("dataset").value;
    if (id === "plan") disabled ||= !deviceState?.provider.configured || !$("scenario").value.trim() || !$("consent").checked;
    if (id === "plan" && !deviceState?.provider.configured) reason = "Synthetic planning is unavailable. Use existing data or upload a table, or configure a provider on the server.";
    if (id === "upload-send") disabled ||= !$("upload-file").files.length || $("choose-upload").disabled;
    return {id, label, disabled, reason: disabled ? reason : "This action does not start an experiment.",
      back: {page: "home"}};
  }
  if (routeState.page !== "run") return null;
  if (routeState.step === "start" && (activeExperiment || startingExperiment)) {
    return {id: "open-active", label: "View log and report", disabled: startingExperiment,
      reason: startingExperiment ? "Starting once. Do not retry an uncertain request." : "Execution is in progress; another start is disabled.",
      back: {page: "run", step: "methods"}};
  }
  if (routeState.step === "start" && lastStartedJob) {
    return {id: "open-report", label: "Open report", disabled: false,
      reason: "Read the execution status, partial evidence and limitations. Completion is not approval.",
      back: {page: "run", step: "methods"}};
  }
  const methods = routeState.step === "methods";
  let reason = methods ? "Review these settings before continuing. No models start on Next." :
    "Start sends this request once. Keep the server running.";
  let disabled = false;
  try { runRequest(!methods); } catch (error) { disabled = true; reason = error.message; }
  if (!methods && startUncertain) { disabled = true; reason = "The start response is uncertain. Refresh status before trying again."; }
  return {id: methods ? "next" : "run", label: methods ? "Next: Start and monitor" : "Start experiment",
    disabled, reason, back: {page: "run", step: methods ? "data" : "methods"}};
}
function renderActions() {
  const action = workflowAction();
  for (const position of ["top", "bottom"]) {
    $(`workflow-actions-${position}`).hidden = !action;
    if (!action) continue;
    const primary = $(position === "top" ? "workflow-primary-top" : "run");
    primary.textContent = action.label;
    primary.disabled = action.disabled || busyActions.size > 0;
    primary.setAttribute("aria-describedby", `workflow-reason-${position}`);
    $(`workflow-reason-${position}`).textContent = busyActions.size ? "Working. Please wait for this action to finish." : action.reason;
    $(`workflow-back-${position}`).disabled = busyActions.size > 0;
  }
  if (routeState.page === "run" && routeState.step === "start") {
    try { $("start-summary").textContent = JSON.stringify(runRequest(), null, 2); }
    catch (error) { $("start-summary").textContent = error.message; }
    $("full-confirmed").disabled = $("mode").value !== "full" || Boolean(activeExperiment) || startingExperiment;
    if ($("tuning-confirmed")) $("tuning-confirmed").disabled =
      $("tier").value !== "tuned" || Boolean(activeExperiment) || startingExperiment;
  }
}
function registerAction(id, handler) {
  actionHandlers[id] = () => busy($(id), handler);
  $(id).onclick = actionHandlers[id];
}
function dispatchWorkflowAction() {
  const action = workflowAction();
  if (!action || action.disabled || busyActions.size) return;
  if (action.id === "next") navigate({page: "run", step: "start"});
  else if (["open-active", "open-report"].includes(action.id)) {
    const job = activeExperiment?.id || lastStartedJob;
    if (job) {
      navigate({page: "results", job});
      $("execution-log").open = action.id === "open-active";
    }
  } else actionHandlers[action.id]();
}
function renderRoute(focus = false) {
  const {page, step} = routeState;
  const data = page === "data" || page === "run" && step === "data";
  const titles = {home: "Home", data: "Datasets", history: "Experiment history", results: "Experiment report",
    backends: "Backends and verification", settings: "Settings and theme",
    verify: "Verify saved evidence", compare: "Compare experiments"};
  const title = page === "run" ? {data: "Choose data", methods: "Methods and settings", start: "Start and monitor"}[step] :
    page === "data" && dataPath === "new" ? "Design new dataset" : titles[page];
  $("page-heading").textContent = title;
  $("workbench-breadcrumb").textContent = page === "run" ? `Run experiment / ${title}` : title;
  document.title = `Quibble | ${title}`;
  for (const name of ["home", "history", "backends", "settings", "verify", "compare"]) $(`${name}-path`).hidden = page !== name;
  $("workflow-steps").hidden = page !== "run";
  $("data-choice").hidden = !data;
  $("existing-path").hidden = dataPath !== "existing";
  $("upload-path").hidden = dataPath !== "upload";
  $("new-path").hidden = !data || dataPath !== "new";
  $("review-path").hidden = !data || dataPath !== "new";
  $("run-path").hidden = page !== "run" || step !== "methods";
  $("start-path").hidden = page !== "run" || step !== "start";
  $("results-path").hidden = page !== "results";
  for (const [id, value] of [["nav-data", "data"], ["nav-run", "run"],
    ["nav-results", "results"]]) {
    const current = {data: "data", run: "methods", results: "start"}[value];
    if (page === "run" && current === step) $(id).setAttribute("aria-current", "step");
    else $(id).removeAttribute("aria-current");
  }
  for (const link of $("workbench-sidebar").querySelectorAll('a[href^="#page="]')) {
    const target = parseWorkbenchRoute(link.hash);
    const current = target.page === page && (target.step || "") === (step || "") &&
      (page !== "data" || (target.source || "existing") === dataPath || target.source === "existing" && dataPath === "upload");
    if (current) {
      link.setAttribute("aria-current", "page");
      if (link.closest("details") && !compactRail()) link.closest("details").open = true;
    }
    else link.removeAttribute("aria-current");
  }
  renderActions();
  if (focus) $("page-heading").focus();
}
function navigate(requested, {replace = false, focus = true} = {}) {
  let route = {...requested}, warning = "";
  if (route.page === "run" && route.step !== "data" && !readyDataset &&
      !(route.step === "start" && activeExperiment)) {
    warning = "Choose and explicitly review data before opening this step.";
    route = {page: "run", step: "data"};
  } else if (route.page === "run" && route.step === "start" && !activeExperiment) {
    try { runRequest(); } catch (error) {
      warning = error.message;
      route = {page: "run", step: "methods"};
    }
  }
  if (route.source && route.source !== dataPath) chooseDataPath(route.source, false);
  routeState = route;
  if (route.page === "verify" && selectedJobId && !$("verify-job").value) {
    if (![...$("verify-job").options].some(item => item.value === selectedJobId)) $("verify-job").append(option(selectedJobId));
    $("verify-job").value = selectedJobId;
    clearVerification();
  }
  if (route.page === "results") {
    failedViewedJob = null;
    selectedJobId = route.job;
    if (![...$("jobs").options].some(item => item.value === route.job)) $("jobs").append(option(route.job));
    $("jobs").value = route.job;
  }
  const hash = workbenchHash(route);
  if (location.hash !== hash) history[replace ? "replaceState" : "pushState"](null, "", hash);
  closeNavigation(false);
  renderRoute(focus);
  if (warning) message(warning, true);
  if (stateReady && route.page === "results") watch().catch(error => message(error.message, true));
}
function restoreRoute() {
  try {
    const route = parseWorkbenchRoute(location.hash);
    if (route) navigate(route, {replace: true, focus: false});
  } catch (error) {
    navigate({page: "home"}, {replace: true, focus: false});
    message(`${error.message} Use Home or Experiment history to continue.`, true);
  }
}
function showWorkflow(view) {
  navigate({page: "run", step: view === "run" ? "methods" : "data", ...(view === "data" ? {source: dataPath} : {})});
}
function clearReadyDataset() {
  readyDataset = "";
  readyFingerprint = null;
  clearFinalApproval();
}
function chooseDataPath(path, navigateToData = true) {
  const changed = dataPath !== path;
  dataPath = path;
  if (changed) clearReadyDataset();
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
  $("choose-upload").setAttribute("aria-pressed", String(path === "upload"));
  setSavedDatasetMode(path !== "new");
  if (navigateToData) {
    if (routeState.page === "data") navigate({page: "data", source: path});
    else showWorkflow("data");
  }
}
function continueToExperiment(name, fingerprint = null) {
  readyDataset = name;
  readyFingerprint = fingerprint;
  $("dataset").value = name;
  $("chosen-dataset").textContent = `Selected dataset: ${name}. The saved rows will be used unchanged.`;
  clearFinalApproval();
  showWorkflow("run");
}
$("choose-existing").onclick = () => chooseDataPath("existing");
$("choose-new").onclick = () => chooseDataPath("new");
$("choose-upload").onclick = () => chooseDataPath("upload");
$("nav-data").onclick = $("change-dataset").onclick = () => showWorkflow("data");
$("nav-run").onclick = () => showWorkflow("run");
$("nav-results").onclick = () => navigate({page: "run", step: "start"});
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
  $("upload-cancel").disabled = !uploadId;
  $("upload-save").disabled = !uploadReview || !$("upload-confirmed").checked;
  $("run-progress").hidden = !activeExperiment && !startingExperiment && !startUncertain;
  $("run-progress-text").textContent = startingExperiment ? "Starting experiment. Please wait..." :
    activeExperiment ? `Execution ${activeExperiment.status}: ${activeExperiment.dataset} | ${activeExperiment.mode} | experiment ${activeExperiment.id}. Keep the Quibble server running.` :
      startUncertain ? "Start response unavailable. Do not retry until you reconnect to server state." : "";
  $("active-job-link").hidden = !activeExperiment;
  if (activeExperiment) $("active-job-link").href = workbenchHash({page: "results", job: activeExperiment.id});
  renderActions();
}
function bookmarkJob(identifier) {
  const url = new URL(location.href);
  url.hash = workbenchHash({page: "results", job: identifier});
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
  tabicl_v2: "Experimental TabICLv2 2.2.0 adapter with pinned, hash-checked classifier/regressor checkpoints. No fine-tuning or automatic device fallback. Weights must already be cached unless the server explicitly enables QUIBBLE_TABICL_ALLOW_DOWNLOAD=1. Real fits remain unverified.",
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
  busyActions.add(button.id);
  renderActions();
  const uploadAction = button.id.startsWith("upload-");
  if (uploadAction) $("upload-summary").classList.remove("error");
  try { await action(); } catch (error) {
    message(error.message, true);
    if (uploadAction) {
      $("upload-summary").textContent = error.message;
      $("upload-summary").classList.add("error");
    }
  }
  finally {
    button.disabled = false;
    $("save").disabled = !previewed || !$("reviewed").checked;
    $("use-existing").disabled = !existingReview || !$("existing-agreed").checked;
    setSavedDatasetMode(usingSavedDataset);
    busyActions.delete(button.id);
    renderActions();
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
async function refreshState(selectedDataset) {
  const state = await api("/api/state");
  if (!state.cuda_exclusions || typeof state.cuda_available !== "boolean") {
    throw new Error("Restart the Quibble web server and reload this page to enable the updated GPU-only device policy.");
  }
  deviceState = state;
  const datasetErrors = state.dataset_errors || [];
  $("dataset-errors").hidden = !datasetErrors.length;
  $("dataset-errors").textContent = datasetErrors.map(item => `${item.name}: ${item.error}`).join(" ");
  if (state.upload_limits) {
    const limits = state.upload_limits;
    $("choose-upload").disabled = false;
    $("upload-limits").textContent = `Limits: ${limits.file_bytes / 1048576} MiB per file; ${limits.rows.toLocaleString()} rows; ${limits.columns} columns; ${limits.decoded_bytes / 1048576} MiB decoded table; ${limits.preview_rows} preview rows; ${limits.cell_chars} characters per preview cell; ${limits.column_name_chars} characters per column name. Parsing: ${limits.timeout_s} seconds and ${limits.worker_bytes / 1073741824} GiB worker memory. At most ${limits.staged_files} staged uploads, expiring after one hour. UTF-8 comma-separated CSV (BOM accepted) or Parquet only.`;
  }
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
      const metadata = state.backend_metadata?.[name];
      let terms = "";
      if (metadata) {
        terms = ` Code: ${metadata.library.license}. Weights: ${metadata.weights.license}. ${metadata.weights.restrictions} Checkpoint: ${metadata.weights.identity_status}.`;
      }
      choice.append(helpButton(name, (methodHelp[name] || "A registered prediction method. Check its documented task, device, and resource limits before running.") + terms));
      $("methods").append(choice);
    }
  }
  updateDeviceMethods();
  const job = selectedJobId;
  $("jobs").replaceChildren(option("", "Select an experiment to view its report"),
    ...state.jobs.map(j => option(j.id, `${j.dataset} | ${j.mode} | ${j.status} | ${j.id}`)));
  if (state.jobs.some(j => j.id === job)) $("jobs").value = job;
  else if (job) {
    $("jobs").append(option(job, job));
    $("jobs").value = job;
  }
  renderCatalogState(state);
  for (const id of ["verify-job", "compare-left", "compare-right"]) {
    const previous = $(id).value;
    $(id).replaceChildren(option("", "Choose saved evidence"), ...state.jobs.map(j =>
      option(j.id, `${j.dataset} | ${j.mode} | ${j.status} | ${j.id}`)));
    if (previous) {
      if (!state.jobs.some(job => job.id === previous)) $(id).append(option(previous));
      $(id).value = previous;
    }
  }
}
function renderCatalogState(state) {
  $("home-setup").textContent = `${state.dataset_catalog.length} available datasets; ${state.backends.length} registered methods. ` +
    (state.provider.configured ? "Synthetic planning is configured. Provider calls still require explicit consent." :
      "Synthetic planning unavailable. Existing data and local uploads still work.");
  for (const id of ["home-jobs", "history-jobs"]) {
    const list = document.createElement("ul");
    for (const job of state.jobs) {
      const item = document.createElement("li"), link = document.createElement("a");
      link.href = workbenchHash({page: "results", job: job.id});
      link.textContent = `${job.dataset} | ${job.mode} | execution ${job.status} | ${job.id}`;
      item.append(link); list.append(item);
    }
    if (state.jobs.length) $(id).replaceChildren(list);
    else $(id).textContent = "No browser experiments in this results directory yet.";
  }
  if (!$("backend-inventory").children.length) {
    for (const name of state.backends) {
      const details = document.createElement("details"), summary = document.createElement("summary"), source = document.createElement("pre");
      summary.textContent = name;
      source.textContent = JSON.stringify(state.backend_metadata?.[name] || {status: "Metadata unavailable"}, null, 2);
      details.append(summary, source); $("backend-inventory").append(details);
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
registerAction("plan", () => propose(false));
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
registerAction("preview", async () => {
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
registerAction("save", async () => {
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
  continueToExperiment(result.name, result.fingerprint);
  message("Dataset saved. Choose methods and start the experiment.");
});
function clearExistingReview() {
  clearReadyDataset();
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
registerAction("inspect-existing", async () => {
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
registerAction("use-existing", async () => {
  if (!existingReview || existingReview.name !== $("dataset").value) throw new Error("Review the selected dataset first.");
  const result = await api("/api/use-existing", {dataset: existingReview.name,
    fingerprint: existingReview.fingerprint, reviewed: $("existing-agreed").checked});
  message(`Using ${result.name} as-is. No rows were regenerated or changed. Choose methods and start the experiment.`);
  continueToExperiment(result.name, result.fingerprint);
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
registerAction("run", async () => {
  if (startingExperiment || activeExperiment || startUncertain) throw new Error("An experiment is already active or its start needs reconnection.");
  const request = runRequest(true);
  startingExperiment = true; updateRunning(activeExperiment);
  try {
  let result;
  try { result = await api("/api/run", request); }
  catch (error) { startUncertain = true; throw error; }
  lastStartedJob = result.id;
  activeExperiment = {id: result.id, dataset: request.dataset, mode: request.mode, status: "queued"};
  await refreshState();
  message("Experiment started once. Monitor it here or open its log and report.");
  await watch();
  } finally {
    startingExperiment = false; updateRunning(activeExperiment);
  }
});
function invalidateUpload() {
  uploadReview = null;
  $("upload-confirmed").checked = false;
  $("upload-confirmed").disabled = true;
  $("upload-save").disabled = true;
  $("upload-validation").textContent = "";
  clearReadyDataset();
}
for (const id of ["upload-name", "upload-target", "upload-task", "upload-drop",
  "upload-categorical", "upload-group", "upload-time", "upload-split", "upload-cv"]) {
  $(id).addEventListener("input", invalidateUpload);
  $(id).addEventListener("change", invalidateUpload);
}
$("upload-file").onchange = () => {
  invalidateUpload();
  $("upload-roles").disabled = true;
};
registerAction("upload-send", async () => {
  const file = $("upload-file").files[0];
  if (!file || !deviceState?.upload_limits) throw new Error("Choose a CSV or Parquet file.");
  if (!file.size || file.size > deviceState.upload_limits.file_bytes) throw new Error("File must be nonempty and no larger than 50 MiB.");
  invalidateUpload();
  $("upload-roles").disabled = true;
  if (uploadId) {
    await api("/api/uploads/cancel", {upload_id: uploadId});
    uploadId = null;
  }
  $("upload-progress").hidden = false;
  $("upload-summary").textContent = "Uploading and inspecting locally. No model or provider call is running.";
  try {
    const response = await fetch("/api/uploads", {method: "POST",
      headers: {"Content-Type": "application/octet-stream", "X-Quibble-Token": token,
        "X-Upload-Name": encodeURIComponent(file.name)}, body: file});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Upload was rejected.");
    if ($("upload-file").files[0] !== file) {
      await api("/api/uploads/cancel", {upload_id: result.upload_id});
      throw new Error("File selection changed during upload. Upload the new selection.");
    }
    uploadId = result.upload_id;
    $("upload-cancel").disabled = false;
    $("upload-roles").disabled = false;
    $("upload-summary").textContent = `${result.rows} rows, ${result.columns.length} columns. ${result.notice}`;
    showSample($("upload-sample"), result);
    for (const id of ["upload-target", "upload-drop", "upload-categorical", "upload-group", "upload-time"]) {
      $(id).replaceChildren();
      if (!["upload-drop", "upload-categorical"].includes(id)) $(id).append(option("", "Choose / none"));
      for (const column of result.schema) {
        $(id).append(option(column.name, `${column.name} (${column.suggested}; ${column.missing} missing)`));
      }
    }
    message("Upload parsed locally. Choose the outcome, column roles and split design, then validate.");
  } finally { $("upload-progress").hidden = true; }
});
function uploadOptions() {
  const selected = id => [...$(id).selectedOptions].map(item => item.value);
  return {upload_id: uploadId,
    name: $("upload-name").value, target: $("upload-target").value, task: $("upload-task").value,
    drop_columns: selected("upload-drop"), categorical_columns: selected("upload-categorical"),
    group_column: $("upload-group").value, time_column: $("upload-time").value,
    split: $("upload-split").value, cv: $("upload-cv").value};
}
registerAction("upload-validate", async () => {
  if (!uploadId) throw new Error("Upload a file first.");
  invalidateUpload();
  $("upload-summary").textContent = "Validating target, roles and evaluation splits locally...";
  const body = uploadOptions();
  const result = await api("/api/uploads/review", body);
  if (JSON.stringify(uploadOptions()) !== JSON.stringify(body)) throw new Error("Column roles changed during validation. Validate again.");
  uploadReview = result;
  $("upload-validation").textContent = JSON.stringify({rows: result.rows, predictors: result.predictors,
    manifest: result.manifest, evaluation: result.cv, target_summary: result.target_summary,
    target_classes: result.target_classes, target_summary_truncated: result.target_summary_truncated,
    fingerprint: result.fingerprint}, null, 2);
  $("upload-confirmed").disabled = false;
  $("upload-summary").textContent = "Validation succeeded. Review the summary and confirm before saving.";
});
$("upload-confirmed").onchange = () => {
  $("upload-save").disabled = !uploadReview || !$("upload-confirmed").checked;
};
registerAction("upload-save", async () => {
  if (!uploadReview) throw new Error("Validate the current column roles first.");
  const result = await api("/api/uploads/confirm", {upload_id: uploadId,
    review_token: uploadReview.review_token, confirmed: $("upload-confirmed").checked});
  uploadId = null;
  $("upload-cancel").disabled = true;
  $("upload-roles").disabled = true;
  invalidateUpload();
  await refreshState(result.name);
  $("cv").value = result.cv;
  continueToExperiment(result.name, result.fingerprint);
  message(`Saved ${result.name} unchanged. Choose methods and run the comparison.`);
});
$("upload-cancel").onclick = () => busy($("upload-cancel"), async () => {
  if (uploadId) await api("/api/uploads/cancel", {upload_id: uploadId});
  uploadId = null;
  invalidateUpload();
  $("upload-file").value = "";
  $("upload-roles").disabled = true;
  $("upload-sample").replaceChildren();
  $("upload-summary").textContent = "Staged upload cancelled. No saved datasets or original files were changed.";
  $("upload-cancel").disabled = true;
});
function invalidateRelational() {
  relationalReview = null;
  relationalSignature = null;
  $("relational-confirmed").checked = false;
  $("relational-confirmed").disabled = true;
  $("relational-save").disabled = true;
  clearReadyDataset();
}
function relationalInputs() {
  return JSON.stringify({name: $("relational-name").value, manifest: $("relational-manifest").value,
    files: [...$("relational-files").files].map(f => [f.name, f.size, f.lastModified])});
}
for (const id of ["relational-name", "relational-manifest"]) $(id).addEventListener("input", invalidateRelational);
$("relational-files").onchange = () => {
  invalidateRelational();
  $("relational-review").disabled = true;
};
$("relational-manifest-file").onchange = () => busy($("relational-manifest-file"), async () => {
  invalidateRelational();
  const file = $("relational-manifest-file").files[0];
  if (!file || file.size > 262144) throw new Error("Choose a manifest JSON smaller than 256 KiB.");
  $("relational-manifest").value = JSON.stringify(JSON.parse(await file.text()), null, 2);
});
$("relational-stage").onclick = () => busy($("relational-stage"), async () => {
  invalidateRelational();
  const files = [...$("relational-files").files];
  if (files.length < 2 || files.length > 16 || files.some(f => !f.size || f.size > 16000000) ||
      files.reduce((total, f) => total + f.size, 0) > 64000000) throw new Error("Choose 2–16 nonempty named tables within 16 MB/file and 64 MB total.");
  const signature = relationalInputs();
  if (relationalStage) await api("/api/relational/cancel", {stage_id: relationalStage});
  relationalStage = (await api("/api/relational/start", {})).stage_id;
  $("relational-cancel").disabled = false;
  $("relational-status").textContent = "Staging bounded local tables. No models or providers are called.";
  const staged = [];
  for (const file of files) {
    const table = file.name.replace(/\.(csv|parquet)$/i, "");
    const response = await fetch("/api/relational/table", {method: "POST",
      headers: {"Content-Type": "application/octet-stream", "X-Quibble-Token": token,
        "X-Upload-Name": encodeURIComponent(file.name), "X-Relational-Stage": relationalStage,
        "X-Relational-Table": table}, body: file});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Table staging failed.");
    staged.push({table, path: result.path, bytes: result.bytes, sha256: result.sha256});
  }
  if (signature !== relationalInputs()) throw new Error("Selected inputs changed while staging. Stage them again.");
  $("relational-summary").textContent = JSON.stringify(staged, null, 2);
  $("relational-review").disabled = false;
  $("relational-status").textContent = "Table identities and hashes preserved. Validate the explicit manifest next.";
});
$("relational-review").onclick = () => busy($("relational-review"), async () => {
  invalidateRelational();
  if (!relationalStage) throw new Error("Stage all named tables first.");
  const signature = relationalInputs();
  const result = await api("/api/relational/review", {stage_id: relationalStage,
    name: $("relational-name").value, manifest: JSON.parse($("relational-manifest").value)});
  if (signature !== relationalInputs()) throw new Error("Schema or inputs changed during review. Validate again.");
  relationalReview = result;
  relationalSignature = signature;
  $("relational-summary").textContent = JSON.stringify(result, null, 2);
  $("relational-confirmed").disabled = false;
  $("relational-status").textContent = "Validated. Review all table/relationship/timestamp/task/split roles and confirm.";
});
$("relational-confirmed").onchange = () => {
  $("relational-save").disabled = !relationalReview || !$("relational-confirmed").checked;
};
$("relational-save").onclick = () => busy($("relational-save"), async () => {
  if (!relationalReview || relationalSignature !== relationalInputs() || !$("relational-confirmed").checked)
    throw new Error("Validate and explicitly confirm the current relational schema first.");
  const result = await api("/api/relational/confirm", {stage_id: relationalStage,
    fingerprint: relationalReview.fingerprint, reviewed: true});
  relationalStage = null;
  invalidateRelational();
  $("relational-review").disabled = true;
  $("relational-cancel").disabled = true;
  await refreshState(result.name);
  $("cv").value = "holdout";
  $("device").value = "cpu";
  continueToExperiment(result.name, result.fingerprint);
  message(`Saved ${result.name} atomically. Use CPU local trees and official holdout; no relational-native claim.`);
});
$("relational-cancel").onclick = () => busy($("relational-cancel"), async () => {
  if (relationalStage) await api("/api/relational/cancel", {stage_id: relationalStage});
  relationalStage = null;
  invalidateRelational();
  $("relational-review").disabled = true;
  $("relational-cancel").disabled = true;
  $("relational-summary").textContent = "";
  $("relational-status").textContent = "Relational stage cancelled; no saved dataset was changed.";
});
async function refreshJob(supplied = null) {
  if (routeState.page !== "results" || !selectedJobId) return null;
  const identifier = selectedJobId;
  if (failedViewedJob === identifier) return null;
  if ($("job-return").dataset.job !== identifier) {
    renderedReport = null;
    $("job-return").replaceChildren();
    $("job-return").dataset.job = "";
    $("report").textContent = "Loading selected experiment...";
    $("report-sections").replaceChildren();
    $("download-report").disabled = true;
    reportText = "";
    savedThresholds = null;
    $("job-policy").hidden = true;
    $("job-status").textContent = "";
    $("log").textContent = "Loading selected experiment...";
  }
  let job;
  try { job = supplied?.id === identifier ? supplied : await api(`/api/jobs/${encodeURIComponent(identifier)}`); }
  catch (error) {
    if (selectedJobId === identifier && routeState.page === "results") {
      failedViewedJob = identifier;
      $("report").textContent = `Cannot open this experiment: ${error.message}. Use Experiment history to select saved evidence.`;
      $("job-return").replaceChildren();
    }
    throw error;
  }
  if (selectedJobId !== identifier || routeState.page !== "results") return null;
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
  if ($("job-return").dataset.job !== identifier) {
    const link = document.createElement("a");
    link.href = bookmarkJob(job.id); link.textContent = `Bookmark experiment ${job.id}`;
    $("job-return").replaceChildren(link, document.createTextNode(
      ` | ${job.backends?.join(", ") || "Selected methods"} | Campaign tier: ${job.campaign_tier || job.tier || "legacy/unrecorded"}${job.tier === "tuned" ? ` (${job.tune_trials} trials × ${job.tune_folds} folds; ${job.tune_metric}; approved=${job.tuning_confirmed})` : ""} | Saved files: ${job.directory}. Return using this link and the same Quibble server address. Reports persist after restart when using the same output directory; interrupted jobs do not resume.`));
    $("job-return").dataset.job = identifier;
  }
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
  const signature = JSON.stringify([job.id, job.report_html, reportText, job.status]);
  if (signature !== renderedReport) {
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
  renderedReport = signature;
  $("report-sections").replaceChildren();
  for (const section of job.report_sections || []) {
    const link = document.createElement("a");
    link.href = `#${section.id}`; link.textContent = section.title;
    link.dataset.reportSection = section.id;
    $("report-sections").append(link);
  }
  }
  $("download-report").disabled = !reportText;
  return job;
}
function watch() {
  if (watchPromise) return watchPromise;
  clearTimeout(polling);
  const viewedAtStart = selectedJobId;
  let failed = false;
  watchPromise = (async () => {
    let active = null;
    if (activeExperiment) {
      active = await api(`/api/jobs/${encodeURIComponent(activeExperiment.id)}`);
      if (["queued", "running"].includes(active.status)) updateRunning(active);
      else {
        updateRunning(null);
        await refreshState();
        message(active.status === "completed" ? "Execution completed. Read the report and its limitations." :
          `Execution ${active.status}. Read the log and partial report.`, active.status !== "completed");
      }
    }
    let viewed = null;
    try { viewed = await refreshJob(active); }
    catch (error) { message(`Could not refresh the selected report: ${error.message}`, true); }
    if (!activeExperiment && viewed && ["queued", "running"].includes(viewed.status)) updateRunning(viewed);
  })().catch(error => {
    failed = true;
    if (activeExperiment || startUncertain) {
      $("run-progress").hidden = false;
      $("run-progress-text").textContent = "Status connection lost. This does not mean execution stopped. Keep the server running and use Refresh status to reconnect.";
    }
    throw error;
  }).finally(() => {
    watchPromise = null;
    if ((!failed && activeExperiment) || routeState.page === "results" && selectedJobId !== viewedAtStart) {
      polling = setTimeout(() => watch().catch(error => message(error.message, true)),
        selectedJobId !== viewedAtStart ? 0 : 2500);
    }
    renderActions();
  });
  return watchPromise;
}
$("jobs").onchange = () => {
  if ($("jobs").value) navigate({page: "results", job: $("jobs").value});
  else navigate({page: "history"});
};
async function reconnect() {
  await refreshState();
  startUncertain = false;
  failedViewedJob = null;
  updateRunning(activeExperiment);
  await watch();
}
$("refresh").onclick = () => busy($("refresh"), reconnect);
$("reconnect").onclick = () => busy($("reconnect"), reconnect);
$("download-report").onclick = () => {
  const url = URL.createObjectURL(new Blob([reportText], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = "quibble-report.md"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("print-report").onclick = () => window.print();
$("report-sections").onclick = event => {
  const link = event.target.closest("a[data-report-section]");
  if (!link) return;
  event.preventDefault();
  const target = [...$("report").querySelectorAll("[id]")].find(node => node.id === link.dataset.reportSection);
  if (target) { target.tabIndex = -1; target.focus(); target.scrollIntoView(); }
};
$("collect-environment").onclick = () => busy($("collect-environment"), async () => {
  $("environment-result").textContent = "Collecting requested local package inventory…";
  environmentResult = await api("/api/environment", {requested: true});
  $("environment-result").textContent = JSON.stringify(environmentResult, null, 2);
  $("download-environment").disabled = false;
});
$("download-environment").onclick = () => {
  if (environmentResult) downloadJSON(environmentResult, "quibble-diagnostics.json");
};
function clearVerification() {
  verificationJob = null; verificationArtifact = null;
  $("verify-swap").disabled = $("verify-gate").disabled = $("download-verification").disabled = true;
  $("verify-context").textContent = "Load this experiment before checking.";
  $("verification-result").textContent = "";
  $("verify-status").textContent = "";
  $("verification-history").replaceChildren();
}
$("verify-job").onchange = clearVerification;
function showVerification(artifact) {
  verificationArtifact = artifact;
  $("verification-result").textContent = JSON.stringify(artifact, null, 2);
  $("verify-status").textContent = artifact.status === "completed" ?
    `${artifact.kind.toUpperCase()}: ${artifact.result.overall} | ${artifact.inputs.candidate} vs ${artifact.inputs.incumbent}. Separate saved-evidence check; original report and policy unchanged.` :
    `ERROR: ${artifact.error || "No scientific verdict."}`;
  $("download-verification").disabled = false;
}
function showVerificationHistory(job) {
  $("verification-history").replaceChildren();
  for (const artifact of job.verifications || []) {
    const button = document.createElement("button"); button.className = "secondary";
    button.textContent = `${artifact.kind} | ${artifact.result?.overall || "ERROR"} | ${artifact.created_at}`;
    button.onclick = () => showVerification(artifact);
    $("verification-history").append(button);
  }
}
$("load-verify-job").onclick = () => busy($("load-verify-job"), async () => {
  clearVerification();
  const id = $("verify-job").value;
  if (!id) throw new Error("Choose a saved experiment first.");
  const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
  if ($("verify-job").value !== id) return;
  verificationJob = job;
  for (const control of ["verify-incumbent", "verify-candidate"]) {
    $(control).replaceChildren(...(job.backends || []).map(name => option(name)));
  }
  $("verify-candidate").selectedIndex = Math.min(1, (job.backends || []).length - 1);
  $("verify-context").textContent = `${job.dataset} | ${job.mode} | ${job.status} | Saved policy: ${JSON.stringify(job.thresholds)}. All recorded methods must have complete evidence.`;
  $("verify-swap").disabled = $("verify-gate").disabled =
    job.mode !== "full" || job.status !== "completed" || (job.backends || []).length < 2;
  showVerificationHistory(job);
});
async function verifySaved(kind) {
  if (verificationPending) throw new Error("Wait for the current saved-evidence check.");
  if (!verificationJob || verificationJob.id !== $("verify-job").value) throw new Error("Load selected evidence first.");
  const id = verificationJob.id;
  const body = {job_id: id, incumbent: $("verify-incumbent").value, candidate: $("verify-candidate").value};
  if (kind === "gate") body.min_gain = Number($("verify-min-gain").value);
  verificationPending = true;
  for (const control of ["verify-job", "load-verify-job", "verify-incumbent", "verify-candidate", "verify-min-gain", "verify-swap", "verify-gate"]) $(control).disabled = true;
  $("verify-status").textContent = "Checking saved evidence only; no training…";
  try {
    const artifact = await api(`/api/verify/${kind}`, body);
    if ($("verify-job").value !== id || verificationJob?.id !== id) return;
    showVerification(artifact);
    verificationJob.verifications = [artifact, ...(verificationJob.verifications || [])];
    showVerificationHistory(verificationJob);
  } catch (error) {
    $("verify-status").textContent = `REFUSED / ERROR: ${error.message}. No scientific verdict.`;
    throw error;
  } finally {
    verificationPending = false;
    for (const control of ["verify-job", "load-verify-job", "verify-incumbent", "verify-candidate", "verify-min-gain"]) $(control).disabled = false;
    $("verify-swap").disabled = $("verify-gate").disabled =
      !verificationJob || verificationJob.mode !== "full" || verificationJob.status !== "completed" ||
      (verificationJob.backends || []).length < 2;
  }
}
for (const id of ["verify-incumbent", "verify-candidate", "verify-min-gain"]) $(id).onchange = () => {
  verificationArtifact = null; $("verification-result").textContent = "";
  $("download-verification").disabled = true;
  $("verify-status").textContent = "Inputs changed. Run a new separate check; saved checks remain unchanged.";
};
$("verify-swap").onclick = () => busy($("verify-swap"), () => verifySaved("swap"));
$("verify-gate").onclick = () => busy($("verify-gate"), () => verifySaved("gate"));
$("download-verification").onclick = () => {
  if (verificationArtifact) downloadJSON(verificationArtifact, `quibble-${verificationArtifact.kind}-${verificationArtifact.id}.json`);
};
$("compare-jobs").onclick = () => busy($("compare-jobs"), async () => {
  const job_ids = [$("compare-left").value, $("compare-right").value];
  if (job_ids.some(id => !id) || job_ids[0] === job_ids[1]) throw new Error("Choose exactly two distinct saved experiments.");
  $("comparison-status").textContent = "Checking evidence compatibility…";
  $("comparison-result").replaceChildren(); comparisonResult = null;
  $("download-comparison").disabled = true;
  try {
    const result = await api("/api/compare-jobs", {job_ids});
    if ($("compare-left").value !== job_ids[0] || $("compare-right").value !== job_ids[1]) return;
    comparisonResult = result;
    $("comparison-status").textContent = `${result.compatibility.toUpperCase()} | ${result.presentation}. ${result.caveat} ${result.differences.join(" ")}`;
    for (const job of result.jobs) {
      const article = document.createElement("article"), heading = document.createElement("h4");
      const link = document.createElement("a"), content = document.createElement("pre");
      heading.textContent = job.id;
      link.href = workbenchHash({page: "results", job: job.id}); link.textContent = "Open complete original report";
      const visible = structuredClone(job);
      if (result.presentation === "configuration only") for (const cell of visible.cells) delete cell.metrics;
      content.textContent = JSON.stringify(visible, null, 2);
      article.append(heading, link, content); $("comparison-result").append(article);
    }
    $("download-comparison").disabled = false;
  } catch (error) {
    $("comparison-status").textContent = `REFUSED / ERROR: ${error.message}. No comparison inferred.`;
    throw error;
  }
});
for (const id of ["compare-left", "compare-right"]) $(id).onchange = () => {
  comparisonResult = null; $("comparison-result").replaceChildren();
  $("download-comparison").disabled = true; $("comparison-status").textContent = "Selection changed. Compare again.";
};
$("download-comparison").onclick = () => {
  if (comparisonResult) downloadJSON(comparisonResult, "quibble-descriptive-comparison.json");
};
document.querySelector("#methods").previousElementSibling.append(
  helpButton("Prediction methods", "Choose one to four methods to compare on identical evaluation rows. Installed packages, first-use downloads, authentication, and resource limits vary by method. These are prediction methods, not the language model used to draft a scenario."));
const mobileNavigation = matchMedia("(max-width: 900px)");
let railTrigger = null;
function compactRail() {
  return document.documentElement.dataset.rail === "collapsed" && !mobileNavigation.matches;
}
function closeRailDisclosure(restoreFocus = false) {
  document.body.classList.remove("rail-disclosure");
  if (compactRail()) document.querySelectorAll("[data-nav-group]").forEach(group => { group.open = false; });
  if (restoreFocus && railTrigger) railTrigger.focus();
  railTrigger = null;
}
function syncRail() {
  closeRailDisclosure();
  if (!compactRail()) document.querySelectorAll("[data-nav-group]").forEach(group => { group.open = true; });
}
syncRail();
document.addEventListener("quibble-appearance", syncRail);
document.querySelectorAll("[data-nav-group] > summary").forEach(summary => {
  summary.addEventListener("click", event => {
    if (!compactRail()) return;
    event.preventDefault();
    if (railTrigger === summary) { closeRailDisclosure(true); return; }
    closeRailDisclosure();
    railTrigger = summary;
    summary.parentElement.open = true;
    document.body.classList.add("rail-disclosure");
  });
});
document.addEventListener("click", event => {
  if (railTrigger && !event.target.closest("#workbench-sidebar")) closeRailDisclosure();
});
document.addEventListener("focusin", event => {
  if (railTrigger && !event.target.closest("#workbench-sidebar")) closeRailDisclosure();
});
function closeNavigation(restoreFocus = true) {
  const opened = document.body.classList.contains("navigation-open");
  document.body.classList.remove("navigation-open");
  $("open-navigation").setAttribute("aria-expanded", "false");
  $("workbench-sidebar").removeAttribute("role");
  $("workbench-sidebar").removeAttribute("aria-modal");
  for (const node of [$("workbench-content"), document.querySelector("body > header"),
    document.querySelector("body > footer"), document.querySelector(".skip-link"), $("open-navigation")]) node.inert = false;
  if (opened && restoreFocus) $("open-navigation").focus();
}
$("open-navigation").onclick = () => {
  if (!mobileNavigation.matches) return;
  document.body.classList.add("navigation-open");
  $("open-navigation").setAttribute("aria-expanded", "true");
  $("workbench-sidebar").setAttribute("role", "dialog");
  $("workbench-sidebar").setAttribute("aria-modal", "true");
  for (const node of [$("workbench-content"), document.querySelector("body > header"),
    document.querySelector("body > footer"), document.querySelector(".skip-link"), $("open-navigation")]) node.inert = true;
  $("close-navigation").focus();
};
$("close-navigation").onclick = () => closeNavigation();
mobileNavigation.addEventListener("change", () => {
  syncRail();
  if (!mobileNavigation.matches) closeNavigation(false);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && railTrigger) {
    event.preventDefault(); closeRailDisclosure(true); return;
  }
  if (!document.body.classList.contains("navigation-open")) return;
  if (event.key === "Escape") { event.preventDefault(); closeNavigation(); }
  if (event.key === "Tab") {
    const controls = [...$("workbench-sidebar").querySelectorAll("a[href], button, summary")]
      .filter(node => node.getClientRects().length && !node.disabled);
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});
document.addEventListener("click", event => {
  const link = event.target.closest("a");
  if (!link || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || link.target) return;
  const url = new URL(link.href, location.href);
  if (url.origin !== location.origin || url.pathname !== location.pathname || !url.hash) return;
  if ($("report").contains(link) && !url.hash.includes("=")) {
    const anchor = document.getElementById(decodeURIComponent(url.hash.slice(1)));
    if (anchor && $("report").contains(anchor)) { event.preventDefault(); anchor.scrollIntoView(); }
    return;
  }
  try {
    const route = parseWorkbenchRoute(url.hash);
    if (route) { event.preventDefault(); navigate(route); }
  } catch (error) { event.preventDefault(); message(error.message, true); }
});
for (const event of ["input", "change"]) document.addEventListener(event, e => {
  if (e.target.closest("#run-path")) clearFinalApproval();
  if (e.target.closest("#run-path")) updateTuning();
  renderActions();
});
$("workflow-primary-top").onclick = $("run").onclick = dispatchWorkflowAction;
for (const position of ["top", "bottom"]) $(`workflow-back-${position}`).onclick = () => {
  const action = workflowAction();
  if (action && !busyActions.size) navigate(action.back);
};
addEventListener("popstate", restoreRoute);
addEventListener("hashchange", restoreRoute);
renderRoute();
refreshState().then(() => {
  stateReady = true;
  message("Choose existing data, upload CSV / Parquet, or design a new dataset to begin.");
  restoreRoute();
  return watch();
})
  .catch(error => message(error.message, true));
