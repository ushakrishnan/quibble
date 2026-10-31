"""Exercise browser-only appearance behavior without adding a JavaScript test dependency."""
import shutil
import subprocess
import re

import pytest

from quibble_ml.web import ASSETS


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed for browser asset unit tests")


def run_script(script, asset):
    result = subprocess.run([NODE, "-e", script, str(ASSETS / asset)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_independent_preferences_prepaint_all_controls_and_storage_failures():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
function setup(saved = {}, dark = false, failure = null) {
  const events = {}, warnings = [], stored = {}, controls = {};
  for (const name of ["theme", "accent", "contrast", "rail"]) {
    controls[`[data-${name}-select]`] = [0, 1].map(() =>
      ({hidden: true, addEventListener(type, fn) { this[type] = fn; }}));
  }
  const button = {setAttribute(name, value) { this[name] = value; },
    addEventListener(type, fn) { this[type] = fn; }};
  controls["[data-rail-toggle]"] = [button];
  const notice = {hidden: true};
  controls["[data-appearance-warning]"] = [notice];
  const media = {matches: dark, addEventListener(type, fn) { this[type] = fn; }};
  let ready = false;
  const document = {documentElement: {dataset: {}},
    querySelectorAll: selector => ready ? controls[selector] || [] : [],
    dispatchEvent() {},
    addEventListener(type, fn) { events[type] = () => { ready = true; fn(); }; }};
  const storage = {
    getItem(key) { if (failure) throw failure; return saved[key] ?? null; },
    setItem(key, value) {
      if (failure) throw failure;
      stored[key] = value;
    }
  };
  vm.runInNewContext(source, {document, DOMException, Event, localStorage: storage,
    console: {warn: (...args) => warnings.push(args)}, matchMedia: () => media,
    addEventListener: (type, fn) => { events[type] = fn; }});
  return {document, controls, media, events, stored, warnings, notice, button, storage};
}
for (const saved of [null, "system", "light", "dark", "invalid"]) {
  for (const dark of [true, false]) {
    const state = setup({"quibble-theme": saved}, dark);
    const expected = ["light", "dark"].includes(saved) ? saved : dark ? "dark" : "light";
    assert.equal(state.document.documentElement.dataset.theme, expected);
    assert.equal(state.document.documentElement.dataset.accent, "blue");
    assert.equal(state.document.documentElement.dataset.contrast, "standard");
    assert.equal(state.document.documentElement.dataset.rail, "expanded");
    state.events.DOMContentLoaded();
    const selects = state.controls["[data-theme-select]"];
    assert.ok(selects.every(s => !s.hidden && s.value === (["light", "dark"].includes(saved) ? saved : "system")));
    selects[1].value = "dark"; selects[1].change();
    assert.ok(selects.every(s => s.value === "dark"));
    assert.equal(state.stored["quibble-theme"], "dark");
    state.media.matches = false; state.media.change();
    assert.equal(state.document.documentElement.dataset.theme, "dark");
    state.events.storage({key: "quibble-theme", newValue: "system"});
    assert.equal(state.document.documentElement.dataset.theme, "light");
    state.media.matches = true; state.media.change();
    assert.equal(state.document.documentElement.dataset.theme, "dark");
    state.events.storage({key: null, newValue: null});
    assert.ok(selects.every(s => s.value === "system"));
    assert.equal(state.warnings.length, saved === "invalid" ? 1 : 0);
    assert.equal(state.notice.hidden, saved !== "invalid");
  }
}
for (const name of ["accent", "contrast", "rail"]) {
  const choices = {accent: ["blue", "teal", "indigo", "violet", "fuchsia", "slate"],
    contrast: ["standard", "high"], rail: ["expanded", "collapsed"]}[name];
  for (const value of choices) {
    const state = setup({[`quibble-${name}`]: value});
    assert.equal(state.document.documentElement.dataset[name], value, "prepaint");
    state.events.DOMContentLoaded();
    const selects = state.controls[`[data-${name}-select]`];
    assert.ok(selects.every(s => s.value === value));
    selects[0].value = choices[1]; selects[0].change();
    assert.ok(selects.every(s => s.value === choices[1]));
    assert.equal(state.stored[`quibble-${name}`], choices[1]);
    assert.equal(state.document.documentElement.dataset.theme, "light");
    state.events.storage({key: `quibble-${name}`, newValue: value});
    assert.ok(selects.every(s => s.value === value));
    state.events.storage({key: `quibble-${name}`, newValue: "INVALID"});
    assert.ok(selects.every(s => s.value === choices[0]));
    assert.equal(state.warnings.length, 1);
    state.events.storage({key: `quibble-${name}`, newValue: null});
    assert.equal(state.document.documentElement.dataset[name], choices[0]);
    state.events.storage({key: `quibble-${name}`, newValue: value, storageArea: {}});
    assert.equal(state.document.documentElement.dataset[name], choices[0]);
  }
}
const cleared = setup({"quibble-theme": "dark", "quibble-accent": "violet",
  "quibble-contrast": "high", "quibble-rail": "collapsed"});
cleared.events.DOMContentLoaded();
cleared.events.storage({key: null, newValue: null});
assert.deepEqual({...cleared.document.documentElement.dataset},
  {theme: "light", accent: "blue", contrast: "standard", rail: "expanded"});
cleared.button.click();
assert.equal(cleared.button["aria-pressed"], "true");
assert.equal(cleared.stored["quibble-rail"], "collapsed");
for (const name of ["SecurityError", "QuotaExceededError"]) {
  const blocked = setup({}, true, new DOMException("blocked", name));
  blocked.events.DOMContentLoaded();
  for (const [preference, value] of [["theme", "light"], ["accent", "teal"], ["contrast", "high"]]) {
    const select = blocked.controls[`[data-${preference}-select]`][1];
    select.value = value; select.change();
    assert.equal(blocked.document.documentElement.dataset[preference], value);
  }
  assert.equal(blocked.warnings.length, 7);
  assert.equal(blocked.notice.hidden, false);
  assert.match(blocked.notice.textContent, /may not be saved/);
}
assert.throws(() => setup({}, false, new Error("unexpected")), /unexpected/);
assert.throws(() => setup({}, false, new DOMException("unexpected", "InvalidStateError")), /unexpected/);
const unexpectedWrite = setup();
unexpectedWrite.events.DOMContentLoaded();
unexpectedWrite.storage.setItem = () => { throw new Error("unexpected write"); };
assert.throws(() => unexpectedWrite.controls["[data-accent-select]"][0].change(), /unexpected write/);
for (const name of ["theme", "accent", "contrast", "rail"]) {
  const invalid = setup({[`quibble-${name}`]: "__proto__"});
  invalid.events.DOMContentLoaded();
  assert.equal(invalid.warnings.length, 1);
  assert.match(invalid.notice.textContent, new RegExp(`invalid saved ${name}`));
}
""", "theme.js")


def test_native_settings_original_icons_and_single_navigation_markup():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    header = (ASSETS / "header.html").read_text(encoding="utf-8")
    combined = html + header
    ids = re.findall(r'\bid="([^"]+)"', combined)
    assert len(ids) == len(set(ids))
    for name in ("theme", "accent", "contrast"):
        assert f"data-{name}-select" in header
        assert f"data-{name}-select" in html
    for select in re.findall(r"<select\b[^>]*data-(?:theme|accent|contrast|rail)-select[^>]*>", combined):
        assert 'aria-label="Color theme"' in select or re.search(
            r"<label>[^<]+ " + re.escape(select), combined)
    assert '<fieldset class="appearance-settings"><legend>Settings and theme</legend>' in html
    assert 'aria-label="Appearance preview"' in html
    for example in ("Primary button", "Selected item", "Ordinary link", "Validation example",
                    "Focus example", "PASS", "FAIL", "INCONCLUSIVE"):
        assert example in html
    assert combined.count('aria-label="Workbench"') == 1
    assert 'data-rail-toggle aria-pressed="false" aria-label="Compact navigation"' in html
    assert len(re.findall(r'<summary aria-label="[^"]+"><svg', html)) == 5
    assert html.count('class="nav-icon"') == 6
    assert "Original Quibble navigation SVG subset" in html and "CC0-1.0" in html
    for svg in re.findall(r"<svg\b.*?</svg>", html, re.S):
        assert 'aria-hidden="true"' in svg
        assert not re.search(r"<(?:script|use|image|foreignObject)|\son\w+=|href=", svg)
    assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)="https?://', combined)
    for page in ("about.html",):
        content = (ASSETS / page).read_text(encoding="utf-8")
        assert content.index("/theme.js") < content.index("/style.css")
    assert html.index("/theme.js") < html.index("/style.css")


def test_rail_disclosure_is_temporary_bounded_and_keyboard_restorable():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const listeners = {}, classes = new Set(), media = {matches: false};
const groups = [0, 1, 2, 3, 4].map(() => ({open: true}));
let focused;
const summaries = groups.map(group => ({parentElement: group,
  addEventListener(type, fn) { this[type] = fn; }, focus() { focused = this; }}));
const document = {documentElement: {dataset: {rail: "collapsed"}},
  body: {classList: {add: x => classes.add(x), remove: x => classes.delete(x)}},
  querySelectorAll: selector => selector.endsWith("summary") ? summaries : groups,
  addEventListener(type, fn) { listeners[type] = fn; }};
const context = {document, matchMedia: () => media};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("const mobileNavigation ="),
  source.indexOf("function closeNavigation(")), context);
assert.ok(groups.every(g => !g.open));
let prevented = 0;
summaries[0].click({preventDefault() { prevented++; }});
assert.ok(classes.has("rail-disclosure") && groups[0].open);
summaries[1].click({preventDefault() { prevented++; }});
assert.ok(!groups[0].open && groups[1].open);
context.closeRailDisclosure(true);
assert.equal(focused, summaries[1]);
assert.ok(!classes.has("rail-disclosure") && groups.every(g => !g.open));
summaries[2].click({preventDefault() {}});
listeners.focusin({target: {closest: () => null}});
assert.ok(!classes.has("rail-disclosure"));
media.matches = true; context.syncRail();
assert.ok(groups.every(g => g.open));
assert.equal(document.documentElement.dataset.rail, "collapsed", "mobile must not overwrite preference");
media.matches = false; context.syncRail();
assert.ok(groups.every(g => !g.open));
document.documentElement.dataset.rail = "expanded";
listeners["quibble-appearance"]();
assert.ok(groups.every(g => g.open));
assert.equal(prevented, 2);
""", "app.js")


def test_route_rendering_cannot_reopen_nested_links_in_the_compact_rail():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const group = {open: false}, nodes = {};
const link = {hash: "#page=settings", closest: () => group,
  setAttribute(name, value) { this[name] = value; }, removeAttribute(name) { delete this[name]; }};
const context = {routeState: {page: "settings"}, dataPath: "existing", document: {},
  $: id => nodes[id] ||= {setAttribute() {}, removeAttribute() {}, querySelectorAll: () => [link]},
  parseWorkbenchRoute: () => ({page: "settings"}), renderActions() {},
  compactRail: () => true};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function renderRoute("),
  source.indexOf("function navigate(")), context);
context.renderRoute();
assert.equal(group.open, false, "initial route and delayed state responses must keep a rail compact");
assert.equal(link["aria-current"], "page");
context.compactRail = () => false;
context.renderRoute();
assert.equal(group.open, true, "desktop labels and mobile drawer still reveal the current page");
""", "app.js")


def test_drawer_keyboard_trap_and_rail_escape_do_not_change_routes():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
let handler, focused, closed = 0, railClosed = 0;
const controls = [0, 1, 2].map(() => ({getClientRects: () => [1],
  focus() { focused = this; }}));
const document = {body: {classList: {contains: () => true}},
  addEventListener(type, fn) { assert.equal(type, "keydown"); handler = fn; }};
const context = {document, railTrigger: null,
  $: () => ({querySelectorAll: () => controls}),
  closeNavigation() { closed++; }, closeRailDisclosure(restore) {
    assert.equal(restore, true); railClosed++;
  }};
vm.createContext(context);
const start = source.indexOf('document.addEventListener("keydown", event =>',
  source.indexOf("const mobileNavigation ="));
vm.runInContext(source.slice(start, source.indexOf('document.addEventListener("click", event =>', start)), context);
let prevented = 0;
document.activeElement = controls[2];
handler({key: "Tab", preventDefault() { prevented++; }});
assert.equal(focused, controls[0]);
document.activeElement = controls[0];
handler({key: "Tab", shiftKey: true, preventDefault() { prevented++; }});
assert.equal(focused, controls[2]);
handler({key: "Escape", preventDefault() { prevented++; }});
assert.equal(closed, 1);
context.railTrigger = {};
handler({key: "Escape", preventDefault() { prevented++; }});
assert.equal(railClosed, 1);
assert.equal(closed, 1);
assert.equal(prevented, 4);
""", "app.js")


def test_drawer_blocks_entire_background_and_restores_focus():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const nodes = {}, classes = new Set();
let focused;
for (const name of ["workbench-sidebar", "workbench-content", "open-navigation", "close-navigation",
  "body > header", "body > footer", ".skip-link"]) {
  nodes[name] = {inert: false, attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
    focus() { focused = this; }};
}
const context = {$: id => nodes[id], mobileNavigation: {matches: true},
  document: {querySelector: selector => nodes[selector],
    body: {classList: {add: x => classes.add(x), remove: x => classes.delete(x),
      contains: x => classes.has(x)}}}};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function closeNavigation("),
  source.indexOf('mobileNavigation.addEventListener("change"')), context);
nodes["open-navigation"].onclick();
for (const name of ["workbench-content", "open-navigation", "body > header", "body > footer", ".skip-link"])
  assert.equal(nodes[name].inert, true, name);
assert.equal(nodes["workbench-sidebar"].attributes["aria-modal"], "true");
assert.equal(nodes["workbench-sidebar"].attributes.role, "dialog");
assert.equal(focused, nodes["close-navigation"]);
nodes["close-navigation"].onclick();
assert.equal(focused, nodes["open-navigation"]);
assert.ok(Object.values(nodes).every(node => !node.inert));
assert.ok(!classes.has("navigation-open"));
assert.equal(nodes["open-navigation"].attributes["aria-expanded"], "false");
assert.equal(nodes["workbench-sidebar"].attributes.role, undefined);
context.mobileNavigation.matches = false;
nodes["open-navigation"].onclick();
assert.ok(!classes.has("navigation-open"));
""", "app.js")


def test_verdict_badges_preserve_other_report_content():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function decorateReport(root)");
const finish = source.indexOf("function message(", start);
assert.ok(start >= 0 && finish > start);
const context = {document: {createElement: () => ({})}};
vm.createContext(context);
vm.runInContext(source.slice(start, finish), context);
function node(text, children = []) {
  return {textContent: text, children,
    replaceChildren(child) { this.children = [child]; }};
}
const values = [node("PASS"), node(" FAIL "), node("INCONCLUSIVE"), node("constructor"),
  node("PASS is not a guarantee"), node("FAIL", [{}]), node("<script>alert(1)</script>")];
context.decorateReport({querySelectorAll: () => values});
for (let i = 0; i < 3; i++) {
  assert.equal(values[i].children[0].textContent, values[i].textContent.trim());
  assert.equal(values[i].children[0].className,
    "verdict verdict-" + values[i].textContent.trim().toLowerCase());
}
assert.equal(values[3].children.length, 0);
assert.equal(values[4].children.length, 0);
assert.equal(values[5].children.length, 1);
assert.equal(values[6].children.length, 0);
const badge = values[0].children[0];
context.decorateReport({querySelectorAll: () => values});
assert.equal(values[0].children[0], badge);
""", "app.js")


def test_workbench_routes_are_allowlisted_and_legacy_links_normalize():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const context = {URLSearchParams};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function parseWorkbenchRoute("),
  source.indexOf("function clearFinalApproval(")), context);
const parse = hash => JSON.parse(JSON.stringify(context.parseWorkbenchRoute(hash)));
assert.deepEqual(parse(""), {page: "home"});
assert.deepEqual(parse("#job=155102-844648-a205bc"), {page: "results", job: "155102-844648-a205bc"});
assert.equal(context.workbenchHash(parse("#job=old")), "#page=results&job=old");
for (const page of ["home", "data", "history", "settings", "backends"]) {
  assert.deepEqual(parse("#page=" + page), {page});
}
for (const step of ["data", "methods", "start"]) {
  assert.deepEqual(parse("#page=run&step=" + step), {page: "run", step});
}
assert.deepEqual(parse("#page=data&source=upload"), {page: "data", source: "upload"});
for (const hash of ["#page=no", "#page=run&step=bad", "#page=home&source=upload",
  "#page=data&source=remote", "#page=results", "#job=../secret", "#page=home&step=start",
  "#page=home&page=run", "#page=home&token=secret", "#page=run&job=old"]) {
  assert.throws(() => context.parseWorkbenchRoute(hash), undefined, hash);
}
assert.equal(context.parseWorkbenchRoute("#workbench-content"), null);
assert.equal(context.parseWorkbenchRoute("#report-heading"), null);
""", "app.js")


def test_run_draft_and_primary_actions_require_review_and_exact_approval():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const nodes = {
  methods: {querySelectorAll: () => [{value: "histgb"}, {value: "lightgbm"}]},
  mode: {value: "full"}, device: {value: "cpu"}, cv: {value: "3x2"},
  "threshold-mode": {value: "defaults"}, "full-confirmed": {checked: false}
};
const context = {$: id => nodes[id], readyDataset: "", readyFingerprint: "reviewed",
  deviceState: {cuda_available: false}, effectiveThresholds: () => ({}),
  routeState: {page: "run", step: "methods"}, activeExperiment: null, startingExperiment: false,
  startUncertain: false, lastStartedJob: null};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function runRequest("),
  source.indexOf("function renderActions(")), context);
assert.throws(() => context.runRequest(), /review data/);
context.readyDataset = "wine";
assert.equal(context.workflowAction().id, "next");
assert.equal(context.workflowAction().disabled, false);
assert.throws(() => context.runRequest(true), /Approve this exact/);
context.routeState.step = "start";
assert.equal(context.workflowAction().disabled, true);
nodes["full-confirmed"].checked = true;
assert.equal(context.workflowAction().id, "run");
assert.equal(context.workflowAction().disabled, false);
assert.equal(context.runRequest(true).fingerprint, "reviewed");
context.startUncertain = true;
assert.equal(context.workflowAction().disabled, true);
context.startUncertain = false;
context.activeExperiment = {id: "active"};
assert.equal(context.workflowAction().id, "open-active");
context.activeExperiment = null;
context.lastStartedJob = "done";
assert.equal(context.workflowAction().id, "open-report");
context.lastStartedJob = null;
nodes.methods.querySelectorAll = () => [];
assert.throws(() => context.runRequest(), /one to four/);
nodes.methods.querySelectorAll = () => Array.from({length: 5}, () => ({value: "method"}));
assert.throws(() => context.runRequest(), /one to four/);
nodes.methods.querySelectorAll = () => [{value: "histgb"}];
nodes.device.value = "cuda";
assert.throws(() => context.runRequest(), /CUDA is unavailable/);
nodes.device.value = "cpu";
context.effectiveThresholds = () => { throw new Error("Invalid policy"); };
assert.throws(() => context.runRequest(), /Invalid policy/);
nodes.mode.value = "smoke";
assert.equal(context.runRequest(true).thresholds, null);
""", "app.js")


def test_polling_is_single_flight_and_independent_of_the_viewed_report():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
(async () => {
  let release, calls = 0, states = 0, renders = [], timers = [];
  const pending = new Promise(resolve => { release = resolve; });
  const context = {watchPromise: null, polling: null, selectedJobId: "old",
    routeState: {page: "results"}, activeExperiment: {id: "active"}, startUncertain: false,
    api: async path => {
      assert.equal(path, "/api/jobs/active"); calls++;
      return calls === 1 ? pending : {id: "active", status: "completed"};
    },
    refreshState: async () => { states++; },
    refreshJob: async () => { renders.push(context.selectedJobId); return {status: "completed"}; },
    updateRunning: job => { context.activeExperiment = job; },
    renderActions() {}, message() {}, clearTimeout() {},
    setTimeout: (callback, delay) => { timers.push({callback, delay}); return 1; }};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf("function watch()"),
    source.indexOf('$("jobs").onchange')), context);
  const first = context.watch(), repeated = context.watch();
  assert.equal(first, repeated);
  context.selectedJobId = "historical";
  release({id: "active", status: "running"});
  await first;
  assert.equal(calls, 1);
  assert.equal(states, 0); // Heartbeats do not rescan every dataset.
  assert.deepEqual(renders, ["historical"]);
  assert.equal(context.selectedJobId, "historical");
  await timers.shift().callback();
  assert.equal(calls, 2);
  assert.equal(states, 1); // Refresh the catalog once after completion.
  assert.equal(context.activeExperiment, null);
  assert.equal(context.selectedJobId, "historical");
  assert.equal(timers.length, 0);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", "app.js")


def test_obsolete_report_response_cannot_replace_the_new_selection():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
(async () => {
  let release;
  const context = {routeState: {page: "results"}, selectedJobId: "old", failedViewedJob: null,
    $: id => { assert.equal(id, "job-return"); return {dataset: {job: "old"}}; },
    api: () => new Promise(resolve => { release = resolve; })};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf("async function refreshJob("),
    source.indexOf("function watch()")), context);
  const response = context.refreshJob();
  context.selectedJobId = "new";
  release({id: "old"});
  assert.equal(await response, null);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", "app.js")


def test_missing_historical_report_does_not_stop_active_monitoring():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
(async () => {
  const timers = [], messages = [];
  const context = {watchPromise: null, polling: null, selectedJobId: "missing",
    routeState: {page: "results"}, activeExperiment: {id: "active"}, startUncertain: false,
    api: async () => ({id: "active", status: "running"}),
    updateRunning: job => { context.activeExperiment = job; },
    refreshJob: async () => { throw new Error("Not found"); },
    renderActions() {}, message: text => messages.push(text), clearTimeout() {},
    setTimeout: (callback, delay) => { timers.push(delay); return 1; }};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf("function watch()"),
    source.indexOf('$("jobs").onchange')), context);
  await context.watch();
  assert.equal(context.activeExperiment.id, "active");
  assert.deepEqual(timers, [2500]);
  assert.ok(messages[0].includes("Not found"));
})().catch(error => { console.error(error); process.exitCode = 1; });
""", "app.js")


def test_finishing_a_start_request_preserves_subsequent_navigation():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
(async () => {
  let handler, finishRefresh;
  const pending = new Promise(resolve => { finishRefresh = resolve; });
  const context = {startingExperiment: false, activeExperiment: null, startUncertain: false,
    lastStartedJob: null, selectedJobId: "old", routeState: {page: "run", step: "start"},
    registerAction: (id, action) => { assert.equal(id, "run"); handler = action; },
    runRequest: () => ({dataset: "wine", mode: "smoke"}),
    updateRunning() {}, api: async () => ({id: "new"}),
    refreshState: (...args) => { assert.equal(args.length, 0); return pending; },
    message() {}, watch: async () => {},
    navigate: () => { throw new Error("Start completion must not navigate"); }};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('registerAction("run",'),
    source.indexOf("function invalidateUpload()")), context);
  const start = handler();
  await Promise.resolve();
  context.routeState = {page: "results", job: "old"};
  finishRefresh();
  await start;
  assert.equal(context.selectedJobId, "old");
  assert.equal(context.routeState.job, "old");
  assert.equal(context.lastStartedJob, "new");
  assert.equal(context.startingExperiment, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", "app.js")
