"""Exercise browser-only appearance behavior without adding a JavaScript test dependency."""
import shutil
import subprocess

import pytest

from quibble.web import ASSETS


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed for browser asset unit tests")


def run_script(script, asset):
    result = subprocess.run([NODE, "-e", script, str(ASSETS / asset)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_theme_initialization_changes_and_unavailable_storage():
    run_script(r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
function setup(saved, dark, blocked = false) {
  const events = {}, warnings = [], stored = {};
  const select = {hidden: true, addEventListener(type, fn) { this[type] = fn; }};
  const media = {matches: dark, addEventListener(type, fn) { this[type] = fn; }};
  let ready = false;
  const document = {documentElement: {dataset: {}},
    querySelector: () => ready ? select : null,
    addEventListener(type, fn) { events[type] = () => { ready = true; fn(); }; }};
  const storage = {
    getItem() { if (blocked) throw new DOMException("blocked", "SecurityError"); return saved; },
    setItem(key, value) {
      if (blocked) throw new DOMException("blocked", "QuotaExceededError");
      stored[key] = value;
    }
  };
  vm.runInNewContext(source, {document, DOMException, localStorage: storage,
    console: {warn: (...args) => warnings.push(args)}, matchMedia: () => media,
    addEventListener: (type, fn) => { events[type] = fn; }});
  return {document, select, media, events, stored, warnings};
}
for (const saved of [null, "system", "light", "dark", "invalid"]) {
  for (const dark of [true, false]) {
    const state = setup(saved, dark);
    const expected = ["light", "dark"].includes(saved) ? saved : dark ? "dark" : "light";
    assert.equal(state.document.documentElement.dataset.theme, expected);
    state.events.DOMContentLoaded();
    assert.equal(state.select.hidden, false);
    assert.equal(state.select.value, ["light", "dark"].includes(saved) ? saved : "system");
    state.select.value = "dark"; state.select.change();
    assert.equal(state.stored["quibble-theme"], "dark");
    state.media.matches = false; state.media.change();
    assert.equal(state.document.documentElement.dataset.theme, "dark");
    state.events.storage({key: "quibble-theme", newValue: "system"});
    assert.equal(state.document.documentElement.dataset.theme, "light");
    state.media.matches = true; state.media.change();
    assert.equal(state.document.documentElement.dataset.theme, "dark");
    state.events.storage({key: null, newValue: null});
    assert.equal(state.select.value, "system");
    assert.equal(state.warnings.length, saved === "invalid" ? 1 : 0);
  }
}
const blocked = setup(null, true, true);
blocked.events.DOMContentLoaded();
blocked.select.value = "light"; blocked.select.change();
assert.equal(blocked.document.documentElement.dataset.theme, "light");
assert.equal(blocked.warnings.length, 2);
""", "theme.js")


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
