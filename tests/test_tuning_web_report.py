"""Approval boundaries, browser invalidation and separate-root cost presentation."""
import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quibble_ml import report, web
from quibble_ml.config import Config
from quibble_ml.run import run_one


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Arrange
    for key, value in {
        "QUIBBLE_OUT_DIR": str(tmp_path / "out"), "QUIBBLE_CACHE_DIR": str(tmp_path / "cache"),
        "QUIBBLE_DATA_DIR": str(tmp_path / "data"), "QUIBBLE_TIER": "defaults",
        "QUIBBLE_RELATIONAL_REGISTRY": "", "QUIBBLE_TUNE_TRIALS": "20",
        "QUIBBLE_TUNE_FOLDS": "3", "QUIBBLE_TUNE_SEED": "0", "QUIBBLE_TUNE_METRIC": "auto",
        "QUIBBLE_TUNE_TIMEOUT_S": "0", "QUIBBLE_MITRA_ALLOW_DOWNLOAD": "0",
    }.items():
        monkeypatch.setenv(key, value)
    instance = web.Workbench()
    yield instance
    instance.close()


def request(**changes):
    return {"dataset": "wine", "backends": ["histgb"], "mode": "smoke", "device": "cpu",
            "cv": "holdout", "tier": "tuned", "tune_trials": 2, "tune_folds": 2,
            "tune_seed": 3, "tune_metric": "auto", "tune_timeout_s": 0, **changes}


@pytest.mark.parametrize("consent", [None, False, 1, "true"])
def test_given_no_explicit_tuning_approval_when_start_then_refuse_before_worker(app, consent):
    # Act & Assert
    with pytest.raises(ValueError, match="Explicit tuning approval"):
        app.start(request(tuning_confirmed=consent))
    assert app.thread is None
    assert not list(app.root.glob("*/job.json"))


@pytest.mark.parametrize("changes", [
    {"tier": "unknown"}, {"tune_trials": 0}, {"tune_trials": True}, {"tune_trials": 1.5},
    {"tune_folds": 1}, {"tune_folds": 21}, {"tune_seed": -1},
    {"tune_timeout_s": float("inf")}, {"tune_timeout_s": True}, {"tune_timeout_s": 2**5000},
    {"tune_metric": "accuracy"}, {"tune_metric": "rmse"},
    {"backends": ["linear"]}, {"backends": ["histgb", "linear"]},
    {"backends": ["catboost"], "tune_trials": 1000},
])
def test_given_invalid_tuning_when_start_then_no_job_is_created(app, changes):
    # Act & Assert
    with pytest.raises(ValueError):
        app.start(request(tuning_confirmed=True, **changes))
    assert app.thread is None
    assert not list(app.root.glob("*/job.json"))


def test_given_ambient_tuned_env_when_default_web_request_then_pin_defaults(app, monkeypatch):
    # Arrange
    captured = []
    monkeypatch.setenv("QUIBBLE_TIER", "tuned")
    monkeypatch.setenv("QUIBBLE_TUNE_TRIALS", "999")
    monkeypatch.setattr(web.subprocess, "run",
                        lambda cmd, **kw: (captured.append(cmd) or SimpleNamespace(returncode=1)))
    body = request()
    body.pop("tier")

    # Act
    job = app.start(body)
    app.thread.join(10)

    # Assert
    assert job["tier"] == "defaults" and job["tuning_confirmed"] is False
    assert captured[0][captured[0].index("--tier") + 1] == "defaults"
    assert captured[0][captured[0].index("--tune-trials") + 1] == "20"
    assert app.job(job["id"])["tuning_protocol"] == {"status": "not requested"}


def test_given_approved_tuning_when_start_then_freeze_protocol_and_worker_flags(app, monkeypatch):
    # Arrange
    captured = []
    monkeypatch.setattr(web.subprocess, "run",
                        lambda cmd, **kw: (captured.append(cmd) or SimpleNamespace(returncode=1)))

    # Act
    job = app.start(request(tuning_confirmed=True))
    app.thread.join(10)
    saved = app.job(job["id"])

    # Assert
    assert saved["campaign_tier"] == "tuned" and saved["tuning_confirmed"] is True
    assert saved["tuning_protocol"]["trials"] == saved["tune_trials"] == 2
    assert saved["tuning_protocol"]["inner_folds"] == saved["tune_folds"] == 2
    assert saved["tuning_protocol"]["seed"] == 3
    for flag, value in (("--tier", "tuned"), ("--tune-trials", "2"), ("--tune-folds", "2"),
                        ("--tune-seed", "3"), ("--tune-metric", "auto"), ("--tune-timeout-s", "0")):
        assert captured[0][captured[0].index(flag) + 1] == value


def test_given_tuned_full_run_when_only_tuning_approved_then_require_full_consent(app):
    # Act & Assert
    with pytest.raises(ValueError, match="Confirm the full comparison"):
        app.start(request(mode="full", tuning_confirmed=True))


def test_given_changed_controls_when_browser_events_then_invalidate_both_approvals():
    # Arrange
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for JavaScript contract tests")
    script = r"""
const assert = require("node:assert/strict"), vm = require("node:vm");
const source = require("node:fs").readFileSync(process.argv[1], "utf8");
const nodes = {
  methods: {querySelectorAll: () => [{value: "histgb"}]},
  mode: {value: "full"}, device: {value: "cpu"}, cv: {value: "holdout"},
  "threshold-mode": {value: "defaults"}, "full-confirmed": {checked: true},
  tier: {value: "tuned"}, "tune-trials": {value: "2"}, "tune-folds": {value: "2"},
  "tune-seed": {value: "3"}, "tune-metric": {value: "auto"}, "tune-timeout-s": {value: "0"},
  "tuning-confirmed": {checked: false}, "tuning-controls": {}, "tuning-approval": {},
  "tuning-cost": {}
};
const events = {};
const context = {$: id => nodes[id], readyDataset: "wine", readyFingerprint: null,
  deviceState: {cuda_available: false}, effectiveThresholds: () => ({}), lastStartedJob: null,
  document: {addEventListener: (name, fn) => {events[name] = fn;}}, renderActions: () => {}};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function clearFinalApproval("),
  source.indexOf("function workflowAction(")), context);
assert.throws(() => context.runRequest(true), /exact tuning/);
nodes["tuning-confirmed"].checked = true;
assert.equal(context.runRequest(true).tier, "tuned");
assert.equal(context.runRequest(true).tune_trials, 2);
const start = source.lastIndexOf('for (const event of ["input", "change"])');
vm.runInContext(source.slice(start, source.indexOf('$("workflow-primary-top").onclick', start)), context);
for (const id of ["tier", "tune-trials", "tune-folds", "tune-seed", "tune-metric", "tune-timeout-s",
  "methods", "mode", "device", "cv"]) {
  nodes["tuning-confirmed"].checked = true; nodes["full-confirmed"].checked = true;
  events.change({target: {id, closest: () => ({})}});
  assert.equal(nodes["tuning-confirmed"].checked, false, id);
  assert.equal(nodes["full-confirmed"].checked, false, id);
}
nodes.tier.value = "defaults"; nodes["tune-trials"].value = "bad";
assert.equal(context.runRequest().tune_trials, 20);
"""

    # Act
    result = subprocess.run([node, "-e", script, str(web.ASSETS / "app.js")],
                            capture_output=True, text=True, timeout=30)

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


def test_given_overlapping_stage_costs_when_rendered_then_never_doublecharge(tmp_path):
    # Arrange
    record = {"dataset": "example", "backend": "mitra_v2", "split_seed": 0, "model_seed": 0,
              "campaign_tier": "tuned", "tier": "defaults", "prediction_adapts": True,
              "fit_total_s": 3.0, "search_refit_s": 3.0, "search_s": 1.0, "refit_s": 2.0,
              "batch_prediction_s": 7.0, "adaptation_and_prediction_s": 7.0,
              "model_pipeline_s": 10.0, "prediction_only_s": None,
              "feature_costs": {"feature_total_s": 9.0},
              "tuning": {"status": "not supported"}}

    # Act
    text = report.protocol_and_cost_section([record], tmp_path)
    stages = pd.read_csv(tmp_path / "stage_costs.csv")

    # Assert
    assert stages.iloc[0]["model_pipeline_s"] == 10.0
    assert stages.iloc[0]["shared feature_total_s"] == 9.0
    assert "never add them" in text and "not zero" in text
    assert "Fit-only costs must not hide prediction-time fine-tuning" in text
    assert "not supported" in text


def test_given_task_specific_foundation_recipes_when_reported_then_label_actual_adaptation():
    # Arrange
    base = {"backend": "mitra_v2", "split_seed": 0, "model_seed": 0, "tier": "defaults",
            "recipe": {"requested": {"adaptation": {
                "classification": "zero-shot; no parameter fine-tuning",
                "regression": {"num_bag_folds": 2, "max_steps": 50}}}}}

    # Act
    classification = report.training_protocol_row({**base, "task": "classification"})
    regression = report.training_protocol_row({
        **base, "task": "regression", "fine_tuning": True, "prediction_adapts": True})

    # Assert
    assert classification["adaptation"] == "zero-shot; no parameter fine-tuning"
    assert regression["adaptation"] == "fine-tuned at prediction time; num_bag_folds=2, max_steps=50"
    assert classification["actual recipe tier"] == regression["actual recipe tier"] == "defaults"


@pytest.fixture
def tier_roots(app, monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setattr(report, "_plt", lambda: None)
    cfg = Config.from_env()
    defaults = tmp_path / "defaults"
    tuned = tmp_path / "tuned"
    run_one("histgb", "wine", 0, replace(cfg, tier="defaults", out_dir=str(defaults)))
    run_one("histgb", "wine", 0, replace(cfg, tier="tuned", out_dir=str(tuned),
                                      tune_trials=1, tune_folds=2))
    return defaults, tuned


def test_given_separate_roots_when_reported_then_use_tier_qualified_same_backend(
        tier_roots, tmp_path):
    # Arrange
    defaults, tuned = tier_roots
    originals = {p: p.read_bytes() for root in tier_roots for p in (root / "wine").glob("*.json")}

    # Act
    path = report.build_tier_comparison(defaults, tuned, datasets=["wine"],
                                        report_dir=tmp_path / "contrast")
    text = path.read_text()

    # Assert
    assert "histgb[defaults]" in text and "histgb[tuned]" in text
    assert "never merged" in text and "not significance tests" in text
    assert (path.parent / "tier_comparison.json").is_file()
    assert all(path.read_bytes() == value for path, value in originals.items())


def test_given_same_root_when_cross_tier_reported_then_reject(tier_roots, tmp_path):
    # Act & Assert
    with pytest.raises(ValueError, match="separate roots"):
        report.build_tier_comparison(tier_roots[0], tier_roots[0], datasets=["wine"],
                                     report_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_given_tuned_run_when_normal_reported_then_preserve_cost_and_attempt_exports(tier_roots):
    # Act
    path = report.build(tier_roots[1], datasets=["wine"], model_seed=0)
    text = path.read_text()
    exported = json.loads((path.parent / "backend_provenance.json").read_text())

    # Assert
    assert "Recipe tiers, search and full model cost" in text
    assert "attempted fits" in text and "valid trials" in text
    assert "tuned" in text and "search_refit_s" in text
    assert exported["runs"]["wine"][0]["tuning"]["attempted_fits"] == 3


def test_given_minimal_relational_source_when_reported_then_render_audit_without_flat_fields(
        monkeypatch, tmp_path):
    # Arrange
    res = {
        "dataset": "rel", "task": "regression", "metric": "rmse", "model_seed": 0,
        "splits": [0], "backends": ["histgb"], "summary": pd.DataFrame({
            "backend": ["histgb"], "rmse": [1.0], "ci_lo": [0.8], "ci_hi": [1.2]}),
        "dropped": {}, "pairs": [], "calibration": [], "slices": pd.DataFrame(), "flags": [],
        "split": {"kind": "relational"}, "dataset_source": {
            "kind": "relational", "fingerprint": "abc", "builder_version": 1},
        "meta": {"rows": 30, "features": 4, "tier": "small", "cat_share": 0, "missing_share": 0},
    }
    record = {"dataset": "rel", "backend": "histgb", "split_seed": 0, "model_seed": 0,
              "fit_total_s": 1.0, "batch_prediction_s": 0.1, "model_pipeline_s": 1.1,
              "relational_audit": {"purged_examples": 3, "n_features": 4},
              "feature_costs": {"feature_total_s": 2.0}}
    res["adir"] = tmp_path / "rel" / "analysis"
    res["adir"].mkdir(parents=True)
    res["relational_slices"] = pd.DataFrame([{
        "backend_a": "histgb", "backend_b": "lightgbm", "slice": "cold_start",
        "n_rows": 0, "n_groups": 0, "metric": "rmse", "status": "empty",
        "disagreement_rate": None, "quality_a": None, "quality_b": None}])
    res["relational_slices"].to_csv(res["adir"] / "relational_slices.csv", index=False)
    (tmp_path / "rel" / "histgb_ss0_ms0.json").write_text(json.dumps(record))
    monkeypatch.setattr(report, "_plt", lambda: None)
    monkeypatch.setattr(report, "analyse", lambda *a, **k: res)
    monkeypatch.setattr(report, "load_runs", lambda *a: [record])
    cold_stage = {"feature_audit": {"costs": {"feature_total_s": 9.0, "cache_state": "cold"}},
                  "backends": ["histgb"], "sweep_wall_s": 12.0}
    (tmp_path / "relational-feature-stage.json").write_text(json.dumps(cold_stage))

    # Act
    text = report.build(tmp_path, datasets=["rel"]).read_text()

    # Assert
    assert "Relational data source" in text and "purged_examples" in text
    assert "shared feature_total_s" in text
    assert "Local data source" not in text
    assert "Shared relational feature stage (once)" in text and "cold" in text
    assert "Descriptive relational slices" in text
    assert "not intervals, causal claims or swap gates" in text
    assert "Limited or empty support" in text
    assert "relational_slices.csv" in text
    archived = next((tmp_path / "runs").glob("*/*/relational-feature-stage.json"))
    assert json.loads(archived.read_text()) == cold_stage
    assert next((tmp_path / "runs").glob("*/*/report/stage_costs.csv")).is_file()
