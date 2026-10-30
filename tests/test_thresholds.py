"""Default policy resources, validation, export, and report integration."""
import json
import subprocess
import sys
from importlib.resources import files

import pytest

from quibble.thresholds import (DEFAULTS, export_thresholds, load_thresholds,
                                resolve_thresholds, validate_primary)


def test_defaults_are_the_packaged_resource_and_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    packaged = json.loads(files("quibble").joinpath("swap_thresholds.json").read_text(encoding="utf-8"))
    assert resolve_thresholds() == packaged == DEFAULTS
    from quibble.analyze import BIG, HI, MIN_SLICE_ROWS, NEAR
    assert (HI, BIG, NEAR, MIN_SLICE_ROWS) == tuple(
        packaged[key] for key in ("confident_prob", "regression_gap_std", "near_margin", "min_slice_n"))
    merged = resolve_thresholds({"max_quality_drop": 0.02})
    assert merged == {**packaged, "max_quality_drop": 0.02}
    merged["max_quality_drop"] = 0.5
    assert DEFAULTS == packaged


@pytest.mark.parametrize("overrides", [
    [], "limits", 1, {"typo": 0.1}, {"primary": "unknown"}, {"primary": []},
    {"max_quality_drop": True}, {"max_quality_drop": "0.01"}, {"max_quality_drop": None},
    {"max_quality_drop": float("nan")}, {"max_quality_drop": float("inf")},
    {"max_quality_drop": 10 ** 400}, {"min_slice_n": 0}, {"min_slice_n": 1.5},
    {"min_slice_n": True}, {"confident_prob": 1.1}, {"near_margin": -0.1},
    {"max_disagreement": 1.1}, {"max_memory_ratio": 0}, {"regression_gap_std": -1},
])
def test_invalid_policies_fail_explicitly(overrides):
    with pytest.raises(ValueError):
        resolve_thresholds(overrides)


def test_explicit_policy_requirement_and_metric_compatibility():
    for overrides in (None, {}):
        with pytest.raises(ValueError, match="non-empty policy"):
            resolve_thresholds(overrides, require=True)
    assert resolve_thresholds(DEFAULTS, require=True) == DEFAULTS
    validate_primary(resolve_thresholds({"primary": "auc"}), "classification")
    validate_primary(resolve_thresholds({"primary": "rmse"}), "regression")
    with pytest.raises(ValueError, match="not valid for classification"):
        validate_primary(resolve_thresholds({"primary": "rmse"}), "classification")
    with pytest.raises(ValueError, match="not valid for regression"):
        validate_primary(resolve_thresholds({"primary": "accuracy"}), "regression")
    assert resolve_thresholds({"max_quality_drop": -0.01})["max_quality_drop"] == -0.01


@pytest.mark.parametrize("text", ["{", "[]", "null", '{"max_quality_drop": NaN}', '{"typo": 1}'])
def test_loading_invalid_json_reports_the_policy_path(tmp_path, text):
    path = tmp_path / "policy.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load thresholds"):
        load_thresholds(path)


def test_export_cli_works_outside_repository_and_never_overwrites(tmp_path):
    path = tmp_path / "policy.json"
    command = [sys.executable, "-m", "quibble", "swap", "--export-thresholds", str(path)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert load_thresholds(path) == DEFAULTS
    before = path.read_bytes()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and path.read_bytes() == before
    with pytest.raises(FileExistsError):
        export_thresholds(path)


def test_swap_records_custom_policy_and_enforces_requirement(cv_results, tmp_path):
    path = tmp_path / "limits.json"
    path.write_text('{"max_quality_drop": 0.02}', encoding="utf-8")
    command = [sys.executable, "-m", "quibble", "swap", "--incumbent", "histgb",
               "--candidate", "histgb", "--dataset", "breast_cancer", "--out", str(cv_results),
               "--thresholds", str(path), "--require-thresholds"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "THRESHOLDS: custom policy" in result.stdout
    saved = json.loads((cv_results / "breast_cancer" / "analysis" /
                        "swap_histgb_to_histgb.json").read_text())
    assert saved["thresholds"] == {**DEFAULTS, "max_quality_drop": 0.02}
    assert saved["thresholds_default"] is False
    path.write_text("{}", encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "non-empty policy" in result.stderr


@pytest.mark.parametrize("custom", [None, {"max_quality_drop": 0.02}])
def test_report_shows_effective_policy_and_origin(cv_results, custom):
    from quibble.report import build
    text = build(cv_results, ["breast_cancer"], swaps=[("histgb", "histgb")],
                 thresholds=custom).read_text(encoding="utf-8")
    assert "Effective swap thresholds" in text
    assert ("Packaged defaults" if custom is None else "Custom policy") in text
    assert "not a production-safety guarantee" in text
    for key in DEFAULTS:
        assert key in text
    if custom:
        assert "0.02" in text


def test_report_requirement_is_checked_before_writing(tmp_path):
    from quibble.report import build
    with pytest.raises(ValueError, match="non-empty policy"):
        build(tmp_path, require_thresholds=True)
    assert not (tmp_path / "report").exists()
