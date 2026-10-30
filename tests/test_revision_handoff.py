"""Regression coverage for the revision handoff's decision-integrity gaps."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quibble.config import Config
from quibble.datasets import split
from quibble.gate import evaluate as gate
from quibble.run import run_one, stem
from quibble.swap import evaluate


@pytest.mark.parametrize("failure", ["error", "refused", "prediction", "record", "shared"])
def test_gate_rejects_partial_coverage(cv_results, tmp_path, failure):
    out = tmp_path / "results"
    shutil.copytree(cv_results, out)
    path = stem(out, "breast_cancer", "lightgbm", 1, 0).with_suffix(".json")
    if failure in ("error", "refused"):
        record = json.loads(path.read_text())
        record[failure] = "deliberately unsuccessful"
        path.write_text(json.dumps(record))
    elif failure == "prediction":
        path.with_suffix(".pred.npy").unlink()
    elif failure == "record":
        path.unlink()
    else:
        (path.parent / "idx_test_ss1.npy").unlink()
    result = gate("breast_cancer", "lightgbm", "histgb", min_gain=-1.0, out=out)
    assert result["quality_verdict"] == "PASS"
    assert result["overall"] == "FAIL"
    assert result["coverage"]["required"] == [0, 1, 2]
    assert result["coverage"]["missing"] == [1]
    assert result["coverage"]["reasons"]["1"]
    assert result["splits"] == 2
    command = [sys.executable, "-m", "quibble", "gate", "--dataset", "breast_cancer",
               "--candidate", "lightgbm", "--baseline", "histgb", "--min-gain", "-1",
               "--out", str(out)]
    cli = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert cli.returncode == 1 and "coverage" in cli.stdout.lower()
    assert "OVERALL: FAIL" in cli.stdout


def test_gate_honours_requested_noncontiguous_split_ids(cv_results, tmp_path):
    out = tmp_path / "results"
    shutil.copytree(cv_results, out)
    (out / "invocation.json").write_text(json.dumps({"cells": [
        {"dataset": "breast_cancer", "backend": "histgb", "model_seed": 0, "split_seed": 7}
    ]}))
    result = gate("breast_cancer", "histgb", "histgb", out=out)
    assert result["overall"] == "FAIL"
    assert result["coverage"]["required"] == [0, 1, 2, 7]
    assert result["coverage"]["missing"] == [7]
    assert result["coverage"]["baseline_missing"] == [7]


def test_gate_rejects_incomplete_baseline(cv_results, tmp_path):
    out = tmp_path / "results"
    shutil.copytree(cv_results, out)
    stem(out, "breast_cancer", "histgb", 1, 0).with_suffix(".pred.npy").unlink()
    result = gate("breast_cancer", "lightgbm", "histgb", min_gain=-1.0, out=out)
    assert result["quality_verdict"] == "PASS" and result["overall"] == "FAIL"
    assert result["coverage"]["missing"] == []
    assert result["coverage"]["baseline_missing"] == [1]


def test_gate_explicitly_rejects_no_usable_candidate(cv_results, tmp_path):
    from quibble.swap import SwapError
    out = tmp_path / "results"
    shutil.copytree(cv_results, out)
    for path in (out / "breast_cancer").glob("lightgbm*.pred.npy"):
        path.unlink()
    with pytest.raises(SwapError, match="candidate lightgbm has no usable runs"):
        gate("breast_cancer", "lightgbm", "histgb", out=out)


def missing_class_data(kind="temporal"):
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    X.attrs["quibble_split"] = {"kind": kind}
    X.attrs["quibble_times"] = np.arange(30)
    X.attrs["quibble_groups"] = np.array(["past"] * 21 + ["future"] * 9)
    y = pd.Series(([1, 2] * 10) + [1] + [0, 1, 2] * 3)
    return X, y


@pytest.mark.parametrize("kind,split_id", [("temporal", 0), ("group", 1)])
def test_split_explicitly_rejects_missing_global_classes(kind, split_id):
    X, y = missing_class_data(kind)
    with pytest.raises(ValueError, match=r"missing.*class.*0"):
        split(X, y, split_id, "classification")


def test_one_class_temporal_training_is_explicitly_refused():
    X, y = missing_class_data()
    y.iloc[:21] = 1
    with pytest.raises(ValueError, match="missing.*class"):
        split(X, y, 0, "classification")


@pytest.mark.parametrize("backend", ["histgb", "dummy", "linear", "tabpfn", "autogluon"])
def test_missing_class_split_is_refused_before_any_backend_fits(tmp_path, monkeypatch, backend):
    from quibble import datasets
    X, y = missing_class_data()
    monkeypatch.setattr(datasets, "load", lambda *a: (X, y, "classification"))
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    run_one(backend, "example", 0, Config.from_env())
    path = stem(tmp_path, "example", backend, 0, 0)
    result = json.loads(path.with_suffix(".json").read_text())
    assert "missing" in result["refused"] and "0" in result["refused"]
    assert "accuracy" not in result and "error" not in result
    assert not path.with_suffix(".pred.npy").exists()


def test_legacy_group_classification_requires_class_order_validation(tmp_path):
    from quibble.datasets import validate_sources
    directory = tmp_path / "example"
    directory.mkdir()
    source = {"fingerprint": "same", "task": "classification", "group_column": "customer",
              "class_labels": ["a", "b", "c"]}
    path = directory / "dummy_ss0_ms0.json"
    record = {"dataset_source": source}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="class-order validation"):
        validate_sources(tmp_path, "example", source)
    record["prediction_classes"] = [0, 1, 2]
    path.write_text(json.dumps(record))
    validate_sources(tmp_path, "example", source)


def test_saved_probability_columns_match_recorded_classes(cv_results, tmp_path):
    from quibble.pool import load_pool
    out = tmp_path / "results"
    shutil.copytree(cv_results, out)
    path = stem(out, "breast_cancer", "histgb", 0, 0).with_suffix(".json")
    record = json.loads(path.read_text())
    record["prediction_classes"] = [1, 2]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recorded class order"):
        load_pool(out, "breast_cancer", "histgb", 0)


def test_partial_policy_discloses_inherited_limits_in_result_and_json(cv_results, tmp_path):
    result = evaluate("breast_cancer", "histgb", "histgb", out=cv_results,
                      thresholds={"max_quality_drop": 0.02}, require_thresholds=True)
    provenance = result["threshold_provenance"]
    assert provenance["kind"] == "partial"
    assert provenance["provided"] == ["max_quality_drop"]
    assert "max_disagreement" in provenance["inherited"]
    assert "max_quality_drop" not in provenance["inherited"]
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_quality_drop": 0.02}))
    command = [sys.executable, "-m", "quibble", "swap", "--dataset", "breast_cancer",
               "--incumbent", "histgb", "--candidate", "histgb", "--thresholds", str(policy),
               "--require-thresholds", "--out", str(cv_results)]
    cli = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    saved = cv_results / "breast_cancer" / "analysis" / "swap_histgb_to_histgb.json"
    assert json.loads(saved.read_text())["threshold_provenance"] == provenance
    assert "partial" in cli.stdout and "max_disagreement" in cli.stdout


def test_report_preserves_flags_notes_split_and_policy(cv_results, monkeypatch):
    from quibble import report
    flag = {"flag": "time_like_column_random_split", "column": "event_date",
            "detail": "test warning about time ordering"}
    monkeypatch.setattr(report, "_plt", lambda: None)
    monkeypatch.setattr(report, "evaluate", lambda *a: {
        "overall": "PASS", "checks": [{"verdict": "PASS", "check": "quality",
        "observed": "0", "limit": "0.01", "note": "retraining variability not counted"}],
        "thresholds": {"max_quality_drop": 0.01}, "thresholds_default": True,
        "threshold_provenance": {"kind": "defaults", "provided": [],
                                 "inherited": ["max_quality_drop"]},
        "flags": [flag], "split": {"kind": "temporal", "time_column": "event_date"},
    })
    path = report.build(cv_results, ["breast_cancer"], swaps=[("histgb", "histgb")])
    text = path.read_text(encoding="utf-8")
    for phrase in ("retraining variability not counted", "time_like_column_random_split",
                   "test warning about time ordering", "temporal", "event_date",
                   "placeholder", "distinct rows", "not calibrated"):
        assert phrase in text
    archived = list((cv_results / "runs").glob("*/*/report/report.md"))
    assert any("test warning about time ordering" in p.read_text(encoding="utf-8") for p in archived)


def test_reseed_adjustment_keeps_raw_disagreement_visible(cv_results, monkeypatch):
    import quibble.swap as swap
    monkeypatch.setattr(swap, "reseed_rate", lambda *a, **k: 0.2)
    result = evaluate("breast_cancer", "histgb", "histgb", out=cv_results)
    check = next(c for c in result["checks"] if c["check"].startswith("disagreement rate"))
    assert "raw disagreement" in check["note"]
    assert "uncertainty" in check["note"]


def test_usable_corrected_interval_is_identified():
    from quibble.metrics import Boot
    from quibble.swap import interval
    _, lo, hi, notes = interval(Boot(np.arange(20), 30), lambda w: 0.0,
                               [0.0, 0.01, 0.0, -0.01, 0.0], 80, 20)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert any("corrected split interval" in note for note in notes)
    assert not any("not counted" in note for note in notes)


def test_report_warns_when_using_saved_mismatched_results(cv_results, monkeypatch):
    from quibble import report
    monkeypatch.setattr(report, "_plt", lambda: None)
    (cv_results / "invocation.json").write_text(json.dumps({"workflow": "run",
        "datasets": ["breast_cancer"], "backends": ["histgb"], "cells": [
            {"dataset": "breast_cancer", "backend": "histgb", "model_seed": 0,
             "split_seed": 0, "execution": "incompatible settings",
             "differences": ["single_n"]}
        ]}))
    text = report.build(cv_results, ["breast_cancer"]).read_text(encoding="utf-8")
    assert "Saved-setting mismatches" in text
    assert "single_n" in text and "not newly evaluated" in text
