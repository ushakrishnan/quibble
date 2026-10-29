"""Local input contracts, provenance guards, and real CSV/Parquet CLI workflows."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quibble.analyze import AnalysisError, analyse
from quibble.config import Config
from quibble.datasets import known, load, local_names, source_info, split, verify
from quibble.run import run_one, settings_differ, stem
from quibble.swap import SwapError, evaluate


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(root))
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("QUIBBLE_CACHE_DIR", str(tmp_path / "cache"))
    directory = root / "payments"
    directory.mkdir(parents=True)
    frame = pd.DataFrame({
        "id": np.arange(120), "amount": np.arange(120, dtype=float),
        "region": ["east", "west"] * 60, "code": ["001", "002"] * 60,
        "late": ["no"] * 60 + ["yes"] * 60,
    })
    frame.to_csv(directory / "data.csv", index=False)
    spec = {"file": "data.csv", "target": "late", "task": "classification",
            "drop_columns": ["id"], "categorical_columns": ["code"]}
    (directory / "dataset.json").write_text(json.dumps(spec), encoding="utf-8")
    return directory, frame, spec, tmp_path / "results"


def write_spec(directory, spec):
    (directory / "dataset.json").write_text(json.dumps(spec), encoding="utf-8")


def test_csv_loading_keeps_target_out_of_features_and_records_labels(local_data):
    directory, _, _, _ = local_data
    X, y, task = load("payments")
    assert task == "classification" and set(X) == {"amount", "region", "code"}
    assert sorted(X["code"].unique()) == ["001", "002"]
    assert X["region"].dtype == object
    assert set(y) == {0, 1}
    source = X.attrs["quibble_source"]
    assert source["class_labels"] == ["no", "yes"] and source["target"] == "late"
    assert source["path"] == str(directory / "data.csv")
    assert source["fingerprint"] == source_info("payments")["fingerprint"]
    assert known("payments") and "payments" in local_names()


def test_category_vocabulary_is_learned_from_training_rows_only(local_data):
    from quibble.backends import Prep
    X, _, _ = load("payments")
    prep = Prep("ordinal").fit(X[X["region"] == "east"])
    transformed = prep.transform(X[X["region"] == "west"])
    assert transformed["region"].isna().all()


def test_parquet_regression_and_missing_features(local_data):
    directory, frame, spec, _ = local_data
    frame["late"] = np.linspace(1., 30., len(frame))
    frame.loc[0, "amount"] = np.nan
    frame.to_parquet(directory / "data.parquet", index=False)
    spec.update(file="data.parquet", task="regression")
    write_spec(directory, spec)
    X, y, task = load("payments")
    assert task == "regression" and y.dtype == float and X["amount"].isna().sum() == 1
    assert "late" not in X and "id" not in X


@pytest.mark.parametrize("change,message", [
    ({"task": "auto"}, "task must"),
    ({"target": "missing"}, "missing declared columns"),
    ({"drop_columns": ["late"]}, "must not overlap"),
    ({"drop_columns": ["unknown"]}, "missing declared columns"),
    ({"categorical_columns": ["code", "code"]}, "unique column names"),
    ({"file": "..\\outside.csv"}, "inside the dataset folder"),
    ({"file": "data.xlsx"}, ".csv or .parquet"),
    ({"typo": 1}, "only"),
])
def test_invalid_manifests_are_explicit_errors(local_data, change, message):
    directory, _, spec, _ = local_data
    write_spec(directory, {**spec, **change})
    with pytest.raises(ValueError, match=message):
        load("payments")


@pytest.mark.parametrize("problem,message", [
    ("missing_target", "target contains missing"),
    ("single_class", "at least 2 classes"),
    ("infinite_feature", "contains infinity"),
    ("no_features", "at least one feature"),
    ("duplicate_header", "unique, nonempty headers"),
    ("extra_field", "fields, expected"),
    ("nonnumeric_target", "Unable to parse"),
    ("infinite_target", "finite numeric"),
    ("too_few_rows", "at least 4"),
])
def test_invalid_tables_are_not_silently_repaired(local_data, problem, message):
    directory, frame, spec, _ = local_data
    if problem == "missing_target":
        frame.loc[0, "late"] = None
    elif problem == "single_class":
        frame["late"] = "yes"
    elif problem == "infinite_feature":
        frame.loc[0, "amount"] = np.inf
    elif problem == "no_features":
        spec["drop_columns"] = ["id", "amount", "region", "code"]
        spec["categorical_columns"] = []
    elif problem in ("nonnumeric_target", "infinite_target"):
        spec["task"] = "regression"
        frame["late"] = "bad" if problem == "nonnumeric_target" else np.inf
    elif problem == "too_few_rows":
        frame = frame.head(3)
    frame.to_csv(directory / "data.csv", index=False)
    write_spec(directory, spec)
    if problem == "duplicate_header":
        (directory / "data.csv").write_text("amount,amount,late\n1,2,no\n", encoding="utf-8")
    elif problem == "extra_field":
        (directory / "data.csv").write_text("amount,late\n1,no,extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load("payments")


def test_local_cv_requires_enough_examples_per_class(local_data):
    X, y, task = load("payments")
    with pytest.raises(ValueError, match="at least 61 rows per class"):
        split(X, y, 0, task, "61x1")


@pytest.mark.parametrize("name", ["..\\outside", "a,b", "report", "runs", "smoke", "artifacts"])
def test_invalid_local_names_cannot_escape_or_collide_with_output_folders(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid dataset output name"):
        stem(tmp_path, name, "histgb", 0, 0)


def test_registered_names_cannot_silently_load_a_conflicting_local_table(local_data):
    directory, _, _, _ = local_data
    directory.rename(directory.parent / "wine")
    with pytest.raises(ValueError, match="registered dataset name"):
        load("wine")


def test_relocating_unchanged_input_preserves_fingerprint(local_data, tmp_path, monkeypatch):
    import shutil
    directory, _, _, _ = local_data
    source = source_info("payments")
    destination = tmp_path / "moved-data" / "payments"
    shutil.copytree(directory, destination)
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(destination.parent))
    relocated = source_info("payments")
    assert relocated["fingerprint"] == source["fingerprint"]
    assert relocated["path"] != source["path"]


def test_dataset_verification_reports_local_provenance_and_errors(local_data):
    result = verify(["payments", "missing"], None)
    assert result["payments"]["rows"] == 120
    assert result["payments"]["dataset_source"]["kind"] == "local"
    assert "error" in result["missing"]


@pytest.mark.parametrize("change", ["features", "manifest"])
def test_changed_source_is_rejected_by_training_analysis_swap_and_compare(local_data, change):
    directory, frame, spec, out = local_data
    cfg = Config.from_env()
    run_one("histgb", "payments", 0, cfg)
    record_path = out / "payments" / "histgb_ss0_ms0.json"
    original = json.loads(record_path.read_text())
    assert original["dataset_source"]["class_labels"] == ["no", "yes"]
    if change == "features":
        frame["amount"] += 1
        frame.to_csv(directory / "data.csv", index=False)
    else:
        spec["drop_columns"].append("amount")
        write_spec(directory, spec)
    assert "dataset_source" in settings_differ(record_path, cfg)
    with pytest.raises(AnalysisError, match="source differs"):
        analyse("payments", out)
    with pytest.raises(SwapError, match="source differs"):
        evaluate("payments", "histgb", "histgb", out=out)
    result = subprocess.run([sys.executable, "-m", "quibble", "compare", "--dataset", "payments",
                             "--out", str(out)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "source differs" in result.stderr
    run_one("histgb", "payments", 0, cfg, force=True)
    record = json.loads(record_path.read_text())
    assert "source differs" in record["error"]


@pytest.mark.parametrize("task,extension,smoke", [
    ("classification", "csv", True), ("regression", "parquet", False),
])
def test_local_data_cli_end_to_end(local_data, task, extension, smoke):
    directory, frame, spec, out = local_data
    if task == "regression":
        frame["late"] = frame["amount"] * 2 + np.sin(frame["amount"])
        frame.to_parquet(directory / "data.parquet", index=False)
    spec.update(task=task, file=f"data.{extension}")
    write_spec(directory, spec)
    args = [sys.executable, "-m", "quibble", "benchmark", "--datasets", "payments",
            "--backends", "histgb,lightgbm", "--device", "cpu", "--jobs", "1",
            "--out", str(out)]
    args += ["--smoke"] if smoke else ["--cv", "3x1"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    root = next((out / "smoke").iterdir()) if smoke else out
    text = (root / "report" / "report.md").read_text(encoding="utf-8")
    assert "### Interpretation" in text and "payments" in text
    assert "### Local data source" in text and "Data/manifest fingerprint" in text
    assert ("No swap approval" in text) if smoke else ("Swap check:" in text)
    records = list((root / "payments").glob("*_ss*_ms*.json"))
    assert len(records) == (2 if smoke else 6)
    assert all(json.loads(p.read_text())["dataset_source"]["task"] == task for p in records)


def test_dataset_cli_lists_local_names_and_verification_exits_nonzero(local_data, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    for args, code, message in [(["--list"], 0, "payments: local"),
                                (["--verify", "--names", "missing"], 1, "ERROR")]:
        result = subprocess.run([sys.executable, "-m", "quibble", "datasets", *args],
                                 cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == code and message in result.stdout, result.stdout + result.stderr
