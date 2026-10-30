"""Workflow isolation, centralized caches, participation, and reusable model artifacts."""
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from quibble import benchmark, report
from quibble.backends import AutoGluon
from quibble.cache import prepare_caches
from quibble.config import Config
from quibble.datasets import load
from quibble.history import snapshot
from quibble.run import run_one, smoke_positions, stem, sweep


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key, value in {
        "QUIBBLE_OUT_DIR": str(tmp_path / "results"), "QUIBBLE_CACHE_DIR": str(tmp_path / "cache"),
        "QUIBBLE_DATA_HOME": "", "QUIBBLE_KEEP_MODELS": "0", "QUIBBLE_DEVICE": "cpu",
        "QUIBBLE_CV": "holdout", "QUIBBLE_MODEL_SEEDS": "0", "QUIBBLE_SPLIT_SEED": "0",
        "_QUIBBLE_RUN_TYPE": "benchmark", "_QUIBBLE_WORKFLOW": "run",
    }.items():
        monkeypatch.setenv(key, value)
    return tmp_path


def test_cache_paths_are_local_before_imports_and_override_inherited_hf(settings, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(settings / "external"))
    monkeypatch.setenv("HF_HUB_CACHE", str(settings / "external" / "hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(settings / "external" / "transformers"))
    monkeypatch.chdir(settings)
    paths = prepare_caches()
    root = settings / "cache"
    assert paths["root"] == root
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "TORCH_HOME", "TABPFN_MODEL_CACHE_DIR", "SKLEARN_DATA", "XDG_CACHE_HOME",
                "OPENML_CACHE_DIRECTORY"):
        assert Path(os.environ[key]).is_relative_to(root), key
    override = settings / "data"
    monkeypatch.setenv("QUIBBLE_DATA_HOME", str(override))
    assert Config.from_env().data_home == str(override)


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_presets_respect_device_and_keep_models(settings, monkeypatch, device):
    from quibble import config
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    mocked_sweep = Mock(return_value={"cells": []})
    mocked_build = Mock(return_value=settings / "report.md")
    monkeypatch.setattr(benchmark, "sweep", mocked_sweep)
    monkeypatch.setattr(report, "build", mocked_build)
    monkeypatch.setenv("QUIBBLE_DEVICE", device)
    monkeypatch.setattr(sys, "argv", ["quibble benchmark", "--keep-models"])
    assert benchmark.main() == 0
    backends, datasets, cfg, force = mocked_sweep.call_args.args
    assert backends == ["histgb", "lightgbm", "tabfm", "kumo_tabular"]
    assert datasets == ["wine"] and cfg.cv == "3x2" and cfg.model_seeds == (0,)
    assert cfg.device == ("cuda" if device == "auto" else device)
    assert cfg.keep_models and not force
    assert mocked_build.call_args.kwargs["swaps"] == (
        [("histgb", "lightgbm"), ("histgb", "tabfm"), ("histgb", "kumo_tabular")]
        if cfg.device == "cpu" else [("tabfm", "kumo_tabular")])


@pytest.mark.parametrize("args", [["--backends", ""], ["--datasets", "unknown"],
                                  ["--smoke", "--cv", "3x2"], ["--smoke", "--model-seeds", "1"]])
def test_benchmark_rejects_invalid_selections_before_running(settings, monkeypatch, args):
    mock = Mock()
    monkeypatch.setattr(benchmark, "sweep", mock)
    monkeypatch.setattr(sys, "argv", ["quibble benchmark", *args])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    mock.assert_not_called()


def test_benchmark_passes_explicit_policy_and_requirement_to_report(settings, monkeypatch):
    limits = settings / "policy.json"
    limits.write_text('{"max_quality_drop": 0.02}', encoding="utf-8")
    monkeypatch.setattr(benchmark, "sweep", Mock(return_value={"cells": []}))
    build = Mock(return_value=settings / "report.md")
    monkeypatch.setattr(report, "build", build)
    monkeypatch.setattr(sys, "argv", ["quibble benchmark", "--backends", "histgb",
                                    "--thresholds", str(limits), "--require-thresholds"])
    assert benchmark.main() == 0
    assert build.call_args.kwargs["thresholds"] == {"max_quality_drop": 0.02}
    assert build.call_args.kwargs["require_thresholds"] is True


@pytest.mark.parametrize("text,flags", [
    ("{}", ["--require-thresholds"]),
    ("[]", []),
    ('{"max_quality_drop": -Infinity}', []),
    ('{"primary": "rmse"}', []),
    ('{"max_quality_drop": 0.02}', ["--smoke"]),
])
def test_benchmark_rejects_invalid_policy_before_training(settings, monkeypatch, text, flags):
    limits = settings / "policy.json"
    limits.write_text(text, encoding="utf-8")
    run = Mock()
    monkeypatch.setattr(benchmark, "sweep", run)
    monkeypatch.setattr(sys, "argv", ["quibble benchmark", "--backends", "histgb",
                                    "--thresholds", str(limits), *flags])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("cv", ["1x1", "3x0", "-2x1"])
def test_invalid_fold_counts_are_configuration_errors(settings, monkeypatch, capsys, cv):
    monkeypatch.setattr(sys, "argv", ["quibble benchmark", f"--cv={cv}"])
    assert benchmark.main() == 2
    assert "config error" in capsys.readouterr().err
    assert not (settings / "results").exists()


def test_relative_cache_paths_resolve_from_project_not_cwd(settings, monkeypatch):
    from quibble import cache
    monkeypatch.setattr(cache, "PROJECT_ROOT", settings)
    monkeypatch.setenv("QUIBBLE_CACHE_DIR", "downloads")
    elsewhere = settings / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert prepare_caches()["root"] == settings / "downloads"


def test_individual_cache_overrides(settings, monkeypatch):
    from quibble import cache
    monkeypatch.setattr(cache, "PROJECT_ROOT", settings)
    monkeypatch.setenv("QUIBBLE_SKLEARN_CACHE_DIR", "custom-data")
    monkeypatch.setenv("QUIBBLE_HF_CACHE_DIR", "custom-hf")
    monkeypatch.setenv("QUIBBLE_TRANSFORMERS_CACHE_DIR", "custom-transformers")
    paths = prepare_caches()
    assert paths["sklearn"] == settings / "custom-data"
    assert paths["huggingface"] == settings / "custom-hf"
    assert paths["transformers"] == settings / "custom-transformers"
    assert os.environ["HF_HOME"] == str(settings / "custom-hf")
    assert os.environ["HF_HUB_CACHE"] == str(settings / "custom-hf" / "hub")
    assert os.environ["SKLEARN_DATA"] == str(settings / "custom-data")
    assert os.environ["TRANSFORMERS_CACHE"] == str(settings / "custom-transformers")


def test_smoke_rows_are_bounded_deterministic_and_keep_classes():
    y = pd.Series([0] * 500 + [1] * 20 + [2])
    first = smoke_positions(y, 128, 42, "classification")
    assert len(first) == len(set(first)) == 128
    assert set(y.iloc[first]) == {0, 1, 2}
    np.testing.assert_array_equal(first, smoke_positions(y, 128, 42, "classification"))
    with pytest.raises(ValueError, match="cannot represent"):
        smoke_positions(y, 2, 42, "classification")


def test_smoke_cli_is_isolated_and_models_reload(settings):
    root = settings / "results"
    baseline = root / "wine" / "histgb_ss0_ms0.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text('{"marker": "keep benchmark data"}')
    r = subprocess.run(
        [sys.executable, "-m", "quibble", "benchmark", "--backends", "histgb,weak",
         "--datasets", "breast_cancer", "--smoke", "--device", "cpu", "--keep-models",
         "--out", str(root)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(baseline.read_text()) == {"marker": "keep benchmark data"}
    assert not (root / "invocation.json").exists()
    smoke = next((root / "smoke").iterdir())
    records = list((smoke / "breast_cancer").glob("*_ss*_ms*.json"))
    assert len(records) == 2
    for path in records:
        record = json.loads(path.read_text())
        assert record["run_type"] == "smoke"
        assert record["n_train"] <= 128 and record["n_test"] <= 32
        assert record["model_retention"] == "retained"
        model = joblib.load(smoke / record["model_path"] / "model.joblib")
        X, _, _ = load("breast_cancer")
        idx = np.load(smoke / "breast_cancer" / "idx_test_ss0.npy")
        np.testing.assert_allclose(model.predict(X.iloc[idx]), np.load(path.with_suffix(".pred.npy")))
    text = (smoke / "report" / "report.md").read_text(encoding="utf-8")
    assert "SMOKE TEST ONLY" in text
    assert "Interpretation and recommended use" not in text and "Swap check:" not in text
    assert "### Interpretation" in text and "bounded setup check" in text
    assert "not a model-selection" in text and "No swap approval" in text
    assert "Model participation" in text
    archive = next((smoke / "runs").glob("*/*/report/report.md"))
    assert not (archive.parent.parent / "artifacts").exists()
    for path in (smoke / "report" / "report.md", archive):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            assert (path.parent / target).exists(), (path, target)


def test_retention_and_invocation_survive_force_and_history(settings, monkeypatch):
    cfg = replace(Config.from_env(), keep_models=True)
    monkeypatch.setenv("QUIBBLE_KEEP_MODELS", "1")
    sweep(["histgb"], ["wine"], cfg, False)
    path = stem(cfg.out_dir, "wine", "histgb", 0, 0).with_suffix(".json")
    first = json.loads(path.read_text())
    model_path = Path(cfg.out_dir) / first["model_path"] / "model.joblib"
    original = model_path.read_bytes()
    sweep(["histgb"], ["wine"], cfg, False)
    manifest = json.loads((Path(cfg.out_dir) / "invocation.json").read_text())
    assert manifest["cells"][0]["execution"] == "cached"
    sweep(["histgb"], ["wine"], cfg, True)
    assert json.loads(path.read_text())["model_path"] != first["model_path"]
    assert model_path.read_bytes() == original
    archives = list((Path(cfg.out_dir) / "runs").glob("*/*/invocation.json"))
    assert archives and all(not (p.parent / "artifacts").exists() for p in archives)


@pytest.mark.parametrize("backend", ["histgb", "lightgbm", "catboost"])
@pytest.mark.parametrize("dataset", ["wine", "diabetes_reg"])
def test_retained_tree_models_reload_for_both_tasks(settings, backend, dataset):
    cfg = replace(Config.from_env(), keep_models=True)
    run_one(backend, dataset, 0, cfg)
    path = stem(cfg.out_dir, dataset, backend, 0, 0).with_suffix(".json")
    record = json.loads(path.read_text())
    assert "error" not in record, record
    out = Path(cfg.out_dir)
    adapter = joblib.load(out / record["model_path"] / "model.joblib")
    X, _, _ = load(dataset)
    idx = np.load(out / dataset / "idx_test_ss0.npy")
    np.testing.assert_allclose(adapter.predict(X.iloc[idx]), np.load(path.with_suffix(".pred.npy")))


def test_autogluon_keeps_models_and_validation_leaderboard(settings):
    b = AutoGluon()
    source = settings / "training"
    source.mkdir()
    (source / "predictor.pkl").write_bytes(b"trained model")
    b.path = str(source)
    b.p = Mock()
    b.p.leaderboard.return_value = pd.DataFrame({"model": ["GBM"], "score_val": [0.9]})
    csv = settings / "analysis" / "leaderboard.csv"
    assert b.diagnostics(csv)
    b.p.leaderboard.assert_called_once_with(silent=True)
    retained = settings / "retained"
    b.save(retained)
    b.close()
    assert not source.exists()
    assert (retained / "predictor.pkl").read_bytes() == b"trained model"
    assert pd.read_csv(csv).score_val.iloc[0] == 0.9


def test_participation_distinguishes_execution_outcome_and_use(settings):
    def record(name, **extra):
        return dict(backend=name, dataset="wine", split_seed=0, model_seed=0, **extra)
    runs = [record("histgb"), record("lightgbm"), record("tabfm", error="missing module"),
            record("tabpfn", refused="too many rows")]
    cells = [dict(backend=r["backend"], dataset="wine", split_seed=0, model_seed=0,
                  execution="incompatible settings" if r["backend"] == "lightgbm" else
                  ("cached" if r["backend"] == "histgb" else "ran"),
                  differences=["cv"] if r["backend"] == "lightgbm" else [])
             for r in runs]
    inv = dict(backends=[r["backend"] for r in runs], datasets=["wine"], cells=cells)
    res = dict(backends=["histgb"], splits=[0], model_seed=0)
    table = report.participation(settings, ["wine"], {"wine": runs}, {"wine": res}, inv)
    rows = table.set_index("backend")
    assert rows.loc["histgb", "execution"] == "cached"
    assert rows.loc["histgb", "used in comparison"] == 1
    assert rows.loc["lightgbm", "reason"] == "cv"
    assert rows.loc["lightgbm", "used in comparison"] == 0
    assert rows.loc["tabfm", "errors"] == 1 and rows.loc["tabpfn", "refused"] == 1
    assert rows.loc["kumo_tabular", "selected"] == "no"


def test_shared_rows_are_not_overwritten_by_incompatible_cv(settings):
    cfg = Config.from_env()
    run_one("histgb", "wine", 0, cfg)
    idx = Path(cfg.out_dir) / "wine" / "idx_test_ss0.npy"
    original = idx.read_bytes()
    run_one("lightgbm", "wine", 0, replace(cfg, cv="3x1"))
    assert idx.read_bytes() == original
    record = json.loads(stem(cfg.out_dir, "wine", "lightgbm", 0, 0).with_suffix(".json").read_text())
    assert "Existing test rows differ" in record["error"]


def test_full_benchmark_cache_and_incompatible_settings_report(settings):
    root = settings / "results"
    args = [sys.executable, "-m", "quibble", "benchmark", "--backends", "histgb,weak",
            "--datasets", "wine", "--cv", "holdout", "--device", "cpu", "--out", str(root)]
    first = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    before = {p.name: p.read_bytes() for p in (root / "wine").glob("*.npy")}
    cached = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert cached.returncode == 0 and "0 to run, 2 already done" in cached.stdout
    incompatible = subprocess.run(args + ["--keep-models"], capture_output=True, text=True, timeout=120)
    assert incompatible.returncode == 1, incompatible.stdout + incompatible.stderr
    text = (root / "report" / "report.md").read_text(encoding="utf-8")
    assert "incompatible settings" in text and "model_retention" in text
    assert "No requested backend has settings compatible" in text
    assert "Interpretation and recommended use" not in text and "Swap check:" not in text
    for name, content in before.items():
        assert (root / "wine" / name).read_bytes() == content


def test_unsupported_foundation_retention_is_explicit(settings, monkeypatch, capsys):
    from quibble.backends import BACKENDS, HistGB

    class FakeTabFM(HistGB):
        can_save = False

    monkeypatch.setitem(BACKENDS, "tabfm", FakeTabFM)
    cfg = replace(Config.from_env(), keep_models=True)
    run_one("tabfm", "wine", 0, cfg)
    record = json.loads(stem(cfg.out_dir, "wine", "tabfm", 0, 0).with_suffix(".json").read_text())
    assert record["model_retention"] == "unsupported" and "error" not in record
    assert "not supported for tabfm" in capsys.readouterr().err


def test_artifact_io_error_is_not_reported_as_success(settings, monkeypatch, capsys):
    from quibble.backends import HistGB

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(HistGB, "save", fail)
    cfg = replace(Config.from_env(), keep_models=True)
    run_one("histgb", "wine", 0, cfg)
    record = json.loads(stem(cfg.out_dir, "wine", "histgb", 0, 0).with_suffix(".json").read_text())
    assert record["model_retention"] == "failed"
    assert record["error"] == "OSError: disk full"
    output = capsys.readouterr().out
    assert "ERROR OSError: disk full" in output and "accuracy=" not in output
