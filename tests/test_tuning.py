"""Training-only search, split safety, immutable recipes and worker contracts."""
import inspect
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quibble_ml import backends, benchmark, datasets, run, tuning
from quibble_ml.backends import Backend, HistGB, Prep
from quibble_ml.config import Config
from quibble_ml.provenance import check_reuse, recipe_snapshot, validate_recipe


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Keep environment, downloads and outputs local to each test."""
    for key, value in {
        "QUIBBLE_TIER": "defaults", "QUIBBLE_TUNE_TRIALS": "20", "QUIBBLE_TUNE_FOLDS": "3",
        "QUIBBLE_TUNE_SEED": "0", "QUIBBLE_TUNE_METRIC": "auto", "QUIBBLE_TUNE_TIMEOUT_S": "0",
        "QUIBBLE_CACHE_DIR": str(tmp_path / "cache"), "QUIBBLE_OUT_DIR": str(tmp_path / "out"),
        "QUIBBLE_CV": "holdout", "QUIBBLE_MODEL_SEEDS": "0", "QUIBBLE_TIMEOUT_S": "300",
        "_QUIBBLE_WORKFLOW": "run", "_QUIBBLE_RUN_TYPE": "benchmark",
        "QUIBBLE_MITRA_ALLOW_DOWNLOAD": "0",
    }.items():
        monkeypatch.setenv(key, value)
    return replace(Config.from_env(), tier="tuned")


@pytest.fixture
def training():
    return pd.DataFrame({"value": np.arange(120)}), pd.Series(np.tile([0, 1], 60))


class CheapTree(Backend):
    """No optional models or downloads: constant predictions expose deterministic ties."""
    fits = []

    def fit(self, X, y, task, seed, cfg, device):
        assert len(X) == len(y)
        self.fits.append((X.index.tolist(), y.index.tolist(), dict(self.tuning_params)))
        self.task = task
        self.info = {"preprocess_effective": "ordinal", "device_effective": "cpu"}

    def predict(self, X):
        return (np.full((len(X), 2), 0.5) if self.task == "classification"
                else np.zeros(len(X)))


@pytest.mark.parametrize("backend", tuning.TREES)
def test_given_default_budget_when_search_then_each_tree_gets_sixty_inner_fits(
        monkeypatch, cfg, training, backend):
    # Arrange
    monkeypatch.setitem(backends.BACKENDS, backend, CheapTree)
    monkeypatch.setattr(CheapTree, "fits", [])
    X, y = training

    # Act
    winner = tuning.fit_tuned(backend, X, y, "classification", 7, cfg, "cpu")
    evidence = winner.info["tuning"]

    # Assert
    assert len(evidence["attempts"]) == 20
    assert all(len(attempt["folds"]) == 3 for attempt in evidence["attempts"])
    assert evidence["attempted_fits"] == evidence["successful_fits"] == 61
    assert evidence["selected_trial"] == 0
    assert len(CheapTree.fits[-1][0]) == len(X)
    assert len({json.dumps(a["params"], sort_keys=True) for a in evidence["attempts"]}) == 20
    assert evidence["total_s"] >= evidence["search_s"]
    assert evidence["refit"]["status"] == "ok"


def test_given_same_seed_when_search_repeated_then_selection_and_split_ids_match(
        monkeypatch, cfg, training):
    # Arrange
    monkeypatch.setitem(backends.BACKENDS, "histgb", CheapTree)
    X, y = training
    cfg = replace(cfg, tune_trials=4)

    # Act
    a = tuning.fit_tuned("histgb", X, y, "classification", 3, cfg, "cpu").info["tuning"]
    b = tuning.fit_tuned("histgb", X, y, "classification", 3, cfg, "cpu").info["tuning"]

    # Assert
    assert a["selected_params"] == b["selected_params"]
    assert a["inner_splits"] == b["inner_splits"]
    assert [t["params"] for t in a["attempts"]] == [t["params"] for t in b["attempts"]]


def test_given_grouped_training_when_inner_split_then_entities_never_overlap(cfg, training):
    # Arrange
    X, y = training
    X.attrs["quibble_split"] = {"kind": "group"}
    X.attrs["quibble_groups"] = np.repeat(np.arange(20), 6)
    Xtr, _, ytr, _, _ = datasets.split(X, y, 0, "classification")

    # Act
    pairs = tuning.inner_splits(Xtr, ytr, "classification", 3, 0)

    # Assert
    assert len(Xtr.attrs["quibble_groups"]) == len(Xtr)
    for tr, va in pairs:
        groups = Xtr.attrs["quibble_groups"]
        assert not set(groups[tr]) & set(groups[va])
        assert set(ytr.iloc[tr]) == set(ytr.iloc[va]) == {0, 1}


def test_given_time_group_and_purge_when_split_then_both_levels_preserve_constraints(training):
    # Arrange
    X, y = training
    X.attrs["quibble_split"] = {"kind": "temporal", "purge_gap": 3}
    X.attrs["quibble_times"] = np.arange(len(X))
    X.attrs["quibble_groups"] = np.repeat(np.arange(20), 6)
    Xtr, Xte, ytr, _, _ = datasets.split(X, y, 0, "classification")

    # Act
    pairs = tuning.inner_splits(Xtr, ytr, "classification", 3, 0)

    # Assert
    assert max(Xtr.index) + 3 < min(Xte.index)
    assert not set(Xtr.attrs["quibble_groups"]) & set(Xte.attrs["quibble_groups"])
    for tr, va in pairs:
        times, groups = Xtr.attrs["quibble_times"], Xtr.attrs["quibble_groups"]
        assert times[tr].max() + 3 < times[va].min()
        assert not set(groups[tr]) & set(groups[va])


def test_given_tied_times_when_inner_split_then_no_tie_crosses_boundary(training):
    # Arrange
    X, y = training
    X.attrs["quibble_split"] = {"kind": "temporal"}
    X.attrs["quibble_times"] = np.repeat(np.arange(12), 10)

    # Act
    pairs = tuning.inner_splits(X, y, "classification", 3, 0)

    # Assert
    for tr, va in pairs:
        assert X.attrs["quibble_times"][tr].max() < X.attrs["quibble_times"][va].min()


@pytest.mark.parametrize("kind", ["group", "temporal", "random"])
def test_given_missing_classes_when_inner_split_then_refuse_without_random_fallback(training, kind):
    # Arrange
    X, _ = training
    y = pd.Series([0] * 119 + [1])
    X.attrs["quibble_split"] = {"kind": kind}
    if kind == "group":
        X.attrs["quibble_groups"] = np.repeat(np.arange(20), 6)
    if kind == "temporal":
        X.attrs["quibble_times"] = np.arange(120)

    # Act & Assert
    with pytest.raises(ValueError, match="class"):
        tuning.inner_splits(X, y, "classification", 3, 0)


def test_given_stale_structural_attrs_when_search_then_fail_explicitly(training):
    # Arrange
    X, y = training
    X.attrs["quibble_split"] = {"kind": "group"}
    X.attrs["quibble_groups"] = np.arange(121)

    # Act & Assert
    with pytest.raises(ValueError, match="align"):
        tuning.inner_splits(X, y, "classification", 3, 0)


def test_given_relational_attrs_when_inner_search_then_never_flatten_to_random(training):
    # Arrange
    X, y = training
    X.attrs["quibble_relational"] = {"official_split": True}

    # Act & Assert
    with pytest.raises(ValueError, match="Relational inner tuning"):
        tuning.inner_splits(X, y, "classification", 3, 0)


def test_given_impossible_time_order_when_inner_search_then_fail_without_fallback(training):
    # Arrange
    X, y = training
    X.attrs["quibble_split"] = {"kind": "temporal"}
    X.attrs["quibble_times"] = np.ones(len(X))

    # Act & Assert
    with pytest.raises(ValueError, match="empty"):
        tuning.inner_splits(X, y, "classification", 3, 0)


def test_given_regression_when_search_then_freeze_rmse_metric(monkeypatch, cfg, training):
    # Arrange
    monkeypatch.setitem(backends.BACKENDS, "histgb", CheapTree)
    X, y = training

    # Act
    winner = tuning.fit_tuned("histgb", X, y.astype(float), "regression", 0,
                              replace(cfg, tune_trials=1, tune_folds=2), "cpu")

    # Assert
    assert winner.info["tuning"]["metric"] == "rmse"
    assert winner.info["tuning"]["attempted_fits"] == 3


def test_given_wrong_task_metric_when_search_then_fail_before_any_fit(cfg, training):
    # Act & Assert
    with pytest.raises(tuning.TuningFailure, match="incompatible") as failure:
        tuning.fit_tuned("histgb", *training, "classification", 0,
                         replace(cfg, tune_metric="rmse"), "cpu")
    assert failure.value.evidence["attempted_fits"] == 0


def test_given_all_failed_candidates_when_search_then_no_refit_or_default_fallback(
        monkeypatch, cfg, training):
    # Arrange
    class Broken(CheapTree):
        def fit(self, *args):
            raise RuntimeError("deliberate candidate failure")
    monkeypatch.setitem(backends.BACKENDS, "histgb", Broken)
    X, y = training
    cfg = replace(cfg, tune_trials=2, tune_folds=2)

    # Act & Assert
    with pytest.raises(tuning.TuningFailure, match="No valid") as failure:
        tuning.fit_tuned("histgb", X, y, "classification", 0, cfg, "cpu")
    evidence = failure.value.evidence
    assert evidence["attempted_fits"] == 4
    assert evidence["selected_trial"] is None and "refit" not in evidence
    assert all(f["status"] == "failed" and f["failure"]
               for a in evidence["attempts"] for f in a["folds"])


def test_given_nonfinite_candidates_when_search_then_explicit_failure(monkeypatch, cfg, training):
    # Arrange
    class Nonfinite(CheapTree):
        def predict(self, X):
            return np.full((len(X), 2), np.nan)
    monkeypatch.setitem(backends.BACKENDS, "histgb", Nonfinite)

    # Act & Assert
    with pytest.raises(tuning.TuningFailure, match="No valid"):
        tuning.fit_tuned("histgb", *training, "classification", 0,
                         replace(cfg, tune_trials=1), "cpu")


def test_given_deadline_overrun_when_fit_returns_then_worker_result_is_failed(
        monkeypatch, cfg, training):
    # Arrange
    clock = [0.0]
    monkeypatch.setattr(tuning.time, "perf_counter", lambda: clock[0])

    class Slow(CheapTree):
        def fit(self, *args):
            super().fit(*args)
            clock[0] += 2
    monkeypatch.setitem(backends.BACKENDS, "histgb", Slow)

    # Act & Assert
    with pytest.raises(tuning.TuningFailure, match="Cooperative") as failure:
        tuning.fit_tuned("histgb", *training, "classification", 0,
                         replace(cfg, tune_timeout_s=1), "cpu")
    assert failure.value.evidence["deadline_overrun"]
    assert failure.value.evidence["status"] == "failed"
    assert failure.value.evidence["attempted_fits"] == 1


def test_given_sentinel_outer_target_when_run_then_search_cannot_observe_it(
        monkeypatch, cfg, training):
    # Arrange
    X, y = training
    searching = [False]

    class Sentinel(pd.Series):
        def to_numpy(self, *args, **kwargs):
            assert not searching[0], "Outer targets accessed during inner selection/refit"
            return super().to_numpy(*args, **kwargs)

    def outer_split(*args):
        return X.iloc[:90], X.iloc[90:], y.iloc[:90], Sentinel(y.iloc[90:]), np.arange(90, 120)

    real_search = tuning.fit_tuned

    def search(*args):
        searching[0] = True
        try:
            return real_search(*args)
        finally:
            searching[0] = False

    class SentinelTree(HistGB):
        def fit(self, X, y, *args):
            assert X.index.max() < 90 and y.index.max() < 90
            return super().fit(X, y, *args)
    monkeypatch.setitem(backends.BACKENDS, "histgb", SentinelTree)
    monkeypatch.setattr(datasets, "load", lambda *a: (X, y, "classification"))
    monkeypatch.setattr(datasets, "split", outer_split)
    monkeypatch.setattr(tuning, "fit_tuned", search)
    cfg = replace(cfg, tune_trials=1, tune_folds=2)

    # Act
    run.run_one("histgb", "sentinel", 0, cfg)
    saved = json.loads(run.stem(cfg.out_dir, "sentinel", "histgb", 0, 0).with_suffix(".json").read_text())

    # Assert
    assert "error" not in saved and "refused" not in saved
    assert saved["tuning"]["status"] == "ok"
    assert saved["tuning"]["refit"]["train_rows"] == 90
    assert saved["search_refit_s"] >= saved["refit_s"]
    assert saved["recipe"]["resolved"]["parameters"]["early_stopping"] is False
    validate_recipe(saved["recipe"])
    assert not any("test" in name for name in inspect.signature(real_search).parameters)


def test_given_fold_only_category_when_preprocessing_then_validation_is_unseen(
        monkeypatch, cfg, training):
    # Arrange
    X, y = training
    X["cat"] = [f"unique-{i}" for i in range(len(X))]
    seen = []
    original = Prep.fit

    def fit(self, frame):
        seen.append(set(frame["cat"]))
        return original(self, frame)
    monkeypatch.setattr(Prep, "fit", fit)

    # Act
    winner = tuning.fit_tuned("histgb", X, y, "classification", 0,
                              replace(cfg, tune_trials=1, tune_folds=2), "cpu")

    # Assert
    assert len(seen) == 3 and seen[-1] == set(X["cat"])
    assert not seen[0] & seen[1]
    assert winner.prep.cats["cat"].size == len(X)


@pytest.mark.parametrize("change", [
    {"tier": "defaults"}, {"tune_trials": 2}, {"tune_folds": 2}, {"tune_seed": 1},
    {"tune_metric": "rmse"}, {"tune_timeout_s": 1},
])
def test_given_changed_protocol_when_reuse_then_separate_root_required(cfg, change):
    # Arrange
    original = recipe_snapshot("histgb", cfg, 0)
    changed = recipe_snapshot("histgb", replace(cfg, **change), 0)
    tuning.ensure_output_tier(cfg)

    # Act & Assert
    with pytest.raises(ValueError, match="Recipe mismatch"):
        check_reuse({"recipe": original}, changed, writing=True)
    with pytest.raises(ValueError, match="separate"):
        tuning.ensure_output_tier(replace(cfg, **change))


def test_given_changed_domain_when_reuse_then_hash_changes(monkeypatch, cfg):
    # Arrange
    before = recipe_snapshot("histgb", cfg, 0)
    domains = deepcopy(tuning.DOMAINS)
    domains["histgb"]["learning_rate"] = [0.2]
    monkeypatch.setattr(tuning, "DOMAINS", domains)

    # Act
    after = recipe_snapshot("histgb", cfg, 0)

    # Assert
    assert before["algorithm_sha256"] != after["algorithm_sha256"]


def test_given_defaults_when_tuning_knobs_change_then_recipe_remains_compatible(cfg):
    # Arrange
    cfg = replace(cfg, tier="defaults")

    # Act
    before = recipe_snapshot("histgb", cfg, 0)
    after = recipe_snapshot("histgb", replace(cfg, tune_trials=1, tune_seed=19), 0)

    # Assert
    assert before == after
    assert before["requested"]["search"] == {"status": "not requested"}
    assert HistGB().make("classification", 0, cfg, "cpu", []).get_params()["early_stopping"] == "auto"


def test_given_default_run_when_search_is_forbidden_then_output_matches_direct_default(
        monkeypatch, cfg):
    # Arrange
    def forbidden(*args):
        raise AssertionError("Defaults must never enter search")
    monkeypatch.setattr(tuning, "fit_tuned", forbidden)
    cfg = replace(cfg, tier="defaults")
    X, y, task = datasets.load("wine", cfg.data_home)
    Xtr, Xte, ytr, _, _ = datasets.split(X, y, cfg.split_seed, task)
    direct = HistGB()
    direct.fit(Xtr, ytr, task, 0, cfg, "cpu")

    # Act
    run.run_one("histgb", "wine", 0, cfg)
    path = run.stem(cfg.out_dir, "wine", "histgb", 0, 0)

    # Assert
    np.testing.assert_array_equal(np.load(f"{path}.pred.npy"), direct.predict(Xte))
    assert json.loads(path.with_suffix(".json").read_text())["tier"] == "defaults"


def test_given_separate_roots_when_pairing_tiers_then_require_identical_outer_evidence(cfg, tmp_path):
    # Arrange
    default_cfg = replace(cfg, tier="defaults", out_dir=str(tmp_path / "defaults"))
    tuned_cfg = replace(cfg, out_dir=str(tmp_path / "tuned"), tune_trials=1, tune_folds=2)
    run.run_one("histgb", "wine", 0, default_cfg)
    run.run_one("histgb", "wine", 0, tuned_cfg)

    # Act
    pair = tuning.load_tier_pair(default_cfg.out_dir, tuned_cfg.out_dir, "wine", "histgb", 0, 0)

    # Assert
    assert pair["defaults"]["record"]["tier"] == "defaults"
    assert pair["tuned"]["record"]["tier"] == "tuned"
    assert pair["defaults"]["predictions"].shape == pair["tuned"]["predictions"].shape
    assert len(pair["idx_test"]) == len(pair["y_test"])
    targets = run.stem(tuned_cfg.out_dir, "wine", "histgb", 0, 0).parent / "y_test_ss0.npy"
    np.save(targets, np.zeros(len(pair["y_test"])))
    with pytest.raises(ValueError, match="test targets"):
        tuning.load_tier_pair(default_cfg.out_dir, tuned_cfg.out_dir, "wine", "histgb", 0, 0)


def test_given_unsupported_family_when_tuned_then_refuse_and_foundations_remain_labeled(cfg):
    # Act
    run.run_one("weak", "wine", 0, cfg)
    saved = json.loads(run.stem(cfg.out_dir, "wine", "weak", 0, 0).with_suffix(".json").read_text())
    foundation = recipe_snapshot("tabicl_v2", cfg, 0)["requested"]

    # Assert
    assert "does not support" in saved["refused"]
    assert foundation["tier"] == "defaults" and foundation["campaign_tier"] == "tuned"
    assert foundation["parameters"]["fine_tuning"] is False
    assert foundation["search"]["status"] == "not supported"


def test_given_worker_timeout_when_sweep_then_no_partial_success_and_config_is_forwarded(
        monkeypatch, cfg):
    # Arrange
    captured = {}

    def timeout(cmd, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(run, "_worker_process", timeout)
    cfg = replace(cfg, tune_trials=2, tune_folds=2, tune_seed=12, timeout_s=5)

    # Act
    invocation = run.sweep(["histgb"], ["wine"], cfg, False)
    saved = json.loads(run.stem(cfg.out_dir, "wine", "histgb", 0, 0).with_suffix(".json").read_text())

    # Assert
    assert captured["timeout"] == 5
    assert captured["env"]["QUIBBLE_TIER"] == "tuned"
    assert captured["env"]["QUIBBLE_TUNE_TRIALS"] == "2"
    assert captured["env"]["QUIBBLE_TUNE_SEED"] == "12"
    assert saved["error_kind"] == "timeout" and saved["tuning"]["status"] == "incomplete"
    assert invocation["tuning_protocol"]["inner_folds"] == 2
    assert not run.stem(cfg.out_dir, "wine", "histgb", 0, 0).with_suffix(".pred.npy").exists()


@pytest.mark.parametrize("key,value", [
    ("QUIBBLE_TIER", "other"), ("QUIBBLE_TUNE_TRIALS", "0"), ("QUIBBLE_TUNE_FOLDS", "1"),
    ("QUIBBLE_TUNE_TIMEOUT_S", "nan"), ("QUIBBLE_TUNE_METRIC", "accuracy"),
])
def test_given_invalid_settings_when_configured_then_fail_before_search(monkeypatch, cfg, key, value):
    # Arrange
    monkeypatch.setenv(key, value)

    # Act & Assert
    with pytest.raises(ValueError):
        Config.from_env()


def test_given_tuned_benchmark_flags_when_parsed_then_explicit_config_reaches_sweep(
        monkeypatch, cfg, tmp_path):
    # Arrange
    calls = []
    from quibble_ml import report
    monkeypatch.setattr(benchmark, "sweep",
                        lambda *a, **k: (calls.append(a[2]) or {"cells": []}))
    monkeypatch.setattr(report, "build", lambda *a, **k: tmp_path / "report.md")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--preset", "trees", "--tier", "tuned",
                                    "--tune-trials", "2", "--tune-folds", "2",
                                    "--tune-seed", "13", "--timeout", "11"])

    # Act
    result = benchmark.main()

    # Assert
    assert result == 0
    assert (calls[0].tier, calls[0].tune_trials, calls[0].tune_folds,
            calls[0].tune_seed, calls[0].timeout_s) == ("tuned", 2, 2, 13, 11)


def test_given_single_tuned_cli_when_called_then_supervised_worker_enforces_timeout(
        monkeypatch, cfg):
    # Arrange
    calls = []
    monkeypatch.setenv("QUIBBLE_TIER", "tuned")
    monkeypatch.delenv("_QUIBBLE_WORKER", raising=False)
    monkeypatch.setattr(run, "sweep", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["run", "--backend", "histgb", "--dataset", "wine",
                                    "--tier", "tuned", "--model-seed", "7"])

    # Act
    run.main()

    # Assert
    assert calls[0][0][2].model_seeds == (7,)
    assert calls[0][1]["single_split"] is True


def test_given_prediction_adaptation_when_run_then_measure_once_without_latency_probes(
        monkeypatch, cfg):
    # Arrange
    calls = []

    class Adapting(Backend):
        def fit(self, X, y, task, seed, cfg, device):
            self.prediction_adapts = True
            self.info = {"prediction_adapts": True, "adaptation_s": None,
                         "prediction_only_s": None, "weight_load_s": None}

        def predict(self, X):
            calls.append(len(X))
            return np.zeros(len(X))
    monkeypatch.setitem(backends.BACKENDS, "adapting", Adapting)
    cfg = replace(cfg, tier="defaults")

    # Act
    run.run_one("adapting", "diabetes_reg", 0, cfg)
    path = run.stem(cfg.out_dir, "diabetes_reg", "adapting", 0, 0)
    saved = json.loads(path.with_suffix(".json").read_text())

    # Assert
    assert "error" not in saved and "refused" not in saved
    assert calls == [saved["n_test"]]
    assert saved["adaptation_and_prediction_s"] >= 0
    assert saved["single_row_p50_ms"] is None and saved["single_row_p95_ms"] is None
    assert saved["single_row_latency_available"] is False
    assert saved["adaptation_s"] is None and saved["weight_load_s"] is None
    assert saved["prediction_only_s"] is None
    assert saved["predict_ms_per_row_batch"] == pytest.approx(
        saved["adaptation_and_prediction_s"] / saved["n_test"] * 1000)


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_given_mitra_download_consent_when_configured_then_require_explicit_one(
        monkeypatch, cfg, value, expected):
    # Arrange
    monkeypatch.setenv("QUIBBLE_MITRA_ALLOW_DOWNLOAD", value)

    # Act & Assert
    assert Config.from_env().mitra_allow_download is expected


@pytest.mark.parametrize("value", ["yes", "true", "2"])
def test_given_invalid_mitra_consent_when_configured_then_refuse(monkeypatch, cfg, value):
    # Arrange
    monkeypatch.setenv("QUIBBLE_MITRA_ALLOW_DOWNLOAD", value)

    # Act & Assert
    with pytest.raises(ValueError, match="QUIBBLE_MITRA_ALLOW_DOWNLOAD must be 0 or 1"):
        Config.from_env()
