"""Regression tests for the review fixes: statistics, reuse checks, splits, flags, baselines."""
import json
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from quibble_ml.backends import BACKENDS
from quibble_ml.config import Config
from quibble_ml.datasets import load, split
from quibble_ml.metrics import Boot, corrected_interval, widen
from quibble_ml.run import is_failed, package_changes, run_one, save_npy, settings_differ, stem
from quibble_ml.swap import SwapError, evaluate, tri


# ---------------------------------------------------------------- statistics
def test_group_bootstrap_is_wider_than_row_bootstrap_for_correlated_rows():
    rng = np.random.default_rng(0)
    n_groups, per = 40, 10
    group_effect = rng.normal(size=n_groups)
    correct = (group_effect[:, None] + 0.2 * rng.normal(size=(n_groups, per)) > 0).astype(float).ravel()
    rows = np.arange(len(correct))
    groups = np.repeat(np.arange(n_groups), per)
    mean = lambda w: float(np.average(correct, weights=w))  # noqa: E731
    _, lo_r, hi_r = Boot(rows, 400).ci(mean)
    _, lo_g, hi_g = Boot(rows, 400, cluster=groups).ci(mean)
    assert hi_g - lo_g > 1.5 * (hi_r - lo_r)


def test_bootstrap_counts_dropped_resamples_and_withholds_a_thin_interval():
    boot = Boot(np.arange(30), 100)
    pt, lo, hi = boot.ci(lambda w: np.nan if w is not None else 1.0)
    assert np.isnan(lo) and np.isnan(hi) and boot.valid == 0 and boot.dropped == 100


def test_corrected_interval_is_wider_than_naive_and_needs_five_splits():
    vals = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93]
    m, lo, hi = corrected_interval(vals, n_train=80, n_test=40)
    naive = 2.571 * np.std(vals, ddof=1) / np.sqrt(len(vals))
    assert hi - m > naive and m - lo > naive
    assert np.isnan(corrected_interval(vals[:4], 80, 40)[1])


def test_widen_takes_the_outer_bounds_and_ignores_nan():
    assert widen((1, 0.5, 1.5), (1, 0.2, 1.2)) == (1, 0.2, 1.5)
    assert widen((1, np.nan, np.nan), (1, 0.2, 1.2)) == (1, 0.2, 1.2)


def test_missing_interval_is_inconclusive_never_pass():
    assert tri(0.0, np.nan, np.nan, 1.0) == "INCONCLUSIVE"
    assert tri(0.0, -1, 0.5, 1.0) == "PASS" and tri(2.0, 1.5, 3, 1.0) == "FAIL"


def test_swap_refuses_implicit_thresholds_when_required(cv_results):
    with pytest.raises(SwapError, match="non-empty policy"):
        evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results, require_thresholds=True)
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results)
    assert r["thresholds_default"] is True
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results, {"max_quality_drop": 0.05},
                 require_thresholds=True)
    assert r["thresholds_default"] is False


def test_swap_cli_prints_the_default_policy_and_honours_require_flag(cv_results):
    base = [sys.executable, "-m", "quibble_ml", "swap", "--incumbent", "histgb", "--candidate", "lightgbm",
            "--dataset", "breast_cancer", "--out", str(cv_results)]
    out = subprocess.run(base, capture_output=True, text=True).stdout
    assert "THRESHOLDS: packaged defaults (quibble_ml/swap_thresholds.json)" in out
    assert '"max_quality_drop": 0.01' in out
    refused = subprocess.run(base + ["--require-thresholds"], capture_output=True, text=True)
    assert refused.returncode == 1 and "non-empty policy" in refused.stderr


def test_worst_slice_check_covers_the_maximum_over_slices(cv_results):
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results)
    slice_check = [c for c in r["checks"] if c["check"].startswith("worst slice")]
    assert slice_check and "maximum over all slices" in slice_check[0]["note"]


def test_three_splits_say_that_retraining_variability_is_not_counted(cv_results):
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results)
    q = next(c for c in r["checks"] if c["check"].startswith("quality"))
    assert "retraining variability not counted" in q["note"]


def test_reseed_noise_floor_is_used_when_the_incumbent_has_two_seeds(cv_results, tmp_path, monkeypatch):
    out = tmp_path / "seeds"
    shutil.copytree(cv_results, out)
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    monkeypatch.setenv("QUIBBLE_CV", "3x1")
    for ss in range(3):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(ss))
        run_one("histgb", "breast_cancer", 1, Config.from_env())
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, out)
    d = next(c for c in r["checks"] if c["check"].startswith("disagreement rate"))
    assert "above reseed noise" in d["check"] and "disagrees with itself" in d["note"]


# ---------------------------------------------------------------- run records and reuse
def cfg_for(monkeypatch, out, **env):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Config.from_env()


def test_run_record_separates_loading_from_fitting_and_measures_memory_increase(monkeypatch, tmp_path):
    run_one("histgb", "wine", 0, cfg_for(monkeypatch, tmp_path), force=True)
    r = json.loads(stem(tmp_path, "wine", "histgb", 0, 0).with_suffix(".json").read_text())
    assert r["fit_total_s"] >= r["fit_s"] >= 0 and r["load_s"] >= 0
    assert r["seed_controlled"] is True and r["single_n"] == 3
    if "peak_rss_increase_mb" in r:
        assert 0 <= r["peak_rss_increase_mb"] <= r["peak_rss_mb"]


def test_autogluon_is_marked_as_not_seed_controlled_and_every_backend_has_a_label():
    assert BACKENDS["autogluon"].seed_controlled is False
    assert all(b.verification for b in BACKENDS.values())
    assert BACKENDS["tabpfn"].verification.startswith("experimental")


def test_reuse_check_sees_autogluon_budget_tabpfn_limit_and_library_versions(monkeypatch, tmp_path):
    record = tmp_path / "r.json"
    record.write_text(json.dumps({"backend": "autogluon", "ag_time_limit": 120, "ag_preset": "medium_quality"}))
    assert settings_differ(record, cfg_for(monkeypatch, tmp_path, QUIBBLE_AG_TIME_LIMIT="60")) == ["ag_time_limit"]
    record.write_text(json.dumps({"backend": "tabpfn", "tabpfn_max_rows": None}))
    differs = settings_differ(record, cfg_for(monkeypatch, tmp_path, QUIBBLE_TABPFN_MAX_ROWS="500"))
    assert "tabpfn_max_rows" in differs
    record.write_text(json.dumps({"backend": "histgb", "env": {"packages": {"numpy": "0.0.1"}}}))
    differs = settings_differ(record, cfg_for(monkeypatch, tmp_path))
    assert any(d.startswith("package_versions") and "numpy 0.0.1" in d for d in differs)
    assert package_changes({"backend": "histgb", "env": {"packages": {}}}) == []


def test_retry_failed_reruns_only_error_results(monkeypatch, tmp_path):
    cfg = cfg_for(monkeypatch, tmp_path)
    broken = type("Ghost", (BACKENDS["histgb"],), {"name": "ghost",
                                                    "make": lambda self, *a: __import__("not_a_real_package")})
    monkeypatch.setitem(BACKENDS, "ghost", broken)
    run_one("ghost", "wine", 0, cfg, force=True)
    path = stem(tmp_path, "wine", "ghost", 0, 0).with_suffix(".json")
    assert is_failed(path)
    monkeypatch.setitem(BACKENDS, "ghost", type("Ghost", (BACKENDS["histgb"],), {"name": "ghost"}))
    run_one("ghost", "wine", 0, cfg)                                  # plain rerun skips the cached error
    assert is_failed(path)
    run_one("ghost", "wine", 0, cfg, retry_failed=True)               # retry reruns it
    assert not is_failed(path) and "accuracy" in json.loads(path.read_text())


def test_saved_split_files_are_written_atomically(tmp_path):
    target = tmp_path / "idx_test_ss0.npy"
    save_npy(target, np.arange(5))
    assert np.array_equal(np.load(target), np.arange(5))
    assert [p.name for p in tmp_path.iterdir()] == ["idx_test_ss0.npy"]


def test_run_metrics_agree_with_scikit_learn():
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    from quibble_ml.run import metrics
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p1 = np.clip(0.6 * y + 0.2 + rng.normal(0, 0.15, 200), 0.02, 0.98)
    P = np.c_[1 - p1, p1]
    m = metrics(y, P, "classification")
    assert m["accuracy"] == pytest.approx(accuracy_score(y, P.argmax(1)))
    assert m["log_loss"] == pytest.approx(log_loss(y, P), rel=1e-6)
    assert m["auc"] == pytest.approx(roc_auc_score(y, p1))


# ---------------------------------------------------------------- backends
def test_unseen_categories_do_not_trigger_pandas_deprecation_warnings():
    from quibble_ml.backends import Prep
    train = pd.DataFrame({"c": pd.Series(["a", "b", "a", "b"], dtype=object), "x": [1.0, 2, 3, 4]})
    prep = Prep("native").fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = prep.transform(pd.DataFrame({"c": pd.Series(["a", "NEVER_SEEN"], dtype=object), "x": [1.0, 2]}))
    assert out["c"].isna().tolist() == [False, True]


@pytest.mark.parametrize("name", ["dummy", "linear"])
def test_baselines_fit_predict_and_handle_unseen_categories(name):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "c": pd.Series(rng.choice(["x", "y", None], 300), dtype=object)})
    y = pd.Series((X.a > 0).astype(int))

    class C:
        preprocess, threads, tabpfn_max_rows = "ordinal", 1, None

    b = BACKENDS[name]()
    b.fit(X[:200], y[:200], "classification", 0, C, "cpu")
    Xt = X[200:].copy()
    Xt.iloc[0, 1] = "NEVER_SEEN"
    P = b.predict(Xt)
    assert P.shape == (100, 2) and np.allclose(P.sum(1), 1)
    b2 = BACKENDS[name]()
    b2.fit(X[:200], X.a[:200], "regression", 0, C, "cpu")
    assert b2.predict(Xt).shape == (100,)
    if name == "linear":
        assert (b.predict(Xt).argmax(1) == y[200:]).mean() > 0.9
        assert BACKENDS[name].baseline


def test_doctor_prints_backend_verification_labels():
    out = subprocess.run([sys.executable, "-m", "quibble_ml", "doctor"], capture_output=True, text=True).stdout
    assert "backend verification" in out and "experimental" in out


# ---------------------------------------------------------------- slices and flags
def test_slices_follow_mutual_information_and_count_distinct_rows():
    from quibble_ml.slices import slice_masks, unique_rows
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({f"noise{i}": rng.normal(size=n) for i in range(5)})
    y = rng.integers(0, 3, n)
    X["signal"] = y * 10.0 + rng.normal(0, 1, n)  # informative, but not linearly ordered with a scrambled label
    names = slice_masks(X, y, "classification", k=1)
    assert any(k.startswith("signal=") for k in names)
    idx = np.array([0, 0, 1, 1, 2])  # rows 0 and 1 predicted twice each
    assert unique_rows(np.array([True, True, True, True, False]), idx) == 2


def test_data_flags_catch_planted_problems_and_stay_quiet_on_clean_data():
    from sklearn import datasets as d

    from quibble_ml.leakage import check
    X, y = d.load_breast_cancer(as_frame=True, return_X_y=True)
    assert check(X, y, "classification") == []
    X = X.copy()
    X["leak"] = y * 1.0 + np.random.default_rng(0).normal(0, 0.01, len(y))
    X["row_id"] = range(len(X))
    X["order_date"] = range(len(X))
    flags = {(f["flag"], f["column"]) for f in check(X, y, "classification")}
    assert ("single_feature_predicts_target", "leak") in flags
    assert ("id_like_column", "row_id") in flags
    assert ("time_like_column_random_split", "order_date") in flags
    assert not [f for f in check(X, y, "classification", "temporal") if f["flag"].startswith("time_like")]


# ---------------------------------------------------------------- group and temporal splits
@pytest.fixture
def panel(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(root))
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("QUIBBLE_CACHE_DIR", str(tmp_path / "cache"))
    directory = root / "panel"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"customer": rng.integers(0, 40, n).astype(str),
                          "issued": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
                          "amount": rng.normal(size=n)})
    frame["late"] = np.where(frame.amount + rng.normal(0, 0.5, n) > 0, "yes", "no")
    frame.to_csv(directory / "data.csv", index=False)
    spec = {"file": "data.csv", "target": "late", "task": "classification",
            "group_column": "customer"}
    (directory / "dataset.json").write_text(json.dumps(spec))
    return directory, frame, spec


def write(directory, spec):
    (directory / "dataset.json").write_text(json.dumps(spec))


def test_group_split_keeps_every_customer_on_one_side(panel):
    directory, frame, spec = panel
    X, y, task = load("panel")
    assert "customer" not in X and X.attrs["quibble_split"]["kind"] == "group"
    groups = X.attrs["quibble_groups"]
    for cv, ids in (("holdout", [0, 1]), ("4x2", range(8))):
        for sid in ids:
            Xtr, Xte, ytr, yte, te = split(X, y, sid, task, cv, 0)
            assert set(groups[te]).isdisjoint(set(np.delete(groups, te)))
            assert len(te) > 0 and len(Xtr) + len(Xte) == len(X)
    a = split(X, y, 0, task, "4x2", 0)[4]
    b = split(X, y, 4, task, "4x2", 0)[4]   # second repeat uses different folds
    assert not np.array_equal(a, b)


def test_temporal_split_always_tests_on_the_future(panel):
    directory, frame, spec = panel
    spec.update(time_column="issued")
    del spec["group_column"]
    write(directory, spec)
    X, y, task = load("panel")
    times = X.attrs["quibble_times"]
    assert X.attrs["quibble_split"]["kind"] == "temporal" and "issued" not in X
    for cv, ids in (("holdout", [0]), ("4x1", range(4))):
        for sid in ids:
            Xtr, _, _, _, te = split(X, y, sid, task, cv, 0)
            train = Xtr.index.to_numpy()   # positions of the rows the model would train on
            assert len(train) and set(train).isdisjoint(te)
            assert times[train].max() <= times[te].min()   # nothing from the future is in training
    sizes = [len(split(X, y, i, task, "4x1", 0)[0]) for i in range(4)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]   # forward chaining: the training set grows
    with pytest.raises(ValueError, match="no repeats"):
        split(X, y, 0, task, "4x2", 0)


def test_split_scheme_is_part_of_the_dataset_fingerprint(panel):
    from quibble_ml.datasets import source_info
    directory, frame, spec = panel
    grouped = source_info("panel")["fingerprint"]
    write(directory, {**spec, "split": "random"})
    assert source_info("panel")["fingerprint"] != grouped
    X, *_ = load("panel")
    assert X.attrs["quibble_split"]["kind"] == "random"


@pytest.mark.parametrize("change, message", [
    ({"split": "diagonal"}, "split must be one of"),
    ({"split": "temporal"}, "needs time_column"),
    ({"split": "group", "group_column": ""}, "nonempty column name"),
    ({"time_column": "customer"}, "must not overlap"),
    ({"time_column": "late"}, "must not overlap"),
    ({"file": "sub\\data.csv"}, "inside the dataset folder"),
])
def test_invalid_split_manifests_are_explicit_errors(panel, change, message):
    directory, frame, spec = panel
    write(directory, {**spec, **change})
    with pytest.raises(ValueError, match=message):
        load("panel")


def test_unparseable_time_column_is_an_error(panel):
    directory, frame, spec = panel
    frame.loc[3, "issued"] = "not a date"
    frame.to_csv(directory / "data.csv", index=False)
    write(directory, {**spec, "time_column": "issued"})
    with pytest.raises(ValueError, match="time_column"):
        load("panel")


def test_analysis_with_group_split_reports_the_scheme(panel, monkeypatch):
    from quibble_ml.analyze import analyse
    out = panel[0].parent.parent / "results"
    monkeypatch.setenv("QUIBBLE_CV", "4x1")
    for sid in range(4):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(sid))
        cfg = Config.from_env()
        for b in ("histgb", "dummy"):
            run_one(b, "panel", 0, cfg)
    r = analyse("panel", out, quiet=True)
    assert r["split"]["kind"] == "group"
    assert json.loads((r["adir"] / "analysis.json").read_text())["split"]["kind"] == "group"


# ---------------------------------------------------------------- baseline gate
def test_gate_passes_a_real_model_over_a_dummy_and_fails_a_dummy_over_a_model(monkeypatch, tmp_path):
    from quibble_ml.gate import evaluate as gate
    cfg = cfg_for(monkeypatch, tmp_path, QUIBBLE_CV="3x1")
    for ss in range(3):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(ss))
        cfg = Config.from_env()
        for b in ("dummy", "linear", "lightgbm"):
            run_one(b, "breast_cancer", 0, cfg)
    good = gate("breast_cancer", "lightgbm", "dummy", min_gain=0.1, out=tmp_path)
    assert good["overall"] == "PASS" and good["gain"] > 0.3
    bad = gate("breast_cancer", "dummy", "lightgbm", min_gain=0.0, out=tmp_path)
    assert bad["overall"] == "FAIL"
    # a strong linear baseline on an easy dataset: lightgbm cannot show a large gain over it
    demanding = gate("breast_cancer", "lightgbm", "linear", min_gain=0.2, out=tmp_path)
    assert demanding["overall"] == "FAIL"
    with pytest.raises(SwapError, match="no usable runs"):
        gate("breast_cancer", "lightgbm", "catboost", out=tmp_path)


def test_gate_command_exit_codes(monkeypatch, tmp_path):
    cfg_for(monkeypatch, tmp_path, QUIBBLE_CV="3x1")
    for ss in range(3):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(ss))
        for b in ("dummy", "lightgbm"):
            run_one(b, "breast_cancer", 0, Config.from_env())
    base = [sys.executable, "-m", "quibble_ml", "gate", "--dataset", "breast_cancer", "--out", str(tmp_path)]
    ok = subprocess.run(base + ["--candidate", "lightgbm", "--baseline", "dummy", "--min-gain", "0.1"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "OVERALL: PASS" in ok.stdout
    no = subprocess.run(base + ["--candidate", "dummy", "--baseline", "lightgbm"], capture_output=True, text=True)
    assert no.returncode == 1


def test_rate_intervals_stay_inside_zero_and_one(cv_results):
    import re
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results)
    for c in r["checks"]:
        if c["check"] in ("disagreement rate", "confident disagreements"):
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?=%)", c["observed"])]
            assert all(0 <= v <= 100 for v in nums), c
