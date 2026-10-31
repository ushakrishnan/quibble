"""Backend contracts tested without installing models or downloading weights."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from quibble_ml.backends import BACKENDS, KumoTabular, Refused, TabFM, TabPFN
from quibble_ml.config import Config
from quibble_ml.run import run_one, stem, sweep


@pytest.fixture
def packages(monkeypatch):
    estimator = Mock()
    tabfm = SimpleNamespace(
        tabfm_v1_0_0_pytorch=SimpleNamespace(load=Mock(return_value=object())),
        TabFMClassifier=Mock(return_value=estimator),
        TabFMRegressor=Mock(return_value=estimator))

    def table(df, **kwargs):
        return SimpleNamespace(df=df.copy(), to_pandas=lambda: df.copy())

    kumo = Mock()
    sdm = SimpleNamespace(
        infer_stypes=Mock(side_effect=lambda df, **kwargs: dict.fromkeys(df.columns, "inferred")),
        TableTensor=SimpleNamespace(from_pandas=Mock(side_effect=table)),
        models=SimpleNamespace(KumoTabular=Mock(return_value=kumo)))
    generator = Mock()
    generator.manual_seed.return_value = generator
    torch = SimpleNamespace(Generator=Mock(return_value=generator))
    for name, module in (("tabfm", tabfm), ("sdm", sdm), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(tabfm=tabfm, estimator=estimator, sdm=sdm, kumo=kumo,
                           torch=torch, generator=generator, table=table)


def training_data(task):
    X = pd.DataFrame({"n": [1., 2., 3., 4.], "c": ["a", "b", "a", None]},
                     index=[10, 20, 30, 40])
    y = pd.Series([0, 1, 0, 1] if task == "classification" else [1., 2., 3., 4.],
                  index=X.index)
    return X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tabpfn_constructor_and_prediction_contract(monkeypatch, task, device):
    estimator = Mock()
    classifier, regressor = Mock(return_value=estimator), Mock(return_value=estimator)
    monkeypatch.setitem(sys.modules, "tabpfn", SimpleNamespace(
        TabPFNClassifier=classifier, TabPFNRegressor=regressor))
    cfg = SimpleNamespace(tabpfn_max_rows=100, preprocess="native")
    X, y = training_data(task)
    backend = TabPFN()
    backend.fit(X, y, task, 7, cfg, device)
    factory = classifier if task == "classification" else regressor
    factory.assert_called_once_with(device=device, random_state=7)
    fitted_X, fitted_y = estimator.fit.call_args.args
    pd.testing.assert_series_equal(fitted_y, y)
    assert not isinstance(fitted_X.c.dtype, pd.CategoricalDtype)
    expected = np.array([[0.2, 0.8]]) if task == "classification" else np.array([3.5])
    method = estimator.predict_proba if task == "classification" else estimator.predict
    method.return_value = expected
    np.testing.assert_array_equal(
        backend.predict(pd.DataFrame({"n": [5.], "c": ["unseen"]})), expected)
    assert pd.isna(method.call_args.args[0].c.iloc[0])
    assert backend.info == {"preprocess_effective": "ordinal", "device_effective": device}


@pytest.mark.parametrize("limit", ["rows", "features", "classes"])
def test_tabpfn_limits_refuse_before_importing_model(monkeypatch, limit):
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    X = pd.DataFrame(np.ones((11, 501 if limit == "features" else 1)))
    y = pd.Series(range(11) if limit == "classes" else [0, 1] * 5 + [0])
    cfg = SimpleNamespace(tabpfn_max_rows=10 if limit == "rows" else 100)
    with pytest.raises(Refused, match={"rows": "train rows", "features": "features",
                                     "classes": "classes"}[limit]):
        TabPFN().fit(X, y, "classification", 0, cfg, "cpu")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("preprocess", ["ordinal", "native"])
def test_tabfm_contract(packages, monkeypatch, task, device, preprocess):
    monkeypatch.setenv("QUIBBLE_PREPROCESS", preprocess)
    monkeypatch.setenv("QUIBBLE_TABFM_N_ESTIMATORS", "2")
    cfg = Config.from_env()
    X, y = training_data(task)
    backend = TabFM()
    backend.fit(X, y, task, 7, cfg, device)
    packages.tabfm.tabfm_v1_0_0_pytorch.load.assert_called_once_with(
        model_type=task, device=device, use_cache=False)
    factory = (packages.tabfm.TabFMClassifier if task == "classification"
               else packages.tabfm.TabFMRegressor)
    assert factory.call_args.kwargs["random_state"] == 7
    assert factory.call_args.kwargs["n_estimators"] == 2
    fitted_X, fitted_y = packages.estimator.fit.call_args.args
    pd.testing.assert_series_equal(fitted_y, y)
    assert len(fitted_X) == len(X)
    assert isinstance(fitted_X.c.dtype, pd.CategoricalDtype) == (preprocess == "native")
    expected = np.array([[0.2, 0.8]]) if task == "classification" else np.array([3.5])
    method = (packages.estimator.predict_proba if task == "classification"
              else packages.estimator.predict)
    method.return_value = expected
    query = pd.DataFrame({"n": [5.], "c": ["unseen"]})
    np.testing.assert_array_equal(backend.predict(query), expected)
    assert pd.isna(method.call_args.args[0].c.iloc[0])
    assert backend.info == {"preprocess_effective": preprocess, "device_effective": device,
                            "tabfm_n_estimators": 2}


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("preprocess", ["ordinal", "native"])
def test_kumo_fit_contract(packages, monkeypatch, task, device, preprocess):
    monkeypatch.setenv("QUIBBLE_PREPROCESS", preprocess)
    monkeypatch.setenv("QUIBBLE_KUMO_SIZE", "medium")
    monkeypatch.setenv("QUIBBLE_KUMO_N_ESTIMATORS", "2")
    X, y = training_data(task)
    backend = KumoTabular()
    backend.fit(X, y, task, 7, Config.from_env(), device)
    packages.sdm.models.KumoTabular.assert_called_once_with(
        task=task, size="medium", pretrained=True, device=device)
    packages.torch.Generator.assert_called_once_with(device=device)
    packages.generator.manual_seed.assert_called_once_with(7)
    fit = packages.kumo.fit.call_args.kwargs
    assert fit["num_estimators"] == 2 and fit["generator"] is packages.generator
    assert len(fit["x"].df) == len(X)
    assert isinstance(fit["x"].df.c.dtype, pd.CategoricalDtype) == (preprocess == "native")
    pd.testing.assert_series_equal(fit["y"].df["__y__"], y.rename("__y__"))
    assert packages.sdm.infer_stypes.call_args.kwargs["overrides"] == {
        "__y__": "categorical" if task == "classification" else "numerical"}
    assert backend.info == {"preprocess_effective": preprocess, "device_effective": device,
                            "kumo_size": "medium", "kumo_n_estimators": 2}
    backend.close()
    packages.kumo.clear.assert_called_once()


@pytest.mark.parametrize("columns", [[1, 0], ["1", "0"], ["2", "0", "1"]])
def test_kumo_aligns_class_probability_columns(packages, columns):
    X = pd.DataFrame({"n": np.arange(len(columns), dtype=float)})
    backend = KumoTabular()
    backend.fit(X, pd.Series(range(len(columns))), "classification", 0, Config.from_env(), "cpu")
    expected = np.arange(1, len(columns) + 1, dtype=float)
    expected /= expected.sum()
    pred = pd.DataFrame({c: [expected[int(c)]] for c in columns})
    packages.kumo.predict.return_value = packages.table(pred)
    np.testing.assert_allclose(backend.predict(X.iloc[:1]), expected[None, :])
    assert packages.sdm.TableTensor.from_pandas.call_args.kwargs["stypes"] is backend.stypes


@pytest.mark.parametrize("pred", [
    pd.DataFrame({"1": [0.8]}),
    pd.DataFrame([[0.5, 0.5]], columns=["0", "0"]),
    pd.DataFrame({"0": [0.2], "2": [0.8]}),
    pd.DataFrame({"0": [-0.1], "1": [1.1]}),
    pd.DataFrame({"0": [0.8], "1": [0.8]}),
    pd.DataFrame({"0": [np.nan], "1": [0.8]}),
    pd.DataFrame({"0": [np.inf], "1": [0.8]}),
])
def test_kumo_rejects_malformed_probabilities(packages, pred):
    X, y = training_data("classification")
    backend = KumoTabular()
    backend.fit(X, y, "classification", 0, Config.from_env(), "cpu")
    packages.kumo.predict.return_value = packages.table(pred)
    with pytest.raises(ValueError, match="probabilit"):
        backend.predict(X.iloc[:1])


def test_kumo_regression_uses_mean_of_all_quantiles(packages):
    X, y = training_data("regression")
    backend = KumoTabular()
    backend.fit(X, y, "regression", 0, Config.from_env(), "cpu")
    quantiles = np.linspace(0, 1, 999) ** 2
    pred = pd.DataFrame([quantiles], columns=[f"q{i:03d}" for i in range(1, 1000)])
    packages.kumo.predict.return_value = packages.table(pred[pred.columns[::-1]])
    result = backend.predict(X.iloc[:1])
    assert result.shape == (1,)
    assert result[0] == pytest.approx(quantiles.mean())
    assert result[0] != pytest.approx(quantiles[0])
    assert result[0] != pytest.approx(np.median(quantiles))


@pytest.mark.parametrize("malformation", ["missing", "extra", "duplicate", "nan", "inf", "rows"])
def test_kumo_rejects_malformed_quantiles(packages, malformation):
    X, y = training_data("regression")
    backend = KumoTabular()
    backend.fit(X, y, "regression", 0, Config.from_env(), "cpu")
    pred = pd.DataFrame(np.zeros((1, 999)), columns=[f"q{i:03d}" for i in range(1, 1000)])
    if malformation == "missing":
        pred = pred.drop(columns="q001")
    elif malformation == "extra":
        pred["extra"] = 0
    elif malformation == "duplicate":
        pred = pred.rename(columns={"q999": "q001"})
    elif malformation == "rows":
        pred = pd.concat([pred, pred])
    else:
        pred.loc[0, "q001"] = np.nan if malformation == "nan" else np.inf
    packages.kumo.predict.return_value = packages.table(pred)
    with pytest.raises(ValueError, match="quantile|row count"):
        backend.predict(X.iloc[:1])


@pytest.mark.parametrize(("backend", "package"), [("tabfm", "tabfm"), ("kumo_tabular", "sdm")])
def test_missing_optional_package_is_recorded(monkeypatch, tmp_path, backend, package):
    monkeypatch.setitem(sys.modules, package, None)
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    run_one(backend, "wine", 0, Config.from_env())
    record = json.loads(stem(tmp_path, "wine", backend, 0, 0).with_suffix(".json").read_text())
    assert "ModuleNotFoundError" in record["error"]
    assert "accuracy" not in record


@pytest.mark.parametrize("backend", ["tabfm", "kumo_tabular"])
@pytest.mark.parametrize("dataset", ["wine", "diabetes_reg"])
def test_run_one_saves_foundation_predictions(packages, monkeypatch, tmp_path, backend, dataset):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    if dataset == "wine":
        packages.estimator.predict_proba.side_effect = lambda X: np.full((len(X), 3), 1 / 3)
        packages.kumo.predict.side_effect = lambda x: packages.table(
            pd.DataFrame(np.full((len(x.df), 3), 1 / 3), columns=["2", "0", "1"]))
    else:
        packages.estimator.predict.side_effect = lambda X: np.zeros(len(X))
        packages.kumo.predict.side_effect = lambda x: packages.table(pd.DataFrame(
            np.zeros((len(x.df), 999)), columns=[f"q{i:03d}" for i in range(1, 1000)]))
    run_one(backend, dataset, 0, Config.from_env())
    s = stem(tmp_path, dataset, backend, 0, 0)
    record = json.loads(s.with_suffix(".json").read_text())
    assert "error" not in record and "refused" not in record
    P = np.load(f"{s}.pred.npy")
    assert P.shape == ((record["n_test"], 3) if dataset == "wine" else (record["n_test"],))
    assert record["device_effective"] == "cpu"
    assert record["peak_rss_mb"] > 0
    assert {"tabfm", "structured-data-models"} <= record["env"]["packages"].keys()
    if backend == "kumo_tabular":
        packages.kumo.clear.assert_called_once()


def test_kumo_fit_error_is_recorded_and_cache_cleared(packages, monkeypatch, tmp_path):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    packages.kumo.fit.side_effect = RuntimeError("fit failed")
    run_one("kumo_tabular", "wine", 0, Config.from_env())
    record = json.loads(stem(tmp_path, "wine", "kumo_tabular", 0, 0).with_suffix(".json").read_text())
    assert record["error"] == "RuntimeError: fit failed"
    packages.kumo.clear.assert_called_once()


@pytest.mark.parametrize(("backend", "key", "value"), [
    ("tabfm", "QUIBBLE_TABFM_N_ESTIMATORS", "8"),
    ("kumo_tabular", "QUIBBLE_KUMO_N_ESTIMATORS", "8"),
    ("kumo_tabular", "QUIBBLE_KUMO_SIZE", "large"),
])
def test_changed_foundation_settings_require_separate_output(
        packages, monkeypatch, tmp_path, capsys, backend, key, value):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    packages.estimator.predict_proba.side_effect = lambda X: np.full((len(X), 3), 1 / 3)
    packages.kumo.predict.side_effect = lambda x: packages.table(
        pd.DataFrame(np.full((len(x.df), 3), 1 / 3), columns=["0", "1", "2"]))
    run_one(backend, "wine", 0, Config.from_env())
    capsys.readouterr()
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="Recipe mismatch"):
        sweep([backend], ["wine"], Config.from_env(), False)


def test_foundation_results_work_with_analysis_swap_and_report(packages, monkeypatch, tmp_path):
    from quibble_ml.analyze import analyse
    from quibble_ml.report import build
    from quibble_ml.swap import evaluate
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("QUIBBLE_CV", "2x1")
    packages.estimator.predict_proba.side_effect = lambda X: np.full((len(X), 3), 1 / 3)
    packages.kumo.predict.side_effect = lambda x: packages.table(
        pd.DataFrame(np.full((len(x.df), 3), 1 / 3), columns=["2", "0", "1"]))
    for ss in (0, 1):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(ss))
        for backend in ("tabfm", "kumo_tabular"):
            run_one(backend, "wine", 0, Config.from_env())
    result = analyse("wine", out=tmp_path, quiet=True)
    assert [(a, b) for a, b, _ in result["pairs"]] == [("kumo_tabular", "tabfm")]
    swap = evaluate("wine", "tabfm", "kumo_tabular", out=tmp_path)
    assert swap["splits"] == 2
    assert swap["overall"] in ("PASS", "FAIL", "INCONCLUSIVE")
    report = build(tmp_path, ["wine"], swaps=[("tabfm", "kumo_tabular")]).read_text()
    assert "Swap check: tabfm -> kumo_tabular" in report
    assert "not possible" not in report and "Skipped datasets" not in report


def test_new_backends_are_registered_and_scheduled_serially_on_cuda(monkeypatch, tmp_path):
    from dataclasses import replace
    from quibble_ml import run
    assert BACKENDS["tabfm"] is TabFM and BACKENDS["kumo_tabular"] is KumoTabular
    cfg = replace(Config.from_env(), device="cuda", device_requested="cuda",
                  jobs=2, out_dir=str(tmp_path))
    calls = []

    def child(cmd, **kwargs):
        import threading
        calls.append((cmd, threading.current_thread().name))
        import subprocess
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run, "_worker_process", child)
    sweep(["tabfm", "kumo_tabular"], ["wine"], cfg, False)
    assert len(calls) == 2
    assert [cmd[4] for cmd, _ in calls] == ["tabfm", "kumo_tabular"]
    assert all(thread == "MainThread" for _, thread in calls)
