import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quibble.backends import BACKENDS, Refused
from quibble.config import Config
from quibble.run import peak_mb, run_one, stem


def test_peak_memory_is_positive_and_finite():
    peak = peak_mb()
    assert np.isfinite(peak) and peak > 0


def test_windows_peak_memory_uses_high_water_mark(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "resource", None)
    memory = SimpleNamespace(peak_wset=250_000_000, rss=100_000_000)
    process = SimpleNamespace(memory_info=lambda: memory)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: process))
    assert peak_mb() == 250.0


@pytest.mark.parametrize(("platform", "maxrss"), [("linux", 250_000), ("darwin", 250_000_000)])
def test_unix_peak_memory_units(monkeypatch, platform, maxrss):
    monkeypatch.setattr(sys, "platform", platform)

    def getrusage(who):
        assert who == 0
        return SimpleNamespace(ru_maxrss=maxrss)

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=getrusage))
    assert peak_mb() == 250.0


def test_peak_memory_errors_propagate(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def memory_info():
        raise OSError("memory query failed")

    process = SimpleNamespace(memory_info=memory_info)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: process))
    with pytest.raises(OSError, match="memory query failed"):
        peak_mb()


def cfg_for(monkeypatch, out, **env):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Config.from_env()


def result(out, backend, ds="breast_cancer", ss=0, ms=0):
    return json.loads(stem(out, ds, backend, ss, ms).with_suffix(".json").read_text())


@pytest.mark.parametrize("backend", ["histgb", "lightgbm", "catboost"])
def test_determinism_same_seed_same_predictions(monkeypatch, tmp_path, backend):
    preds = []
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        run_one(backend, "breast_cancer", 0, cfg_for(monkeypatch, out), force=True)
        preds.append(np.load(f"{stem(out, 'breast_cancer', backend, 0, 0)}.pred.npy"))
    assert np.array_equal(preds[0], preds[1])


def test_smoke_result_contents(monkeypatch, tmp_path):
    run_one("histgb", "wine", 0, cfg_for(monkeypatch, tmp_path), force=True)
    r = result(tmp_path, "histgb", "wine")
    for k in ("accuracy", "auc", "brier", "fit_s", "single_row_p50_ms", "peak_rss_mb",
              "device", "device_effective", "preprocess_effective", "env"):
        assert k in r
    assert r["env"]["packages"]["numpy"]
    assert np.isfinite(r["peak_rss_mb"]) and r["peak_rss_mb"] > 0
    assert 0.5 < r["accuracy"] <= 1.0


def test_regression_metrics(monkeypatch, tmp_path):
    run_one("lightgbm", "diabetes_reg", 0, cfg_for(monkeypatch, tmp_path), force=True)
    r = result(tmp_path, "lightgbm", "diabetes_reg")
    assert r["task"] == "regression" and {"rmse", "mae", "r2"} <= set(r)


def test_finished_runs_are_skipped_unless_forced(monkeypatch, tmp_path, capsys):
    cfg = cfg_for(monkeypatch, tmp_path)
    run_one("histgb", "wine", 0, cfg)
    capsys.readouterr()
    run_one("histgb", "wine", 0, cfg)
    assert "skip" in capsys.readouterr().out


def test_tabpfn_refusal_is_a_result_not_a_crash(monkeypatch, tmp_path):
    cfg = cfg_for(monkeypatch, tmp_path, QUIBBLE_TABPFN_MAX_ROWS="10")
    run_one("tabpfn", "wine", 0, cfg, force=True)
    r = result(tmp_path, "tabpfn", "wine")
    assert "refused" in r and "error" not in r and "accuracy" not in r


def test_missing_package_is_recorded_as_error(monkeypatch, tmp_path):
    cfg = cfg_for(monkeypatch, tmp_path)
    monkeypatch.setitem(BACKENDS, "ghost", type("Ghost", (BACKENDS["histgb"],), {
        "name": "ghost", "make": lambda self, *a: __import__("not_a_real_package")}))
    run_one("ghost", "wine", 0, cfg, force=True)
    assert "ModuleNotFoundError" in result(tmp_path, "ghost", "wine")["error"]


def test_backend_that_cannot_do_the_task_refuses(monkeypatch, tmp_path):
    class ClfOnly(BACKENDS["histgb"]):
        name, tasks = "clfonly", ("classification",)

    monkeypatch.setitem(BACKENDS, "clfonly", ClfOnly)
    run_one("clfonly", "diabetes_reg", 0, cfg_for(monkeypatch, tmp_path), force=True)
    assert "does not support" in result(tmp_path, "clfonly", "diabetes_reg")["refused"]


def test_native_categoricals_run_and_handle_unseen_values():
    import numpy as np
    import pandas as pd
    from quibble.backends import CatBoost, HistGB, LightGBM
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "c": pd.Series(rng.choice(["x", "y", "z", None], n), dtype="object")})
    y = pd.Series(((X.a > 0) ^ (X.c == "x")).astype(int))

    class C:
        preprocess, threads, tabpfn_max_rows = "native", 1, None

    for B in (HistGB, LightGBM, CatBoost):
        b = B()
        b.fit(X[:300], y[:300], "classification", 0, C, "cpu")
        Xt = X[300:].copy()
        Xt.iloc[0, 1] = "NEVER_SEEN"
        assert b.predict(Xt).shape == (100, 2)
        assert b.info["preprocess_effective"] == "native"


def test_refused_is_an_exception_type():
    assert issubclass(Refused, Exception)


def test_changed_settings_with_same_out_dir_warns_instead_of_silently_reusing(monkeypatch, tmp_path, capsys):
    from quibble.run import sweep
    run_one("histgb", "wine", 0, cfg_for(monkeypatch, tmp_path), force=True)  # preprocess=ordinal
    cfg = cfg_for(monkeypatch, tmp_path, QUIBBLE_PREPROCESS="native")
    capsys.readouterr()
    sweep(["histgb"], ["wine"], cfg, False)  # everything already done -> no subprocess
    out = capsys.readouterr().out
    assert "0 to run, 1 already done" in out and "different settings" in out and "preprocess_requested" in out
