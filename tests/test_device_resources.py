"""Pinned device policies and explicit resource/worker failure accounting."""
import json
import subprocess
from dataclasses import replace

import pytest

from quibble import config, run
from quibble.backends import BACKENDS, HistGB, select_backends
from quibble.config import Config
from quibble.report import device_summary


@pytest.mark.parametrize(("requested", "available", "expected"), [
    ("cpu", True, "cpu"), ("cpu", False, "cpu"),
    ("auto", True, "cuda"), ("auto", False, "cpu"), ("cuda", True, "cuda"),
])
def test_shared_device_policy(monkeypatch, requested, available, expected):
    monkeypatch.setenv("QUIBBLE_DEVICE", requested)
    monkeypatch.setattr(config, "cuda_available", lambda: available)
    assert Config.from_env().device == expected


def test_auto_worker_keeps_cpu_selection_even_if_gpu_becomes_available(monkeypatch):
    monkeypatch.setenv("QUIBBLE_DEVICE", "auto")
    monkeypatch.setenv("_QUIBBLE_RESOLVED_DEVICE", "cpu")
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    cfg = Config.from_env()
    assert cfg.device_requested == "auto" and cfg.device == "cpu"


def test_pinned_gpu_does_not_fall_back_when_unavailable(monkeypatch):
    monkeypatch.setenv("QUIBBLE_DEVICE", "auto")
    monkeypatch.setenv("_QUIBBLE_RESOLVED_DEVICE", "cuda")
    monkeypatch.setattr(config, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        Config.from_env()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_backend_filtering_preserves_model_identity(device):
    names = list(BACKENDS)
    included, excluded = select_backends(names, device)
    if device == "cpu":
        assert included == names and excluded == {}
    else:
        assert included == ["catboost", "tabpfn", "tabfm", "kumo_tabular"]
        assert set(excluded) == {"histgb", "weak", "lightgbm", "autogluon", "dummy", "linear"}
        assert all(excluded.values())


def test_cuda_sweep_excludes_cpu_and_mixed_models_before_workers(monkeypatch, tmp_path, capsys):
    calls = []

    def worker(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(run.subprocess, "run", worker)
    cfg = replace(Config.from_env(), out_dir=str(tmp_path), device_requested="auto", device="cuda")
    invocation = run.sweep(["histgb", "lightgbm", "catboost", "autogluon"], ["wine"], cfg, True)
    assert invocation["backends"] == ["catboost"]
    assert invocation["requested_backends"] == ["histgb", "lightgbm", "catboost", "autogluon"]
    assert set(invocation["excluded_backends"]) == {"histgb", "lightgbm", "autogluon"}
    assert len(calls) == 1 and calls[0][1]["_QUIBBLE_RESOLVED_DEVICE"] == "cuda"
    assert all(c["backend"] == "catboost" for c in invocation["cells"])
    assert "EXCLUDED histgb" in capsys.readouterr().err
    persisted = json.loads((tmp_path / "invocation.json").read_text())
    assert persisted["excluded_backends"] == invocation["excluded_backends"]


def test_all_excluded_stops_before_creating_results(monkeypatch, tmp_path):
    cfg = replace(Config.from_env(), out_dir=str(tmp_path / "new"), device="cuda")
    worker = lambda *a, **k: pytest.fail("Excluded methods must never start a worker")
    monkeypatch.setattr(run.subprocess, "run", worker)
    with pytest.raises(ValueError, match="No selected methods"):
        run.sweep(["histgb", "lightgbm", "weak", "autogluon"], ["wine"], cfg, True)
    assert not (tmp_path / "new").exists()


def test_direct_cuda_cpu_backend_is_refused_before_loading_data(monkeypatch, tmp_path):
    monkeypatch.setattr("quibble.datasets.load", lambda *a: pytest.fail("Must refuse before loading"))
    cfg = replace(Config.from_env(), out_dir=str(tmp_path), device="cuda")
    run.run_one("histgb", "wine", 0, cfg)
    record = json.loads((tmp_path / "wine" / "histgb_ss0_ms0.json").read_text())
    assert "Excluded from CUDA execution" in record["refused"]
    assert record["device_attempted"] == "cuda" and "fit_s" not in record


def test_report_records_exclusions_instead_of_old_cpu_success():
    from quibble.report import participation
    invocation = dict(backends=["catboost"], requested_backends=["histgb", "catboost"],
                      excluded_backends={"histgb": "CPU-only implementation"}, datasets=["wine"])
    records = {"wine": [dict(backend="histgb", split_seed=0, model_seed=0)]}
    table = participation(None, ["wine"], records, {}, invocation).set_index("backend")
    assert table.loc["histgb", "selected"] == "yes"
    assert table.loc["histgb", "execution"] == "excluded"
    assert table.loc["histgb", "completed"] == 0
    assert "GPU-only policy" in table.loc["histgb", "reason"]


@pytest.mark.parametrize("cleanup", [False, True])
def test_resource_failure_and_cleanup_are_recorded(monkeypatch, tmp_path, cleanup):
    class OutOfMemory(HistGB):
        def fit(self, *args):
            raise MemoryError("not enough memory for this fit")

        def close(self):
            if cleanup:
                raise RuntimeError("cleanup failed")

    monkeypatch.setitem(BACKENDS, "histgb", OutOfMemory)
    cfg = replace(Config.from_env(), out_dir=str(tmp_path))
    run.run_one("histgb", "wine", 0, cfg)
    record = json.loads((tmp_path / "wine" / "histgb_ss0_ms0.json").read_text())
    assert record["error_kind"] == "resource" and record["device_attempted"] == "cpu"
    assert record["error"].startswith("MemoryError")
    assert ("cleanup_error" in record) == cleanup


@pytest.mark.parametrize("failure", ["exit", "missing", "spawn", "timeout"])
def test_failed_workers_do_not_hide_old_success_or_stop_other_cells(monkeypatch, tmp_path, failure):
    calls = []

    def worker(cmd, **kwargs):
        calls.append(kwargs["env"])
        if failure == "spawn":
            raise OSError("not enough memory")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 3)
        return subprocess.CompletedProcess(cmd, 3221225495 if failure == "exit" else 0)

    monkeypatch.setattr(run.subprocess, "run", worker)
    cfg = replace(Config.from_env(), out_dir=str(tmp_path), device_requested="auto", device="cpu")
    invocation = run.sweep(["histgb", "lightgbm"], ["wine"], cfg, True)
    assert len(calls) == 2
    for cell in invocation["cells"]:
        record = json.loads(run.stem(tmp_path, "wine", cell["backend"], 0, 0)
                            .with_suffix(".json").read_text())
        assert "error" in record and record["run_id"] == invocation["run_id"]
        assert record["device_requested"] == "auto" and record["device_attempted"] == "cpu"
    assert all(env["_QUIBBLE_RESOLVED_DEVICE"] == "cpu" for env in calls)


def test_mixed_report_warns_and_cpu_auto_is_not_called_mixed():
    cpu = dict(backend="histgb", device_requested="auto", device="cuda", device_effective="cpu")
    gpu = dict(backend="tabfm", device_requested="auto", device="cuda", device_effective="cuda")
    text = device_summary([cpu, gpu])
    assert "MIXED CPU/GPU MEASUREMENTS" in text and "AUTO DEVICE POLICY" in text
    assert "not an equal-device speed comparison" in text
    assert "MIXED CPU/GPU MEASUREMENTS" not in device_summary([dict(cpu, device="cpu")])
    failed = dict(gpu, error="CUDA out of memory")
    assert "not completed" in device_summary([failed])
    assert "MIXED CPU/GPU MEASUREMENTS" not in device_summary([cpu, failed])
