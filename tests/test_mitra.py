import hashlib
import json
import os
import sys
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from quibble_ml import mitra
from quibble_ml.backends import MitraV2, Refused
from quibble_ml.config import Config


@pytest.fixture
def library(monkeypatch, tmp_path):
    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0

        def fit(self, X, y):
            self.X, self.y = X.copy(), y.copy()
            return self

        def predict_proba(self, X):
            self.calls += 1
            return np.tile([.3, .7], (len(X), 1))

        def predict(self, X):
            self.calls += 1
            return np.full(len(X), .5)

    monkeypatch.setattr(mitra, "verify_runtime", lambda: None)
    monkeypatch.setattr(mitra, "cpu_budget", lambda threads: nullcontext())
    checkpoint = Mock(return_value=tmp_path)
    monkeypatch.setattr(mitra, "checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "autogluon.tabular.models.mitra.sklearn_interface",
                        SimpleNamespace(MitraClassifier=Estimator))
    monkeypatch.setitem(sys.modules, "mitra_finetune", SimpleNamespace(MitraFinetune=Estimator))
    return checkpoint


def training(task):
    X = pd.DataFrame({"number": [1., np.nan, 3., 4.], "category": ["a", "b", None, "a"]})
    y = pd.Series(["z", "a", "z", "a"] if task == "classification" else [1., 2., 3., 4.])
    return X, y


def test_single_mitra_cli_uses_outer_worker_deadline(monkeypatch):
    from quibble_ml import run
    worker = Mock()
    monkeypatch.setattr(run, "sweep", worker)
    monkeypatch.delenv("_QUIBBLE_WORKER", raising=False)
    monkeypatch.setattr(sys, "argv", ["quibble-ml", "--backend", "mitra_v2",
                                     "--dataset", "wine", "--model-seed", "7"])
    run.main()
    args, kwargs = worker.call_args
    assert args[:2] == (["mitra_v2"], ["wine"])
    assert args[2].model_seeds == (7,) and kwargs["single_split"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_task_recipe_and_checked_output(library, task):
    cfg = Config.from_env()
    X, y = training(task)
    backend = MitraV2()
    backend.fit(X, y, task, 7, cfg, "cuda")
    library.assert_called_once_with(task, False)
    assert backend.model.kwargs["device"] == "cuda"
    assert backend.info["checkpoint"]["sha256_verified"]
    assert backend.info["weight_load_s"] is None
    assert not backend.info["seed_controlled"]
    assert not backend.native_ok and not backend.can_save
    query = pd.DataFrame({"number": [np.nan, 3.], "category": ["unseen", "a"]})
    pred = backend.predict(query)
    assert np.isfinite(pred).all()
    if task == "classification":
        assert backend.classes == ["a", "z"]
        assert backend.model.y.tolist() == [1, 0, 1, 0]
        assert pred.shape == (2, 2)
        assert backend.model.kwargs["fine_tune"] is False
        assert backend.model.kwargs["fine_tune_steps"] == 0
        assert backend.model.kwargs["seed"] == 7
        assert not backend.prediction_adapts
    else:
        assert pred.shape == (2,)
        assert backend.model.kwargs["num_bag_folds"] == 2
        assert backend.model.kwargs["in_process"] is True
        assert backend.model.kwargs["time_limit"] == cfg.timeout_s
        assert backend.prediction_adapts
        assert backend.info["adaptation_steps_completed"] is None
        assert backend.info["mitra_parameters"]["max_steps"] == 50
    assert pd.isna(backend.transform(query).iloc[0]["category"])
    backend.close()
    assert not hasattr(backend, "model")


@pytest.mark.parametrize("fails", [False, True])
def test_adaptation_workspace_restores_cwd_and_removes_upstream_models(library, monkeypatch, fails):
    monkeypatch.delenv("_QUIBBLE_WORKSPACE", raising=False)
    X, y = training("regression")
    backend = MitraV2()
    backend.fit(X, y, "regression", 0, Config.from_env(), "cuda")
    original, created = Path.cwd(), []

    def predict(query):
        created.append(Path.cwd())
        assert Path.cwd() != original
        Path("AutogluonModels").mkdir()
        Path("AutogluonModels", "temporary-model").write_bytes(b"model")
        if fails:
            raise RuntimeError("adaptation failed")
        return np.ones(len(query))

    backend.model.predict = predict
    if fails:
        with pytest.raises(RuntimeError, match="adaptation failed"):
            backend.predict(X)
    else:
        backend.predict(X)
    assert Path.cwd() == original
    assert not created[0].exists()


@pytest.mark.parametrize("task,bad", [
    ("classification", "shape"), ("classification", "nan"), ("classification", "probability"),
    ("regression", "shape"), ("regression", "nan"),
])
def test_invalid_predictions_fail(library, task, bad):
    X, y = training(task)
    backend = MitraV2()
    backend.fit(X, y, task, 0, Config.from_env(), "cuda")
    if task == "classification":
        values = np.ones((len(X), 1)) if bad == "shape" else np.full((len(X), 2), np.nan)
        if bad == "probability":
            values = np.tile([-.1, 1.1], (len(X), 1))
        backend.model.predict_proba = lambda X: values
    else:
        values = np.ones((len(X), 1)) if bad == "shape" else np.full(len(X), np.nan)
        backend.model.predict = lambda X: values
    with pytest.raises(ValueError, match="Mitra-v2"):
        backend.predict(X)


@pytest.mark.parametrize("device", ["cpu", "auto", "mps"])
def test_unsupported_devices_refuse_before_checkpoint(library, device):
    X, y = training("classification")
    with pytest.raises(Refused, match="explicit CUDA"):
        MitraV2().fit(X, y, "classification", 0, Config.from_env(), device)
    library.assert_not_called()


@pytest.mark.parametrize("rows,features", [(513, 2), (4, 33)])
def test_bounded_resources(library, rows, features):
    X = pd.DataFrame(np.zeros((rows, features)))
    y = pd.Series(np.arange(rows) % 2)
    with pytest.raises(Refused, match="policy limit"):
        MitraV2().fit(X, y, "classification", 0, Config.from_env(), "cuda")
    library.assert_not_called()


@pytest.mark.parametrize("allow", [True, False])
def test_checked_artifact_lookup(monkeypatch, tmp_path, allow):
    files = {"config.json": b'{"task":"classification"}', "model.safetensors": b"test weights"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    artifact = {"repository": "test/model", "revision": "fixed",
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    monkeypatch.setitem(mitra.ARTIFACTS, "classification", artifact)
    lookup = Mock(side_effect=lambda **kw: str(tmp_path / kw["filename"]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lookup))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                        SimpleNamespace(LocalEntryNotFoundError=FileNotFoundError))
    assert mitra.checkpoint("classification", allow) == tmp_path
    assert all(call.kwargs["local_files_only"] is not allow for call in lookup.call_args_list)
    assert all(call.kwargs["revision"] == "fixed" for call in lookup.call_args_list)
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mitra.checkpoint("classification", allow)
    assert (tmp_path / "model.safetensors").read_bytes() == b"corrupt"


def test_missing_weights_refuse_without_substitution(monkeypatch):
    lookup = Mock(side_effect=FileNotFoundError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lookup))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                        SimpleNamespace(LocalEntryNotFoundError=FileNotFoundError))
    with pytest.raises(Refused, match="QUIBBLE_MITRA_ALLOW_DOWNLOAD"):
        mitra.checkpoint("regression", False)
    assert lookup.call_count == 1
    assert lookup.call_args.kwargs["local_files_only"] is True


def test_runtime_requires_verified_packages_and_source(monkeypatch):
    monkeypatch.setattr(mitra, "sys", SimpleNamespace(platform="win32", version_info=(3, 13, 4)))
    monkeypatch.setattr("quibble_ml.env_info._ver", lambda name: None)
    with pytest.raises(Refused, match="requires verifiable"):
        mitra.verify_runtime()
    monkeypatch.setattr("quibble_ml.env_info._ver", lambda name: mitra.VERSIONS[name])
    source = {"url": mitra.SOURCE_URL, "vcs_info": {"commit_id": "wrong"}}
    monkeypatch.setattr(mitra.metadata, "distribution",
                        lambda name: SimpleNamespace(read_text=lambda name: json.dumps(source)))
    with pytest.raises(Refused, match="source revision"):
        mitra.verify_runtime()
    source["vcs_info"]["commit_id"] = mitra.SOURCE_REVISION
    for key in list(os.environ):
        if key.startswith("MITRA_"):
            monkeypatch.delenv(key)
    mitra.verify_runtime()
    monkeypatch.setenv("MITRA_FT_STEPS", "1")
    with pytest.raises(Refused, match="runtime overrides"):
        mitra.verify_runtime()


def test_config_download_opt_in_is_strict(monkeypatch):
    monkeypatch.delenv("QUIBBLE_MITRA_ALLOW_DOWNLOAD", raising=False)
    assert not Config.from_env().mitra_allow_download
    monkeypatch.setenv("QUIBBLE_MITRA_ALLOW_DOWNLOAD", "1")
    assert Config.from_env().mitra_allow_download
    monkeypatch.setenv("QUIBBLE_MITRA_ALLOW_DOWNLOAD", "yes")
    with pytest.raises(ValueError, match="QUIBBLE_MITRA_ALLOW_DOWNLOAD"):
        Config.from_env()


def test_recipe_records_each_task_and_changed_checkpoint(monkeypatch):
    from copy import deepcopy
    from quibble_ml.provenance import backend_metadata, check_reuse, recipe_snapshot
    cfg = Config.from_env()
    before = recipe_snapshot("mitra_v2", cfg, 0)
    weights = backend_metadata("mitra_v2")["weights"]
    assert weights["license"] == "Apache-2.0"
    assert weights["artifacts"]["regression"]["revision"] == mitra.ARTIFACTS["regression"]["revision"]
    assert before["requested"]["adaptation"]["regression"]["max_steps"] == 50
    assert before["requested"]["parameters"]["classification"]["fine_tune"] is False
    changed = deepcopy(mitra.ARTIFACTS["regression"])
    changed["files"]["model.safetensors"] = "0" * 64
    monkeypatch.setitem(mitra.ARTIFACTS, "regression", changed)
    after = recipe_snapshot("mitra_v2", cfg, 0)
    assert before["algorithm_sha256"] != after["algorithm_sha256"]
    with pytest.raises(ValueError, match="Recipe mismatch"):
        check_reuse({"recipe": before}, after, writing=True)


@pytest.mark.parametrize("existing", [None, "9"])
def test_cpu_budget_uses_supported_override_and_restores_after_failure(monkeypatch, existing):
    state = {"threads": 4}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        get_num_threads=lambda: state["threads"],
        set_num_threads=lambda n: state.update(threads=n)))
    if existing is None:
        monkeypatch.delenv("AG_CPU_COUNT", raising=False)
    else:
        monkeypatch.setenv("AG_CPU_COUNT", existing)
    with pytest.raises(RuntimeError, match="test failure"):
        with mitra.cpu_budget(2):
            assert os.environ["AG_CPU_COUNT"] == "2" and state["threads"] == 2
            raise RuntimeError("test failure")
    assert state["threads"] == 4 and os.environ.get("AG_CPU_COUNT") == existing
