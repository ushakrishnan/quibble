import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from quibble_ml import tabicl
from quibble_ml.backends import Refused, TabICLv2
from quibble_ml.config import Config
from quibble_ml.provenance import backend_metadata, recipe_snapshot
from quibble_ml.run import run_one


@pytest.fixture(autouse=True)
def hub(monkeypatch):
    class LocalEntryNotFoundError(OSError):
        pass
    errors = SimpleNamespace(LocalEntryNotFoundError=LocalEntryNotFoundError)
    module = SimpleNamespace(hf_hub_download=Mock(), errors=errors)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return module


@pytest.fixture
def library(monkeypatch, tmp_path):
    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _load_model(self):
            self.loaded = True

        def fit(self, X, y):
            self._load_model()
            self.training = X.copy()
            self.classes_ = np.sort(y.unique())[::-1]
            return self

        def predict_proba(self, X):
            return np.tile([0.8, 0.2], (len(X), 1))

        def predict(self, X, output_type):
            assert output_type == "mean"
            return np.ones(len(X))

    module = SimpleNamespace(TabICLClassifier=Estimator, TabICLRegressor=Estimator)
    monkeypatch.setitem(sys.modules, "tabicl", module)
    monkeypatch.setattr("quibble_ml.env_info._ver", lambda name: tabicl.VERSION)
    loader = Mock(return_value=tmp_path / "weights.ckpt")
    monkeypatch.setattr(tabicl, "checkpoint", loader)
    return loader


def training(task="classification"):
    return (pd.DataFrame({"n": [1., 2., np.nan, 4.], "c": ["a", "b", "a", None]}),
            pd.Series([0, 1, 0, 1] if task == "classification" else [1., 2., 3., 4.]))


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("preprocess", ["ordinal", "native"])
def test_adapter_contract(library, monkeypatch, task, device, preprocess):
    monkeypatch.setenv("QUIBBLE_PREPROCESS", preprocess)
    backend = TabICLv2()
    X, y = training(task)
    backend.fit(X, y, task, 7, Config.from_env(), device)
    library.assert_called_once_with(task, False)
    kwargs = backend.model.kwargs
    assert kwargs["device"] == device and kwargs["random_state"] == 7
    assert kwargs["checkpoint_version"] == tabicl.ARTIFACTS[task]["filename"]
    assert kwargs["allow_auto_download"] is False
    assert kwargs["offload_mode"] is False and kwargs["kv_cache"] is False
    assert kwargs["n_estimators"] == kwargs["batch_size"] == 8
    assert backend.info["preprocess_effective"] == "ordinal"
    assert backend.info["checkpoint"]["sha256_verified"] is True
    assert backend.load_s >= backend.model.quibble_weight_load_s >= 0
    query = pd.DataFrame({"n": [2.], "c": ["unknown"]})
    result = backend.predict(query)
    np.testing.assert_allclose(result, [[0.2, 0.8]] if task == "classification" else [1.])
    assert pd.isna(backend.transform(query)["c"].iloc[0])
    backend.close()
    backend.close()
    assert not hasattr(backend, "model")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "range", "sum", "classes"])
def test_bad_outputs_are_rejected(library, monkeypatch, task, bad):
    X, y = training(task)
    backend = TabICLv2()
    backend.fit(X, y, task, 0, Config.from_env(), "cpu")
    if task == "regression":
        value = np.array([[1.]]) if bad == "shape" else np.array([np.nan])
        monkeypatch.setattr(backend.model, "predict", lambda *args, **kwargs: value)
    else:
        value = {"shape": [[0.2]], "nan": [[np.nan, 0.8]], "inf": [[np.inf, 0.8]],
                 "range": [[-0.1, 1.1]], "sum": [[0.8, 0.8]], "classes": [[0.2, 0.8]]}[bad]
        if bad == "classes":
            backend.model.classes_ = [0, 2]
        monkeypatch.setattr(backend.model, "predict_proba", lambda *args: value)
    with pytest.raises(ValueError):
        backend.predict(X.iloc[:1])


@pytest.mark.parametrize("case", ["rows", "features", "device", "version"])
def test_unsupported_settings_refuse_before_checkpoint_access(library, monkeypatch, case):
    X, y = training()
    device = "cpu"
    if case == "rows":
        X = pd.DataFrame(np.zeros((1001, 1)))
        y = pd.Series(np.arange(1001) % 2)
    elif case == "features":
        X = pd.DataFrame(np.zeros((4, 501)))
    elif case == "device":
        device = "mps"
    else:
        monkeypatch.setattr("quibble_ml.env_info._ver", lambda name: "3.0")
    with pytest.raises(Refused):
        TabICLv2().fit(X, y, "classification", 0, Config.from_env(), device)
    library.assert_not_called()


@pytest.mark.parametrize("device,rows", [("cpu", 1000), ("cpu", 1001),
                                        ("cuda", 10_000), ("cuda", 10_001)])
def test_row_policy_at_boundary(library, device, rows):
    X, y = pd.DataFrame({"n": np.arange(rows)}), pd.Series(np.arange(rows) % 2)
    if rows in (1001, 10_001):
        with pytest.raises(Refused):
            TabICLv2().fit(X, y, "classification", 0, Config.from_env(), device)
    else:
        TabICLv2().fit(X, y, "classification", 0, Config.from_env(), device)


@pytest.mark.parametrize("columns", [500, 501])
def test_feature_policy_at_boundary(library, columns):
    X, y = pd.DataFrame(np.zeros((4, columns))), pd.Series([0, 1, 0, 1])
    if columns > 500:
        with pytest.raises(Refused):
            TabICLv2().fit(X, y, "classification", 0, Config.from_env(), "cpu")
    else:
        TabICLv2().fit(X, y, "classification", 0, Config.from_env(), "cpu")


def test_missing_dependency_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tabicl", None)
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    run_one("tabicl_v2", "wine", 0, Config.from_env())
    record = json.loads((tmp_path / "wine" / "tabicl_v2_ss0_ms0.json").read_text())
    assert "tabicl" in record["error"]
    assert record["backend_metadata"]["library"]["package"] == "tabicl"


@pytest.mark.parametrize("allow", [False, True])
def test_checkpoint_uses_pinned_revision_hash_and_download_policy(monkeypatch, tmp_path, allow):
    path = tmp_path / "weights"
    path.write_bytes(b"fixture checkpoint")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(tabicl.ARTIFACTS["classification"], "sha256", expected)
    download = Mock(return_value=str(path))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert tabicl.checkpoint("classification", allow) == path
    download.assert_called_once_with(repo_id=tabicl.REPOSITORY,
                                     filename=tabicl.ARTIFACTS["classification"]["filename"],
                                     revision=tabicl.REVISION, local_files_only=not allow)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        tabicl.checkpoint("classification", allow)


def test_uncached_checkpoint_refuses_without_fallback(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    download = Mock(side_effect=LocalEntryNotFoundError("not cached"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(Refused, match="QUIBBLE_TABICL_ALLOW_DOWNLOAD"):
        tabicl.checkpoint("regression", False)
    assert download.call_count == 1


@pytest.mark.parametrize("value", ["yes", "-1", "2"])
def test_download_opt_in_is_validated(monkeypatch, value):
    monkeypatch.setenv("QUIBBLE_TABICL_ALLOW_DOWNLOAD", value)
    with pytest.raises(ValueError, match="QUIBBLE_TABICL_ALLOW_DOWNLOAD"):
        Config.from_env()


def test_registry_and_recipe_capture_both_task_artifacts():
    metadata = backend_metadata("tabicl_v2")
    assert metadata["weights"]["revision"] == tabicl.REVISION
    assert metadata["weights"]["artifacts"] == tabicl.ARTIFACTS
    recipe = recipe_snapshot("tabicl_v2", Config.from_env(), 0)
    assert recipe["requested"]["checkpoint"]["artifacts"] == tabicl.ARTIFACTS
    assert recipe["requested"]["parameters"]["kv_cache"] is False
    assert recipe["identity_complete"] is False


def test_changed_artifact_changes_recipe_identity(monkeypatch):
    before = recipe_snapshot("tabicl_v2", Config.from_env(), 0)
    monkeypatch.setitem(tabicl.ARTIFACTS["classification"], "sha256", "0" * 64)
    after = recipe_snapshot("tabicl_v2", Config.from_env(), 0)
    assert before["algorithm_sha256"] != after["algorithm_sha256"]


def test_run_records_verified_artifact_without_inventing_real_model_evidence(library, monkeypatch, tmp_path):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    run_one("tabicl_v2", "breast_cancer", 0, Config.from_env())
    record = json.loads((tmp_path / "breast_cancer" / "tabicl_v2_ss0_ms0.json").read_text())
    assert "error" not in record
    assert record["backend_metadata"]["weights"]["sha256"] == tabicl.ARTIFACTS["classification"]["sha256"]
    assert record["checkpoint"]["sha256_verified"] is True
    assert "experimental" in record["backend_metadata"]["verification"]
