import pytest

from quibble_ml import config
from quibble_ml.config import Config


def test_cpu_toggle(monkeypatch):
    monkeypatch.setenv("QUIBBLE_DEVICE", "cpu")
    c = Config.from_env()
    assert (c.device_requested, c.device) == ("cpu", "cpu")


def test_cuda_fails_loudly_when_unavailable(monkeypatch):
    monkeypatch.setattr(config, "cuda_available", lambda: False)
    monkeypatch.setenv("QUIBBLE_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="not available"):
        Config.from_env()


def test_auto_follows_availability(monkeypatch):
    monkeypatch.setenv("QUIBBLE_DEVICE", "auto")
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    assert Config.from_env().device == "cuda"
    monkeypatch.setattr(config, "cuda_available", lambda: False)
    assert Config.from_env().device == "cpu"


@pytest.mark.parametrize("key,val", [("QUIBBLE_DEVICE", "tpu"), ("QUIBBLE_PREPROCESS", "magic"),
                                     ("QUIBBLE_CV", "five"), ("QUIBBLE_TABFM_N_ESTIMATORS", "0"),
                                     ("QUIBBLE_KUMO_N_ESTIMATORS", "-1"), ("QUIBBLE_KUMO_SIZE", "huge")])
def test_invalid_values_rejected(monkeypatch, key, val):
    monkeypatch.setenv(key, val)
    with pytest.raises(ValueError):
        Config.from_env()


def test_foundation_model_settings(monkeypatch):
    c = Config.from_env()
    assert (c.tabfm_n_estimators, c.kumo_size, c.kumo_n_estimators) == (4, "small", 4)
    monkeypatch.setenv("QUIBBLE_TABFM_N_ESTIMATORS", "2")
    monkeypatch.setenv("QUIBBLE_KUMO_SIZE", "medium")
    monkeypatch.setenv("QUIBBLE_KUMO_N_ESTIMATORS", "8")
    c = Config.from_env()
    assert (c.tabfm_n_estimators, c.kumo_size, c.kumo_n_estimators) == (2, "medium", 8)


def test_split_ids(monkeypatch):
    monkeypatch.setenv("QUIBBLE_CV", "5x2")
    assert Config.from_env().split_ids == tuple(range(10))
    monkeypatch.setenv("QUIBBLE_CV", "holdout")
    monkeypatch.setenv("QUIBBLE_SPLIT_SEED", "7")
    assert Config.from_env().split_ids == (7,)


def test_parallel_jobs_divide_threads(monkeypatch):
    monkeypatch.setenv("QUIBBLE_THREADS", "0")
    monkeypatch.setenv("QUIBBLE_JOBS", "2")
    monkeypatch.setattr(config.os, "cpu_count", lambda: 8)
    assert Config.from_env().threads == 4


def test_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nQUIBBLE_X1=from_file  # trailing\nQUIBBLE_X2='quoted'\n")
    monkeypatch.setenv("QUIBBLE_X2", "from_env")
    monkeypatch.delenv("QUIBBLE_X1", raising=False)
    config.load_dotenv(str(f))
    import os
    assert os.environ["QUIBBLE_X1"] == "from_file"
    assert os.environ["QUIBBLE_X2"] == "from_env"
    monkeypatch.delenv("QUIBBLE_X1")
