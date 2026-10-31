import json
from types import SimpleNamespace
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from quibble_ml import datasets


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("QUIBBLE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "data"))
    X = pd.DataFrame({"value": np.arange(60, dtype=float),
                      "category": pd.Categorical(["01", "02", None] * 20)})
    y = pd.Series(pd.Categorical(["yes", "no"] * 30), name="class")
    b = SimpleNamespace(data=X, target=y, details={
        "id": "31", "name": "credit-g", "version": "1", "file_id": "31",
        "default_target_attribute": "class", "md5_checksum": "a" * 32,
        "licence": "Public", "status": "active", "url": "https://openml.org/data/v1/download/31",
    })
    monkeypatch.setattr(datasets.skd, "fetch_openml", lambda **kwargs: b)
    return b


def test_loader_records_identity_schema_missingness_and_label_order(bundle):
    X, y, task = datasets.load("credit-g")
    source = X.attrs["quibble_source"]
    assert task == datasets.C and y.tolist() == [1, 0] * 30
    assert source["class_labels"] == ["no", "yes"]
    assert source["identity_complete"] is True
    assert source["data_id"] == 31 and source["file_id"] == "31"
    assert source["md5_checksum"] == "a" * 32
    assert len(source["sha256"]) == len(source["fingerprint"]) == 64
    assert X["category"].dtype == bundle.data["category"].dtype
    assert X["category"].isna().sum() == 20
    assert source["fingerprint"] == datasets.source_info("credit-g")["fingerprint"]


@pytest.mark.parametrize("change", ["values", "target", "order", "categories", "version", "checksum"])
def test_fingerprint_detects_loaded_content_and_upstream_identity_changes(bundle, change):
    before = datasets.source_info("credit-g")["fingerprint"]
    if change == "values":
        bundle.data.loc[0, "value"] = 0.0001
    elif change == "target":
        bundle.target.iloc[0] = "no"
    elif change == "order":
        bundle.data = bundle.data.iloc[::-1].reset_index(drop=True)
        bundle.target = bundle.target.iloc[::-1].reset_index(drop=True)
    elif change == "categories":
        bundle.data["category"] = bundle.data["category"].cat.reorder_categories(["02", "01"])
    elif change == "version":
        bundle.details["version"] = "2"
    else:
        bundle.details["md5_checksum"] = "b" * 32
    assert datasets.source_info("credit-g")["fingerprint"] != before


def test_fingerprint_distinguishes_missing_from_empty_string(bundle):
    bundle.data["category"] = bundle.data["category"].astype(object)
    before = datasets.source_info("credit-g")["sha256"]
    bundle.data.loc[2, "category"] = ""
    assert datasets.source_info("credit-g")["sha256"] != before


@pytest.mark.parametrize("change", ["name", "id", "alignment", "multitarget", "missing",
                                   "target_name", "target_in_features", "infinity", "one_class"])
def test_ordinary_loading_rejects_bad_public_data_not_only_verification(bundle, change):
    if change in ("name", "id"):
        bundle.details[change] = "unexpected"
    elif change == "alignment":
        bundle.target.index = bundle.target.index + 1
    elif change == "multitarget":
        bundle.target = bundle.target.to_frame()
    elif change == "missing":
        bundle.target.iloc[0] = np.nan
    elif change == "target_name":
        bundle.target.name = "another_target"
    elif change == "target_in_features":
        bundle.data["class"] = bundle.target
    elif change == "infinity":
        bundle.data.loc[0, "value"] = np.inf
    else:
        bundle.target[:] = "yes"
    with pytest.raises(ValueError):
        datasets.load("credit-g")


@pytest.mark.parametrize("values", [["1", "2"] * 30, ["1", "bad"] * 30, [1, np.inf] * 30])
def test_regression_uses_declared_numeric_target_without_relabeling(bundle, values):
    bundle.details.update(id="183", name="abalone")
    bundle.target = pd.Series(values, name="class")
    if "bad" in values or np.inf in values:
        with pytest.raises(ValueError, match="finite numeric"):
            datasets.load("abalone")
    else:
        X, y, task = datasets.load("abalone")
        assert task == datasets.R and y.tolist() == [1.0, 2.0] * 30
        assert "class_labels" not in X.attrs["quibble_source"]


def test_verification_uses_normal_loader_and_records_failures(bundle, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["datasets", "--verify", "--names", "credit-g"])
    assert datasets.main() == 0
    metadata = json.loads((tmp_path / "datasets_meta.json").read_text())
    assert metadata["credit-g"]["dataset_source"]["class_labels"] == ["no", "yes"]
    bundle.details["name"] = "unexpected"
    assert datasets.main() == 1
    assert "mismatch" in json.loads((tmp_path / "datasets_meta.json").read_text())["credit-g"]["error"]
    def failed(**kwargs):
        raise URLError("download unavailable")
    monkeypatch.setattr(datasets.skd, "fetch_openml", failed)
    assert datasets.main() == 1
    assert "download unavailable" in (tmp_path / "datasets_meta.json").read_text()


def test_verify_respects_cache_root_when_data_home_is_blank(bundle, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = []
    def fetched(**kwargs):
        seen.append(kwargs["data_home"])
        return bundle
    monkeypatch.setattr(datasets.skd, "fetch_openml", fetched)
    monkeypatch.setattr("sys.argv", ["datasets", "--verify", "--names", "credit-g"])
    assert datasets.main() == 0
    assert seen == [str(tmp_path / "cache" / "openml")]


def test_incomplete_upstream_identity_is_explicit(bundle):
    del bundle.details["md5_checksum"]
    assert datasets.source_info("credit-g")["identity_complete"] is False


def test_changed_openml_source_cannot_mix_with_saved_evidence(bundle, tmp_path):
    out = tmp_path / "results"
    directory = out / "credit-g"
    directory.mkdir(parents=True)
    saved = datasets.source_info("credit-g")
    (directory / "histgb_ss0_ms0.json").write_text(json.dumps({"dataset_source": saved}))
    datasets.validate_sources(out, "credit-g", saved)
    bundle.data.loc[0, "value"] = 9
    with pytest.raises(ValueError, match="OpenML dataset source differs"):
        datasets.validate_sources(out, "credit-g", datasets.source_info("credit-g"))


def test_public_source_survives_fitting_analysis_and_report(bundle, monkeypatch, tmp_path):
    from quibble_ml.config import Config
    from quibble_ml.report import build
    from quibble_ml.run import run_one
    out = tmp_path / "results"
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    monkeypatch.setenv("QUIBBLE_BOOT", "20")
    cfg = Config.from_env()
    for backend in ("histgb", "weak"):
        run_one(backend, "credit-g", 0, cfg)
        record = json.loads((out / "credit-g" / f"{backend}_ss0_ms0.json").read_text())
        assert "error" not in record
        assert record["dataset_source"]["class_labels"] == ["no", "yes"]
    path = build(str(out), ["credit-g"], swaps=[("histgb", "weak")])
    report = path.read_text(encoding="utf-8")
    assert "### OpenML data source" in report
    assert "Upstream ARFF MD5" in report and "Loaded-content SHA-256" in report
    assert "### Local data source" not in report
    analysis = json.loads((out / "credit-g" / "analysis" / "analysis.json").read_text())
    assert analysis["dataset_source"]["file_id"] == "31"
