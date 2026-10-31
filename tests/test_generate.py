"""Synthetic specification contracts, reproducibility, and local-dataset integration."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quibble_ml import datasets
from quibble_ml.generate import generate, main, preview, recommendation, save, validate_spec


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_reproducible_exact_shape_and_preview(task):
    spec = recommendation(rows=200, fields=6, seed=18, task=task)
    a = generate(spec)
    pd.testing.assert_frame_equal(a, generate(copy.deepcopy(spec)))
    assert a.shape == (200, 7)
    assert list(a) == [f["name"] for f in spec["features"]] + [spec["target"]]
    assert preview(spec)["sample"] == json.loads(a.head(20).to_json(orient="records"))
    changed = {**spec, "seed": 19}
    assert not generate(changed).equals(a)
    assert preview(spec)["predictors"] == 6


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(rows=True), lambda s: s.update(rows=49), lambda s: s.update(rows=50001),
    lambda s: s.update(seed=-1), lambda s: s.update(version=2),
    lambda s: s.update(name="../escape"), lambda s: s.update(name="CON"),
    lambda s: s.update(name="wine"), lambda s: s.update(target=s["features"][0]["name"]),
    lambda s: s["features"][0].update(weight=float("nan")),
    lambda s: s["features"][0].update(missing_rate=0.9),
    lambda s: s["features"][0].update(distribution="eval"),
    lambda s: s["features"][0].update(effect="none", weight=2),
    lambda s: s["features"][0].update(minimum=10, maximum=1),
    lambda s: s["features"][0].update(code="print('not executed')"),
    lambda s: s.update(interactions=[{"left": "unknown", "right": "unknown", "weight": 1}]),
    lambda s: s.update(features=s["features"] * 20),
    lambda s: s.update(extra="unsupported"),
])
def test_invalid_specs_fail_explicitly(mutation):
    spec = recommendation()
    mutation(spec)
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_resource_limit():
    with pytest.raises(ValueError, match="limits"):
        recommendation(50000, 100)


def test_categories_missingness_and_interactions():
    spec = recommendation(rows=300, fields=3, task="regression")
    spec["features"][0].update(distribution="categorical", categories=["urban", "rural"],
                               missing_rate=0.2)
    spec["features"][1].update(distribution="integer", effect="threshold")
    spec["features"][2].update(distribution="normal", effect="quadratic")
    spec["interactions"] = [{"left": spec["features"][0]["name"],
                              "right": spec["features"][1]["name"], "weight": 1}]
    frame = generate(spec)
    assert 20 < frame.iloc[:, 0].isna().sum() < 100
    assert set(frame.iloc[:, 0].dropna()) == {"urban", "rural"}
    assert frame[spec["target"]].notna().all()
    assert np.isfinite(frame[spec["target"]]).all()
    assert set(frame) == {f["name"] for f in spec["features"]} | {spec["target"]}


def test_extreme_target_rejected_without_retry():
    spec = recommendation()
    spec["outcome"]["intercept"] = 10000
    with pytest.raises(ValueError, match="10 rows each"):
        generate(spec)
    spec = recommendation(task="regression")
    spec["outcome"]["noise"] = 0
    for f in spec["features"]:
        f.update(effect="none", weight=0)
    with pytest.raises(ValueError, match="constant"):
        generate(spec)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_save_load_fingerprint_no_overwrite(tmp_path, monkeypatch, task):
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path))
    spec = recommendation(rows=100, fields=5, task=task)
    result = save(spec)
    directory = tmp_path / spec["name"]
    X, y, actual_task = datasets.load(spec["name"])
    assert X.shape == (100, 5) and len(y) == 100 and actual_task == task
    record = json.loads((directory / "generation.json").read_text())
    assert record["synthetic"] is True
    assert record["specification"] == spec
    assert record["sha256"] == hashlib.sha256((directory / "data.csv").read_bytes()).hexdigest() == result["sha256"]
    assert set(X) == {f["name"] for f in spec["features"]}
    assert X.attrs["quibble_source"]["sha256"] == record["sha256"]
    assert X.attrs["quibble_source"]["synthetic"]["seed"] == spec["seed"]
    before = (directory / "data.csv").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        save(spec)
    assert (directory / "data.csv").read_bytes() == before
    assert list(tmp_path.iterdir()) == [directory]


def test_save_failure_leaves_no_discoverable_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path))
    def fail(*args):
        raise ValueError("loader rejected")
    monkeypatch.setattr(datasets, "load", fail)
    with pytest.raises(ValueError, match="loader rejected"):
        save(recommendation())
    assert not list(tmp_path.iterdir())


def test_changed_generation_metadata_changes_source_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path))
    spec = recommendation(rows=100)
    save(spec)
    original = datasets.source_info(spec["name"])
    path = tmp_path / spec["name"] / "generation.json"
    metadata = json.loads(path.read_text())
    metadata["specification"]["assumptions"].append("Changed explanation.")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert datasets.source_info(spec["name"])["fingerprint"] != original["fingerprint"]
    data = tmp_path / spec["name"] / "data.csv"
    data.write_bytes(data.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metadata does not match"):
        datasets.load(spec["name"])


def test_cli_example_and_save(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("sys.argv", ["quibble-ml generate", "--example"])
    assert main() == 0
    spec = json.loads(capsys.readouterr().out)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr("sys.argv", ["quibble-ml generate", "--spec", str(spec_path)])
    assert main() == 0
    assert datasets.known(spec["name"])


def test_cli_rejects_ambiguous_generation_source():
    from quibble_ml.generate import create_parser
    with pytest.raises(SystemExit) as caught:
        create_parser().parse_args(["--example", "--spec", "ignored.json"])
    assert caught.value.code == 2
