"""Code/weight distinctions, immutable recipe evidence, and exported snapshots."""
import json
import shutil
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from quibble_ml.backends import BACKENDS, Backend
from quibble_ml.config import Config
from quibble_ml.provenance import (
    backend_metadata, check_reuse, license_label, recipe_hashes, recipe_snapshot,
    validate_pool_recipes, validate_recipe,
)
from quibble_ml.run import run_one, stem, sweep


def record(recipe, **extra):
    cell = recipe["requested"]["cell"]
    return {"backend": recipe["requested"]["backend"], **cell, "recipe": recipe, **extra}


def test_registry_covers_all_backends_without_downloads():
    for name in BACKENDS:
        metadata = backend_metadata(name)
        assert metadata["library"]["license"] != "unknown"
        assert metadata["library"]["source_url"].startswith("https://")
        assert metadata["reviewed"] and metadata["verification"]
        assert {"revision", "sha256", "restrictions", "identity_status"} <= metadata["weights"].keys()
    for name in ("tabfm", "tabpfn"):
        metadata = backend_metadata(name)
        assert metadata["library"]["license"] == "Apache-2.0"
        assert "non-commercial" in metadata["weights"]["restrictions"].lower()
        assert metadata["weights"]["revision"] is None
        assert metadata["weights"]["sha256"] is None
        assert "Reference" in metadata["weights"]["terms_scope"]


def test_unregistered_provenance_is_explicitly_unknown(monkeypatch):
    monkeypatch.setitem(BACKENDS, "custom", Backend)
    metadata = backend_metadata("custom")
    assert metadata["library"]["license"] == "unknown"
    assert metadata["weights"]["license"] == "unknown"
    assert metadata["verification"] == "unverified custom backend"
    assert metadata["reviewed"] is None


def test_normalized_recipe_distinguishes_cells_not_comparable_family():
    cfg = Config.from_env()
    first = recipe_snapshot("histgb", cfg, 0)
    second = recipe_snapshot("histgb", replace(cfg, split_seed=1), 2)
    assert first["sha256"] != second["sha256"]
    assert first["family_sha256"] == second["family_sha256"]
    assert first["algorithm_sha256"] == second["algorithm_sha256"]
    validate_pool_recipes([record(first), record(second)])


@pytest.mark.parametrize("field,value", [
    ("tier", "tuned"),
    ("checkpoint", {"checkpoint_id": "different", "revision": "new", "sha256": "new"}),
    ("search", {"version": 2, "inner_splits": 3}),
    ("adaptation", "new ensemble"),
    ("packages", {"scikit-learn": "different"}),
])
def test_changed_recipe_requires_separate_output(field, value):
    original = recipe_snapshot("histgb", Config.from_env(), 0)
    changed = deepcopy(original)
    changed["requested"][field] = value
    changed.update(recipe_hashes(changed["requested"]))
    with pytest.raises(ValueError, match="Recipe mismatch"):
        check_reuse(record(original), changed)
    with pytest.raises(ValueError, match="Mixed recipe"):
        validate_pool_recipes([record(original), record(changed)])


def test_preprocessing_changes_only_reuse_cached_evidence_with_warning_policy():
    cfg = Config.from_env()
    old = recipe_snapshot("histgb", cfg, 0)
    new = recipe_snapshot("histgb", replace(cfg, preprocess="native"), 0)
    check_reuse(record(old), new)
    with pytest.raises(ValueError, match="Recipe family mismatch"):
        check_reuse(record(old), new, writing=True)


def test_legacy_is_readable_but_not_silently_upgraded():
    current = recipe_snapshot("histgb", Config.from_env(), 0)
    legacy = {"backend": "histgb", "split_seed": 0, "model_seed": 0}
    assert "legacy" in validate_pool_recipes([legacy])
    assert "legacy" in license_label(None)
    check_reuse(legacy, current)
    with pytest.raises(ValueError, match="Legacy recipe"):
        check_reuse(legacy, current, writing=True)
    with pytest.raises(ValueError, match="Mixed legacy"):
        validate_pool_recipes([legacy, record(current)])
    check_reuse({**legacy, "error": "dependency unavailable"}, current, writing=True)


def test_digest_and_cell_identity_are_checked():
    recipe = recipe_snapshot("histgb", Config.from_env(), 0)
    recipe["requested"]["tier"] = "edited"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_recipe(recipe)
    recipe.update(recipe_hashes(recipe["requested"]))
    with pytest.raises(ValueError, match="cell identity"):
        validate_pool_recipes([record(recipe, model_seed=3)])


def test_success_refusal_and_failure_retain_snapshots(monkeypatch, tmp_path):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("QUIBBLE_TABPFN_MAX_ROWS", "1")
    cfg = Config.from_env()
    run_one("histgb", "wine", 0, cfg)
    run_one("tabpfn", "wine", 0, cfg)

    class Broken(Backend):
        def fit(self, *args):
            raise RuntimeError("intentional failure")
    monkeypatch.setitem(BACKENDS, "broken", Broken)
    run_one("broken", "wine", 0, cfg)
    for name, outcome in (("histgb", None), ("tabpfn", "refused"), ("broken", "error")):
        saved = json.loads(stem(tmp_path, "wine", name, 0, 0).with_suffix(".json").read_text())
        assert saved["backend_metadata"] == backend_metadata(name)
        validate_recipe(saved["recipe"])
        if outcome:
            assert outcome in saved
        else:
            assert saved["recipe"]["resolved"]["parameters"]["random_state"] == 0
            assert saved["recipe"]["resolved"]["context"]["test_indices_sha256"]
            edited = deepcopy(saved["recipe"])
            edited["resolved"]["parameters"]["max_iter"] = 1
            with pytest.raises(ValueError, match="Resolved recipe"):
                validate_recipe(edited)


def test_worker_failure_and_exclusions_keep_metadata(monkeypatch, tmp_path):
    from quibble_ml import run
    import subprocess
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    monkeypatch.setattr(run, "_worker_process",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("worker", 1)))
    invocation = sweep(["histgb"], ["wine"], Config.from_env(), False)
    saved = json.loads(stem(tmp_path, "wine", "histgb", 0, 0).with_suffix(".json").read_text())
    assert saved["error_kind"] == "timeout"
    assert saved["backend_metadata"] == invocation["backend_metadata"]["histgb"]
    validate_recipe(saved["recipe"])
    gpu = replace(Config.from_env(), device="cuda", device_requested="cuda",
                  out_dir=str(tmp_path / "gpu"))
    invocation = sweep(["histgb", "catboost"], ["wine"], gpu, False)
    assert "histgb" in invocation["excluded_backends"]
    assert invocation["backend_metadata"]["histgb"] == backend_metadata("histgb")
    assert invocation["backend_metadata"]["catboost"] == backend_metadata("catboost")


def test_report_csv_json_and_archive_preserve_runtime_metadata(cv_results, tmp_path, monkeypatch):
    from quibble_ml.report import build
    out = tmp_path / "out"
    shutil.copytree(cv_results, out)
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    build(out, datasets=["breast_cancer"])
    report = out / "report"
    participants = pd.read_csv(report / "participation.csv")
    histgb = participants[participants["backend"] == "histgb"].iloc[0]
    assert histgb["license source"] == "runtime snapshot"
    assert "BSD-3-Clause" in histgb["license"]
    full = json.loads((report / "backend_provenance.json").read_text())
    original = json.loads(stem(out, "breast_cancer", "histgb", 0, 0).with_suffix(".json").read_text())
    exported = next(r for r in full["runs"]["breast_cancer"]
                    if r["backend"] == "histgb" and r["split_seed"] == 0)
    assert exported["backend_metadata"] == original["backend_metadata"]
    assert exported["recipe"] == original["recipe"]
    analysis = json.loads((out / "breast_cancer" / "analysis" / "analysis.json").read_text())
    assert analysis["backend_provenance"]["histgb"][0]["recipe"] == original["recipe"]
    archived = list((out / "runs").glob("*/*/report/backend_provenance.json"))
    assert archived and json.loads(archived[-1].read_text()) == full
    assert archived[-1].with_name("participation.csv").read_bytes() == (report / "participation.csv").read_bytes()
