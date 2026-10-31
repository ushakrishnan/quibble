"""Dated snapshots preserve old predictions, reports, figures, and relative links."""
import json
import os
import re
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from quibble_ml.config import Config
from quibble_ml.history import snapshot
from quibble_ml.run import run_one, stem, sweep


def saved_outputs(out, version=1, report_dir=None):
    directory = out / "wine"
    analysis = directory / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    record = {"backend": "histgb", "dataset": "wine", "split_seed": 0, "model_seed": 0,
              "run_id": f"run-{version}", "version": version}
    (directory / "histgb_ss0_ms0.json").write_text(json.dumps(record))
    np.save(directory / "histgb_ss0_ms0.pred.npy", np.full((2, 2), version))
    np.save(directory / "y_test_ss0.npy", np.array([0, version]))
    np.save(directory / "idx_test_ss0.npy", np.array([0, version]))
    csv = analysis / "disagreements_histgb_vs_lightgbm.csv"
    csv.write_text(f"version\n{version}\n")
    report_dir = report_dir or out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    link = os.path.relpath(csv, report_dir)
    (report_dir / "report.md").write_text(
        f"# Report {version}\n\n![Quality](fig_wine_quality_cost.png)\n[Rows]({link})\n",
        encoding="utf-8")
    (report_dir / "fig_wine_quality_cost.png").write_bytes(f"figure {version}".encode())
    return directory


def manifests(out):
    return list((out / "runs").glob("*/*/snapshot.json"))


def test_empty_results_do_not_create_empty_snapshots(tmp_path):
    assert snapshot(tmp_path / "missing", "before-run") is None
    assert not (tmp_path / "missing").exists()


def test_snapshots_keep_older_results_and_reports_without_recursive_history(tmp_path):
    out = tmp_path / "results"
    saved_outputs(out, 1)
    legacy = out / "report" / "legacy"
    legacy.mkdir()
    (legacy / "old.md").write_text("legacy")
    legacy_report = legacy / "demo_results" / "report" / "report.md"
    legacy_report.parent.mkdir(parents=True)
    legacy_report.write_text("Legacy report")
    cache = out / ".cache"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"not results")
    first = snapshot(out, "report")
    original = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    saved_outputs(out, 2)
    second = snapshot(out, "report")
    assert first != second and first.parent == second.parent
    assert len(manifests(out)) == 2
    for relative, content in original.items():
        assert (first / relative).read_bytes() == content
    np.testing.assert_array_equal(np.load(first / "wine" / "histgb_ss0_ms0.pred.npy"), 1)
    np.testing.assert_array_equal(np.load(second / "wine" / "histgb_ss0_ms0.pred.npy"), 2)
    for archive in (first, second):
        assert not (archive / "runs").exists()
        assert not (archive / "report" / "legacy").exists()
        assert not (archive / ".cache").exists()
        meta = json.loads((archive / "snapshot.json").read_text())
        assert meta["datasets"] == ["wine"] and meta["run_ids"]
        report = archive / "report" / "report.md"
        for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
            assert (report.parent / target).exists(), target
    index = (out / "report" / "history.md").read_text()
    assert first.name in index and second.name in index
    assert "[demo_results](legacy/demo_results/report/report.md)" in index


def test_external_report_directory_has_self_contained_archived_links(tmp_path):
    out, report_dir = tmp_path / "results", tmp_path / "external"
    saved_outputs(out, report_dir=report_dir)
    archive = snapshot(out, "report", report_dir)
    report = archive / "report" / "report.md"
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        assert (report.parent / target).exists(), target
    assert "../wine/analysis/" in report.read_text()


def test_archived_report_does_not_link_to_uncopied_history(tmp_path):
    saved_outputs(tmp_path)
    report = tmp_path / "report" / "report.md"
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\nArchived copy: [old](../runs/old/report/report.md). "
                     "[Run history](history.md).\n")
    archive = snapshot(tmp_path, "before-sweep")
    text = (archive / "report" / "report.md").read_text()
    assert "Archived copy:" not in text and "Run history" not in text
    assert "[Rows]" in text


def test_raw_snapshot_does_not_attach_an_older_report(tmp_path):
    saved_outputs(tmp_path)
    archive = snapshot(tmp_path, "sweep", include_report=False)
    assert (archive / "wine" / "histgb_ss0_ms0.json").exists()
    assert not (archive / "report").exists()
    assert "snapshot.json" in (tmp_path / "report" / "history.md").read_text()


def test_archive_failure_stops_before_changing_originals(monkeypatch, tmp_path):
    from quibble_ml import history
    saved_outputs(tmp_path)
    original = (tmp_path / "wine" / "histgb_ss0_ms0.json").read_bytes()

    def fail(*args, **kwargs):
        raise PermissionError("cannot archive")

    monkeypatch.setattr(history.shutil, "copytree", fail)
    with pytest.raises(PermissionError, match="cannot archive"):
        snapshot(tmp_path, "before-run")
    assert (tmp_path / "wine" / "histgb_ss0_ms0.json").read_bytes() == original
    assert not manifests(tmp_path)


def test_force_preserves_previous_run_and_cached_repeat_does_not_retrain(monkeypatch, tmp_path):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path))
    cfg = Config.from_env()
    run_one("histgb", "wine", 0, cfg)
    path = stem(tmp_path, "wine", "histgb", 0, 0).with_suffix(".json")
    first = json.loads(path.read_text())
    assert first["run_id"] and first["timestamp"]
    before = len(manifests(tmp_path))
    run_one("histgb", "wine", 0, cfg)
    assert len(manifests(tmp_path)) == before
    assert json.loads(path.read_text())["run_id"] == first["run_id"]
    first["marker"] = "previous"
    path.write_text(json.dumps(first))
    old_prediction = path.with_suffix(".pred.npy").read_bytes()
    run_one("histgb", "wine", 0, cfg, force=True)
    assert json.loads(path.read_text())["run_id"] != first["run_id"]
    previous = [p.parent for p in manifests(tmp_path)
                if json.loads(p.read_text())["reason"] == "before-run"]
    assert len(previous) == 1
    archived = previous[0] / "wine" / path.name
    assert json.loads(archived.read_text())["marker"] == "previous"
    assert archived.with_suffix(".pred.npy").read_bytes() == old_prediction


def test_timeout_has_provenance_and_preserves_old_results(monkeypatch, tmp_path):
    from quibble_ml import run
    saved_outputs(tmp_path)
    cfg = replace(Config.from_env(), out_dir=str(tmp_path), model_seeds=(0,), timeout_s=1)
    from quibble_ml.provenance import recipe_snapshot
    path = tmp_path / "wine" / "histgb_ss0_ms0.json"
    previous = json.loads(path.read_text())
    previous["recipe"] = recipe_snapshot("histgb", cfg, 0)
    path.write_text(json.dumps(previous))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(run, "_worker_process", timeout)
    sweep(["histgb"], ["wine"], cfg, True)
    record = json.loads((tmp_path / "wine" / "histgb_ss0_ms0.json").read_text())
    assert record["error"] == "timeout after 1s"
    assert record["run_id"] and record["timestamp"]
    old = next(p.parent for p in manifests(tmp_path)
               if json.loads(p.read_text())["reason"] == "before-sweep")
    assert json.loads((old / "wine" / "histgb_ss0_ms0.json").read_text())["version"] == 1
