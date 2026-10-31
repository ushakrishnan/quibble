"""Saved-evidence HTTP contracts, real CPU runs and browser asset boundaries."""
import hashlib
import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from quibble_ml import generate
from quibble_ml.thresholds import DEFAULTS
from quibble_ml.web import ASSETS, Server, Workbench


@pytest.fixture(scope="module")
def saved_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("saved")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QUIBBLE_OUT_DIR", str(root / "results"))
        patch.setenv("QUIBBLE_DATA_DIR", str(root / "data"))
        patch.setenv("QUIBBLE_N_BOOT", "100")
        spec = generate.recommendation(rows=100, fields=4, seed=42, task="classification")
        generate.save(spec)
        app = Workbench()
        identifiers = []
        try:
            for _ in range(2):
                job = app.start({"dataset": spec["name"], "backends": ["histgb", "linear"],
                                 "mode": "full", "device": "cpu", "cv": "holdout",
                                 "full_confirmed": True, "tier": "defaults"})
                app.thread.join(120)
                assert not app.thread.is_alive()
                finished = app.job(job["id"])
                assert finished["status"] == "completed", finished["log"]
                identifiers.append(job["id"])
        finally:
            app.close()
    return root, identifiers, spec["name"]


@pytest.fixture
def evidence(saved_runs, tmp_path, monkeypatch):
    root, identifiers, dataset = saved_runs
    shutil.copytree(root / "results", tmp_path / "results")
    shutil.copytree(root / "data", tmp_path / "data")
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "data"))
    app = Workbench()
    server = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def request(route, body=None, headers=None):
        base = f"http://127.0.0.1:{server.server_port}"
        selected = {"Content-Type": "application/json", "X-Quibble-Token": server.token}
        selected.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + route, data=data, headers=selected)
        with urllib.request.urlopen(req, timeout=150) as response:
            return json.loads(response.read())

    yield app, identifiers, dataset, request, tmp_path
    server.shutdown()
    server.server_close()
    thread.join()
    app.close()


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file() and "verifications" not in p.parts
            and p.name != ".workbench.lock"}


def verification(ids):
    return {"job_id": ids[0], "incumbent": "linear", "candidate": "histgb"}


def rewrite(path, key, value):
    record = json.loads(path.read_text(encoding="utf-8"))
    record[key] = value
    path.write_text(json.dumps(record), encoding="utf-8")


def test_real_saved_swap_and_gate_are_separate_immutable_artifacts(evidence):
    app, ids, _, request, _ = evidence
    before = hashes(app.root)
    for kind in ("swap", "gate"):
        result = request(f"/api/verify/{kind}", verification(ids))
        assert result["status"] == "completed", result
        assert result["result"]["overall"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert result["policy"] == DEFAULTS
        assert result["coverage"]["linear"]["required"] == [0]
        path = app.root / ids[0] / "verifications" / (result["id"] + ".json")
        assert json.loads(path.read_text()) == result
    assert hashes(app.root) == before
    assert len(request(f"/api/jobs/{ids[0]}")["verifications"]) == 2


def test_unknown_report_sections_and_duplicate_headings_are_preserved(evidence):
    app, ids, _, request, _ = evidence
    report = app.root / ids[0] / "output" / "report" / "report.md"
    text = "# Future evidence\n\nKeep me.\n\n## Unknown section\n\nfirst\n\n## Unknown section\n\nlast\n"
    report.write_text(text, encoding="utf-8")
    job = request(f"/api/jobs/{ids[0]}")
    assert job["report"] == text
    assert [section["id"] for section in job["report_sections"]] == [
        "future-evidence", "unknown-section", "unknown-section-1"]
    assert "Keep me." in job["report_html"] and "last" in job["report_html"]


def test_diagnostics_are_explicit_and_do_not_collect_secrets(evidence, monkeypatch):
    _, ids, _, request, _ = evidence
    monkeypatch.setenv("API_KEY", "DO-NOT-EXPOSE-CREDENTIAL")
    monkeypatch.setenv("QUIBBLE_PRIVATE_REGISTRY_KEY", "DO-NOT-EXPOSE-PRIVATE-ROWS")
    monkeypatch.setattr("quibble_ml.env_info.collect", lambda *a, **k: pytest.fail("No path/environment collector"))
    request(f"/api/jobs/{ids[0]}")
    result = request("/api/environment", {"requested": True})
    text = json.dumps(result)
    assert result["collected_at"] and result["packages"]
    assert "DO-NOT-EXPOSE" not in text
    assert not ({"cache_dir", "cpu", "gpu", "environment", "provider", "credential"} & result.keys())
    with pytest.raises(urllib.error.HTTPError) as error:
        request("/api/environment", {})
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError):
        request("/api/environment")


@pytest.mark.parametrize("route,body", [
    ("/api/environment", {"requested": True}),
    ("/api/verify/swap", {}),
    ("/api/compare-jobs", {"job_ids": ["../outside", "bad"]}),
])
@pytest.mark.parametrize("headers", [
    {"X-Quibble-Token": "invalid"}, {"Origin": "https://outside.invalid"},
])
def test_new_post_routes_preserve_token_origin_boundary(evidence, route, body, headers):
    *_, request, _ = evidence
    with pytest.raises(urllib.error.HTTPError) as error:
        request(route, body, headers)
    assert error.value.code == 403


@pytest.mark.parametrize("damage", ["predictions", "source", "labels", "rows", "recipe", "partial", "smoke"])
def test_verification_refuses_incomplete_or_incompatible_evidence(evidence, damage):
    app, ids, dataset, request, tmp = evidence
    out = app.root / ids[0] / "output"
    cells = out / dataset
    if damage == "predictions":
        (cells / "histgb_ss0_ms0.pred.npy").unlink()
    elif damage == "source":
        csv = tmp / "data" / dataset / "data.csv"
        csv.write_bytes(csv.read_bytes() + b"\n")
    elif damage in ("labels", "rows"):
        path = cells / ("y_test_ss0.npy" if damage == "labels" else "idx_test_ss0.npy")
        values = np.load(path)
        np.save(path, values[::-1])
    elif damage == "recipe":
        path = cells / "histgb_ss0_ms0.json"
        record = json.loads(path.read_text())
        record["recipe"]["requested"]["cv_seed"] = 99
        path.write_text(json.dumps(record))
    elif damage == "partial":
        invocation = json.loads((out / "invocation.json").read_text())
        invocation["cells"].append({"dataset": dataset, "backend": "histgb",
                                    "split_seed": 1, "model_seed": 0})
        (out / "invocation.json").write_text(json.dumps(invocation))
    else:
        rewrite(app.root / ids[0] / "job.json", "mode", "smoke")
    with pytest.raises(urllib.error.HTTPError) as error:
        request("/api/verify/swap", verification(ids))
    assert error.value.code == 400
    assert not list((app.root / ids[0]).glob("verifications/*.json"))


def test_changed_policy_requires_separate_provenance_not_request_override(evidence):
    app, ids, _, request, _ = evidence
    with pytest.raises(urllib.error.HTTPError):
        request("/api/verify/swap", {**verification(ids), "thresholds": {"max_quality_drop": 999}})
    changed = {**DEFAULTS, "max_quality_drop": DEFAULTS["max_quality_drop"] + 0.1}
    rewrite(app.root / ids[1] / "job.json", "thresholds", changed)
    (app.root / ids[1] / "thresholds.json").write_text(json.dumps(changed))
    comparison = request("/api/compare-jobs", {"job_ids": ids})
    assert comparison["compatibility"] == "differences"
    assert "Different policy_sha256." in comparison["differences"]
    assert comparison["paired_verdict"] is None


@pytest.mark.parametrize("damage", ["stored-id", "policy-file", "hidden-method"])
def test_saved_identity_policy_and_requested_method_contracts_are_not_bypassable(evidence, damage):
    app, ids, _, request, _ = evidence
    directory = app.root / ids[0]
    if damage == "stored-id":
        rewrite(directory / "job.json", "id", "../outside")
    elif damage == "policy-file":
        (directory / "thresholds.json").unlink()
    else:
        rewrite(directory / "job.json", "backends", ["linear"])
    with pytest.raises(urllib.error.HTTPError) as error:
        request("/api/verify/swap", verification(ids))
    assert error.value.code == 400
    assert not list(directory.glob("verifications/*.json"))


def test_two_real_default_jobs_match_descriptively_without_claiming_winner(evidence):
    _, ids, _, request, _ = evidence
    comparison = request("/api/compare-jobs", {"job_ids": ids})
    assert comparison["compatibility"] == "matched", comparison
    assert comparison["presentation"] == "descriptive only"
    assert comparison["paired_verdict"] is None
    assert len(comparison["jobs"]) == 2
    assert all(job["cells"] and job["coverage"] for job in comparison["jobs"])


@pytest.mark.parametrize("damage", ["source", "smoke", "splits", "labels", "partial"])
def test_comparison_explains_mismatch_without_paired_claim(evidence, damage):
    app, ids, dataset, request, _ = evidence
    directory = app.root / ids[1]
    if damage == "source":
        record = json.loads((directory / "job.json").read_text())
        record["dataset_source"]["fingerprint"] = "different-source"
        (directory / "job.json").write_text(json.dumps(record))
    elif damage == "smoke":
        rewrite(directory / "job.json", "mode", "smoke")
    elif damage in ("splits", "labels"):
        path = directory / "output" / dataset / (
            "idx_test_ss0.npy" if damage == "splits" else "y_test_ss0.npy")
        np.save(path, np.load(path)[::-1])
    else:
        (directory / "output" / dataset / "linear_ss0_ms0.pred.npy").unlink()
    result = request("/api/compare-jobs", {"job_ids": ids})
    assert result["compatibility"] != "matched"
    assert result["differences"]
    assert result["paired_verdict"] is None
    if damage == "source":
        assert result["presentation"] == "configuration only"
        assert all("metrics" not in cell for job in result["jobs"] for cell in job["cells"])


def test_evaluator_failure_is_error_not_scientific_verdict(evidence, monkeypatch):
    app, ids, _, request, _ = evidence
    def failed(*args, **kwargs):
        raise OSError("private registry credential must not be returned")
    monkeypatch.setattr("quibble_ml.web.subprocess.run", failed)
    result = request("/api/verify/gate", verification(ids))
    assert result["status"] == "error"
    assert "result" not in result and "credential" not in json.dumps(result)
    assert (app.root / ids[0] / "verifications" / (result["id"] + ".json")).is_file()


@pytest.mark.parametrize("identifier", ["../outside", "/absolute", "a%2fb", "not-a-job"])
def test_evidence_rejects_arbitrary_paths(evidence, identifier):
    _, ids, _, request, _ = evidence
    with pytest.raises(urllib.error.HTTPError) as error:
        request("/api/compare-jobs", {"job_ids": [identifier, ids[1]]})
    assert error.value.code == 400


def test_new_ui_actions_are_explicit_and_preserve_full_report():
    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    markup = (ASSETS / "index.html").read_text(encoding="utf-8")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    assert js.count('api("/api/environment"') == 1
    assert '$("collect-environment").onclick' in js
    assert 'job.report_sections || []' in js
    assert '$("report").innerHTML = job.report_html' in js
    assert "reportText = job.report" in js and 'window.print()' in js
    for page in ("verify", "compare"):
        assert f'href="#page={page}"' in markup
        assert f'id="{page}-path"' in markup
    assert "result.presentation === \"configuration only\"" in js
    assert "result.paired_verdict" not in js
    assert "#report pre" in css and "max-height: none" in css
