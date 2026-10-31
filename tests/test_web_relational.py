"""Offline relational browser intake, atomic publication and review guards."""

import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from quibble_ml import datasets, relational
from quibble_ml.relational_synthetic import write_fixture
from quibble_ml.web import Server, Workbench


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "saved"))
    monkeypatch.delenv("QUIBBLE_RELATIONAL_REGISTRY", raising=False)
    monkeypatch.delenv("QUIBBLE_RELATIONAL_CACHE", raising=False)
    instance = Workbench()
    yield instance
    instance.close()


def stage(app, tmp_path, name="browser_relational"):
    registry = write_fixture(tmp_path / ("source_" + name))
    spec = json.loads((registry.parent / "manifest.json").read_text(encoding="utf-8"))
    identifier = app.action("/api/relational/start", {})["stage_id"]
    for table, entry in spec["tables"].items():
        path = registry.parent / entry["path"]
        raw = path.read_bytes()
        app.relational_uploads.receive(identifier, table, path.name, io.BytesIO(raw), len(raw))
    return {"stage_id": identifier, "name": name, "manifest": spec}


def review(app, body):
    result = app.action("/api/relational/review", body)
    return {"stage_id": body["stage_id"], "fingerprint": result["fingerprint"], "reviewed": True}


def test_given_reviewed_named_tables_when_confirmed_then_atomic_dataset_loads_and_keys_align(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)
    confirmation = review(app, body)

    # Act
    saved = app.action("/api/relational/confirm", confirmation)
    X, y, task = datasets.load(saved["name"])
    train, test, _, _, idx = datasets.split(X, y, 0, task)

    # Assert
    assert len(X) == 60 and len(train) == 24 and len(test) == len(idx) == 12
    assert X.attrs["quibble_source"]["fingerprint"] == saved["fingerprint"]
    assert app.reviewed_sources[saved["name"]] == saved["fingerprint"]
    assert datasets.known(saved["name"])
    assert datasets.task_of(saved["name"]) == task
    assert datasets.source_info(saved["name"])["fingerprint"] == saved["fingerprint"]
    assert saved["name"] in app.state()["datasets"]
    inspected = app.action("/api/inspect", {"dataset": saved["name"]})
    assert inspected["relational"]["n_examples"] == 60
    assert "synthetic_person" not in json.dumps(inspected)
    with pytest.raises(ValueError, match="unknown"):
        app.relational_uploads.directory(body["stage_id"])


def test_given_unreviewed_stage_when_confirmed_then_nothing_is_published(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)

    # Act & Assert
    with pytest.raises(ValueError):
        app.action("/api/relational/confirm", {"stage_id": body["stage_id"], "reviewed": True,
                                              "fingerprint": "forged"})
    assert not (datasets.data_root() / body["name"]).exists()


@pytest.mark.parametrize("filename", ["manifest.json", "state.json", "registry.json", "tables/events.csv"])
def test_given_mutation_after_review_when_confirmed_then_stale_approval_cannot_publish(app, tmp_path, filename):
    # Arrange
    body = stage(app, tmp_path)
    confirmation = review(app, body)
    path = app.relational_uploads.directory(body["stage_id"]) / filename
    path.write_bytes(path.read_bytes() + b"\n")

    # Act & Assert
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        app.action("/api/relational/confirm", confirmation)
    assert not (datasets.data_root() / body["name"]).exists()


def test_given_invalid_new_review_when_old_confirmation_replayed_then_refused(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)
    previous = review(app, body)
    body["manifest"]["recipe"]["aggregations"][1]["column"] = "answer"

    # Act & Assert
    with pytest.raises(ValueError, match="never IDs/targets"):
        app.action("/api/relational/review", body)
    with pytest.raises(ValueError):
        app.action("/api/relational/confirm", previous)


def test_given_existing_destination_when_confirmed_then_never_overwritten(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)
    confirmation = review(app, body)
    target = datasets.data_root() / body["name"]
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="never overwritten"):
        app.action("/api/relational/confirm", confirmation)
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_given_registry_publication_failure_when_saved_then_directory_rolls_back(app, tmp_path, monkeypatch):
    # Arrange
    body = stage(app, tmp_path)
    confirmation = review(app, body)
    original = app.relational_uploads.write
    def fail_registry(path, value):
        if path.name == "relational-registry.json":
            raise OSError("simulated write failure")
        original(path, value)
    monkeypatch.setattr(app.relational_uploads, "write", fail_registry)

    # Act & Assert
    with pytest.raises(OSError, match="simulated"):
        app.action("/api/relational/confirm", confirmation)
    assert not (datasets.data_root() / body["name"]).exists()
    assert not (datasets.data_root() / ".relational-registry.lock").exists()
    assert not list(datasets.data_root().glob(".relational-*"))


@pytest.mark.parametrize("table", ["../escape", "bad-table", "a/b"])
def test_given_unsafe_table_name_when_uploaded_then_refused_before_read(app, table):
    # Arrange
    identifier = app.action("/api/relational/start", {})["stage_id"]

    # Act & Assert
    with pytest.raises(ValueError, match="table name"):
        app.relational_uploads.receive(identifier, table, "table.csv", io.BytesIO(b"x\n"), 2)


def test_given_duplicate_table_identity_when_uploaded_then_no_replacement(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)

    # Act & Assert
    with pytest.raises(ValueError, match="already staged"):
        app.relational_uploads.receive(body["stage_id"], "events", "events.csv", io.BytesIO(b"x\n"), 2)


def test_given_manifest_different_table_identity_when_reviewed_then_refused(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)
    body["manifest"]["tables"]["renamed"] = body["manifest"]["tables"].pop("events")

    # Act & Assert
    with pytest.raises(ValueError, match="identities differ"):
        app.action("/api/relational/review", body)


def test_given_saved_relational_source_without_session_review_when_run_then_no_bypass(app, tmp_path):
    # Arrange
    body = stage(app, tmp_path)
    saved = app.action("/api/relational/confirm", review(app, body))
    app.reviewed_sources.clear()

    # Act & Assert
    with pytest.raises(ValueError, match="not reviewed"):
        app.start({"dataset": saved["name"], "backends": ["histgb", "lightgbm"],
                   "device": "cpu", "mode": "full", "cv": "holdout", "full_confirmed": True,
                   "fingerprint": saved["fingerprint"]})


@pytest.mark.parametrize(("backends", "cv"), [(["histgb", "kumo_tabular"], "holdout"),
                                             (["histgb", "lightgbm"], "3x1")])
def test_given_unsupported_relational_route_when_run_then_explicit_refusal(app, tmp_path, backends, cv):
    # Arrange
    body = stage(app, tmp_path)
    saved = app.action("/api/relational/confirm", review(app, body))

    # Act & Assert
    with pytest.raises(ValueError, match="local trees"):
        app.start({"dataset": saved["name"], "backends": backends, "device": "cpu", "mode": "full",
                   "cv": cv, "full_confirmed": True, "fingerprint": saved["fingerprint"]})


def test_given_relational_http_upload_when_token_missing_or_oversized_then_rejected(app):
    # Arrange
    server = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/relational/table"

    # Act & Assert
    try:
        for token, size, status in (("", 2, 403), (server.token, 16000001, 413)):
            request = urllib.request.Request(url, data=b"x\n", headers={
                "Content-Type": "application/octet-stream", "X-Quibble-Token": token,
                "Content-Length": str(size)}, method="POST")
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            assert error.value.code == status
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
