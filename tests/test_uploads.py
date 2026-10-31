"""Staged local imports, real bounded parsers, and unchanged original data."""
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from quibble_ml import datasets, uploads
from quibble_ml.web import Server, Workbench


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "data"))
    instance = Workbench()
    yield instance
    instance.close()


def table():
    return pd.DataFrame({"id": range(60), "number": np.arange(60) * 0.1,
                         "category": ["01", "02"] * 30, "target": ["no", "yes"] * 30})


def upload(app, fmt="csv"):
    buffer = io.BytesIO()
    if fmt == "csv":
        buffer.write(table().to_csv(index=False).encode("utf-8-sig"))
    else:
        table().to_parquet(buffer, index=False)
    raw = buffer.getvalue()
    return app.uploads.receive(io.BytesIO(raw), len(raw), f"input.{fmt}"), raw


def roles(identifier, **kwargs):
    return {"upload_id": identifier, "name": "reviewed_input", "target": "target",
            "task": "classification", "drop_columns": ["id"],
            "categorical_columns": ["category"], "split": "random", "cv": "3x1", **kwargs}


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_upload_review_confirm_compare_and_reopen(app, fmt):
    initial, raw = upload(app, fmt)
    assert app.thread is None
    assert initial["rows"] == 60 and len(initial["sample"]) == 20
    assert initial["sample"][0]["category"] == "01"
    assert not datasets.known("reviewed_input")
    reviewed = app.action("/api/uploads/review", roles(initial["upload_id"]))
    assert reviewed["predictors"] == 2
    assert reviewed["target_summary"] == [{"label": "no", "count": 30}, {"label": "yes", "count": 30}]
    confirmed = app.action("/api/uploads/confirm", {"upload_id": initial["upload_id"],
                            "review_token": reviewed["review_token"], "confirmed": True})
    assert app.thread is None
    X, y, task = datasets.load(confirmed["name"])
    source = X.attrs["quibble_source"]
    assert source["fingerprint"] == reviewed["fingerprint"] == confirmed["fingerprint"]
    assert source["uploaded"]["parser"]["format"] == fmt
    assert source["uploaded"]["manifest"] == reviewed["manifest"]
    assert list(X) == ["number", "category"] and set(X["category"]) == {"01", "02"}
    assert set(y) == {0, 1}
    assert (datasets.data_root() / confirmed["name"] / f"data.{fmt}").read_bytes() == raw
    assert not list(app.uploads.root.iterdir())
    with pytest.raises(ValueError, match="already saved"):
        app.action("/api/uploads/confirm", {"upload_id": initial["upload_id"], "confirmed": True})
    record = app.start({"dataset": confirmed["name"], "fingerprint": confirmed["fingerprint"],
                        "backends": ["histgb", "lightgbm"], "device": "cpu",
                        "mode": "full", "cv": "holdout", "full_confirmed": True})
    app.thread.join(120)
    assert not app.thread.is_alive()
    finished = app.job(record["id"])
    assert finished["status"] == "completed", finished["log"]
    assert finished["dataset_source"]["fingerprint"] == confirmed["fingerprint"]
    assert "Interpretation" in finished["report"]
    app.close()
    reopened = Workbench()
    try:
        assert reopened.job(record["id"])["report"] == finished["report"]
        with pytest.raises(ValueError, match="reviewed"):
            reopened.start({"dataset": confirmed["name"], "fingerprint": confirmed["fingerprint"],
                            "backends": ["histgb"], "device": "cpu", "mode": "smoke", "cv": "holdout"})
        path = datasets.data_root() / confirmed["name"] / f"data.{fmt}"
        path.write_bytes(path.read_bytes() + b"\n")
        state = reopened.state()
        assert any(item["name"] == confirmed["name"] for item in state["dataset_errors"])
        assert reopened.job(record["id"])["report"] == finished["report"]
    finally:
        reopened.close()


@pytest.mark.parametrize("raw", [b"a,a\n1,2\n", b",b\n1,2\n", b"a,b\n1,2,3\n",
                                b"\xff,b\n1,2\n", b"PAR1not-valid-parquet"])
def test_bad_files_do_not_publish_or_remain_staged(app, raw):
    with pytest.raises(ValueError):
        app.uploads.receive(io.BytesIO(raw), len(raw), "broken.csv")
    assert not list(app.uploads.root.iterdir())
    assert "reviewed_input" not in datasets.local_names()


@pytest.mark.parametrize("name", ["../input.csv", "a\\input.csv", "input.zip", "a:input.csv"])
def test_browser_filenames_are_not_paths(app, name):
    with pytest.raises(ValueError, match="filename"):
        app.uploads.receive(io.BytesIO(b"x\n1\n"), 4, name)
    assert not list(app.uploads.root.iterdir())


def test_review_changes_invalidate_confirmation_and_mutation_is_detected(app):
    initial, _ = upload(app)
    identifier = initial["upload_id"]
    first = app.uploads.review(roles(identifier))
    with pytest.raises(ValueError):
        app.uploads.review(roles(identifier, target="missing"))
    with pytest.raises(ValueError, match="Validate"):
        app.uploads.confirm({"upload_id": identifier, "review_token": first["review_token"], "confirmed": True})
    second = app.uploads.review(roles(identifier))
    path = app.uploads.directory(identifier) / "data.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        app.uploads.confirm({"upload_id": identifier, "review_token": second["review_token"], "confirmed": True})
    app.uploads.cancel({"upload_id": identifier})
    assert not list(app.uploads.root.iterdir())


@pytest.mark.parametrize("name", ["wine", "WINE", "report", "../other", "grocery_sample"])
def test_reserved_dataset_names_are_rejected(app, name):
    initial, _ = upload(app)
    with pytest.raises(ValueError, match="unused"):
        app.uploads.review(roles(initial["upload_id"], name=name))


def test_existing_destination_and_unconfirmed_import_are_never_overwritten(app):
    initial, _ = upload(app)
    reviewed = app.uploads.review(roles(initial["upload_id"]))
    request = {"upload_id": initial["upload_id"], "review_token": reviewed["review_token"]}
    with pytest.raises(ValueError, match="confirm"):
        app.uploads.confirm(request)
    destination = datasets.data_root() / "reviewed_input"
    destination.mkdir(parents=True)
    (destination / "keep.txt").write_text("unchanged")
    with pytest.raises(ValueError, match="already exists"):
        app.uploads.confirm({**request, "confirmed": True})
    assert (destination / "keep.txt").read_text() == "unchanged"
    assert not (destination / "dataset.json").exists()


def test_interrupted_uploads_timeouts_expiry_and_staging_cap(app, monkeypatch):
    with pytest.raises(ValueError, match="interrupted"):
        app.uploads.receive(io.BytesIO(b"x"), 10, "input.csv")
    assert not list(app.uploads.root.iterdir())
    first, _ = upload(app)
    monkeypatch.setitem(uploads.LIMITS, "staged_files", 1)
    with pytest.raises(ValueError, match="full"):
        upload(app)
    directory = app.uploads.directory(first["upload_id"])
    os.utime(directory, (time.time() - 7200,) * 2)
    app.uploads.cleanup()
    assert not directory.exists()
    result = app.uploads.cancel({"upload_id": first["upload_id"]})
    assert result["cancelled"] is False and "absent" in result["status"]


@pytest.mark.parametrize("count", [250_000, 250_001])
def test_actual_row_limit(tmp_path, count):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n" + "1,2\n" * count)
    if count > uploads.LIMITS["rows"]:
        with pytest.raises(ValueError, match="row"):
            datasets.read_table(path, limits=uploads.LIMITS)
    else:
        frame, _ = datasets.read_table(path, limits=uploads.LIMITS)
        assert len(frame) == count


@pytest.mark.parametrize("count", [1000, 1001])
def test_actual_column_limit(tmp_path, count):
    path = tmp_path / "data.csv"
    path.write_text(",".join(f"c{i}" for i in range(count)) + "\n" +
                    (",".join("1" for _ in range(count)) + "\n") * 4)
    if count > uploads.LIMITS["columns"]:
        with pytest.raises(ValueError, match="column"):
            datasets.read_table(path, limits=uploads.LIMITS)
    else:
        frame, _ = datasets.read_table(path, limits=uploads.LIMITS)
        assert len(frame.columns) == count


def test_compressed_parquet_cannot_bypass_decoded_limit(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "data.parquet"
    array = pa.array(np.zeros(17_000))
    pq.write_table(pa.table({f"c{i}": array for i in range(1000)}), path)
    assert path.stat().st_size < uploads.LIMITS["file_bytes"]
    with pytest.raises(ValueError, match="decoded"):
        datasets.read_table(path, limits=uploads.LIMITS)


def test_decoded_limit_exact_boundary(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
    frame, _ = datasets.read_table(path)
    size = int(frame.memory_usage(deep=True).sum())
    limits = {**uploads.LIMITS, "decoded_bytes": size}
    datasets.read_table(path, limits=limits)
    with pytest.raises(ValueError, match="decoded"):
        datasets.read_table(path, limits={**limits, "decoded_bytes": size - 1})


def test_actual_upload_byte_boundary(app, monkeypatch):
    class Stream:
        def read(self, size):
            return b"x" * size
    monkeypatch.setattr(app.uploads, "worker", lambda *args: {"rows": 4})
    result = app.uploads.receive(Stream(), uploads.LIMITS["file_bytes"], "input.csv")
    directory = app.uploads.directory(result["upload_id"])
    assert (directory / "data.csv").stat().st_size == 50 * 1024 * 1024
    with pytest.raises(ValueError, match="50 MiB"):
        app.uploads.receive(Stream(), uploads.LIMITS["file_bytes"] + 1, "input.csv")
    app.uploads.cancel({"upload_id": result["upload_id"]})


@pytest.mark.parametrize("kind,column", [("group", "group_column"), ("temporal", "time_column")])
def test_structural_roles_share_loader_and_split_policy(app, kind, column):
    initial, _ = upload(app)
    result = app.uploads.review(roles(initial["upload_id"], drop_columns=[], split=kind,
                                    cv="3x1", **{column: "id"}))
    assert result["manifest"][column] == "id"
    assert result["source"]["uploaded"]["manifest"]["split"] == kind


def test_bad_regression_targets_do_not_dump_raw_values(app):
    initial, _ = upload(app)
    with pytest.raises(ValueError, match="finite numeric") as caught:
        app.uploads.review(roles(initial["upload_id"], task="regression"))
    assert '"yes"' not in str(caught.value)


def test_quoted_csv_bom_missingness_and_loader_agree(app):
    frame = table()
    frame["category"] = ["comma,inside", "two\nlines"] * 30
    frame.loc[0, "number"] = np.nan
    raw = frame.to_csv(index=False).encode("utf-8-sig")
    initial = app.uploads.receive(io.BytesIO(raw), len(raw), "quoted.csv")
    assert initial["sample"][0]["category"] == "comma,inside"
    assert initial["sample"][1]["category"] == "two\nlines"
    assert initial["sample"][0]["number"] is None
    reviewed = app.uploads.review(roles(initial["upload_id"]))
    app.uploads.confirm({"upload_id": initial["upload_id"],
                        "review_token": reviewed["review_token"], "confirmed": True})
    X, _, _ = datasets.load("reviewed_input")
    assert X["number"].isna().sum() == 1
    assert set(X["category"]) == {"comma,inside", "two\nlines"}


def test_worker_memory_ceiling_is_enforced(tmp_path):
    code = ("from quibble_ml.uploads import memory_limit\n"
            "handle=memory_limit(100*1024**2)\n"
            "try:\n data=bytearray(200*1024**2)\n"
            "except MemoryError:\n print('bounded')\n"
            "else:\n raise RuntimeError('memory limit not enforced')\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "bounded" in result.stdout


def test_worker_timeout_is_explicit_and_cleans_owned_stage(app, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("upload worker", uploads.LIMITS["timeout_s"])
    monkeypatch.setattr(uploads.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="time limit"):
        upload(app)
    assert not list(app.uploads.root.iterdir())


def test_reviewed_source_guard_prevents_fit(app, monkeypatch):
    from unittest.mock import Mock
    from quibble_ml.backends import HistGB
    from quibble_ml.config import Config
    from quibble_ml.run import run_one, stem
    initial, _ = upload(app)
    reviewed = app.uploads.review(roles(initial["upload_id"]))
    app.uploads.confirm({"upload_id": initial["upload_id"],
                        "review_token": reviewed["review_token"], "confirmed": True})
    monkeypatch.setenv("_QUIBBLE_REVIEWED_SOURCE", json.dumps({
        "dataset": "reviewed_input", "fingerprint": "changed-since-web-review"}))
    fit = Mock()
    monkeypatch.setattr(HistGB, "fit", fit)
    cfg = Config.from_env()
    run_one("histgb", "reviewed_input", 0, cfg)
    saved = json.loads(stem(cfg.out_dir, "reviewed_input", "histgb", 0, 0).with_suffix(".json").read_text())
    assert "changed after browser review" in saved["error"]
    fit.assert_not_called()


def test_http_upload_protections_and_size_rejection(app):
    server = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Content-Type": "application/octet-stream", "X-Upload-Name": "input.csv",
               "X-Quibble-Token": server.token}
    try:
        for overrides, code in [({"X-Quibble-Token": ""}, 403), ({"Origin": "https://evil.example"}, 403),
                                ({"Content-Type": "application/zip"}, 415),
                                ({"Content-Length": str(uploads.LIMITS["file_bytes"] + 1)}, 413)]:
            request = urllib.request.Request(base + "/api/uploads", data=b"x",
                                             headers={**headers, **overrides})
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=60)
            assert caught.value.code == code
        raw = table().to_csv(index=False).encode()
        request = urllib.request.Request(base + "/api/uploads", data=raw, headers=headers)
        result = json.loads(urllib.request.urlopen(request, timeout=60).read())
        assert result["rows"] == 60
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
