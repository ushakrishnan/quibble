"""Opt-in bounded CC-BY-4.0 RelBench arXiv sampled pilot, NOT a full benchmark.

No network is used on import. Run with ``--download`` to fetch only the five
pinned files (9,237,686 bytes), never the 158 MB papers table. Source authors:
Tang, Haiming; He, Sirui; Li, Mengjie; Guo, Zhimao (2024), arXiv-physics.
https://github.com/PKUTHM/arxiv-physics ; https://creativecommons.org/licenses/by/4.0/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

from . import relational as rel

REVISION = "6b02dae348f68ecf5b08d8587ccdc839c2c33e3e"
FILES = {
    "db/authors.parquet": (4449181, "3b987ccbf6474e15555ba01b134301ad0a3eda5d466b5eb71446987a1c744c44"),
    "db/paperAuthors.parquet": (3367953, "0b769e2faee764a353ff084de73d996503c8b8ec0ec86dea1fd255e03ed765eb"),
    "tasks/author-publication/train.parquet": (916602, "7aa669af6342585753345c09a934884e97f99efd2d59e61ee8e0d142ea3b9cfc"),
    "tasks/author-publication/val.parquet": (249438, "4fabb302f545d545455a207f014b9e5a22d873ca0679b6201ff23a200b630f1a"),
    "tasks/author-publication/test.parquet": (254512, "bbfc2421a06a43473b5c98ee97176a63af2fa1d350bf1ee6b57e81fccdd7ecbe"),
}


def fetch_sources(root: Path) -> None:
    """Fetch only pinned, hash-verified CC-BY source artifacts under 10 MB total."""
    for relative, (size, sha) in FILES.items():
        path = root / relative
        if path.exists():
            rel._require(path.stat().st_size == size and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                         "existing pilot source hash mismatch")
            continue
        url = f"https://huggingface.co/datasets/stanford-star/relbench-v2-extra/resolve/{REVISION}/rel-arxiv/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(size + 1)
        rel._require(len(data) == size and hashlib.sha256(data).hexdigest() == sha,
                     "downloaded pilot source size/hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare(source: Path, output: Path, *, per_split: int = 64) -> Path:
    """Sample by deterministic label-blind key hash within official split tables."""
    rel._require(16 <= per_split <= 256, "pilot sample budget must be 16..256 examples per split")
    rel._require(not output.exists(), "pilot output already exists")
    for relative, (size, sha) in FILES.items():
        path = source / relative
        rel._require(path.is_file() and path.stat().st_size == size
                     and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                     "pinned pilot source size/hash mismatch; no unverified attribution")
    examples = []
    inventory = {}
    for part in ("train", "val", "test"):
        table = pq.ParquetFile(source / f"tasks/author-publication/{part}.parquet")
        rows = table.read().to_pandas()
        rel._require({"Author_ID", "date", "publication_count"} <= set(rows), "official task schema changed")
        inventory[part] = {"rows": len(rows), "min_cutoff": str(rows["date"].min()),
                           "max_cutoff": str(rows["date"].max())}
        # Source datetime columns are timezone-naive; the explicit pilot assumption
        # is calendar dates interpreted at UTC midnight, NOT ingestion timestamps.
        rows["cutoff"] = pd.to_datetime(rows["date"], utc=True)
        rows["rank"] = [hashlib.sha256(f"{part}|{a}|{t.isoformat()}".encode()).hexdigest()
                        for a, t in zip(rows["Author_ID"], rows["cutoff"])]
        rows = rows.sort_values("rank").head(per_split)
        rows["entity_id"] = rows["Author_ID"].astype(str)
        rows["label_available"] = rows["cutoff"] + pd.Timedelta(182, unit="D")
        rows["split"] = part
        rows["target"] = rows["publication_count"]
        rows["cutoff"] = rows["cutoff"].map(lambda t: t.isoformat())
        rows["label_available"] = rows["label_available"].map(lambda t: t.isoformat())
        examples.append(rows[["entity_id", "cutoff", "label_available", "split", "target"]])
    examples = pd.concat(examples, ignore_index=True)
    entities = set(examples["entity_id"])
    upstream_ids = set()
    for batch in pq.ParquetFile(source / "db/authors.parquet").iter_batches(
            batch_size=20000, columns=["Author_ID"]):
        upstream_ids.update(str(v) for v in batch.column(0).to_pylist() if str(v) in entities)
    rel._require(upstream_ids == entities, "sampled task contains unknown author foreign key")
    history = []
    for batch in pq.ParquetFile(source / "db/paperAuthors.parquet").iter_batches(batch_size=20000):
        frame = batch.to_pandas()
        rel._require({"Author_ID", "Paper_ID", "Submission_Date"} <= set(frame), "official history schema changed")
        frame = frame[frame["Author_ID"].astype(str).isin(entities)].copy()
        if len(frame):
            history.append(frame)
    child = pd.concat(history, ignore_index=True)
    # De-duplicate identical source authorship edges; record this transformation.
    before = len(child)
    child = child.drop_duplicates(["Author_ID", "Paper_ID", "Submission_Date"])
    child["entity_id"] = child["Author_ID"].astype(str)
    child["paper_id"] = child["Paper_ID"].astype(str)
    child["event_time"] = pd.to_datetime(child["Submission_Date"], utc=True).map(lambda t: t.isoformat())
    child["available_time"] = child["event_time"]
    child["publication"] = 1.0
    child = child[["entity_id", "paper_id", "event_time", "available_time", "publication"]]
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    frames = {"authors": pd.DataFrame({"entity_id": sorted(entities)}),
              "authorships": child, "examples": examples}
    schemas = {
        "authors": (["entity_id"], "entity", {"entity_id": "id"}),
        "authorships": (["entity_id", "paper_id", "event_time"], "event", {
            "entity_id": "id", "paper_id": "id", "event_time": "timestamp",
            "available_time": "timestamp", "publication": "observed"}),
        "examples": (["entity_id", "cutoff"], "examples", {
            "entity_id": "id", "cutoff": "timestamp", "label_available": "timestamp",
            "split": "split", "target": "target"}),
    }
    spec = {
        "schema_version": 1, "tables": {},
        "relations": [{"name": "history", "parent": "authors", "parent_key": "entity_id",
                       "child": "authorships", "foreign_key": "entity_id", "kind": "events"}],
        "task": {"type": "regression", "entity_table": "authors", "entity_key": "entity_id",
                 "examples_table": "examples", "entity_column": "entity_id", "prediction_time": "cutoff",
                 "target": "target", "label_available_time": "label_available", "horizon_seconds": 182 * 86400},
        "recipe": {"tie_policy": "exclude", "lookback_seconds": 365 * 86400, "aggregations": [
            {"name": "past_year_publications", "relation": "history", "op": "count"},
            {"name": "publication_recency", "relation": "history", "op": "recency_seconds"}]},
        "split": {"kind": "official_temporal", "entity_overlap": "allowed", "embargo_seconds": 0},
        "source": {"kind": "RelBench deterministic sampled pilot NOT official full-corpus benchmark",
                   "license": "CC-BY-4.0 source; CDLA-Permissive-2.0 RelBench conversion",
                   "version": "relbench-3.1.0", "dataset": "rel-arxiv", "task": "author-publication",
                   "revision": REVISION, "url": "https://github.com/PKUTHM/arxiv-physics",
                   "official_val_timestamp": "2022-01-01", "official_test_timestamp": "2023-01-01",
                   "availability_assumption": "Submission calendar date interpreted as UTC; available=event. Not verified ingestion; sampled pilot only."},
    }
    for name, frame in frames.items():
        path = output / "tables" / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        pk, role, columns = schemas[name]
        spec["tables"][name] = {"path": f"tables/{name}.parquet",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "primary_key": pk, "role": role, "columns": columns}
        if name == "authorships":
            spec["tables"][name].update(event_time="event_time", available_time="available_time")
    (output / "manifest.json").write_bytes(rel._json_bytes(spec))
    registry = {"schema_version": 1, "datasets": {"rel_arxiv_sample": {
        "manifest": "manifest.json", "identity_key": secrets.token_hex(32)}}}
    path = output / "registry.json"
    path.write_bytes(rel._json_bytes(registry))
    record = {"warning": "NOT official full-corpus benchmark", "inventory": inventory,
              "per_split_sample": per_split, "sampling": "SHA256(part|Author_ID|UTC cutoff), ascending, label-blind",
              "deduplicated_authorship_edges": before - len(child),
              "source_files": {name: {"bytes": n, "sha256": h} for name, (n, h) in FILES.items()},
              "attribution": "Tang, Haiming; He, Sirui; Li, Mengjie; Guo, Zhimao (2024), arXiv-physics. CC BY 4.0. RelBench conversions CDLA-Permissive-2.0.",
              "license_review": {"date": "2026-10-30",
                                 "upstream_license_blob": "b2c3ce4e61335a1ea45b5adcf571f669acb6c49c",
                                 "url": "https://github.com/PKUTHM/arxiv-physics/blob/main/LICENSE.txt"},
              "changes": "ID-only author projection, sampled examples, selected authorship history, UTC/date availability assumption, 365-day features."}
    (output / "pilot-evidence.json").write_bytes(rel._json_bytes(record))
    rel.materialize_registered("rel_arxiv_sample", path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="explicitly permit the pinned <10MB CC-BY download")
    args = parser.parse_args(argv)
    try:
        if args.download:
            fetch_sources(args.source)
        print(prepare(args.source, args.out))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Pilot refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
