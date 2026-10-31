"""Write a small, public synthetic proof fixture (no network or private data).

Usage: ``python -m quibble_ml.relational_synthetic DIRECTORY``.
The directory must not exist, preventing accidental replacement of real inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


def write_fixture(root: Path, *, task_type: str = "classification", entities: int = 12,
                  repeated_test: bool = False) -> Path:
    """Create registered CSV tables with deliberate future/late/revision traps."""
    root = Path(root)
    if task_type not in {"classification", "regression"} or not 4 <= entities <= 100:
        raise ValueError("fixture requires classification/regression and 4..100 entities")
    root.mkdir(parents=True, exist_ok=False)
    (root / "tables").mkdir()
    ids = [f"synthetic_person_{i:03d}" for i in range(entities)]
    tables = {"entities": pd.DataFrame({"entity_id": ids})}
    events, dimensions, examples = [], [], []
    for i, entity in enumerate(ids):
        events.extend([
            [f"early_{i}", entity, "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", i % 2, 900 + i],
            [f"future_{i}", entity, "2024-01-25T00:00:00Z", "2024-01-25T00:00:00Z", 10000 + i % 2, 900 + i],
            [f"late_{i}", entity, "2024-01-02T00:00:00Z", "2024-01-15T00:00:00Z", 20000, 900 + i],
            [f"tie_{i}", entity, "2024-01-10T00:00:00Z", "2024-01-10T00:00:00Z", 30000, 900 + i],
        ])
        dimensions.extend([
            [f"dim_old_{i}", entity, "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", i],
            [f"dim_new_{i}", entity, "2024-01-08T00:00:00Z", "2024-01-08T00:00:00Z", 99999],
        ])
        schedule = [(10, "train", 15), (20, "train", 25),
                    (29, "train", 45), (40, "val", 45), (60, "test", 65)]
        if repeated_test:
            schedule.append((70, "test", 75))
        for day, split, label_day in schedule:
            origin = pd.Timestamp("2024-01-01", tz="UTC")
            cutoff = (origin + pd.Timedelta(day - 1, unit="D")).isoformat()
            label_available = (origin + pd.Timedelta(label_day - 1, unit="D")).isoformat()
            label = str(i % 2) if task_type == "classification" else float(i % 2) + i * 0.1
            examples.append([entity, cutoff, label_available, split, label])
    tables["events"] = pd.DataFrame(events, columns=["event_id", "entity_id", "event_time",
                                                     "available_time", "value", "answer"])
    tables["dimensions"] = pd.DataFrame(dimensions, columns=["revision_id", "entity_id",
                                                              "event_time", "available_time", "score"])
    tables["examples"] = pd.DataFrame(examples, columns=["entity_id", "cutoff", "label_available",
                                                        "split", "target"])
    columns = {
        "entities": {"entity_id": "id"},
        "events": {"event_id": "id", "entity_id": "id", "event_time": "timestamp",
                   "available_time": "timestamp", "value": "observed", "answer": "target_derived"},
        "dimensions": {"revision_id": "id", "entity_id": "id", "event_time": "timestamp",
                       "available_time": "timestamp", "score": "observed"},
        "examples": {"entity_id": "id", "cutoff": "timestamp", "label_available": "timestamp",
                     "split": "split", "target": "target"},
    }
    manifest = {
        "schema_version": 1, "tables": {}, "relations": [
            {"name": "history", "parent": "entities", "parent_key": "entity_id", "child": "events",
             "foreign_key": "entity_id", "kind": "events"},
            {"name": "profile", "parent": "entities", "parent_key": "entity_id", "child": "dimensions",
             "foreign_key": "entity_id", "kind": "frozen_dimension", "frozen_as_of": "2024-01-05T00:00:00Z"},
        ],
        "task": {"type": task_type, "entity_table": "entities", "entity_key": "entity_id",
                 "examples_table": "examples", "entity_column": "entity_id", "prediction_time": "cutoff",
                 "target": "target", "label_available_time": "label_available", "horizon_seconds": 432000},
        "recipe": {"tie_policy": "exclude", "lookback_seconds": None, "aggregations": [
            {"name": "history_count", "relation": "history", "op": "count"},
            {"name": "history_value", "relation": "history", "op": "sum", "column": "value"},
            {"name": "history_recency", "relation": "history", "op": "recency_seconds"},
            {"name": "profile_score", "relation": "profile", "op": "value", "column": "score"},
        ]},
        "split": {"kind": "official_temporal", "entity_overlap": "allowed", "embargo_seconds": 0},
        "source": {"kind": "synthetic", "license": "MIT", "version": "quibble-relational-proof-v1"},
    }
    for name, frame in tables.items():
        path = root / "tables" / f"{name}.csv"
        frame.to_csv(path, index=False)
        role = {"entities": "entity", "examples": "examples", "events": "event", "dimensions": "dimension"}[name]
        pk = {"entities": ["entity_id"], "examples": ["entity_id", "cutoff"],
              "events": ["event_id"], "dimensions": ["revision_id"]}[name]
        table = {"path": f"tables/{name}.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "primary_key": pk, "role": role, "columns": columns[name]}
        if name in {"events", "dimensions"}:
            table.update(event_time="event_time", available_time="available_time")
        manifest["tables"][name] = table
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {"schema_version": 1, "datasets": {"relational_synthetic": {
        "manifest": "manifest.json",
        # Deliberately public deterministic key ONLY for this synthetic fixture.
        "identity_key": hashlib.sha256(b"quibble-public-synthetic-fixture-only").hexdigest(),
    }}}
    path = root / "registry.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Create the standalone fixture without overwriting an existing directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--repeat-test-entities", action="store_true",
                        help="include two distinct test cutoffs per entity for clustered uncertainty proof")
    args = parser.parse_args(argv)
    try:
        print(write_fixture(args.directory, task_type=args.task, repeated_test=args.repeat_test_entities))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Fixture refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
