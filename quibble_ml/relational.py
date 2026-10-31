"""Bounded, opt-in point-in-time relational baselines.

Usage: ``python -m quibble_ml.relational inspect NAME --registry registry.json``.
Only explicitly registered manifests are accepted. This module does not infer
relations, download data, learn transforms, or invoke a relational-native model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd

VERSION = "relational-v1"
MAX_MANIFEST_BYTES = 262_144
LIMITS = {
    "tables": 16, "table_bytes": 16_000_000, "total_bytes": 64_000_000,
    "table_rows": 100_000, "total_rows": 250_000, "examples": 20_000,
    "features": 128, "matches": 2_000_000, "decoded_bytes": 128_000_000,
    "columns": 256,
}
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
DATASET_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class RelationalError(ValueError):
    """A manifest, input, or integration contract was refused."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RelationalError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _fields(obj: dict, required: set[str], optional: set[str], context: str) -> None:
    _require(isinstance(obj, dict), f"{context}: expected object")
    _require(required <= obj.keys(), f"{context}: missing fields {sorted(required - obj.keys())}")
    _require(not (obj.keys() - required - optional),
             f"{context}: unknown fields {sorted(obj.keys() - required - optional)}")


def _read_json(path: Path) -> dict:
    _require(path.is_file() and path.stat().st_size <= MAX_MANIFEST_BYTES,
             "JSON input missing or exceeds manifest byte budget")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RelationalError("invalid JSON input") from exc


def _inside(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "paths must be nonempty portable relative paths")
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts and ":" not in relative,
             "path traversal or absolute path refused")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root.resolve()), "path or symlink escapes dataset root")
    return resolved


def _timestamps(values: pd.Series, context: str) -> pd.Series:
    # Requiring an offset avoids guessing the timezone or DST fold of naive data.
    _require(values.notna().all(), f"{context}: missing timestamp")
    texts = values.astype(str)
    _require(texts.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all(),
             f"{context}: timestamps require explicit timezone offsets")
    try:
        result = pd.to_datetime(texts, utc=True, format="mixed", errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise RelationalError(f"{context}: invalid timestamp") from exc
    _require(result.notna().all(), f"{context}: invalid timestamp")
    return result


def _timestamp(value: str, context: str) -> pd.Timestamp:
    return _timestamps(pd.Series([value]), context).iloc[0]


def _duration(value: object, context: str, *, positive: bool = False) -> pd.Timedelta:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and (value > 0 if positive else value >= 0),
             f"{context}: expected finite {'positive' if positive else 'nonnegative'} seconds")
    try:
        return pd.Timedelta(value, unit="s")
    except (ValueError, OverflowError) as exc:
        raise RelationalError(f"{context}: duration out of range") from exc


@dataclass
class Materialized:
    """Features and private in-memory targets; audit is safe to export."""

    X: pd.DataFrame
    y: pd.Series
    task: str
    audit: dict

    def as_dataset(self) -> tuple[pd.DataFrame, pd.Series, str]:
        """Return the normal dataset-loader tuple without dropping metadata."""
        return self.X, self.y, self.task


def registered_manifest(name: str, registry: Path) -> tuple[Path, bytes]:
    """Resolve an explicit registry entry and private HMAC identity key.

    Registry paths are relative to the registry directory. The private key must
    be random 32-byte hex, kept out of reports and shared evidence bundles.
    """
    registry = Path(registry).resolve()
    spec = _read_json(registry)
    _fields(spec, {"schema_version", "datasets"}, set(), "registry")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1 and isinstance(spec["datasets"], dict),
             "unsupported registry schema")
    _require(all(DATASET_NAME.fullmatch(key) is not None
                 and key.lower() not in {"report", "runs", "smoke", "artifacts"}
                 for key in spec["datasets"]), "invalid registered dataset name")
    _require(name in spec["datasets"], "dataset is not explicitly registered")
    entry = spec["datasets"][name]
    _fields(entry, {"manifest", "identity_key"}, set(), "registry entry")
    _require(isinstance(entry["identity_key"], str)
             and re.fullmatch(r"[a-fA-F0-9]{64}", entry["identity_key"]) is not None,
             "registry identity_key must be 32 random bytes encoded as hex")
    return _inside(registry.parent, entry["manifest"]), bytes.fromhex(entry["identity_key"])


def registered_names(registry: Path) -> list[str]:
    """Enumerate only an explicitly chosen registry; do not scan folders."""
    spec = _read_json(Path(registry))
    _fields(spec, {"schema_version", "datasets"}, set(), "registry")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1 and isinstance(spec["datasets"], dict),
             "unsupported registry schema")
    _require(all(DATASET_NAME.fullmatch(key) is not None
                 and key.lower() not in {"report", "runs", "smoke", "artifacts"}
                 for key in spec["datasets"]), "invalid registered dataset name")
    return sorted(spec["datasets"])


def _validated(manifest: Path) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    spec = _read_json(manifest)
    _fields(spec, {"schema_version", "tables", "relations", "task", "recipe", "split"},
            {"limits", "source"}, "manifest")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
             "unsupported relational schema_version")
    if "source" in spec:
        _fields(spec["source"], {"kind", "license", "version"},
                {"dataset", "task", "url", "revision", "official_val_timestamp",
                 "official_test_timestamp", "availability_assumption"}, "source")
        _require(all(isinstance(value, str) and len(value) <= 1000
                     for value in spec["source"].values()), "source metadata must be bounded strings")
    limits = dict(LIMITS)
    _require(isinstance(spec.get("limits", {}), dict), "limits must be an object")
    for key, value in spec.get("limits", {}).items():
        _require(key in limits and isinstance(value, int) and not isinstance(value, bool)
                 and 0 < value <= limits[key], "limits may only tighten built-in budgets")
        limits[key] = value
    tables = spec["tables"]
    _require(isinstance(tables, dict) and 2 <= len(tables) <= limits["tables"],
             "table count outside budget")
    frames, audit = {}, {}
    total_bytes = total_rows = decoded_bytes = 0
    for name, table in tables.items():
        _require(isinstance(name, str) and NAME.fullmatch(name) is not None, "invalid table name")
        _fields(table, {"path", "sha256", "primary_key", "role", "columns"},
                {"event_time", "available_time"}, f"table {name}")
        _require(all(isinstance(value, str) for key, value in table.items()
                     if key not in {"columns", "primary_key"}), f"table {name}: fields must be strings")
        _require(table["role"] in {"entity", "event", "dimension", "examples"},
                 f"table {name}: unsupported role")
        columns = table["columns"]
        _require(isinstance(columns, dict) and columns
                 and len(columns) <= limits["columns"]
                 and all(isinstance(c, str) and NAME.fullmatch(c) for c in columns),
                 f"table {name}: invalid columns")
        _require(all(isinstance(v, str) and v in {"id", "timestamp", "observed", "target", "target_derived", "split"}
                     for v in columns.values()), f"table {name}: unknown column role")
        pk = table["primary_key"]
        _require(isinstance(pk, list) and pk and all(isinstance(c, str) for c in pk)
                 and len(pk) == len(set(pk))
                 and all(c in columns for c in pk), f"table {name}: invalid primary key")
        path = _inside(manifest.parent, table["path"])
        _require(path.is_file() and path.suffix.lower() in {".csv", ".parquet"},
                 f"table {name}: missing or unsupported CSV/Parquet file")
        size = path.stat().st_size
        total_bytes += size
        _require(size <= limits["table_bytes"] and total_bytes <= limits["total_bytes"],
                 "input byte budget exceeded")
        # Parse exactly the bytes whose hash was checked (not a second read).
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        _require(sha == table["sha256"], f"table {name}: declared hash mismatch; re-review inputs")
        if path.suffix.lower() == ".csv":
            # Preflight decoded shape before pandas allocates an object grid.
            try:
                reader = csv.reader(io.StringIO(raw.decode("utf-8")), strict=True)
                header = next(reader, [])
                _require(len(header) == len(set(header)) and set(header) == set(columns),
                         f"table {name}: columns differ from declared schema")
                row_count = 0
                for row in reader:
                    row_count += 1
                    _require(len(row) == len(columns), f"table {name}: ragged CSV row")
                    _require(row_count <= limits["table_rows"], f"table {name}: row budget exceeded")
                    estimate = len(raw) + row_count * len(columns) * 64
                    _require(estimate + decoded_bytes <= limits["decoded_bytes"],
                             "CSV decoded memory budget exceeded")
            except (UnicodeError, csv.Error) as exc:
                raise RelationalError(f"table {name}: invalid UTF-8 CSV") from exc
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                                nrows=limits["table_rows"] + 1)
            # pandas mangles duplicate CSV headers; exact schema rejects those too.
        else:
            import pyarrow.parquet as pq
            metadata = pq.ParquetFile(io.BytesIO(raw)).metadata
            _require(metadata.num_rows <= limits["table_rows"], f"table {name}: row budget exceeded")
            expanded = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
            _require(len(metadata.schema.names) == len(columns)
                     and set(metadata.schema.names) == set(columns), "Parquet columns differ from schema")
            estimate = expanded + metadata.num_rows * len(columns) * 64
            _require(estimate + decoded_bytes <= limits["decoded_bytes"], "Parquet decoded byte budget exceeded")
            frame = pd.read_parquet(io.BytesIO(raw))
        total_rows += len(frame)
        decoded_bytes += int(frame.memory_usage(index=True, deep=True).sum())
        _require(len(frame) <= limits["table_rows"] and total_rows <= limits["total_rows"],
                 "row budget exceeded")
        _require(decoded_bytes <= limits["decoded_bytes"], "decoded memory budget exceeded")
        _require(set(frame.columns) == set(columns) and len(frame.columns) == len(columns),
                 f"table {name}: columns differ from declared schema")
        for col, role in columns.items():
            if role in {"id", "split"}:
                _require(frame[col].notna().all() and frame[col].astype(str).str.len().gt(0).all(),
                         f"table {name}: missing identifier/split")
                frame[col] = frame[col].astype(str)
            elif role == "timestamp":
                frame[col] = _timestamps(frame[col], f"table {name}.{col}")
            elif role == "observed":
                try:
                    frame[col] = pd.to_numeric(frame[col].replace("", np.nan), errors="raise")
                except (ValueError, TypeError) as exc:
                    raise RelationalError(f"table {name}.{col}: observed features must be numeric") from exc
                _require(not np.isinf(frame[col].to_numpy(dtype=float)).any(),
                         f"table {name}.{col}: infinite feature")
        _require(not frame[pk].isna().any().any() and not frame.duplicated(pk).any(),
                 f"table {name}: null or duplicate primary key")
        # Stable reduction order even when physical child rows are shuffled.
        frame = frame.sort_values(pk, kind="stable").reset_index(drop=True)
        for key in ("event_time", "available_time"):
            if key in table:
                _require(columns.get(table[key]) == "timestamp", f"table {name}: invalid {key}")
        if table["role"] in {"event", "dimension"}:
            _require("event_time" in table and "available_time" in table,
                     f"table {name}: explicit event_time and available_time required")
            _require((frame[table["available_time"]] >= frame[table["event_time"]]).all(),
                     f"table {name}: availability precedes event")
        frames[name] = frame
        audit[name] = {"sha256": sha, "rows": len(frame), "role": table["role"]}
    return spec, frames, {"tables": audit, "limits": limits}


def _contracts(spec: dict, frames: dict, limits: dict) -> tuple[pd.DataFrame, list[dict]]:
    task = spec["task"]
    _fields(task, {"type", "entity_table", "entity_key", "examples_table", "entity_column",
                   "prediction_time", "target", "label_available_time", "horizon_seconds"},
            set(), "task")
    _require(all(isinstance(value, str) for key, value in task.items() if key != "horizon_seconds"),
             "task column/table/type fields must be strings")
    _require(task["type"] in {"classification", "regression"}, "unsupported task type")
    tables = spec["tables"]
    entity, examples = task["entity_table"], task["examples_table"]
    _require(entity in tables and examples in tables, "unknown task table")
    _require(tables[entity]["role"] == "entity" and tables[examples]["role"] == "examples",
             "task requires entity and examples roles")
    _require(tables[entity]["primary_key"] == [task["entity_key"]]
             and tables[entity]["columns"].get(task["entity_key"]) == "id",
             "entity requires a single identifier primary key")
    roles = tables[examples]["columns"]
    for key, role in (("entity_column", "id"), ("prediction_time", "timestamp"),
                      ("target", "target"), ("label_available_time", "timestamp")):
        _require(roles.get(task[key]) == role, f"task {key}: incorrect column role")
    _require(tables[examples]["primary_key"] == [task["entity_column"], task["prediction_time"]],
             "examples primary key must be exactly (entity, prediction_time)")
    _require(roles.get("split") == "split", "examples require explicit split column")
    ex = frames[examples].sort_values([task["entity_column"], task["prediction_time"]]).reset_index(drop=True)
    _require(0 < len(ex) <= limits["examples"], "example row budget exceeded or empty")
    _require(ex[task["entity_column"]].isin(frames[entity][task["entity_key"]]).all(),
             "examples: orphan foreign key")
    horizon = _duration(task["horizon_seconds"], "horizon_seconds", positive=True)
    _require((ex[task["label_available_time"]] >= ex[task["prediction_time"]] + horizon).all(),
             "label availability precedes outcome horizon")
    split = spec["split"]
    _fields(split, {"kind", "entity_overlap", "embargo_seconds"}, set(), "split")
    _require(split["kind"] == "official_temporal" and split["entity_overlap"] in {"allowed", "disjoint"},
             "explicit official_temporal split and entity overlap policy required")
    _duration(split["embargo_seconds"], "embargo_seconds")
    _require(set(ex["split"]) == {"train", "val", "test"}, "official train/val/test rows required")
    previous_max = None
    seen = set()
    for part in ("train", "val", "test"):
        rows = ex[ex["split"] == part]
        low, high = rows[task["prediction_time"]].min(), rows[task["prediction_time"]].max()
        _require(previous_max is None or previous_max < low, "split cutoff ranges overlap or tie")
        previous_max = high
        entities = set(rows[task["entity_column"]])
        _require(split["entity_overlap"] == "allowed" or not seen.intersection(entities),
                 "entity-disjoint policy violated")
        seen.update(entities)
    _require(isinstance(spec["relations"], list) and len(spec["relations"]) <= limits["tables"],
             "invalid relations")
    relation_names = set()
    for rel in spec["relations"]:
        _fields(rel, {"name", "parent", "parent_key", "child", "foreign_key", "kind"},
                {"frozen_as_of"}, "relation")
        _require(all(isinstance(value, str) for value in rel.values()), "relation fields must be strings")
        name, child = rel["name"], rel["child"]
        _require(isinstance(name, str) and NAME.fullmatch(name) and name not in relation_names,
                 "duplicate or invalid relation name")
        relation_names.add(name)
        _require(rel["parent"] == entity and rel["parent_key"] == task["entity_key"]
                 and child in tables and child not in {entity, examples},
                 f"relation {name}: only direct entity-to-child relations supported")
        _require(rel["kind"] in {"events", "frozen_dimension"}, "unsupported relation kind")
        _require(tables[child]["role"] == ("event" if rel["kind"] == "events" else "dimension"),
                 "relation/table role mismatch")
        _require(tables[child]["columns"].get(rel["foreign_key"]) == "id",
                 "foreign key must be an identifier")
        _require(frames[child][rel["foreign_key"]].isin(frames[entity][task["entity_key"]]).all(),
                 f"relation {name}: orphan foreign key")
        if rel["kind"] == "frozen_dimension":
            _require("frozen_as_of" in rel, "dimension requires frozen_as_of")
            _require(_timestamp(rel["frozen_as_of"], "frozen_as_of") <= ex[task["prediction_time"]].min(),
                     "dimension snapshot must be frozen before all prediction cutoffs")
        else:
            _require("frozen_as_of" not in rel, "event relation cannot declare frozen snapshot")
    recipe = spec["recipe"]
    _fields(recipe, {"tie_policy", "lookback_seconds", "aggregations"}, set(), "recipe")
    _require(recipe["tie_policy"] in {"exclude", "include"}, "explicit tie_policy required")
    if recipe["lookback_seconds"] is not None:
        _duration(recipe["lookback_seconds"], "lookback_seconds", positive=True)
    _require(isinstance(recipe["aggregations"], list) and recipe["aggregations"],
             "explicit aggregation recipe required")
    names = set()
    rels = {r["name"]: r for r in spec["relations"]}
    for agg in recipe["aggregations"]:
        _fields(agg, {"name", "relation", "op"}, {"column"}, "aggregation")
        _require(all(isinstance(value, str) for value in agg.values()), "aggregation fields must be strings")
        _require(isinstance(agg["name"], str) and NAME.fullmatch(agg["name"])
                 and agg["name"] not in names, "duplicate or invalid feature name")
        names.add(agg["name"])
        _require(agg["relation"] in rels, "unknown aggregation relation")
        rel = rels[agg["relation"]]
        allowed = {"count", "recency_seconds", "sum", "mean", "min", "max"}
        if rel["kind"] == "frozen_dimension":
            allowed = {"value"}
        _require(agg["op"] in allowed, "unsupported aggregation")
        if agg["op"] in {"count", "recency_seconds"}:
            _require("column" not in agg, "count/recency cannot reference a column")
        else:
            _require(tables[rel["child"]]["columns"].get(agg.get("column")) == "observed",
                     "predictors must be explicitly observed numeric columns, never IDs/targets")
    _require(len(names) <= limits["features"], "feature budget exceeded")
    return ex, spec["relations"]


def _build(spec: dict, frames: dict, ex: pd.DataFrame, relations: list, limits: dict) -> tuple[list, dict]:
    task, recipe = spec["task"], spec["recipe"]
    include = recipe["tie_policy"] == "include"
    lookback = recipe["lookback_seconds"]
    groups, metadata = {}, {}
    for rel in relations:
        table = spec["tables"][rel["child"]]
        child = frames[rel["child"]]
        if rel["kind"] == "frozen_dimension":
            frozen = _timestamp(rel["frozen_as_of"], "frozen_as_of")
            child = child[(child[table["event_time"]] <= frozen)
                          & (child[table["available_time"]] <= frozen)]
            # Freeze one latest revision per entity. Equal event times are ambiguous.
            _require(not child.duplicated([rel["foreign_key"], table["event_time"]]).any(),
                     "dimension snapshot has ambiguous revision ties")
            child = child.sort_values(table["event_time"]).drop_duplicates(rel["foreign_key"], keep="last")
        groups[rel["name"]] = {key: rows for key, rows in child.groupby(rel["foreign_key"], sort=False)}
        metadata[rel["name"]] = table
    output, history, recencies, missing = [], [], [], []
    matched = scanned = 0
    for _, example in ex.iterrows():
        entity, cutoff = example[task["entity_column"]], example[task["prediction_time"]]
        selected, counts, ages = {}, [], []
        for rel in relations:
            table = metadata[rel["name"]]
            rows = groups[rel["name"]].get(entity, frames[rel["child"]].iloc[:0])
            # Charge work before selection: a huge rejected history is not free.
            scanned += len(rows)
            _require(scanned <= limits["matches"], "join scan/cardinality budget exceeded")
            if rel["kind"] == "events":
                event, available = rows[table["event_time"]], rows[table["available_time"]]
                mask = ((event <= cutoff) & (available <= cutoff) if include
                        else (event < cutoff) & (available < cutoff))
                if lookback is not None:
                    mask &= event >= cutoff - pd.Timedelta(lookback, unit="s")
                rows = rows[mask]
                counts.append(len(rows))
                if len(rows):
                    ages.append((cutoff - rows[table["event_time"]].max()).total_seconds())
            selected[rel["name"]] = rows
            matched += len(rows)
        features = []
        for agg in recipe["aggregations"]:
            rows = selected[agg["relation"]]
            op = agg["op"]
            if op == "count":
                value = len(rows)
            elif op == "recency_seconds":
                col = metadata[agg["relation"]]["event_time"]
                value = (cutoff - rows[col].max()).total_seconds() if len(rows) else None
            elif op == "value":
                value = rows[agg["column"]].iloc[0] if len(rows) else None
            else:
                values = rows[agg["column"]]
                value = getattr(values, op)() if values.notna().any() else None
            if pd.isna(value):
                features.append(None)
            else:
                _require(math.isfinite(float(value)), "aggregate overflow/nonfinite feature")
                features.append(float(value))
        output.append(features)
        history.append(sum(counts))
        recencies.append(min(ages) if ages else None)
        missing.append(sum(len(rows) == 0 for rows in selected.values()))
    return output, {"history_length": history, "recency_seconds": recencies,
                    "missing_relations": missing, "cold_start": [n == 0 for n in history],
                    "matched_rows": matched, "scanned_rows": scanned}


def _split_positions(spec: dict, ex: pd.DataFrame) -> tuple[dict, dict]:
    task = spec["task"]
    embargo = _duration(spec["split"]["embargo_seconds"], "embargo_seconds")
    positions, purged = {}, {}
    for part, following in (("train", "val"), ("val", "test"), ("test", None)):
        mask = ex["split"] == part
        if following:
            boundary = ex.loc[ex["split"] == following, task["prediction_time"]].min() - embargo
            keep = ex[task["label_available_time"]] < boundary
            purged[part] = int((mask & ~keep).sum())
            mask &= keep
        positions[part] = np.flatnonzero(mask.to_numpy()).tolist()
        _require(positions[part], f"{part}: no rows remain after label-horizon purge")
    return positions, purged


def materialize_registered(name: str, registry: Path, cache_dir: Path | None = None) -> Materialized:
    """Validate all inputs, build or reuse features, and preserve explicit splits.

    Cold and warm calls both rehash and validate every table. Cache payloads are
    JSON, never executable pickle. Targets never enter the aggregation path.
    """
    started = time.perf_counter()
    manifest, secret = registered_manifest(name, registry)
    spec, frames, audit = _validated(manifest)
    ex, relations = _contracts(spec, frames, audit["limits"])
    positions, purged = _split_positions(spec, ex)
    fingerprint = _digest({"version": VERSION, "manifest": spec,
                           "tables": audit["tables"],
                           "identity_namespace": hashlib.sha256(secret).hexdigest()})
    validated_s = time.perf_counter() - started
    cache_path = Path(cache_dir) / f"{fingerprint}.json" if cache_dir is not None else None
    cache_state = "disabled" if cache_path is None else "cold"
    feature_start = time.perf_counter()
    if cache_path is not None and cache_path.exists():
        _require(cache_path.stat().st_size <= LIMITS["decoded_bytes"], "feature cache exceeds budget")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        _require(payload.get("fingerprint") == fingerprint, "feature cache fingerprint mismatch")
        _require(payload.get("checksum") == _digest(payload.get("data")), "feature cache checksum mismatch")
        values, slices = payload["data"]["values"], payload["data"]["slices"]
        _require(len(values) == len(ex) and all(len(row) == len(spec["recipe"]["aggregations"])
                                               for row in values), "feature cache shape mismatch")
        cache_state = "warm"
    else:
        task = spec["task"]
        feature_examples = ex[[task["entity_column"], task["prediction_time"]]]
        values, slices = _build(spec, frames, feature_examples, relations, audit["limits"])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"values": values, "slices": slices}
            payload = {"fingerprint": fingerprint, "data": data, "checksum": _digest(data)}
            raw = _json_bytes(payload)
            _require(len(raw) <= LIMITS["decoded_bytes"], "feature cache byte budget exceeded")
            temporary = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(raw)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
    feature_s = time.perf_counter() - feature_start
    task = spec["task"]
    X = pd.DataFrame(values, columns=[a["name"] for a in spec["recipe"]["aggregations"]], dtype=float)
    target = ex[task["target"]]
    _require(target.notna().all() and target.astype(str).str.len().gt(0).all(), "missing target")
    if task["type"] == "regression":
        try:
            y = pd.to_numeric(target, errors="raise")
        except (ValueError, TypeError) as exc:
            raise RelationalError("regression target must be numeric") from exc
        _require(np.isfinite(y.to_numpy(dtype=float)).all(), "nonfinite regression target")
    else:
        target = target.astype(str)
        classes = sorted(target.iloc[positions["train"]].unique())
        _require(len(classes) >= 2, "classification training requires at least two labels")
        _require(target.isin(classes).all(), "evaluation label not present in training label vocabulary")
        y = target.map({label: i for i, label in enumerate(classes)}).astype(int)
    y = y.reset_index(drop=True)
    entity_tokens = [hmac.new(secret, _json_bytes(["entity", v]), "sha256").hexdigest()
                     for v in ex[task["entity_column"]]]
    cutoffs = [v.isoformat() for v in ex[task["prediction_time"]]]
    example_keys = [hmac.new(secret, _json_bytes(["example", e, t]), "sha256").hexdigest()
                    for e, t in zip(ex[task["entity_column"]], cutoffs)]
    elapsed = time.perf_counter() - started
    cost = {"validation_s": validated_s, "feature_build_s": feature_s if cache_state != "warm" else 0.0,
            "feature_cache_read_s": feature_s if cache_state == "warm" else 0.0,
            "feature_total_s": elapsed, "cache_state": cache_state,
            "accounting": "shared once per dataset; do not silently assign to one backend"}
    audit.update(schema_version=1, builder_version=VERSION, fingerprint=fingerprint,
                 task_type=task["type"], n_examples=len(ex), n_features=X.shape[1],
                 task_contract=task, source=spec.get("source"),
                 feature_recipe=spec["recipe"], relations=spec["relations"], split=spec["split"],
                 purged_labels=purged, split_counts={k: len(v) for k, v in positions.items()},
                 repeated_entities=len(ex) != len(set(entity_tokens)),
                 temporal_caveat="Entity-clustered uncertainty does not remove temporal dependence.",
                 costs=cost)
    # Public audit never includes IDs, labels, raw joined rows, registry secrets or paths.
    X.attrs["quibble_source"] = {"kind": "relational", "fingerprint": fingerprint,
                                 "schema_version": 1, "builder_version": VERSION,
                                 "task": task["type"]}
    X.attrs["quibble_groups"] = np.asarray(entity_tokens)
    X.attrs["quibble_times"] = pd.DatetimeIndex(ex[task["prediction_time"]]).asi8
    X.attrs["quibble_split"] = {"kind": "relational", "evaluation": "test",
                                "policy": spec["split"]}
    X.attrs["quibble_relational"] = {
        "schema_version": 1, "fingerprint": fingerprint, "positions": positions,
        "example_keys": example_keys, "prediction_timestamps": cutoffs,
        "slices": slices, "audit": audit, "costs": cost,
    }
    return Materialized(X, y, task["type"], audit)


def load_registered(name: str, registry: Path, cache_dir: Path | None = None):
    """Dataset loader integration hook; never infer registration from a file."""
    return materialize_registered(name, registry, cache_dir).as_dataset()


def split_registered(X: pd.DataFrame, y: pd.Series, *, evaluation: str = "test",
                     cv: str = "holdout", split_id: int = 0):
    """Use purged official train vs validation/test, never random folds.

    Validation is not silently folded into training. Callers must explicitly
    request validation when tuning; test labels are not used in that process.
    """
    _require(evaluation in {"val", "test"}, "evaluation must be val or test")
    _require(cv == "holdout" and split_id == 0,
             "relational official protocol requires holdout and split_id=0; random CV is unsupported")
    info = X.attrs.get("quibble_relational")
    _require(info is not None, "relational metadata missing")
    _require("positions" in info, "official split positions missing; cannot re-split a subset")
    _require(len(X) == len(y) == len(info["example_keys"]), "relational split row alignment mismatch")
    train = np.asarray(info["positions"]["train"], dtype=int)
    test = np.asarray(info["positions"][evaluation], dtype=int)
    return subset_registered(X, train), subset_registered(X, test), y.iloc[train], y.iloc[test], test


def subset_registered(X: pd.DataFrame, positions) -> pd.DataFrame:
    """Slice relational identity/slice arrays alongside ordinary structural attrs.

    Official positions are removed from subsets: they address the complete
    canonical frame and must never accidentally be reused against a subset.
    """
    positions = np.asarray(positions, dtype=int)
    result = X.iloc[positions].copy()
    result.attrs = dict(X.attrs)
    for key in ("quibble_groups", "quibble_times"):
        if key in X.attrs:
            _require(len(X.attrs[key]) == len(X), f"{key}: row alignment mismatch")
            result.attrs[key] = np.asarray(X.attrs[key])[positions]
    original = X.attrs["quibble_relational"]
    info = dict(original)
    info.pop("positions", None)
    for key in ("example_keys", "prediction_timestamps"):
        _require(len(original[key]) == len(X), f"{key}: row alignment mismatch")
        info[key] = [original[key][i] for i in positions]
    info["slices"] = {
        key: [value[i] for i in positions] if isinstance(value, list) else value
        for key, value in original["slices"].items()
    }
    result.attrs["quibble_relational"] = info
    return result


def analysis_traits(X: pd.DataFrame) -> pd.DataFrame:
    """Return label-free, row-aligned relational disagreement slice attributes.

    Buckets are fixed rules, not learned from held-out labels or quantiles.
    Entity grouping is in ``quibble_groups``; it is not a predictor or a claim
    of independent temporal observations.
    """
    info = X.attrs.get("quibble_relational")
    _require(info is not None, "relational metadata missing")
    slices = info["slices"]
    _require(all(len(slices[key]) == len(X) for key in
                 ("history_length", "recency_seconds", "missing_relations", "cold_start")),
             "relational analysis traits are not row aligned")
    length = np.asarray(slices["history_length"])
    recency = np.asarray([np.nan if v is None else v for v in slices["recency_seconds"]], dtype=float)
    return pd.DataFrame({
        "relational_history_length": length,
        "relational_history_bucket": np.select([length == 0, length < 5, length < 20],
                                                ["none", "1-4", "5-19"], default="20+"),
        "relational_recency_seconds": recency,
        "relational_recency_bucket": np.select(
            [np.isnan(recency), recency < 86400, recency < 30 * 86400],
            ["missing", "under_1_day", "1_to_29_days"], default="30_days_or_more"),
        "relational_missing_relations": slices["missing_relations"],
        "relational_cold_start": slices["cold_start"],
    }, index=X.index)


def analysis_contract(X: pd.DataFrame) -> dict:
    """Provide safe prediction-unit/group semantics for parent report hooks."""
    info = X.attrs.get("quibble_relational")
    _require(info is not None, "relational metadata missing")
    return {"prediction_unit": "canonical (entity, UTC prediction timestamp)",
            "uncertainty_unit": "entity cluster",
            "entity_overlap": info["audit"]["split"]["entity_overlap"],
            "temporal_caveat": info["audit"]["temporal_caveat"],
            "example_keys": info["example_keys"],
            "prediction_timestamps": info["prediction_timestamps"],
            "entity_groups": np.asarray(X.attrs["quibble_groups"]).tolist(),
            "traits": {key: [None if pd.isna(v) else v for v in values]
                       for key, values in analysis_traits(X).to_dict(orient="list").items()},
            "feature_costs": info["costs"], "audit": info["audit"]}


def slice_masks(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return full-frame boolean masks for the normal analysis slice dictionary.

    Use in qualified analysis diagnostics when the relational attribute is
    present, not as additional acceptance gates. No labels, predictions, learned
    thresholds or IDs enter these definitions. Existing minimum-size and
    multiplicity qualifications still apply.
    """
    traits = analysis_traits(X)
    masks = {
        f"relational/history={value}":
            (traits["relational_history_bucket"] == value).to_numpy()
        for value in ("none", "1-4", "5-19", "20+")
    }
    masks.update({
        f"relational/recency={value}":
            (traits["relational_recency_bucket"] == value).to_numpy()
        for value in ("missing", "under_1_day", "1_to_29_days", "30_days_or_more")
    })
    masks["relational/missing_relation"] = (traits["relational_missing_relations"] > 0).to_numpy()
    masks["relational/cold_start"] = traits["relational_cold_start"].to_numpy(dtype=bool)
    return masks


def diagnostic_masks(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed descriptive masks for qualified diagnostics, never extra swap gates."""
    return slice_masks(X)


def create_parser() -> argparse.ArgumentParser:
    """Create the explicit-registration CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "materialize", "run"):
        child = sub.add_parser(command)
        child.add_argument("name")
        child.add_argument("--registry", type=Path, required=True)
        child.add_argument("--cache-dir", type=Path)
        if command == "run":
            child.add_argument("--backends", nargs="+", default=["histgb", "lightgbm"])
            child.add_argument("--out", type=Path, required=True)
    return parser


def run_registered(name: str, registry: Path, out: Path, backends: list[str],
                   cache_dir: Path | None = None) -> dict:
    """Invoke the existing sweep in a fresh process with explicit registration.

    Requires parent-owned dataset registry/split integration. No runtime
    monkeypatching, flattening into a random-split dataset, remote adapters, or
    overwriting an existing evidence directory is permitted.
    """
    _require(len(backends) >= 2 and len(set(backends)) == len(backends)
             and all(b in {"histgb", "lightgbm", "catboost"} for b in backends),
             "choose at least two distinct local tree routes: histgb/lightgbm/catboost")
    _require(not Path(out).exists(), "run output already exists; choose an isolated evidence directory")
    built = materialize_registered(name, registry, cache_dir)
    env = {**os.environ, "QUIBBLE_RELATIONAL_REGISTRY": str(Path(registry).resolve()),
           "QUIBBLE_RELATIONAL_CACHE": str(Path(cache_dir).resolve()) if cache_dir else "",
           "PYTHONDONTWRITEBYTECODE": "1"}
    # Refuse before starting model workers if the normal loader lacks integration.
    check = (
        "import sys; from quibble_ml.datasets import load,split; "
        "X,y,t=load(sys.argv[1]); "
        "assert X.attrs.get('quibble_source',{}).get('fingerprint') == sys.argv[2], "
        "'normal loader does not resolve the reviewed relational fingerprint'; "
        "info=X.attrs['quibble_relational']; "
        "tr,te,_,_,idx=split(X,y,0,t,'holdout',0); "
        "assert idx.tolist()==info['positions']['test'], 'official test split changed'; "
        "assert tr.index.tolist()==info['positions']['train'], 'purged training split changed'"
    )
    ready = subprocess.run([sys.executable, "-B", "-c", check, name, built.audit["fingerprint"]],
                           env=env, capture_output=True, text=True, timeout=60, check=False)
    _require(ready.returncode == 0,
             "normal relational registry/load/split integration unavailable; no models run")
    command = [sys.executable, "-B", "-m", "quibble_ml.run", "--sweep",
               "--backends", ",".join(backends), "--datasets", name,
               "--out", str(out), "--device", "cpu", "--jobs", "1", "--threads", "1",
               "--model-seeds", "0", "--split-seed", "0", "--cv", "holdout",
               "--tier", "defaults", "--timeout", "120"]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, timeout=600, check=False)
    _require(completed.returncode == 0, "normal relational sweep failed")
    records = []
    for path in Path(out).rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("dataset") == name and record.get("backend") in backends:
            records.append(record)
    _require(set(r["backend"] for r in records) == set(backends)
             and all("error" not in r and "refused" not in r and r.get("status") != "failed"
                     and r.get("dataset_source", {}).get("fingerprint") == built.audit["fingerprint"]
                     for r in records), "sweep evidence incomplete, failed or mismatched")
    evidence = {"feature_audit": built.audit, "backends": backends,
                "sweep_wall_s": time.perf_counter() - started,
                "execution": "existing tabular sweep; not relational-native"}
    # Preserve the shared front-stage cold/warm cost, not just worker reloads.
    # This file contains no raw identifiers, labels, registry secrets or paths.
    (Path(out) / "relational-feature-stage.json").write_bytes(_json_bytes(evidence))
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Inspect, materialize or execute through the normal integrated sweep."""
    args = create_parser().parse_args(argv)
    try:
        if args.command == "run":
            audit = run_registered(args.name, args.registry, args.out, args.backends, args.cache_dir)
        else:
            audit = materialize_registered(args.name, args.registry, args.cache_dir).audit
        print(json.dumps(audit, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (RelationalError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"Relational input refused: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except subprocess.TimeoutExpired:
        print("Relational execution timed out; no passing comparison claimed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
