"""Hash-bound, atomic, loopback-only relational intake; no schema inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import time

from . import relational as rel


class RelationalUploads:
    """Keep staged tables private until schema validation and explicit review."""

    def __init__(self, root: Path):
        self.root = Path(root) / ".relational-uploads"
        self.root.mkdir(exist_ok=True)
        rel._require(not self.root.is_symlink(), "relational staging cannot be a symlink")

    def directory(self, identifier: str, *, allow_expired: bool = False) -> Path:
        rel._require(isinstance(identifier, str) and re.fullmatch(r"r[0-9a-f]{32}", identifier),
                     "invalid relational stage identifier")
        path = self.root / identifier
        rel._require(path.is_dir() and not path.is_symlink(), "unknown or expired relational stage")
        rel._require(allow_expired or time.time() - path.stat().st_mtime < 3600,
                     "relational stage expired; cancel it and start again")
        return path

    @staticmethod
    def read(path: Path) -> dict:
        return rel._read_json(path)

    @staticmethod
    def write(path: Path, value: dict) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_bytes(rel._json_bytes(value))
        temp.replace(path)

    def start(self, body: dict) -> dict:
        rel._require(len(list(self.root.iterdir())) < 4, "relational staging full; cancel an old stage")
        path = self.root / ("r" + secrets.token_hex(16))
        path.mkdir()
        (path / "tables").mkdir()
        self.write(path / "state.json", {"tables": {}, "identity_key": secrets.token_hex(32)})
        return {"stage_id": path.name, "limits": rel.LIMITS,
                "notice": "Local tables only. No provider/model call; explicit schema review required."}

    def receive(self, identifier: str, table: str, filename: str, stream, size: int) -> dict:
        path = self.directory(identifier)
        rel._require(isinstance(table, str) and rel.NAME.fullmatch(table), "invalid logical table name")
        rel._require(isinstance(filename, str) and filename and len(filename) <= 160
                     and not any(c in filename for c in "/\\:\x00")
                     and Path(filename).suffix.lower() in {".csv", ".parquet"}, "invalid table filename")
        state = self.read(path / "state.json")
        rel._require(table not in state["tables"], "table identity already staged; start a fresh stage to replace it")
        rel._require(len(state["tables"]) < rel.LIMITS["tables"], "too many tables")
        rel._require(0 < size <= rel.LIMITS["table_bytes"]
                     and size + sum(t["bytes"] for t in state["tables"].values()) <= rel.LIMITS["total_bytes"],
                     "per-table or total encoded byte budget exceeded")
        target = path / "tables" / (table + Path(filename).suffix.lower())
        remaining = size
        try:
            with target.open("xb") as writer:
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    rel._require(bool(chunk), "interrupted table upload")
                    writer.write(chunk)
                    remaining -= len(chunk)
            info = {"path": "tables/" + target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "bytes": size, "filename": filename}
            state["tables"][table] = info
            self.write(path / "state.json", state)
            (path / "approval.json").unlink(missing_ok=True)
            return {"stage_id": identifier, "table": table, **info, "tables": state["tables"]}
        except (OSError, ValueError):
            target.unlink(missing_ok=True)
            raise

    def _checked(self, path: Path, spec: dict, state: dict) -> None:
        rel._require(isinstance(spec, dict) and isinstance(spec.get("tables"), dict),
                     "provide the complete relational manifest")
        rel._require(set(spec["tables"]) == set(state["tables"]), "manifest table identities differ from staged tables")
        for name, table in spec["tables"].items():
            rel._require(isinstance(table, dict), "each table schema must be an object")
            saved = state["tables"][name]
            rel._require(table.get("path") == saved["path"] and table.get("sha256") == saved["sha256"],
                         "manifest path/hash differs from staged identity; re-stage changed tables")
        self.write(path / "manifest.json", spec)

    @staticmethod
    def _inventory(path: Path, state: dict) -> None:
        expected = {"state.json", "manifest.json", "registry.json", "approval.json", "tables"}
        rel._require(all(p.name in expected and not p.is_symlink() for p in path.iterdir()),
                     "unexpected staged files or symlinks")
        allowed = {Path(t["path"]).name for t in state["tables"].values()}
        rel._require(all(p.is_file() and not p.is_symlink() and p.name in allowed
                         for p in (path / "tables").iterdir()), "unexpected table files or symlinks")

    def review(self, body: dict) -> dict:
        from . import datasets
        path = self.directory(body["stage_id"])
        # A failed new review cannot preserve an older approval.
        (path / "approval.json").unlink(missing_ok=True)
        name = body.get("name")
        rel._require(datasets.valid_name(name) and rel.DATASET_NAME.fullmatch(name)
                     and name.lower() not in {n.lower() for n in
                         (*datasets.BUILTIN, *datasets.OPENML, *datasets.local_names(), *datasets.relational_names())},
                     "choose an unused dataset name")
        state = self.read(path / "state.json")
        self._inventory(path, state)
        self._checked(path, body.get("manifest"), state)
        registry = {"schema_version": 1, "datasets": {name: {
            "manifest": "manifest.json", "identity_key": state["identity_key"]}}}
        self.write(path / "registry.json", registry)
        built = rel.materialize_registered(name, path / "registry.json")
        approval = {"name": name, "fingerprint": built.audit["fingerprint"],
                    "manifest_hash": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                    "state_hash": hashlib.sha256((path / "state.json").read_bytes()).hexdigest(),
                    "registry_hash": hashlib.sha256((path / "registry.json").read_bytes()).hexdigest()}
        self.write(path / "approval.json", approval)
        return {"stage_id": path.name, "name": name, "fingerprint": approval["fingerprint"],
                "tables": {n: {"columns": t["columns"], "primary_key": t["primary_key"],
                              "role": t["role"], **built.audit["tables"][n]} for n, t in body["manifest"]["tables"].items()},
                "audit": built.audit, "supported_routes": ["histgb", "lightgbm", "catboost"],
                "notice": "Flattened local tabular routes only; not relational-native. Confirm all roles and relations."}

    def confirm(self, body: dict) -> dict:
        from . import datasets
        rel._require(body.get("reviewed") is True, "explicit relational schema review confirmation required")
        path = self.directory(body["stage_id"])
        approval = self.read(path / "approval.json")
        self._inventory(path, self.read(path / "state.json"))
        rel._require(body.get("fingerprint") == approval["fingerprint"], "review fingerprint changed")
        for filename, key in (("manifest.json", "manifest_hash"), ("state.json", "state_hash"),
                              ("registry.json", "registry_hash")):
            rel._require(hashlib.sha256((path / filename).read_bytes()).hexdigest() == approval[key],
                         "staged schema/identity changed after review; review again")
        name = approval["name"]
        built = rel.materialize_registered(name, path / "registry.json")
        rel._require(built.audit["fingerprint"] == approval["fingerprint"], "tables changed since review")
        root = datasets.data_root().resolve()
        root.mkdir(parents=True, exist_ok=True)
        registry = datasets.relational_registry() or root / "relational-registry.json"
        rel._require(registry.parent.resolve() == root, "browser registry must be inside the dataset root")
        target = root / name
        temporary = root / (".relational-" + secrets.token_hex(16))
        lock_path = root / ".relational-registry.lock"
        # Cross-workbench writer exclusion; stale locks require local inspection.
        marker = lock_path.open("xb")
        marker.close()
        try:
            published = False
            try:
                rel._require(not target.exists() and name.lower() not in
                             {p.name.lower() for p in root.iterdir()}, "saved datasets are never overwritten")
                current = self.read(registry) if registry.exists() else {"schema_version": 1, "datasets": {}}
                rel._require(name.lower() not in {n.lower() for n in current["datasets"]},
                             "dataset already registered")
                shutil.copytree(path, temporary, symlinks=False)
                copied = rel.materialize_registered(name, temporary / "registry.json")
                rel._require(copied.audit["fingerprint"] == approval["fingerprint"], "inputs changed during atomic save")
                for extra in ("approval.json", "state.json", "registry.json"):
                    (temporary / extra).unlink()
                temporary.rename(target)
                published = True
                state = self.read(path / "state.json")
                current["datasets"][name] = {
                    "manifest": f"{name}/manifest.json", "identity_key": state["identity_key"]}
                self.write(registry, current)
                os.environ["QUIBBLE_RELATIONAL_REGISTRY"] = str(registry)
            except BaseException:
                if published:
                    shutil.rmtree(target)
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        finally:
            lock_path.unlink(missing_ok=True)
        shutil.rmtree(path)
        return {"name": name, "fingerprint": approval["fingerprint"],
                "notice": "Named relational tables saved atomically; registry contains a private identity key."}

    def cancel(self, body: dict) -> dict:
        path = self.directory(body["stage_id"], allow_expired=True)
        self._inventory(path, self.read(path / "state.json"))
        shutil.rmtree(path)
        return {"cancelled": True}
