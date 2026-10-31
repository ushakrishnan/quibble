"""Local, staged browser imports. Parsing runs in a bounded child process."""
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

LIMITS = {"file_bytes": 50 * 1024 * 1024, "rows": 250_000, "columns": 1000,
          "preview_rows": 20, "cell_chars": 160, "column_name_chars": 256,
          "decoded_bytes": 128 * 1024 * 1024, "worker_bytes": 2 * 1024**3,
          "timeout_s": 45, "staged_files": 4, "expiry_s": 3600}
OWNED = ("data.csv", "data.parquet", "options.json", "result.json", "dataset.json",
         "upload.json", "review.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approval(directory):
    return {p.name: file_hash(p) for p in directory.iterdir()
            if p.name in ("data.csv", "data.parquet", "dataset.json", "upload.json")}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")


class Uploads:
    def __init__(self, root):
        self.root = Path(root) / ".uploads"
        self.root.mkdir(exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("Upload staging cannot be a symbolic link.")
        self.cleanup()

    def directory(self, identifier, required=True):
        if not isinstance(identifier, str) or not re.fullmatch(r"u[0-9a-f]{32}", identifier):
            raise ValueError("Invalid upload identifier.")
        directory = self.root / identifier
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()) \
                or (required and not directory.is_dir()):
            raise ValueError("Upload is unknown, expired, cancelled or already saved.")
        return directory

    def remove(self, directory):
        directory = self.directory(directory.name)
        if any(p.name not in OWNED or not p.is_file() or p.is_symlink() for p in directory.iterdir()):
            raise ValueError("Upload staging contains unexpected files; inspect it locally.")
        shutil.rmtree(directory)

    def cleanup(self):
        for directory in self.root.iterdir():
            if re.fullmatch(r"u[0-9a-f]{32}", directory.name) and directory.is_dir() \
                    and time.time() - directory.stat().st_mtime > LIMITS["expiry_s"]:
                self.remove(directory)

    def worker(self, directory, options):
        write_json(directory / "options.json", {**options, "limits": LIMITS})
        (directory / "result.json").unlink(missing_ok=True)
        env = {**os.environ, "QUIBBLE_DATA_DIR": str(self.root),
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            completed = subprocess.run([sys.executable, "-m", "quibble_ml.uploads", str(directory)],
                                       env=env, cwd=directory, capture_output=True,
                                       timeout=LIMITS["timeout_s"], check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Upload parsing exceeded its time limit. Use a smaller file.") from exc
        if completed.returncode or not (directory / "result.json").is_file():
            raise ValueError("Upload parsing failed or exceeded its memory limit. Use a smaller, valid table.")
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def receive(self, stream, size, filename):
        self.cleanup()
        if len(list(self.root.iterdir())) >= LIMITS["staged_files"]:
            raise ValueError("Staging is full. Cancel an upload or wait for its one-hour expiry.")
        if not isinstance(filename, str) or not filename or len(filename) > 160 \
                or any(c in filename for c in "/\\:\x00") \
                or Path(filename).suffix.lower() not in (".csv", ".parquet"):
            raise ValueError("Choose a CSV or Parquet filename without path components.")
        if not 0 < size <= LIMITS["file_bytes"]:
            raise ValueError("Upload exceeds the 50 MiB file limit or is empty.")
        directory = self.root / ("u" + secrets.token_hex(16))
        directory.mkdir()
        try:
            path = directory / ("data" + Path(filename).suffix.lower())
            with path.open("xb") as target:
                remaining = size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload was interrupted; no dataset was saved.")
                    target.write(chunk)
                    remaining -= len(chunk)
            result = self.worker(directory, {"filename": filename, "file": path.name})
            write_json(directory / "review.json", {"filename": filename, "file": path.name})
            return {**result, "upload_id": directory.name}
        except (ValueError, OSError):
            self.remove(directory)
            raise

    def review(self, body):
        from . import datasets
        directory = self.directory(body["upload_id"])
        saved = json.loads((directory / "review.json").read_text(encoding="utf-8"))
        name = body.get("name")
        if not datasets.valid_name(name) or name.lower() in {
                n.lower() for n in (*datasets.BUILTIN, *datasets.OPENML, *datasets.local_names())} \
                or (datasets.data_root() / name).exists():
            raise ValueError("Choose a valid, unused dataset name; reserved names cannot be overwritten.")
        (directory / "dataset.json").unlink(missing_ok=True)
        (directory / "upload.json").unlink(missing_ok=True)
        saved.pop("review_token", None)
        write_json(directory / "review.json", saved)
        result = self.worker(directory, {"filename": saved["filename"], "file": saved["file"],
                                        "review": body})
        nonce = secrets.token_hex(24)
        write_json(directory / "review.json", {**saved, "name": name, "review_token": nonce,
                                              "approval": approval(directory), "result": result})
        return {**result, "review_token": nonce, "upload_id": directory.name}

    def confirm(self, body):
        from . import datasets
        directory = self.directory(body["upload_id"])
        saved = json.loads((directory / "review.json").read_text(encoding="utf-8"))
        if body.get("confirmed") is not True or not saved.get("review_token") \
                or not secrets.compare_digest(str(body.get("review_token", "")), saved["review_token"]):
            raise ValueError("Validate the current roles and explicitly confirm review before saving.")
        if approval(directory) != saved["approval"]:
            raise ValueError("Upload changed since review. Validate it again.")
        root = datasets.data_root()
        root.mkdir(parents=True, exist_ok=True)
        if saved["name"].lower() in {n.lower() for n in datasets.local_names()}:
            raise ValueError("Dataset name already exists; choose another name and review again.")
        destination = root / saved["name"]
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise ValueError("Dataset name already exists; nothing was overwritten.") from exc
        published = []
        try:
            # The manifest is the publication marker: readers never see a partial dataset.
            for name in (saved["file"], "upload.json", "dataset.json"):
                target = destination / (".dataset.tmp" if name == "dataset.json" else name)
                with (directory / name).open("rb") as source, target.open("xb") as output:
                    published.append(target)
                    shutil.copyfileobj(source, output)
            published[-1].replace(destination / "dataset.json")
            published[-1] = destination / "dataset.json"
        except OSError:
            for path in published:
                path.unlink()
            destination.rmdir()
            raise
        result = {"name": saved["name"], "fingerprint": saved["result"]["fingerprint"],
                  "source": datasets.source_info(saved["name"]), "cv": saved["result"]["cv"]}
        self.remove(directory)
        return result

    def cancel(self, body):
        directory = self.directory(body["upload_id"], required=False)
        if not directory.exists():
            return {"cancelled": False, "status": "already absent (expired or previously cancelled)"}
        self.remove(directory)
        return {"cancelled": True}


def memory_limit(limit):
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("processes", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", ctypes.c_ulonglong * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.process_memory = limit
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) \
            or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise OSError("Unable to enforce the upload worker memory limit.")
    return handle


def parse(directory, options):
    import numpy as np
    import pandas as pd
    from . import datasets
    limits = options["limits"]
    path = directory / options["file"]
    body = options.get("review")
    spec = None
    if body:
        spec = {"file": path.name, "target": body.get("target"), "task": body.get("task"),
                "drop_columns": body.get("drop_columns", []),
                "categorical_columns": body.get("categorical_columns", []),
                "split": body.get("split", "random")}
        for role in ("group_column", "time_column"):
            if body.get(role):
                spec[role] = body[role]
        write_json(directory / "dataset.json", spec)
        spec, _ = datasets.local_spec(directory.name)
    frame, raw = datasets.read_table(path, spec, limits)
    if len(frame) < 4:
        raise ValueError("At least four rows are required for labeled evaluation.")
    result = {"rows": len(frame), "columns": list(frame.columns),
              "schema": [{"name": c, "type": str(frame[c].dtype),
                          "suggested": "numeric" if pd.to_numeric(frame[c].dropna(), errors="coerce").notna().all()
                          else "categorical", "missing": int(frame[c].isna().sum())} for c in frame],
              "sample": [{c: None if pd.api.types.is_scalar(v) and pd.isna(v) else str(v)[:limits["cell_chars"]]
                          for c, v in row.items()}
                         for row in frame.head(limits["preview_rows"]).to_dict(orient="records")],
              "sha256": hashlib.sha256(raw).hexdigest(),
              "notice": "Local preview only; text is truncated for display, not training. No rows were repaired or dropped."}
    if body:
        cv = body.get("cv", "holdout")
        if cv not in ("holdout", "3x1", "3x2"):
            raise ValueError("Choose holdout, 3x1 or 3x2 evaluation.")
        cv_seed = int(os.getenv("QUIBBLE_CV_SEED", "0"))
        split_seed = int(os.getenv("QUIBBLE_SPLIT_SEED", "0"))
        write_json(directory / "upload.json", {"version": 1, "original_name": options["filename"],
                   "sha256": result["sha256"], "manifest": spec, "rows": len(frame),
                   "parser": {"format": path.suffix[1:], "encoding": "utf-8-sig" if path.suffix == ".csv" else None,
                              "missing_values": "pandas defaults", "pandas_version": pd.__version__},
                   "schema": result["schema"],
                   "reviewed_evaluation": {"cv": cv, "cv_seed": cv_seed, "split_seed": split_seed}})
        X, y, task = datasets.load(directory.name)
        from .config import parse_cv
        ids = range(np.prod(parse_cv(cv))) if cv != "holdout" else (split_seed,)
        for ss in ids:
            datasets.split(X, y, int(ss), task, cv, cv_seed)
        source = {k: v for k, v in X.attrs["quibble_source"].items() if k != "class_labels"}
        result.update(fingerprint=source["fingerprint"], source=source,
                      manifest=spec, cv=cv, predictors=X.shape[1],
                      target_summary=[{"label": str(k)[:limits["cell_chars"]], "count": int(v)}
                                      for k, v in frame[spec["target"]].value_counts().head(100).items()]
                      if task == "classification" else {"min": float(y.min()), "max": float(y.max())})
        if task == "classification":
            result.update(target_classes=int(y.nunique()), target_summary_truncated=y.nunique() > 100)
    return result


def main():
    directory = Path(sys.argv[1])
    options = json.loads((directory / "options.json").read_text(encoding="utf-8"))
    handle = memory_limit(options["limits"]["worker_bytes"])
    try:
        result = parse(directory, options)
    except (ValueError, TypeError, KeyError, OSError, ImportError, MemoryError) as exc:
        result = {"error": str(exc)[:700] if not isinstance(exc, MemoryError) else "Decoded table exceeded memory limits."}
    write_json(directory / "result.json", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
