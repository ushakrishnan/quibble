"""One backend x dataset x model-seed per process (honest peak memory).

  python -m quibble_ml.run --backend lightgbm --dataset credit-g --model-seed 0
  python -m quibble_ml.run --sweep --backends histgb,catboost --datasets credit-g,phoneme
  python -m quibble_ml.run --sweep --backends all --datasets breast_cancer --device cuda --jobs 4
  python -m quibble_ml.run --sweep --backends histgb,lightgbm --datasets wine --cv 5x2 --model-seeds 0
Settings: .env / environment; CLI flags override. Finished runs are skipped (--force to redo).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from .history import new_run_id, snapshot, timestamp
from .metrics import score as metric_score
from .provenance import backend_metadata, check_reuse, digest, package_names, recipe_snapshot, resolved_recipe

FLAGS = {"device": "QUIBBLE_DEVICE", "jobs": "QUIBBLE_JOBS", "preprocess": "QUIBBLE_PREPROCESS",
         "split_seed": "QUIBBLE_SPLIT_SEED", "model_seeds": "QUIBBLE_MODEL_SEEDS",
         "out": "QUIBBLE_OUT_DIR", "timeout": "QUIBBLE_TIMEOUT_S", "threads": "QUIBBLE_THREADS",
         "cv": "QUIBBLE_CV", "cv_seed": "QUIBBLE_CV_SEED", "keep_models": "QUIBBLE_KEEP_MODELS",
         "tier": "QUIBBLE_TIER", "tune_trials": "QUIBBLE_TUNE_TRIALS",
         "tune_folds": "QUIBBLE_TUNE_FOLDS", "tune_seed": "QUIBBLE_TUNE_SEED",
         "tune_metric": "QUIBBLE_TUNE_METRIC", "tune_timeout_s": "QUIBBLE_TUNE_TIMEOUT_S"}


def stem(out, dataset, backend, ss, ms):
    from .datasets import valid_name
    if not valid_name(dataset):
        raise ValueError(f"Invalid dataset output name: {dataset!r}")
    return Path(out) / dataset / f"{backend}_ss{ss}_ms{ms}"


def _worker_process(cmd, *, timeout, env):
    """Keep hard deadlines authoritative for a worker and its subprocesses."""
    with subprocess.Popen(cmd, env=env, start_new_session=sys.platform != "win32") as worker:
        try:
            return subprocess.CompletedProcess(cmd, worker.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                import psutil
                try:
                    descendants = psutil.Process(worker.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                # Windows virtualenv launchers can outlive or be outlived by real Python.
                for child in reversed(descendants):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                worker.kill()
                _, alive = psutil.wait_procs(descendants, timeout=10)
                if alive:
                    raise OSError(f"Timed-out worker descendants did not exit: {[p.pid for p in alive]}")
            else:
                import signal
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.wait()
            raise


def settings_differ(json_path, cfg):
    """Which device, preprocessing, CV or foundation-model settings differ from a saved run."""
    try:
        r = json.loads(Path(json_path).read_text())
    except Exception:
        return []
    cur = {"device": cfg.device, "preprocess_requested": cfg.preprocess, "cv": cfg.cv,
           "cv_seed": cfg.cv_seed, "run_type": cfg.run_type}
    r.setdefault("cv_seed", 0)
    r.setdefault("run_type", "benchmark")
    if r.get("backend") == "tabfm":
        cur["tabfm_n_estimators"] = cfg.tabfm_n_estimators
    elif r.get("backend") == "kumo_tabular":
        cur.update(kumo_size=cfg.kumo_size, kumo_n_estimators=cfg.kumo_n_estimators)
    elif r.get("backend") == "autogluon":
        cur.update(ag_preset=cfg.ag_preset, ag_time_limit=cfg.ag_time_limit)
    differs = [k for k, v in cur.items() if r.get(k) not in (None, v)]
    if r.get("backend") == "tabpfn" and "tabpfn_max_rows" in r and r["tabpfn_max_rows"] != cfg.tabpfn_max_rows:
        differs.append("tabpfn_max_rows")
    if r.get("single_n") not in (None, cfg.single_n):
        differs.append("single_n")
    if changed := package_changes(r):
        differs.append("package_versions (" + ", ".join(changed) + ")")
    if r.get("dataset"):
        from .datasets import source_info
        try:
            source = source_info(r["dataset"])
        except (ValueError, OSError):
            differs.append("dataset_source unavailable")
        else:
            saved = r.get("dataset_source")
            if (source or saved) and (not source or not saved
                                      or saved["fingerprint"] != source["fingerprint"]):
                differs.append("dataset_source")
    if cfg.keep_models and r.get("backend") in ("histgb", "weak", "lightgbm", "catboost", "autogluon") \
            and r.get("model_retention") != "retained":
        differs.append("model_retention")
    return differs


def package_changes(record):
    """Libraries whose installed version differs from the one that produced a saved result."""
    from .env_info import _ver
    saved = ((record.get("env") or {}).get("packages")) or {}
    names = ("numpy", "pandas") + package_names(record.get("backend"))
    return [f"{n} {saved[n]}->{_ver(n)}" for n in names if saved.get(n) and _ver(n) != saved[n]]


def is_failed(json_path):
    """True when a saved result is an error record (refusals are deterministic and are not retried)."""
    try:
        return "error" in json.loads(Path(json_path).read_text())
    except (OSError, ValueError):
        return False


STALE_MSG = ("WARNING: {n} existing result(s) were produced with different settings ({keys}) "
             "and are being SKIPPED. Use --force to redo them, or a different --out per configuration.")


def peak_mb():
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def current_rss_mb():
    """Resident memory of this process now (not the high-water mark), or None if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, AttributeError):
        return None


def save_npy(path, array):
    """Write an .npy file atomically, so parallel workers never read a half-written split file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def error_kind(error):
    message = str(error).lower()
    if isinstance(error, MemoryError) or any(term in message for term in
            ("out of memory", "outofmemory", "cannot allocate memory", "bad allocation",
             "not enough memory", "paging file is too small")):
        return "resource"
    return "execution"


def metrics(y, P, task):
    """Per-split metrics, computed by quibble_ml.metrics.score so run records and analysis agree."""
    names = ("rmse", "mae", "r2") if task == "regression" else ("accuracy", "log_loss", "brier", "auc")
    out = {n: metric_score(n, y, P) for n in names}
    return {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in out.items()}


def smoke_positions(y, limit, seed, task):
    """Choose the same bounded rows for every backend, preserving classification labels."""
    if len(y) <= limit:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    required = []
    if task == "classification":
        labels = y.to_numpy()
        classes = np.unique(labels)
        if len(classes) > limit:
            raise ValueError(f"Smoke limit {limit} cannot represent {len(classes)} classes")
        required = [rng.choice(np.flatnonzero(labels == label)) for label in classes]
    remaining = np.setdiff1d(np.arange(len(y)), required)
    return np.sort(np.concatenate([required, rng.choice(
        remaining, limit - len(required), replace=False)]).astype(int))


def run_one(backend, dataset, ms, cfg, force=False, retry_failed=False):
    from .backends import BACKENDS, Refused, device_exclusion
    from .datasets import UnsupportedSplit, load, split, subset_rows, validate_sources
    from .env_info import collect
    from .tuning import TREES, TuningFailure, ensure_output_tier, fit_tuned
    ensure_output_tier(cfg)
    s = stem(cfg.out_dir, dataset, backend, cfg.split_seed, ms)
    js = s.with_suffix(".json")
    if js.exists() and retry_failed and is_failed(js):
        force = True
    recipe = recipe_snapshot(backend, cfg, ms)
    for path in s.parent.glob(f"{backend}_ss*_ms*.json"):
        check_reuse(json.loads(path.read_text()), recipe,
                    writing=force or not js.exists() or bool(device_exclusion(backend, cfg.device)))
    if js.exists() and not force and not device_exclusion(backend, cfg.device):
        d = settings_differ(js, cfg)
        print(f"skip (done): {js.name}" + (f"  [WARNING: saved with different {', '.join(d)}]" if d else ""))
        return
    parent_managed = os.environ.get("_QUIBBLE_HISTORY_OUT") == str(Path(cfg.out_dir).resolve())
    if not parent_managed:
        snapshot(cfg.out_dir, "before-run")
    s.parent.mkdir(parents=True, exist_ok=True)
    res = {"backend": backend, "dataset": dataset, "split_seed": cfg.split_seed,
           "run_id": os.environ.get("_QUIBBLE_RUN_ID") or new_run_id(), "timestamp": timestamp(),
           "model_seed": ms, "cv": cfg.cv, "device_requested": cfg.device_requested, "device": cfg.device,
           "threads": cfg.threads, "jobs": cfg.jobs, "preprocess_requested": cfg.preprocess,
           "cv_seed": cfg.cv_seed, "run_type": cfg.run_type, "cache_dir": cfg.cache_dir,
           "single_n": cfg.single_n, "seed_controlled": BACKENDS[backend].seed_controlled,
           "backend_verification": BACKENDS[backend].verification,
           "backend_metadata": backend_metadata(backend), "recipe": recipe,
           "campaign_tier": cfg.tier, "tier": recipe["requested"]["tier"],
           "env": collect(cfg.device == "cuda")}
    if backend == "tabpfn":
        res["tabpfn_max_rows"] = cfg.tabpfn_max_rows
    attempted = cfg.device
    res["device_attempted"] = attempted
    b = None
    try:
        if reason := device_exclusion(backend, attempted):
            raise Refused(f"Excluded from CUDA execution: {reason} Choose --device cpu to run {backend}.")
        if cfg.tier == "tuned" and backend not in TREES:
            from .provenance import WEIGHTS
            if backend not in WEIGHTS:
                raise Refused(f"Tuned tier does not support {backend}; choose defaults or a supported tree.")
            res["tuning"] = {"status": "not supported",
                             "reason": "Foundation adapter unchanged; no tree parameter search",
                             "total_s": 0.0}
        if cfg.threads:
            from threadpoolctl import threadpool_limits
            threadpool_limits(cfg.threads)
        X, y, task = load(dataset, cfg.data_home)
        source = X.attrs.get("quibble_source")
        reviewed = os.environ.get("_QUIBBLE_REVIEWED_SOURCE")
        if reviewed:
            expected = json.loads(reviewed)
            if expected["dataset"] != dataset or (source or {}).get("fingerprint") != expected["fingerprint"]:
                raise ValueError("Dataset changed after browser review; review again before running.")
        if source:
            res["dataset_source"] = source
        relational = X.attrs.get("quibble_relational")
        if relational:
            res.update(relational_audit=relational["audit"], feature_costs=relational["costs"])
        validate_sources(cfg.out_dir, dataset, source)
        res.update(task=task, split=X.attrs.get("quibble_split") or {"kind": "random"})
        if task == "classification":
            res["prediction_classes"] = sorted(y.unique().tolist())
        Xtr, Xte, ytr, yte, te_idx = split(X, y, cfg.split_seed, task, cfg.cv, cfg.cv_seed)
        res.update(n_train_available=len(Xtr), n_test_available=len(Xte))
        if cfg.run_type == "smoke":
            tr = smoke_positions(ytr, 128, cfg.cv_seed + cfg.split_seed, task)
            te = smoke_positions(yte, 32, cfg.cv_seed + cfg.split_seed, task)
            Xtr, ytr = subset_rows(Xtr, tr), ytr.iloc[tr]
            Xte, yte, te_idx = subset_rows(Xte, te), yte.iloc[te], te_idx[te]
        for name, expected in (("idx_test", te_idx), ("y_test", yte.to_numpy())):
            path = s.parent / f"{name}_ss{cfg.split_seed}.npy"
            if path.exists() and not np.array_equal(np.load(path), expected):
                raise ValueError("Existing test rows differ from this split. Use matching CV/seeds "
                                 "or an isolated --out; shared predictions must not be overwritten.")
        res.update(task=task, n_train=len(Xtr), n_test=len(Xte))
        if "quibble_relational" in Xte.attrs:
            selected = Xte.attrs["quibble_relational"]
            res.update(prediction_keys=np.asarray(selected["example_keys"]).tolist(),
                       prediction_timestamps=np.asarray(selected["prediction_timestamps"]).tolist(),
                       entity_groups=np.asarray(Xte.attrs["quibble_groups"]).tolist())
        vram = False
        if attempted == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats()
            vram = True
        rss_before = current_rss_mb()
        t = time.perf_counter()
        if cfg.tier == "tuned" and backend in TREES:
            b = fit_tuned(backend, Xtr, ytr, task, ms, cfg, attempted)
        else:
            b = BACKENDS[backend]()
            b.fit(Xtr, ytr, task, ms, cfg, attempted)
        res["fit_total_s"] = time.perf_counter() - t
        res["load_s"] = float(b.load_s)
        # Tuned fits include inner validation/selection as well as final training.
        res["fit_s"] = max(res["fit_total_s"] - res["load_s"], 0.0)
        if cfg.tier == "tuned" and backend in TREES:
            res.update(search_s=b.info["tuning"]["search_s"],
                       refit_s=b.info["tuning"]["refit"]["duration_s"],
                       search_refit_s=b.info["tuning"]["total_s"])
        if backend in ("tabicl_v2", "mitra_v2"):
            checkpoint = b.info["checkpoint"]
            res["backend_metadata"]["weights"].update(
                checkpoint_id=checkpoint["repository"],
                checkpoint_file=checkpoint["filename"], revision=checkpoint["revision"],
                sha256=checkpoint["sha256"], identity_status="SHA-256 verified before loading",
                terms_scope="Pinned artifact verified locally; usage terms remain separate.")
            if backend == "mitra_v2":
                weights = res["backend_metadata"]["weights"]
                weights["config_sha256"] = checkpoint["config_sha256"]
                weights["terms_url"] = weights["task_terms_urls"][task]
        t = time.perf_counter()
        P = b.predict(Xte)
        batch_s = time.perf_counter() - t
        res["batch_prediction_s"] = batch_s
        res["model_pipeline_s"] = res["fit_total_s"] + batch_s
        res["predict_ms_per_row_batch"] = batch_s / len(Xte) * 1e3
        if getattr(b, "prediction_adapts", False):
            # One predict call may load weights and fit query-dependent models.
            # Repeated warm-up/latency probes would repeat that adaptation.
            res.update(
                prediction_adapts=True, adaptation_and_prediction_s=batch_s,
                prediction_only_s=None, weight_load_s=None, adaptation_s=None,
                single_row_p50_ms=None, single_row_p95_ms=None,
                single_row_latency_available=False,
                predict_ms_per_row_batch_scope="inclusive prediction-time loading/adaptation/inference",
                single_row_latency_unavailable_reason="Prediction adapts; repeated probes are disabled")
        else:
            for i in range(min(3, len(Xte))):  # warm-up: lazy initialisation must not count as latency
                b.predict(Xte.iloc[[i]])
            lat = []
            for i in range(min(cfg.single_n, len(Xte))):
                t = time.perf_counter()
                b.predict(Xte.iloc[[i]])
                lat.append((time.perf_counter() - t) * 1e3)
            res["single_row_p50_ms"] = float(np.percentile(lat, 50))
            res["single_row_p95_ms"] = float(np.percentile(lat, 95))
        res["peak_rss_mb"] = peak_mb()
        if rss_before is not None:
            res["rss_before_fit_mb"] = rss_before
            res["peak_rss_increase_mb"] = max(res["peak_rss_mb"] - rss_before, 0.0)
        if vram:  # only counts PyTorch allocations made in this process
            res["peak_torch_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        res.update(b.info)
        if getattr(b, "prediction_adapts", False):
            res["prediction_only_s"] = None
        else:
            res["prediction_only_s"] = batch_s
        res["recipe"] = resolved_recipe(recipe, b, backend, {
            "dataset": dataset, "task": task, "dataset_fingerprint": (source or {}).get("fingerprint"),
            "split": res["split"], "train_indices_sha256": digest(Xtr.index.tolist()),
            "test_indices_sha256": digest(te_idx.tolist())})
        res.update(metrics(yte.to_numpy(), P, task))
        leaderboard = s.parent / "analysis" / f"leaderboard_{s.name}.csv"
        if b.diagnostics(leaderboard):
            res["leaderboard"] = leaderboard.relative_to(Path(cfg.out_dir)).as_posix()
        res["model_retention"] = "disabled"
        if cfg.keep_models:
            if b.can_save:
                res["model_retention"] = "failed"
                directory = Path(cfg.out_dir) / "artifacts" / dataset / s.name / res["run_id"]
                directory.parent.mkdir(parents=True, exist_ok=True)
                b.save(directory)
                res.update(model_retention="retained",
                           model_path=directory.relative_to(Path(cfg.out_dir)).as_posix())
            else:
                res["model_retention"] = "unsupported"
                print(f"WARNING: fitted-model retention is not supported for {backend}; "
                      "pretrained download caches are retained separately.", file=sys.stderr)
        save_npy(f"{s}.pred.npy", P)
        save_npy(s.parent / f"y_test_ss{cfg.split_seed}.npy", yte.to_numpy())
        save_npy(s.parent / f"idx_test_ss{cfg.split_seed}.npy", te_idx)
    except TuningFailure as e:
        res["tuning"] = e.evidence
        res["error"] = f"TuningFailure: {e}"
        res["error_kind"] = error_kind(e.__cause__ or e)
    except (Refused, UnsupportedSplit) as e:
        res["refused"] = str(e)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["error_kind"] = error_kind(e)
    finally:
        if b is not None:
            try:
                b.close()
            except Exception as e:
                res["cleanup_error"] = f"{type(e).__name__}: {e}"
                if "error" not in res:
                    res["error"] = f"Cleanup failed: {res['cleanup_error']}"
                    res["error_kind"] = error_kind(e)
    js.write_text(json.dumps(res, indent=2))
    if not parent_managed:
        snapshot(cfg.out_dir, "run", include_report=False)
    key = next((k for k in ("accuracy", "rmse") if k in res), None) \
        if "error" not in res and "refused" not in res else None
    status = f"{key}={res[key]:.4f}" if key else (
        "REFUSED " + res["refused"] if "refused" in res else "ERROR " + res.get("error", "?"))
    effective = res.get("device_effective", attempted)
    sys.stdout.write(f"{backend:10s} {dataset:16s} ss{cfg.split_seed} ms{ms} [{effective}] {status}\n")
    sys.stdout.flush()


def sweep(backends, datasets, cfg, force, retry_failed=False, single_split=False):
    from .backends import BACKENDS, select_backends
    from dataclasses import replace
    from .tuning import ensure_output_tier, protocol
    requested_backends = list(backends)
    backends, excluded = select_backends(requested_backends, cfg.device)
    for name, reason in excluded.items():
        print(f"EXCLUDED {name} (device={cfg.device}): {reason}", file=sys.stderr)
    if not backends:
        raise ValueError("No selected methods support GPU-only execution. "
                         "Select CatBoost, TabPFN, TabFM or KumoTabular, or choose --device cpu.")
    ensure_output_tier(cfg)
    cells = [(b, d, ms, ss) for d in datasets for b in backends for ms in cfg.model_seeds
             for ss in ((cfg.split_seed,) if single_split else cfg.split_ids)]
    def finished(c):
        return stem(cfg.out_dir, c[1], c[0], c[3], c[2]).with_suffix(".json")

    for b in backends:
        requested = recipe_snapshot(b, cfg, cfg.model_seeds[0])
        for d in datasets:
            writing = force or any(not finished(c).exists() or
                                  (retry_failed and is_failed(finished(c)))
                                  for c in cells if c[0] == b and c[1] == d)
            for path in (Path(cfg.out_dir) / d).glob(f"{b}_ss*_ms*.json"):
                check_reuse(json.loads(path.read_text()), requested, writing=writing)
    retry = {c for c in cells if retry_failed and finished(c).exists() and is_failed(finished(c))}
    todo = [c for c in cells if force or c in retry or not finished(c).exists()]
    if retry and not force:
        print(f"retrying {len(retry)} cell(s) whose saved result is an error (--retry-failed)")
    print(f"{len(todo)} to run, {len(cells) - len(todo)} already done "
          f"(device={cfg.device}, jobs={cfg.jobs}, threads={cfg.threads or 'all'})")
    stale = {k for c in cells if c not in todo
             for k in settings_differ(stem(cfg.out_dir, c[1], c[0], c[3], c[2]).with_suffix(".json"), cfg)}
    if stale:
        n = sum(1 for c in cells if c not in todo and settings_differ(
            stem(cfg.out_dir, c[1], c[0], c[3], c[2]).with_suffix(".json"), cfg))
        print(STALE_MSG.format(n=n, keys=", ".join(sorted(stale))))
    if todo:
        snapshot(cfg.out_dir, "before-sweep")
    run_id = new_run_id()
    invocation = {
        "run_id": run_id, "timestamp": timestamp(), "run_type": cfg.run_type,
        "workflow": os.environ.get("_QUIBBLE_WORKFLOW", "run"),
        "device_requested": cfg.device_requested, "device_resolved": cfg.device,
        "backends": list(backends), "requested_backends": requested_backends,
        "excluded_backends": excluded, "datasets": list(datasets), "cells": [],
        "backend_metadata": {b: backend_metadata(b) for b in requested_backends},
        "campaign_tier": cfg.tier,
        "tuning_protocol": protocol(cfg) if cfg.tier == "tuned" else {"status": "not requested"},
        "worker_timeout_s": cfg.timeout_s,
    }

    def record_invocation():
        rows = []
        for b, d, ms, ss in cells:
            path = stem(cfg.out_dir, d, b, ss, ms).with_suffix(".json")
            record = json.loads(path.read_text()) if path.exists() else {}
            differences = settings_differ(path, cfg) if record else []
            cached = (b, d, ms, ss) not in todo
            rows.append({"backend": b, "dataset": d, "model_seed": ms, "split_seed": ss,
                         "execution": ("incompatible settings" if differences else "cached")
                         if cached else ("ran" if record.get("run_id") == run_id else "not run"),
                         "differences": differences})
        invocation["cells"] = rows
        directory = Path(cfg.out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "invocation.json").write_text(json.dumps(invocation, indent=2), encoding="utf-8")

    record_invocation()

    def work(c):
        b, d, ms, ss = c
        cmd = [sys.executable, "-m", "quibble_ml.run", "--backend", b, "--dataset", d,
               "--model-seed", str(ms)] + (["--force"] if force or c in retry else [])

        def failure(message, kind, **extra):
            s = stem(cfg.out_dir, d, b, ss, ms)
            s.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "backend": b, "dataset": d, "split_seed": ss, "model_seed": ms,
                "run_id": run_id, "timestamp": timestamp(), "run_type": cfg.run_type,
                "device_requested": cfg.device_requested, "device": cfg.device,
                "device_attempted": cfg.device,
                "cv": cfg.cv, "cv_seed": cfg.cv_seed, "preprocess_requested": cfg.preprocess,
                "backend_metadata": invocation["backend_metadata"][b],
                "recipe": recipe_snapshot(b, replace(cfg, split_seed=ss), ms),
                "campaign_tier": cfg.tier,
                "tier": recipe_snapshot(b, replace(cfg, split_seed=ss), ms)["requested"]["tier"],
                "tuning": {"status": "incomplete", "attempts_recoverable": False,
                           "reason": message} if cfg.tier == "tuned" else {"status": "not requested"},
                "error": message, "error_kind": kind, **extra,
            }
            s.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"{b:10s} {d:16s} ss{ss} ms{ms} ERROR {message}", file=sys.stderr)

        workspace = tempfile.TemporaryDirectory(prefix="quibble-worker-")
        try:
            completed = _worker_process(cmd, timeout=cfg.timeout_s or None,
                           env={**os.environ, "QUIBBLE_SPLIT_SEED": str(ss),
                                "_QUIBBLE_WORKSPACE": workspace.name,
                                "QUIBBLE_DEVICE": cfg.device_requested,
                                "_QUIBBLE_RESOLVED_DEVICE": cfg.device,
                                "QUIBBLE_CV": cfg.cv, "QUIBBLE_CV_SEED": str(cfg.cv_seed),
                                "QUIBBLE_OUT_DIR": cfg.out_dir,
                                "QUIBBLE_KEEP_MODELS": "1" if cfg.keep_models else "0",
                                "QUIBBLE_THREADS": str(cfg.threads), "QUIBBLE_JOBS": str(cfg.jobs),
                                "QUIBBLE_TIMEOUT_S": str(cfg.timeout_s),
                                "QUIBBLE_TIER": cfg.tier, "QUIBBLE_TUNE_TRIALS": str(cfg.tune_trials),
                                "QUIBBLE_TUNE_FOLDS": str(cfg.tune_folds),
                                "QUIBBLE_TUNE_SEED": str(cfg.tune_seed),
                                "QUIBBLE_TUNE_METRIC": cfg.tune_metric,
                                "QUIBBLE_TUNE_TIMEOUT_S": str(cfg.tune_timeout_s),
                                "_QUIBBLE_RUN_TYPE": cfg.run_type,
                                "_QUIBBLE_HISTORY_OUT": str(Path(cfg.out_dir).resolve()),
                                "_QUIBBLE_WORKER": "1",
                                "_QUIBBLE_RUN_ID": run_id})
            path = stem(cfg.out_dir, d, b, ss, ms).with_suffix(".json")
            record = json.loads(path.read_text()) if path.exists() else {}
            if completed.returncode != 0:
                failure(f"Worker exited with code {completed.returncode}; possible resource "
                        "exhaustion or native-library crash; inspect worker output.",
                        "worker_exit", worker_exit_code=completed.returncode)
            elif record.get("run_id") != run_id:
                failure("Worker exited without a result for this invocation; inspect worker output.",
                        "missing_result")
        except subprocess.TimeoutExpired:
            failure(f"timeout after {cfg.timeout_s}s", "timeout")
        except OSError as e:
            failure(f"Could not start worker: {type(e).__name__}: {e}", error_kind(e))
        finally:
            workspace.cleanup()

    # GPU-capable backends run one at a time on cuda (they would fight over VRAM)
    gpu = [c for c in todo if cfg.device == "cuda" and BACKENDS[c[0]].gpu]
    cpu = [c for c in todo if c not in gpu]
    with ThreadPoolExecutor(cfg.jobs) as ex:
        list(ex.map(work, cpu))
    for c in gpu:
        work(c)
    record_invocation()
    if todo:
        snapshot(cfg.out_dir, "sweep", include_report=False)
    return invocation


def main():
    ap = argparse.ArgumentParser()
    from .tuning import add_arguments
    add_arguments(ap)
    ap.add_argument("--backend")
    ap.add_argument("--dataset")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--backends", help="comma list or 'all'")
    ap.add_argument("--datasets", help="comma list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="rerun only cells whose saved result is an error (refusals are kept)")
    ap.add_argument("--keep-models", action="store_const", const="1", default=None,
                    help="retain fitted tree/AutoGluon models; foundation estimators are not serialized")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--preprocess", choices=["ordinal", "native"])
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--cv", help="holdout (default) or KxR, e.g. 5x2 = 10 splits")
    ap.add_argument("--cv-seed", type=int)
    ap.add_argument("--model-seeds")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int)
    a = ap.parse_args()
    for dest, env in FLAGS.items():  # CLI overrides .env by writing into the environment
        v = getattr(a, dest, None)
        if v is not None:
            os.environ[env] = str(v)
    from .backends import BACKENDS
    from .config import Config
    try:
        cfg = Config.from_env()
    except (RuntimeError, ValueError) as e:
        sys.exit(f"config error: {e}")
    if a.sweep:
        bs = [n for n, B in BACKENDS.items() if not getattr(B, "internal", False)] \
            if a.backends == "all" else a.backends.split(",")
        try:
            sweep(bs, a.datasets.split(","), cfg, a.force, retry_failed=a.retry_failed)
        except ValueError as e:
            sys.exit(f"selection error: {e}")
    else:
        if (cfg.tier == "tuned" or a.backend == "mitra_v2") \
                and os.environ.get("_QUIBBLE_WORKER") != "1":
            # Search and prediction-time fitting need an outer deadline, even for one cell.
            from dataclasses import replace
            sweep([a.backend], [a.dataset], replace(cfg, model_seeds=(a.model_seed,)),
                  a.force, a.retry_failed, single_split=True)
        else:
            run_one(a.backend, a.dataset, a.model_seed, cfg, a.force, a.retry_failed)


if __name__ == "__main__":
    main()
