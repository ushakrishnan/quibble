"""Versioned equal-trial tree search. This API never accepts outer-test data.

The worker's subprocess timeout is the hard limit. Optional search deadlines
are cooperative checks around fits, not interruptible wall-time guarantees.
"""
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

VERSION = "tree-equal-trials-v1"
TREES = ("histgb", "lightgbm", "catboost")
# Deliberately finite and frozen: changing a value requires a new VERSION.
DOMAINS = {
    "histgb": {
        "max_leaf_nodes": [15, 31, 63], "max_depth": [None, 4, 8],
        "learning_rate": [0.03, 0.1], "max_iter": [50, 100, 200],
        "l2_regularization": [0.0, 1.0, 10.0], "min_samples_leaf": [10, 20, 40],
        # Library-internal random early-stopping splits would violate time/group structure.
        "early_stopping": [False],
    },
    "lightgbm": {
        "num_leaves": [15, 31, 63], "max_depth": [4, 8, -1],
        "learning_rate": [0.03, 0.1], "n_estimators": [50, 100, 200],
        "reg_lambda": [0.0, 1.0, 10.0], "min_child_samples": [10, 20, 40],
    },
    "catboost": {
        "depth": [4, 6, 8], "learning_rate": [0.03, 0.1],
        "iterations": [50, 100, 200], "l2_leaf_reg": [1.0, 3.0, 10.0],
        "min_data_in_leaf": [10, 20, 40], "grow_policy": ["Depthwise"],
        "use_best_model": [False],
    },
}


class TuningFailure(RuntimeError):
    """A failed/incomplete search, with all completed attempt provenance attached."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def protocol(cfg, backend=None):
    """JSON-stable requested protocol, also included in recipe hashes."""
    return {
        "version": VERSION, "protocol": "equal trials, not equal realized compute",
        "trials": cfg.tune_trials, "inner_folds": cfg.tune_folds, "seed": cfg.tune_seed,
        "metric": cfg.tune_metric, "auto_metric": {"classification": "log_loss", "regression": "rmse"},
        "direction": "minimize", "aggregation": "unweighted mean of inner-fold scores",
        "tie_breaker": "lowest trial index (seeded candidate order)",
        "sampling": "numpy RandomState permutation of sorted sklearn ParameterGrid; no replacement",
        "domains": deepcopy(DOMAINS[backend] if backend else DOMAINS),
        "inner_split": "class-safe stratified/group/strict forward time; purge rows; group disjoint",
        "early_stopping": "disabled; fixed searched iteration counts",
        "cooperative_timeout_s": cfg.tune_timeout_s,
        "timeout_semantics": "checks around inner fits and refit; outer worker timeout is authoritative",
    }


def add_arguments(parser):
    parser.add_argument("--tier", choices=("defaults", "tuned"),
                        help="defaults preserves existing fits; tuned opts into tree-only search")
    parser.add_argument("--tune-trials", type=int, help="candidate budget per tree (default 20)")
    parser.add_argument("--tune-folds", type=int, help="inner fold budget (default 3)")
    parser.add_argument("--tune-seed", type=int, help="candidate and inner split seed (default 0)")
    parser.add_argument("--tune-metric", choices=("auto", "log_loss", "rmse"))
    parser.add_argument("--tune-timeout-s", type=float,
                        help="cooperative search deadline; NOT a hard fit timeout (default disabled)")


def ensure_output_tier(cfg):
    """One tier/protocol per root, including failed records and unchanged foundations."""
    from .provenance import digest
    root = Path(cfg.out_dir)
    marker = root / ".quibble-tier.json"
    expected = {"tier": cfg.tier}
    if cfg.tier == "tuned":
        expected["search_sha256"] = digest(protocol(cfg))
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Tier/search protocol mismatch: use a separate --out root.")
        return
    for path in root.glob("*/*_ss*_ms*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        saved = record.get("campaign_tier", record.get("recipe", {}).get("requested", {}).get(
            "tier", "defaults"))
        if saved != cfg.tier:
            raise ValueError("Tier mismatch: defaults and tuned require separate --out roots.")
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is safe when several independently launched workers share a root.
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, sort_keys=True)
    except FileExistsError:
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Concurrent tier/search mismatch: use a separate --out root.")


def load_tier_pair(defaults_root, tuned_root, dataset, backend, split_seed, model_seed):
    """Read an explicit paired cross-tier cell without copying or pooling roots.

    Returns the two records and prediction arrays plus shared test targets/row
    IDs. Reports can calculate paired contrasts from this read-only bundle.
    Incompatible outer protocols, sources, rows or targets are rejected.
    """
    from .provenance import validate_recipe
    from .run import stem
    if backend not in TREES or Path(defaults_root).resolve() == Path(tuned_root).resolve():
        raise ValueError("Tier comparisons require a supported tree and two separate roots")
    bundle = {}
    requested = []
    contexts = []
    arrays = []
    for tier, root in (("defaults", defaults_root), ("tuned", tuned_root)):
        path = stem(root, dataset, backend, split_seed, model_seed)
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if "error" in record or "refused" in record:
            raise ValueError(f"{tier} cell is failed/refused; no partial tier comparison")
        recipe = validate_recipe(record.get("recipe"))
        request = recipe["requested"]
        if request["tier"] != tier or record.get("campaign_tier", request["tier"]) != tier:
            raise ValueError(f"Expected {tier} evidence in its explicitly selected root")
        if request["backend"] != backend or record.get("backend") != backend or \
                record.get("dataset") != dataset or request["cell"] != {
                    "split_seed": split_seed, "model_seed": model_seed}:
            raise ValueError("Tier comparison cell identity mismatch")
        if recipe["resolved"].get("status") != "fitted":
            raise ValueError("Tier comparison needs completed fitted recipes")
        if tier == "tuned" and record.get("tuning", {}).get("status") != "ok":
            raise ValueError("Tuned cell has no complete search evidence")
        requested.append({k: request[k] for k in (
            "preprocess", "device", "cv", "cv_seed", "run_type", "cell", "packages")})
        contexts.append({
            "context": recipe["resolved"]["context"], "split": record.get("split"),
            "source": (record.get("dataset_source") or {}).get("fingerprint"),
            "task": record.get("task"), "classes": record.get("prediction_classes"),
        })
        targets = np.load(path.parent / f"y_test_ss{split_seed}.npy", allow_pickle=False)
        indices = np.load(path.parent / f"idx_test_ss{split_seed}.npy", allow_pickle=False)
        prediction = np.load(f"{path}.pred.npy", allow_pickle=False)
        if len(targets) != len(indices) or len(prediction) != len(targets):
            raise ValueError("Tier comparison prediction/target/row lengths differ")
        arrays.append((targets, indices, prediction.shape))
        bundle[tier] = {"record": record, "predictions": prediction}
    if requested[0] != requested[1] or contexts[0] != contexts[1]:
        raise ValueError("Tier comparison outer protocol/source/context differs")
    if not np.array_equal(arrays[0][0], arrays[1][0]) or \
            not np.array_equal(arrays[0][1], arrays[1][1]) or arrays[0][2] != arrays[1][2]:
        raise ValueError("Tier comparison test targets/rows/probability shapes differ")
    bundle.update(y_test=arrays[0][0], idx_test=arrays[0][1])
    return bundle


def inner_splits(X, y, task, folds, seed):
    """Only outer-training positions, rejecting unsafe or impossible folds."""
    from sklearn.model_selection import KFold, StratifiedKFold
    from .datasets import _balanced_group_folds
    if folds < 2 or len(X) != len(y):
        raise ValueError("Inner search needs aligned training data and at least two folds")
    if "quibble_relational" in X.attrs:
        raise ValueError("Relational inner tuning requires an official training-only split and "
                         "label-horizon purge contract; random fallback is forbidden")
    spec = X.attrs.get("quibble_split") or {"kind": "random"}
    kind = spec["kind"]
    groups = X.attrs.get("quibble_groups")
    times = X.attrs.get("quibble_times")
    for key, values in (("groups", groups), ("times", times)):
        if values is not None and len(values) != len(X):
            raise ValueError(f"Inner {key} metadata does not align with outer training")
    idx = np.arange(len(X))
    if kind == "temporal":
        if times is None:
            raise ValueError("Temporal inner splits require training-only times")
        times = np.asarray(times)
        order = np.argsort(times, kind="stable")
        edges = np.linspace(0, len(X), folds + 2).round().astype(int)
        gap = spec.get("purge_gap", 0)
        if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
            raise ValueError("purge_gap must be a nonnegative row count")
        pairs = []
        for f in range(folds):
            va = order[edges[f + 1]:edges[f + 2]]
            tr = order[:max(0, edges[f + 1] - gap)]
            if len(va):
                tr = tr[times[tr] < times[va].min()]  # ties cannot straddle the boundary
            if groups is not None:
                tr = tr[~np.isin(np.asarray(groups)[tr], np.asarray(groups)[va])]
            pairs.append((tr, va))
    elif kind == "group":
        if groups is None:
            raise ValueError("Group inner splits require training-only groups")
        assignment = _balanced_group_folds(np.asarray(groups), folds, seed)
        pairs = [(idx[assignment != f], idx[assignment == f]) for f in range(folds)]
    elif kind == "random":
        if groups is not None or times is not None:
            raise ValueError("Structural metadata cannot be silently ignored by random inner splits")
        if task == "classification":
            if y.value_counts().min() < folds:
                raise ValueError("Inner classification needs at least inner_folds rows per class")
            pairs = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
        else:
            pairs = list(KFold(folds, shuffle=True, random_state=seed).split(X))
    else:
        raise ValueError(f"Unsupported inner split kind: {kind}")
    expected = set(y.unique())
    for tr, va in pairs:
        if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
            raise ValueError("Inner fold is empty or overlaps")
        if groups is not None and np.intersect1d(np.asarray(groups)[tr], np.asarray(groups)[va]).size:
            raise ValueError("Inner fold has group overlap")
        if kind == "temporal" and not np.max(times[tr]) < np.min(times[va]):
            raise ValueError("Inner fold has time reversal or tied boundary")
        if task == "classification" and (len(expected) < 2 or set(y.iloc[tr].unique()) != expected
                                         or set(y.iloc[va].unique()) != expected):
            raise ValueError("Inner fold is missing classification classes; no random fallback")
    return pairs


def fit_tuned(backend, X, y, task, model_seed, cfg, device):
    """Return a refitted winner; never inspect or accept an outer test partition."""
    from sklearn.model_selection import ParameterGrid
    from .backends import BACKENDS
    from .datasets import subset_rows
    from .metrics import score
    from .provenance import digest
    started = time.perf_counter()
    evidence = {
        "status": "searching", "protocol": protocol(cfg, backend),
        "model_seed": model_seed, "search_seed": cfg.tune_seed,
        "attempts": [], "selected_trial": None, "attempted_fits": 0, "successful_fits": 0,
        "requested_candidate_fits": cfg.tune_trials * cfg.tune_folds,
        "requested_refit_fits": 1, "deadline_overrun": False,
    }
    active = None

    def check_deadline():
        if cfg.tune_timeout_s and time.perf_counter() - started >= cfg.tune_timeout_s:
            evidence["deadline_overrun"] = True
            raise TimeoutError("Cooperative tuning deadline exceeded; search/refit incomplete")

    try:
        metric = {"classification": "log_loss", "regression": "rmse"}[task]
        if cfg.tune_metric not in ("auto", metric):
            raise ValueError(f"Selection metric {cfg.tune_metric} is incompatible with {task}")
        evidence["metric"] = metric
        evidence["split_spec"] = deepcopy(X.attrs.get("quibble_split") or {"kind": "random"})
        evidence["outer_training_indices_sha256"] = digest(X.index.tolist())
        pairs = inner_splits(X, y, task, cfg.tune_folds, cfg.tune_seed)
        evidence["inner_splits"] = [
            {"id": f, "train_rows": len(tr), "validation_rows": len(va),
             "train_indices_sha256": digest(X.index[tr].tolist()),
             "validation_indices_sha256": digest(X.index[va].tolist())}
            for f, (tr, va) in enumerate(pairs)]
        grid = list(ParameterGrid(DOMAINS[backend]))
        if not 1 <= cfg.tune_trials <= len(grid):
            raise ValueError(f"Requested trials must be 1..{len(grid)} for {backend}")
        candidates = np.random.RandomState(cfg.tune_seed).permutation(len(grid))[:cfg.tune_trials]
        best = None
        for trial, candidate in enumerate(candidates):
            check_deadline()
            attempt = {"trial": trial, "candidate": int(candidate), "params": grid[candidate],
                       "model_seed": model_seed, "search_seed": cfg.tune_seed,
                       "metric": metric, "folds": [], "status": "running"}
            evidence["attempts"].append(attempt)
            trial_started = time.perf_counter()
            for f, (tr, va) in enumerate(pairs):
                check_deadline()
                fold = {"split_id": f, "model_seed": model_seed, "status": "running"}
                attempt["folds"].append(fold)
                fold_started = time.perf_counter()
                evidence["attempted_fits"] += 1
                try:
                    active = BACKENDS[backend]()
                    active.tuning_params = dict(attempt["params"])
                    active.fit(subset_rows(X, tr), y.iloc[tr], task, model_seed, cfg, device)
                    prediction = active.predict(subset_rows(X, va))
                    value = score(metric, y.iloc[va].to_numpy(), prediction)
                    if value is None or not np.isfinite(value):
                        raise ValueError("Non-finite inner selection metric")
                    fold.update(status="ok", value=float(value))
                    evidence["successful_fits"] += 1
                except Exception as exc:
                    fold.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                finally:
                    fold["duration_s"] = time.perf_counter() - fold_started
                    if active is not None:
                        try:
                            active.close()
                        except Exception as exc:
                            fold.update(status="failed", cleanup_failure=f"{type(exc).__name__}: {exc}")
                    active = None
                check_deadline()
            attempt["duration_s"] = time.perf_counter() - trial_started
            if all(f["status"] == "ok" for f in attempt["folds"]):
                attempt.update(status="ok", value=float(np.mean([f["value"] for f in attempt["folds"]])))
                if best is None or attempt["value"] < best["value"]:
                    best = attempt
            else:
                attempt["status"] = "failed"
        evidence["search_s"] = time.perf_counter() - started
        if best is None:
            raise ValueError("No valid tuning candidate; final refit was not attempted")
        evidence["selected_trial"] = best["trial"]
        evidence["selected_params"] = dict(best["params"])
        check_deadline()
        active = BACKENDS[backend]()
        active.tuning_params = dict(best["params"])
        refit_started = time.perf_counter()
        evidence["refit"] = {"status": "running", "model_seed": model_seed,
                             "params": dict(best["params"]), "train_rows": len(X)}
        evidence["attempted_fits"] += 1
        try:
            active.fit(X, y, task, model_seed, cfg, device)
            evidence["successful_fits"] += 1
            evidence["refit"]["status"] = "ok"
        except Exception as exc:
            evidence["refit"].update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            evidence["refit"]["duration_s"] = time.perf_counter() - refit_started
        check_deadline()
        evidence.update(status="ok", total_s=time.perf_counter() - started)
        active.info = {**active.info, "tuning": evidence}
        return active
    except Exception as exc:
        for attempt in evidence["attempts"]:
            if attempt["status"] == "running":
                attempt.update(status="incomplete", failure=f"{type(exc).__name__}: {exc}",
                               duration_s=time.perf_counter() - trial_started)
        evidence.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                        total_s=time.perf_counter() - started)
        evidence.setdefault("search_s", evidence["total_s"])
        if active is not None:
            try:
                active.close()
            except Exception as cleanup:
                evidence["cleanup_failure"] = str(cleanup)
        raise TuningFailure(str(exc), evidence) from exc
