"""Load saved predictions and pool them across splits, aligned between backends."""
import json
from pathlib import Path

import numpy as np


def load_pool(out, dataset, backend, ms):
    """-> ({split_id: {idx, y, P}}, [result json dicts]) for one backend and model seed."""
    d, splits, runs = Path(out) / dataset, {}, []
    for p in sorted(d.glob(f"{backend}_ss*_ms{ms}.json")):
        r = json.loads(p.read_text())
        runs.append(r)
        ss, pred = r["split_seed"], Path(f"{p.with_suffix('')}.pred.npy")
        idx, yy = d / f"idx_test_ss{ss}.npy", d / f"y_test_ss{ss}.npy"
        if "error" not in r and "refused" not in r and pred.exists() and idx.exists() and yy.exists():
            P = np.load(pred)
            classes = r.get("prediction_classes")
            if classes is not None and (P.ndim != 2 or classes != list(range(P.shape[1]))):
                raise ValueError(f"{p.name}: prediction columns do not match recorded class order; "
                                 "regenerate this result.")
            splits[ss] = {"idx": np.load(idx), "y": np.load(yy), "P": P}
    return splits, runs


def coverage_status(required, completed, runs):
    """Describe missing required splits without hiding errors or incomplete artifacts."""
    required = set(required)
    missing = sorted(required - set(completed))
    records = {r["split_seed"]: r for r in runs}
    reasons = {}
    for ss in missing:
        record = records.get(ss)
        if record is None:
            reason = "missing run record"
        elif "error" in record:
            reason = "error: " + record["error"]
        elif "refused" in record:
            reason = "refused: " + record["refused"]
        else:
            reason = "incomplete prediction or shared test artifacts"
        reasons[str(ss)] = reason
    return {"required": sorted(required), "completed": sorted(required & set(completed)),
            "missing": missing, "reasons": reasons}


def seeds_available(out, dataset, backend):
    """Model seeds that have saved results for this backend and dataset, ascending."""
    seeds = set()
    for p in (Path(out) / dataset).glob(f"{backend}_ss*_ms*.json"):
        tail = p.stem.rsplit("_ms", 1)[-1]
        if tail.isdigit() and p.stem.startswith(f"{backend}_ss"):
            seeds.add(int(tail))
    return sorted(seeds)


def align(pools):
    """Keep only splits every backend completed; stack them. Row order is identical across
    backends because a split id fixes the test rows."""
    common = sorted(set.intersection(*[set(p) for p in pools]))
    if not common:
        return None
    sp, idx, y, Ps = [], [], [], [[] for _ in pools]
    for s in common:
        base = pools[0][s]["idx"]
        for k, p in enumerate(pools):
            if not np.array_equal(p[s]["idx"], base):
                raise ValueError(f"split {s}: backends saw different test rows (mixed settings?)")
            Ps[k].append(p[s]["P"])
        sp.append(np.full(len(base), s))
        idx.append(base)
        y.append(pools[0][s]["y"])
    idx = np.concatenate(idx)
    return {"splits": common, "split": np.concatenate(sp), "idx": idx, "y": np.concatenate(y),
            "Ps": [np.concatenate(p) for p in Ps], "row_pos": np.unique(idx, return_inverse=True)[1]}
