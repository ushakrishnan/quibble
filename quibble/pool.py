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
            splits[ss] = {"idx": np.load(idx), "y": np.load(yy), "P": np.load(pred)}
    return splits, runs


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
