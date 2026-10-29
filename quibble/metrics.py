"""Weighted metrics (weights = bootstrap counts) and a cluster bootstrap over data rows."""
import numpy as np
from sklearn.metrics import roc_auc_score

HIGHER = {"accuracy": True, "auc": True, "r2": True, "log_loss": False, "brier": False,
          "rmse": False, "mae": False, "ece": False}
RELATIVE = {"rmse", "mae"}  # degradation of these is reported as a relative increase


def norm(P):
    P = np.clip(P, 1e-7, 1)
    return P / P.sum(1, keepdims=True)


def ece(y, P, w=None, bins=10):
    """Top-label expected calibration error plus the reliability table."""
    P = norm(P)
    w = np.ones(len(y)) if w is None else w
    conf, ok = P.max(1), (P.argmax(1) == y).astype(float)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    tot, e, rows = w.sum(), 0.0, []
    for k in range(bins):
        m = b == k
        wk = w[m].sum()
        if wk == 0:
            continue
        acc, mc = np.average(ok[m], weights=w[m]), np.average(conf[m], weights=w[m])
        e += wk / tot * abs(acc - mc)
        rows.append((k, float(wk), float(mc), float(acc)))
    return float(e), rows


def score(name, y, P, w=None):
    w = np.ones(len(y)) if w is None else w
    if w.sum() == 0:
        return np.nan
    if name in ("rmse", "mae", "r2"):
        e = y - P
        if name == "rmse":
            return float(np.sqrt(np.average(e ** 2, weights=w)))
        if name == "mae":
            return float(np.average(np.abs(e), weights=w))
        den = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        return float(1 - np.sum(w * e ** 2) / den) if den > 0 else np.nan
    P, yi = norm(P), y.astype(int)
    if name == "accuracy":
        return float(np.average(P.argmax(1) == yi, weights=w))
    if name == "log_loss":
        return float(np.average(-np.log(P[np.arange(len(yi)), yi]), weights=w))
    if name == "brier":
        return float(np.average(((P - np.eye(P.shape[1])[yi]) ** 2).sum(1), weights=w))
    if name == "ece":
        return ece(yi, P, w)[0]
    if name == "auc":
        try:
            if P.shape[1] == 2:
                return float(roc_auc_score(yi, P[:, 1], sample_weight=w))
            return float(roc_auc_score(yi, P, multi_class="ovr", labels=list(range(P.shape[1])),
                                       sample_weight=w))
        except ValueError:
            return np.nan
    raise KeyError(name)


def degradation(name, a, b):
    """How much worse b (candidate) is than a (incumbent). Positive = worse.
    Absolute for most metrics, relative for rmse/mae."""
    if HIGHER[name]:
        return a - b
    return (b - a) / a if name in RELATIVE and a else b - a


class Boot:
    """Cluster bootstrap: resample unique data rows, so a row repeated across CV repeats
    stays together. Captures test-set sampling noise, NOT training-set variability."""

    def __init__(self, row_pos, n_boot=500, seed=0):
        self.row_pos = row_pos
        nu = int(row_pos.max()) + 1
        self.counts = np.random.default_rng(seed).multinomial(nu, np.full(nu, 1 / nu), size=n_boot)

    def ci(self, fn, alpha=0.05):
        """fn(weights_or_None) -> float. Returns (point, lo, hi)."""
        pt = fn(None)
        v = np.array([fn(c[self.row_pos]) for c in self.counts], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 20:
            return pt, np.nan, np.nan
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return pt, float(lo), float(hi)
