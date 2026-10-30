"""Weighted metrics (weights = bootstrap counts) and a cluster bootstrap over data rows."""
import numpy as np
from sklearn.metrics import roc_auc_score

HIGHER = {"accuracy": True, "auc": True, "r2": True, "log_loss": False, "brier": False,
          "rmse": False, "mae": False, "ece": False}
MIN_SPLITS = 5  # fewest splits for the retraining-variability interval
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
    """Cluster bootstrap: resample units of data, so correlated rows stay together.

    By default a unit is a unique data row (a row repeated across CV repeats is one unit).
    With ``cluster`` (one id per unique row, e.g. a customer or account id) a unit is a whole
    cluster, so correlated rows are never treated as independent.
    Captures test-set sampling noise, NOT training-set variability (see corrected_interval).
    After each ``ci`` call, ``valid`` and ``dropped`` say how many replicates were usable."""

    def __init__(self, row_pos, n_boot=500, seed=0, cluster=None):
        self.row_pos = row_pos
        nu = int(row_pos.max()) + 1
        if cluster is None:
            unit = np.arange(nu)
        else:
            unit = np.asarray(cluster)
            if len(unit) != nu:
                raise ValueError("cluster needs one id per unique row")
            unit = np.unique(unit, return_inverse=True)[1]
        nc = int(unit.max()) + 1
        self.counts = np.random.default_rng(seed).multinomial(nc, np.full(nc, 1 / nc), size=n_boot)
        self.unit_of_pred = unit[row_pos]
        self.valid = self.dropped = 0

    def ci(self, fn, alpha=0.05):
        """fn(weights_or_None) -> float. Returns (point, lo, hi); (pt, nan, nan) if fewer than
        20 replicates give a finite value. Dropped replicates are counted, never hidden."""
        pt = fn(None)
        v = np.array([fn(c[self.unit_of_pred]) for c in self.counts], dtype=float)
        ok = np.isfinite(v)
        self.valid, self.dropped = int(ok.sum()), int((~ok).sum())
        v = v[ok]
        if len(v) < 20:
            return pt, np.nan, np.nan
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return pt, float(lo), float(hi)


def corrected_interval(values, n_train, n_test, alpha=0.05):
    """Approximate corrected interval for the mean of a per-split quantity.

    Nadeau-Bengio corrected resampled t interval: the plain variance of the J split values is
    inflated by (1/J + n_test/n_train), because CV training sets overlap. Returns
    (mean, lo, hi), or (mean, nan, nan) with fewer than 5 usable splits.
    The cutoff is an implementation rule, not a precision guarantee. Exchangeability and
    covariance assumptions need validation, especially for forward-chaining temporal folds."""
    from scipy.stats import t
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m, j = float(v.mean()), len(v)
    if j < MIN_SPLITS or not n_train or not n_test or n_train <= 0:
        return m, np.nan, np.nan
    half = float(t.ppf(1 - alpha / 2, j - 1) * np.sqrt(np.var(v, ddof=1) * (1 / j + n_test / n_train)))
    return m, m - half, m + half


def widen(a, b):
    """Union of two (point, lo, hi) intervals' bounds: the wider interval. NaN bounds are ignored,
    and the first point estimate is kept."""
    pt = a[0]
    los = [x for x in (a[1], b[1]) if np.isfinite(x)]
    his = [x for x in (a[2], b[2]) if np.isfinite(x)]
    return pt, (min(los) if los else np.nan), (max(his) if his else np.nan)
