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


def class_calibration(y, P, classes, row_ids=None, bins=10):
    """One-vs-rest reliability, with equal total weight per distinct source row.

    Repeated predictions divide their row's weight, without averaging probabilities.
    Empty bins and one-sided class support have no fabricated calibration estimate.
    These are descriptive estimates, not confidence intervals or verdict checks.
    """
    y, P, classes = np.asarray(y), np.asarray(P, dtype=float), list(classes)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("Class calibration requires a positive integer bin count")
    if P.ndim != 2 or y.ndim != 1 or len(y) != len(P) or P.shape[1] != len(classes):
        raise ValueError("Class calibration requires matching targets, probabilities and class columns")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Class calibration requires at least two distinct recorded class labels")
    if not set(y.tolist()).issubset(set(classes)):
        raise ValueError("Class calibration targets contain labels absent from the probability columns")
    if not np.isfinite(P).all() or (P < 0).any() or (P > 1).any() \
            or not np.allclose(P.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Class calibration requires finite probabilities in [0,1] summing to one")
    ids = np.arange(len(y)) if row_ids is None else np.asarray(row_ids)
    if ids.ndim != 1 or len(ids) != len(y):
        raise ValueError("Class calibration requires one source-row identifier per prediction")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1 / counts[inverse] if len(y) else np.empty(0)
    for label in classes:
        positive = y == label
        if np.intersect1d(inverse[positive], inverse[~positive]).size:
            raise ValueError("Repeated source rows must have consistent class targets")
    summary, table = [], []
    for col, label in enumerate(classes):
        label = label.item() if isinstance(label, np.generic) else label
        positive = y == label
        n_pos, n_neg = len(np.unique(inverse[positive])), len(np.unique(inverse[~positive]))
        status = ("no_predictions" if not len(y) else "absent_positive" if not n_pos
                  else "absent_negative" if not n_neg else "rare_positive" if n_pos < 20
                  else "rare_negative" if n_neg < 20 else "ok")
        base = {"class_column": col, "class_label": label, "positive_rows": n_pos,
                "negative_rows": n_neg, "entries": len(y), "n_rows": len(counts),
                "weighting": "equal_unique_row", "support_status": status}
        assigned = np.minimum((P[:, col] * bins).astype(int), bins - 1)
        error = 0.0
        for k in range(bins):
            mask = assigned == k
            mass = float(w[mask].sum())
            mean = float(np.average(P[mask, col], weights=w[mask])) if mass else None
            observed = float(np.average(positive[mask], weights=w[mask])) if mass else None
            if mass:
                error += mass / len(counts) * abs(mean - observed)
            table.append({**base, "bin": k, "bin_lower": k / bins, "bin_upper": (k + 1) / bins,
                          "upper_inclusive": k == bins - 1, "count": int(mask.sum()),
                          "unique_rows": len(np.unique(inverse[mask])), "weight": mass,
                          "positive_weight": float(w[mask & positive].sum()),
                          "negative_weight": float(w[mask & ~positive].sum()),
                          "mean_probability": mean, "observed_frequency": observed})
        value = float(error) if n_pos and n_neg else None
        summary.append({**base, "class_ece": value, "bins": bins})
        for record in table[-bins:]:
            record["class_ece"] = value
    return summary, table


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
