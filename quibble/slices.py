"""Row traits and slice definitions, computed on the full dataset (diagnostics only).

Slices are exploratory: they are chosen using the target (mutual information) and ranked by
the spread between backends, so a large spread in a small slice is often noise."""
import numpy as np
import pandas as pd

from .backends import cat_cols

RARE = 0.01  # a category value seen in under 1% of rows counts as rare


def row_traits(X):
    t = pd.DataFrame(index=range(len(X)))
    t["n_missing"] = X.isna().sum(axis=1).to_numpy()
    t["has_missing"] = t["n_missing"] > 0
    rare = np.zeros(len(X), dtype=bool)
    for c in cat_cols(X):
        col = X[c].astype(object)
        freq = col.map(col.value_counts(normalize=True)).astype(float)
        rare |= ((freq < RARE) & col.notna()).to_numpy()
    t["rare_category"] = rare
    return t


def _top_numeric(num, y, task, k):
    """The k numeric features most informative about the target, by mutual information.

    Mutual information needs no ordering of the labels, so multiclass codes are not treated as
    numbers, and it does not assume a linear relationship. Deterministic (fixed seed)."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    X = num.fillna(num.median()).fillna(0.0)
    if len(X) > 5000:
        keep = np.random.default_rng(0).choice(len(X), 5000, replace=False)
        X, y = X.iloc[keep], np.asarray(y)[keep]
    y = np.asarray(y)
    fn = mutual_info_classif if task == "classification" else mutual_info_regression
    mi = pd.Series(fn(X.to_numpy(), y.astype(int) if task == "classification" else y.astype(float),
                      random_state=0), index=num.columns)
    return list(mi.sort_values(ascending=False, kind="stable").index[:k])


def slice_masks(X, y, task, k=3):
    """{slice name: boolean mask over dataset rows}: missingness, rare categories, and terciles
    of the k numeric features most informative about the target (by mutual information)."""
    t, m = row_traits(X), {}
    if t["has_missing"].any() and (~t["has_missing"]).any():
        m["missing=none"], m["missing=some"] = (~t["has_missing"]).to_numpy(), t["has_missing"].to_numpy()
    if t["rare_category"].any():
        m["rare_category=yes"] = t["rare_category"].to_numpy()
    num = X.select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1]:
        for c in _top_numeric(num, y, task, k):
            v = num[c]
            q1, q2 = v.quantile([1 / 3, 2 / 3])
            m[f"{c}=low"], m[f"{c}=mid"], m[f"{c}=high"] = (
                (v <= q1).to_numpy(), ((v > q1) & (v <= q2)).to_numpy(), (v > q2).to_numpy())
    return m


def unique_rows(sel, idx):
    """Number of distinct dataset rows among the pooled predictions picked by ``sel``.
    With repeated CV one row is predicted several times; slice-size limits count rows, not predictions."""
    return int(len(np.unique(idx[sel])))
