"""Row traits and slice definitions, computed on the full dataset (diagnostics only)."""
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


def slice_masks(X, y, task, k=3):
    """{slice name: boolean mask over dataset rows}: missingness, rare categories, and terciles
    of the k numeric features most correlated with the target."""
    t, m = row_traits(X), {}
    if t["has_missing"].any() and (~t["has_missing"]).any():
        m["missing=none"], m["missing=some"] = (~t["has_missing"]).to_numpy(), t["has_missing"].to_numpy()
    if t["rare_category"].any():
        m["rare_category=yes"] = t["rare_category"].to_numpy()
    num = X.select_dtypes(include="number")
    if num.shape[1]:
        yv = pd.Series(np.asarray(y, dtype=float), index=num.index)
        cor = num.apply(lambda s: abs(s.corr(yv))).fillna(0).sort_values(ascending=False)
        for c in cor.index[:k]:
            v = num[c]
            q1, q2 = v.quantile([1 / 3, 2 / 3])
            m[f"{c}=low"], m[f"{c}=mid"], m[f"{c}=high"] = (
                (v <= q1).to_numpy(), ((v > q1) & (v <= q2)).to_numpy(), (v > q2).to_numpy())
    return m
