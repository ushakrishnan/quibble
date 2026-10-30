"""Cheap pre-fit data checks. They flag, never block: every flag travels with the verdicts.

  flags = check(X, y, task, split_kind="random")

Checks: a feature that predicts the target almost perfectly on its own, ID-like columns,
duplicate feature rows, and time-like columns when the split is random.
"""
import re

import numpy as np
import pandas as pd

TIME_NAME = re.compile(r"(date|time|timestamp|year|month|week|day|period|quarter)", re.I)
SOLO_AUC, SOLO_ACC, SOLO_R2 = 0.99, 0.99, 0.98   # a single feature this good is a red flag
MAX_FEATURES, MAX_ROWS = 60, 4000


def _solo_score(x, y, task, seed=0):
    """Cross-validated score of a depth-3 tree on one feature (AUC, accuracy or R^2)."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    x = x.to_numpy().reshape(-1, 1)
    if task == "classification":
        n_classes = len(np.unique(y))
        scoring = "roc_auc" if n_classes == 2 else "accuracy"
        model = DecisionTreeClassifier(max_depth=3, random_state=seed)
    else:
        scoring, model = "r2", DecisionTreeRegressor(max_depth=3, random_state=seed)
    try:
        return float(np.mean(cross_val_score(model, x, y, cv=3, scoring=scoring))), scoring
    except ValueError:
        return float("nan"), scoring


def check(X, y, task, split_kind="random"):
    """-> list of {"flag", "column", "detail"} dicts (empty when nothing looks wrong)."""
    flags = []
    y = np.asarray(y)
    n = len(X)
    if n > MAX_ROWS:
        keep = np.random.default_rng(0).choice(n, MAX_ROWS, replace=False)
        Xs, ys = X.iloc[keep].reset_index(drop=True), y[keep]
    else:
        Xs, ys = X.reset_index(drop=True), y

    # a feature that nearly is the answer
    limit = {"roc_auc": SOLO_AUC, "accuracy": SOLO_ACC, "r2": SOLO_R2}
    for c in list(Xs.columns)[:MAX_FEATURES]:
        col = Xs[c]
        if col.nunique(dropna=True) < 2:
            continue
        if not pd.api.types.is_numeric_dtype(col):
            col = pd.Series(pd.factorize(col)[0], index=col.index).astype(float)
        col = col.fillna(col.median() if col.notna().any() else 0.0)
        score, kind = _solo_score(col, ys, task)
        if np.isfinite(score) and score >= limit[kind]:
            flags.append({"flag": "single_feature_predicts_target", "column": c,
                          "detail": f"{kind} {score:.3f} from this feature alone; check it is known "
                                    "at prediction time and is not derived from the target"})

    # ID-like columns
    for c in X.columns:
        col = X[c]
        if n >= 20 and col.nunique(dropna=True) >= 0.98 * n and not pd.api.types.is_float_dtype(col):
            flags.append({"flag": "id_like_column", "column": c,
                          "detail": "almost every value is unique; an ID can let a model memorise rows"})

    # duplicate rows
    dup = float(X.duplicated().mean()) if n else 0.0
    if dup >= 0.05:
        flags.append({"flag": "duplicate_rows", "column": "",
                      "detail": f"{dup:.1%} of rows have identical features; duplicates can appear on both "
                                "sides of a split and inflate scores"})

    # time-like columns with a random split
    if split_kind == "random":
        for c in X.columns:
            if TIME_NAME.search(str(c)):
                flags.append({"flag": "time_like_column_random_split", "column": c,
                              "detail": "the name suggests time ordering but the split is random; a random split "
                                        "lets the model train on the future. Declare time_column in dataset.json "
                                        "and use a temporal split"})
    return flags


def format_flags(flags):
    return [f"[{f['flag']}] {f['column']}: {f['detail']}".replace(" : ", ": ") for f in flags]
