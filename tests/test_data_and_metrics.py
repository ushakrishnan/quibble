import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from quibble.datasets import load, split
from quibble.metrics import Boot, degradation, ece, score


def test_same_split_id_gives_identical_rows_and_ids_differ():
    X, y, task = load("breast_cancer")
    a = split(X, y, 0, task)[4]
    assert np.array_equal(a, split(X, y, 0, task)[4])
    assert not np.array_equal(a, split(X, y, 1, task)[4])


def test_cv_folds_partition_the_data():
    X, y, task = load("breast_cancer")
    tests = [split(X, y, i, task, "3x2", 0)[4] for i in range(6)]
    for rep in (tests[:3], tests[3:]):  # each repeat covers every row exactly once
        allrows = np.concatenate(rep)
        assert sorted(allrows) == list(range(len(X)))
    Xtr, Xte, _, _, te = split(X, y, 0, task, "3x2", 0)
    assert not set(Xtr.index) & set(Xte.index)


def test_regression_split_not_stratified():
    X, y, task = load("diabetes_reg")
    assert task == "regression"
    split(X, y, 0, task)  # must not raise


def test_scores_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    P = np.c_[1 - p, p]
    assert score("accuracy", y, P) == pytest.approx(accuracy_score(y, P.argmax(1)))
    assert score("auc", y, P) == pytest.approx(roc_auc_score(y, p))
    yr, pr = rng.normal(size=100), rng.normal(size=100)
    assert score("rmse", yr, pr) == pytest.approx(mean_squared_error(yr, pr) ** 0.5)


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 50)
    P = rng.dirichlet([1, 1], 50)
    w = rng.integers(0, 4, 50)
    dup = np.repeat(np.arange(50), w)
    for m in ("accuracy", "log_loss", "brier", "ece", "auc"):
        assert score(m, y, P, w) == pytest.approx(score(m, y[dup], P[dup]))


def test_ece_perfectly_calibrated_is_small_and_bounded():
    y = np.array([0] * 50 + [1] * 50)
    P = np.c_[np.full(100, 0.5), np.full(100, 0.5)]
    assert 0 <= ece(y, P)[0] <= 1
    assert ece(y, P)[0] == pytest.approx(0.0, abs=1e-9)


def test_degradation_sign_and_relative():
    assert degradation("accuracy", 0.9, 0.8) == pytest.approx(0.1)  # candidate worse -> positive
    assert degradation("accuracy", 0.8, 0.9) == pytest.approx(-0.1)
    assert degradation("rmse", 10.0, 11.0) == pytest.approx(0.1)  # relative
    assert degradation("brier", 0.2, 0.3) == pytest.approx(0.1)  # absolute


def test_bootstrap_ci_brackets_point_and_keeps_rows_together():
    y = np.array([0, 1] * 50)
    P = np.c_[np.full(100, 0.4), np.full(100, 0.6)]
    row_pos = np.tile(np.arange(50), 2)  # each row appears twice (CV repeats)
    b = Boot(row_pos, 200)
    assert b.counts.shape == (200, 50)  # resamples unique rows, not entries
    pt, lo, hi = b.ci(lambda w: score("accuracy", y, P, w))
    assert lo <= pt <= hi
