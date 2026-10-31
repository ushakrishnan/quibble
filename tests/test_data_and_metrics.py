import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from quibble_ml.datasets import load, split
from quibble_ml.metrics import Boot, class_calibration, degradation, ece, score


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


def test_class_calibration_manual_bins_and_label_column_order():
    y = np.array(["yes", "no", "yes", "no"])
    P = np.array([[.8, .2], [.4, .6], [.6, .4], [.2, .8]])
    summary, table = class_calibration(y, P, ["yes", "no"], bins=2)
    assert [r["class_ece"] for r in summary] == pytest.approx([.3, .3])
    assert [r["class_label"] for r in summary] == ["yes", "no"]
    assert all(r["support_status"] == "rare_positive" for r in summary)
    yes = table[:2]
    assert [r["count"] for r in yes] == [2, 2]
    assert [r["mean_probability"] for r in yes] == pytest.approx([.3, .7])
    assert [r["observed_frequency"] for r in yes] == [0, 1]
    reversed_summary, _ = class_calibration(y, P[:, ::-1], ["no", "yes"], bins=2)
    assert [r["class_ece"] for r in reversed_summary] == pytest.approx([.3, .3])


def test_class_calibration_repeats_have_equal_row_weight():
    y, P = np.array([0, 1]), np.array([[.8, .2], [.1, .9]])
    original, _ = class_calibration(y, P, [0, 1], bins=2)
    repeated, table = class_calibration(y[[0, 0, 0, 1]], P[[0, 0, 0, 1]],
                                        [0, 1], row_ids=[7, 7, 7, 9], bins=2)
    assert [r["class_ece"] for r in repeated] == pytest.approx(
        [r["class_ece"] for r in original])
    assert repeated[0]["positive_rows"] == repeated[0]["negative_rows"] == 1
    assert sum(r["weight"] for r in table[:2]) == pytest.approx(2)
    assert sum(r["count"] for r in table[:2]) == 4
    with pytest.raises(ValueError, match="consistent class"):
        class_calibration([0, 1], P, [0, 1], row_ids=[7, 7])


def test_class_calibration_absent_and_empty_are_explicit():
    summary, table = class_calibration(["a", "a"], [[1, 0, 0], [.8, .1, .1]],
                                        ["a", "b", "c"], bins=4)
    assert summary[1]["support_status"] == "absent_positive"
    assert summary[0]["support_status"] == "absent_negative"
    assert all(r["class_ece"] is None for r in summary)
    assert all(r["mean_probability"] is None and r["observed_frequency"] is None
               for r in table if r["count"] == 0)
    empty, table = class_calibration([], np.empty((0, 2)), ["a", "b"])
    assert all(r["support_status"] == "no_predictions" for r in empty)
    assert len(table) == 20 and all(r["class_ece"] is None for r in table)


@pytest.mark.parametrize("P", [
    [[.4, .5]], [[-.1, 1.1]], [[float("nan"), .5]], [[float("inf"), .5]], [[.5]],
])
def test_class_calibration_rejects_malformed_probabilities(P):
    with pytest.raises(ValueError):
        class_calibration([0], P, [0, 1])


def test_class_calibration_multiclass_imbalance_keeps_every_bin():
    y = np.array(["rare"] + ["usual"] * 29)
    P = np.tile([.1, .8, .1], (30, 1))
    summary, table = class_calibration(y, P, ["absent", "usual", "rare"])
    assert len(summary) == 3 and len(table) == 30
    assert [r["support_status"] for r in summary] == [
        "absent_positive", "rare_negative", "rare_positive"]
    assert summary[2]["class_ece"] == pytest.approx(abs(.1 - 1 / 30))


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
