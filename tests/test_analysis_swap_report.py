import subprocess
import sys

import numpy as np
import pytest

from quibble.analyze import analyse
from quibble.swap import evaluate


def test_analyse_classification(cv_results):
    r = analyse("breast_cancer", cv_results, quiet=True)
    assert r["task"] == "classification" and r["metric"] == "accuracy"
    assert len(r["splits"]) == 3 and set(r["backends"]) == {"histgb", "lightgbm", "weak"}
    s = r["summary"].set_index("backend")
    assert (s.ci_lo <= s.accuracy).all() and (s.accuracy <= s.ci_hi).all()
    assert s.loc["weak", "accuracy"] < s.loc["histgb", "accuracy"] - 0.2
    assert (r["adir"] / "calibration.csv").exists() and (r["adir"] / "analysis.json").exists()
    assert len(r["pairs"]) == 3


def test_analyse_regression(cv_results):
    r = analyse("diabetes_reg", cv_results, quiet=True)
    assert r["task"] == "regression" and r["metric"] == "rmse"
    assert {"mae", "r2"} <= set(r["summary"].columns)


def test_disagreement_csv_holds_only_disputed_rows(cv_results):
    import pandas as pd
    r = analyse("breast_cancer", cv_results, quiet=True)
    a, b, s = next(p for p in r["pairs"] if p[:2] == ("histgb", "weak"))
    df = pd.read_csv(r["adir"] / s["csv"])
    assert len(df) == s["disputed"] > 0 and df["disputed"].all()
    assert any(c.startswith("x_") for c in df.columns)
    assert np.all(df.pred_a != df.pred_b)


def test_self_swap_passes(cv_results):
    r = evaluate("breast_cancer", "histgb", "histgb", 0, cv_results)
    assert r["overall"] == "PASS"
    assert all(c["verdict"] == "PASS" for c in r["checks"])


@pytest.mark.parametrize("ds", ["breast_cancer", "diabetes_reg"])
def test_deliberately_weak_candidate_fails(cv_results, ds):
    r = evaluate(ds, "histgb", "weak", 0, cv_results)
    assert r["overall"] == "FAIL"
    assert next(c for c in r["checks"] if c["check"].startswith("quality"))["verdict"] == "FAIL"


def test_threshold_inside_the_interval_is_inconclusive(cv_results):
    import re
    base = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results)
    q = next(c for c in base["checks"] if c["check"].startswith("quality"))
    pt, lo, hi = (float(x) for x in re.findall(r"[-+]\d+\.\d+", q["observed"]))
    assert lo < hi
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, cv_results, {"max_quality_drop": (lo + hi) / 2})
    q2 = next(c for c in r["checks"] if c["check"].startswith("quality"))
    assert q2["verdict"] == "INCONCLUSIVE"  # limit sits inside the 95% interval


def test_threshold_outside_the_interval_is_decisive(cv_results):
    r = evaluate("breast_cancer", "histgb", "weak", 0, cv_results, {"max_quality_drop": 0.9})
    q = next(c for c in r["checks"] if c["check"].startswith("quality"))
    assert q["verdict"] == "PASS"  # the whole interval is below a 0.9 allowance


def test_candidate_missing_splits_fails_coverage(cv_results, tmp_path):
    import shutil
    out = tmp_path / "copy"
    shutil.copytree(cv_results, out)
    for p in (out / "breast_cancer").glob("lightgbm_ss1_ms0.*"):
        p.unlink()
    r = evaluate("breast_cancer", "histgb", "lightgbm", 0, out)
    cov = next(c for c in r["checks"] if c["check"] == "coverage")
    assert cov["verdict"] == "FAIL" and r["overall"] == "FAIL"


def test_swap_cli_exit_codes(cv_results):
    def code(cand):
        return subprocess.run([sys.executable, "-m", "quibble", "swap", "--incumbent", "histgb",
                               "--candidate", cand, "--dataset", "breast_cancer",
                               "--out", str(cv_results)], capture_output=True).returncode
    assert code("histgb") == 0
    assert code("weak") == 1


def test_report_has_all_sections(cv_results):
    from quibble.report import build
    path = build(cv_results, swaps=[("histgb", "weak")])
    text = path.read_text()
    for needle in ("# quibble report", "## breast_cancer", "## diabetes_reg", "### Quality and cost",
                   "### Interpretation and recommended use", "We should",
                   "### Where the backends disagree", "What failed or refused", "Swap check: histgb -> weak",
                   "**FAIL**", "How to read this"):
        assert needle in text, needle
    pytest.importorskip("matplotlib")
    assert (path.parent / "fig_breast_cancer_quality_cost.png").exists()
    assert (path.parent / "fig_breast_cancer_disagreement.png").exists()


def test_report_shows_refusals_in_failure_matrix(cv_results, tmp_path, monkeypatch):
    import shutil
    from quibble.config import Config
    from quibble.report import build
    from quibble.run import run_one
    out = tmp_path / "r"
    shutil.copytree(cv_results, out)
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    monkeypatch.setenv("QUIBBLE_TABPFN_MAX_ROWS", "10")
    monkeypatch.setenv("QUIBBLE_CV", "3x1")
    for ss in range(3):
        monkeypatch.setenv("QUIBBLE_SPLIT_SEED", str(ss))
        run_one("tabpfn", "breast_cancer", 0, Config.from_env())
    text = build(out, ["breast_cancer"]).read_text()
    assert "refused 3/3" in text and "tabpfn" in text
