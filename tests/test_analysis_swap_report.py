import subprocess
import sys

import numpy as np
import pytest

from quibble_ml.analyze import analyse
from quibble_ml.swap import evaluate


def test_analyse_classification(cv_results):
    r = analyse("breast_cancer", cv_results, quiet=True)
    assert r["task"] == "classification" and r["metric"] == "accuracy"
    assert len(r["splits"]) == 3 and set(r["backends"]) == {"histgb", "lightgbm", "weak"}
    s = r["summary"].set_index("backend")
    assert (s.ci_lo <= s.accuracy).all() and (s.accuracy <= s.ci_hi).all()
    assert s.loc["weak", "accuracy"] < s.loc["histgb", "accuracy"] - 0.2
    assert (r["adir"] / "calibration.csv").exists() and (r["adir"] / "analysis.json").exists()
    assert len(r["pairs"]) == 3
    assert r["relational_slices"] is None
    assert not (r["adir"] / "relational_slices.csv").exists()
    assert len(r["class_calibration_summary"]) == 6
    assert len(r["class_calibration_bins"]) == 60
    assert all(q["n_rows"] == r["n_rows"] for q in r["class_calibration_summary"])
    assert (r["adir"] / "calibration_classwise.csv").exists()
    assert (r["adir"] / "calibration_classwise_summary.csv").exists()


def test_analyse_regression(cv_results):
    r = analyse("diabetes_reg", cv_results, quiet=True)
    assert r["task"] == "regression" and r["metric"] == "rmse"
    assert {"mae", "r2"} <= set(r["summary"].columns)


def test_relational_diagnostics_keep_support_groups_and_do_not_add_gates(monkeypatch, tmp_path):
    import json
    import pandas as pd
    from quibble_ml.config import Config
    from quibble_ml.relational_synthetic import write_fixture
    from quibble_ml.run import run_one
    registry = write_fixture(tmp_path / "tables", repeated_test=True)
    monkeypatch.setenv("QUIBBLE_RELATIONAL_REGISTRY", str(registry))
    monkeypatch.setenv("QUIBBLE_RELATIONAL_CACHE", str(tmp_path / "feature-cache"))
    out = tmp_path / "results"
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(out))
    monkeypatch.setenv("QUIBBLE_CV", "holdout")
    cfg = Config.from_env()
    for backend in ("histgb", "lightgbm"):
        run_one(backend, "relational_synthetic", 0, cfg)
    result = analyse("relational_synthetic", out, quiet=True)
    slices = result["relational_slices"]
    assert len(slices) == 10
    assert set(slices.status) <= {"empty", "limited support", "descriptive"}
    assert (slices.n_rows <= result["n_rows"]).all()
    assert (slices.n_groups <= slices.n_rows).all()
    assert ((slices.n_groups < slices.n_rows) & (slices.n_rows > 0)).any()
    empty = slices[slices.n_rows == 0]
    assert len(empty) and empty[["quality_a", "quality_b", "disagreement_rate"]].isna().all().all()
    saved = json.loads((result["adir"] / "analysis.json").read_text())["relational_slices"]
    assert len(saved) == 10 and (result["adir"] / "relational_slices.csv").exists()
    swap = evaluate("relational_synthetic", "histgb", "lightgbm", 0, out)
    assert not any(row["slice"] in check["check"] for row in saved for check in swap["checks"])
    from quibble_ml.report import build
    report = build(out, ["relational_synthetic"])
    assert "Descriptive relational slices" in report.read_text()
    archives = list((out / "runs").glob("*/*/relational_synthetic/analysis/relational_slices.csv"))
    assert archives and len(pd.read_csv(archives[-1])) == 10


def test_class_calibration_legacy_labels_remain_unknown(cv_results, tmp_path):
    import json
    import shutil
    out = tmp_path / "legacy"
    shutil.copytree(cv_results, out)
    for path in (out / "breast_cancer").glob("*_ss*_ms*.json"):
        record = json.loads(path.read_text())
        record.pop("prediction_classes", None)
        path.write_text(json.dumps(record))
    result = analyse("breast_cancer", out, quiet=True)
    assert all(r["class_mapping"] == "legacy_unrecorded" and r["class_label"] is None
               for r in result["class_calibration_summary"])
    assert all(r["class_ece"] is not None for r in result["class_calibration_summary"])


def test_class_calibration_complete_exports_survive_archive(cv_results):
    import json
    from quibble_ml.report import build
    path = build(cv_results, ["breast_cancer"])
    archives = sorted((path.parent.parent / "runs").glob("*/*/snapshot.json"))
    assert archives
    archive = next(p.parent for p in reversed(archives)
                   if json.loads(p.read_text())["reason"] == "report")
    saved = archive / "breast_cancer" / "analysis"
    assert (saved / "calibration_classwise.csv").exists()
    assert (saved / "calibration_classwise_summary.csv").exists()
    assert len(json.loads((saved / "analysis.json").read_text())["class_calibration_bins"]) == 60
    assert "calibration_classwise.csv" in (archive / "report" / "report.md").read_text()


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


def test_prediction_adaptation_cannot_skip_unavailable_latency_budget(cv_results, tmp_path):
    import json
    import shutil
    out = tmp_path / "adaptation"
    shutil.copytree(cv_results, out)
    for path in (out / "diabetes_reg").glob("histgb_ss*_ms*.json"):
        record = json.loads(path.read_text())
        record.update(prediction_adapts=True, single_row_latency_available=False,
                      single_row_p50_ms=None, single_row_p95_ms=None)
        path.write_text(json.dumps(record))
    result = evaluate("diabetes_reg", "histgb", "histgb", 0, out)
    latency = next(c for c in result["checks"] if c["check"].startswith("latency ratio"))
    assert latency["verdict"] == "INCONCLUSIVE"
    assert "Prediction-time adaptation" in latency["note"]
    assert result["overall"] == "INCONCLUSIVE"


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
        return subprocess.run([sys.executable, "-m", "quibble_ml", "swap", "--incumbent", "histgb",
                               "--candidate", cand, "--dataset", "breast_cancer",
                               "--out", str(cv_results)], capture_output=True).returncode
    assert code("histgb") == 0
    assert code("weak") == 1


def test_report_has_all_sections(cv_results):
    from quibble_ml.report import build
    path = build(cv_results, swaps=[("histgb", "weak")])
    text = path.read_text()
    for needle in ("# quibble report", "## breast_cancer", "## diabetes_reg", "### Quality and cost",
                   "### Interpretation and recommended use", "We should",
                   "### Where the backends disagree", "What failed or refused", "Swap check: histgb -> weak",
                   "**FAIL**", "How to read this"):
        assert needle in text, needle
    assert "### Per-class calibration" in text
    assert "calibration_classwise_summary.csv" in text
    assert "no new verdict" in text or "new verdict" in text
    pytest.importorskip("matplotlib")
    assert (path.parent / "fig_breast_cancer_quality_cost.png").exists()
    assert (path.parent / "fig_breast_cancer_disagreement.png").exists()
    assert (path.parent / "fig_breast_cancer_class_calibration.png").exists()


def test_report_shows_refusals_in_failure_matrix(cv_results, tmp_path, monkeypatch):
    import shutil
    from quibble_ml.config import Config
    from quibble_ml.report import build
    from quibble_ml.run import run_one
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
