"""Evidence-based recommendation wording and matching of quality/cost run populations."""
import numpy as np
import pandas as pd
import pytest

from quibble.report import build, interpretation


def analysis(metric="accuracy", values=(0.90, 0.95), names=("fast", "quality")):
    return {
        "dataset": "example", "task": "classification" if metric == "accuracy" else "regression",
        "metric": metric, "model_seed": 0, "splits": [0], "backends": list(names),
        "summary": pd.DataFrame({"backend": names, metric: values,
                                 "ci_lo": np.array(values) - 0.1,
                                 "ci_hi": np.array(values) + 0.1}),
        "dropped": {}, "pairs": [], "calibration": [], "slices": pd.DataFrame(),
        "meta": {"rows": 100, "features": 4, "tier": "small",
                 "cat_share": 0.0, "missing_share": 0.0},
    }


def costs(names=("fast", "quality"), latency=(1., 10.), memory=(100., 500.)):
    return pd.DataFrame({"backend": names, "single_row_p50_ms": latency, "peak_rss_mb": memory})


@pytest.mark.parametrize(("metric", "values", "direction"), [
    ("accuracy", (0.90, 0.95), "highest"),
    ("rmse", (10., 5.), "lowest"),
])
def test_quality_and_resource_use_cases_are_grounded_in_measurements(metric, values, direction):
    text = interpretation(analysis(metric, values), costs(), [], [])
    assert f"shortlist `quality`, since it has the {direction} observed `{metric}`" in text
    assert "evaluate `fast` for latency-sensitive single-row prediction" in text
    assert "is 1 ms" in text
    assert "evaluate `fast` for memory-constrained processes" in text
    assert "is 100 MB" in text
    assert f"Measured quality: `fast`: {metric}={values[0]:.4g}" in text
    assert "only if that quality meets the application's requirements" in text
    assert "No evaluated swap verdict is available" in text


def test_tied_quality_prefers_lower_cost_without_claiming_equivalence():
    res = analysis(values=(0.9, 0.95, 0.95), names=("fast", "kumo_tabular", "tabfm"))
    cost = costs(("fast", "kumo_tabular", "tabfm"), (1., 30., 1000.), (100., 600., 10000.))
    text = interpretation(res, cost, [], [])
    assert "shortlist `kumo_tabular`, `tabfm`, since they share" in text
    assert "prefer `kumo_tabular` for further evaluation" in text
    assert "not proof of superiority or equivalence" in text


def test_conflicting_cost_tradeoffs_do_not_invent_a_tie_break_winner():
    res = analysis(values=(0.95, 0.95))
    text = interpretation(res, costs(memory=(500., 100.)), [], [])
    assert "prefer `" not in text
    assert "evaluate `fast` for latency-sensitive" in text
    assert "evaluate `quality` for memory-constrained" in text


def test_parallel_measurements_do_not_produce_resource_recommendations():
    text = interpretation(analysis(values=(0.95, 0.95)), costs(), [{"jobs": 2}], [])
    assert "rerun with `--jobs 1`" in text and "contention" in text
    assert "for latency-sensitive single-row prediction, since" not in text
    assert "for memory-constrained processes, since" not in text
    assert "prefer `" not in text


@pytest.mark.parametrize("value", [np.nan, np.inf, 0., -1.])
def test_invalid_resource_measurements_do_not_create_recommendations(value):
    text = interpretation(analysis(), costs(latency=(value, value), memory=(value, value)), [], [])
    assert "No latency-sensitive single-row prediction shortlist is available" in text
    assert "No memory-constrained processes shortlist is available" in text
    assert "We should evaluate" not in text


def test_missing_cost_fields_are_explained():
    text = interpretation(analysis(), pd.DataFrame({"backend": ["fast", "quality"]}), [], [])
    assert "`single_row_p50_ms` measurements are missing" in text
    assert "`peak_rss_mb` measurements are missing" in text


def test_nonfinite_quality_estimates_do_not_create_a_winner():
    text = interpretation(analysis(values=(np.nan, np.inf)), costs(), [], [])
    assert "No model recommendation is available" in text
    assert "shortlist `" not in text


def test_internal_weak_backend_is_not_recommended_even_if_fast_or_passing():
    res = analysis(values=(1., 0.9), names=("weak", "quality"))
    swap = {"overall": "PASS", "checks": []}
    text = interpretation(res, costs(names=("weak", "quality")), [], [("quality", "weak", swap)])
    assert "treat `quality` as an evaluation candidate" in text
    assert "evaluate `quality` for latency-sensitive" in text
    assert "not use `weak` for deployment" in text
    assert "consider `weak` as a replacement" not in text


def test_single_candidate_is_not_called_a_comparative_winner():
    text = interpretation(analysis(values=(0.9,), names=("only",)),
                          costs(("only",), (1.,), (100.,)), [], [])
    assert "only non-internal backend" in text
    assert "not a comparative winner" in text


@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_swap_recommendations_follow_verdicts_and_explain_blockers(verdict):
    check = {"check": "latency ratio (single-row p50)", "observed": "10x",
             "limit": "<= 2x", "verdict": verdict}
    result = {"overall": verdict, "checks": [check]}
    text = interpretation(analysis(), costs(), [], [("fast", "quality", result)])
    if verdict == "PASS":
        assert "consider `quality` as a replacement for `fast` under the configured limits" in text
        assert "every evaluated swap check passed" in text
        assert "not a production-safety guarantee" in text
    elif verdict == "FAIL":
        assert "keep `fast` rather than replace it with `quality`" in text
        assert "latency ratio (single-row p50): 10x (limit <= 2x)" in text
        assert "policy-limit failure, not a failed model run" in text
    else:
        assert "defer replacing `fast` with `quality`" in text
        assert "latency ratio (single-row p50): 10x (limit <= 2x)" in text
        assert "Gather more paired evidence" in text


def test_partial_coverage_is_not_called_deployment_ready():
    res = analysis()
    res["dropped"] = {"quality": [1]}
    text = interpretation(res, costs(), [{"error": "failed"}], [])
    assert "common successful splits" in text and "partial coverage is not deployment readiness" in text


def test_smoke_interpretation_explains_measurements_without_recommending_winners():
    res = analysis(values=(1., 1.))
    res["pairs"] = [("fast", "quality", {"disputed": 0, "entries": 32})]
    runs = [{"backend": name, "split_seed": 0, "n_train": 128, "n_test": 32}
            for name in res["backends"]]
    text = interpretation(res, costs(), runs, [], smoke=True)
    assert "### Interpretation" in text
    assert "128 train/32 test rows" in text and "`accuracy`=1" in text
    assert "single-row p50=1 ms" in text and "process peak RSS=100 MB" in text
    assert "same observed primary score" in text and "0/32" in text
    assert "not additional independent observations" in text
    assert "without `--smoke`" in text and "No swap approval" in text
    assert "shortlist" not in text and "prefer `" not in text


def test_unanalyzable_dataset_still_has_interpretation(monkeypatch, tmp_path):
    from quibble import report
    monkeypatch.setattr(report, "analyse", lambda *a, **k: (_ for _ in ()).throw(
        report.AnalysisError("no usable predictions")))
    monkeypatch.setattr(report, "_plt", lambda: None)
    text = build(tmp_path, ["example"]).read_text()
    assert "## example\n\n### Interpretation" in text
    assert "No usable aligned analysis" in text and "no usable predictions" in text
    assert "No quality ranking or replacement recommendation" in text


def test_build_matches_costs_to_analyzed_model_seed_and_splits(monkeypatch, tmp_path):
    from quibble import report
    res = analysis(values=(0.9,), names=("only",))
    runs = [
        {"backend": "only", "model_seed": 0, "split_seed": 0,
         "single_row_p50_ms": 20., "peak_rss_mb": 200.},
        {"backend": "only", "model_seed": 1, "split_seed": 0,
         "single_row_p50_ms": 1., "peak_rss_mb": 1.},
        {"backend": "only", "model_seed": 0, "split_seed": 1,
         "single_row_p50_ms": 1., "peak_rss_mb": 1.},
    ]
    monkeypatch.setattr(report, "analyse", lambda *args, **kwargs: res)
    monkeypatch.setattr(report, "load_runs", lambda *args: runs)
    monkeypatch.setattr(report, "_plt", lambda: None)
    text = build(tmp_path, ["example"]).read_text()
    assert "### Interpretation and recommended use" in text
    assert "is 20 ms" in text and "is 200 MB" in text
    assert "Figures skipped" in text
