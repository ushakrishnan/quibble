"""Exploratory interaction discovery, privacy, and analysis/report integration."""
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quibble_ml.interactions import (
    CSV_FILE, JSON_FILE, InteractionSpec, discover, fit_schema, partition_rows, rule_mask,
    transform, write_exports,
)


@pytest.fixture
def planted():
    # Arrange: disagreement occurs only in a two-feature quadrant.
    X = pd.DataFrame(np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (200, 1)),
                     columns=["left", "right"])
    disputed = (X.left == 1) & (X.right == 1)
    paired = pd.DataFrame({"row": np.arange(len(X)), "disputed": disputed,
                           "a_correct": True, "b_correct": ~disputed})
    spec = InteractionSpec(("alpha", "beta"), ("left", "right"), min_rows=8)
    recipes = {"alpha": [{"recipe": {"family_sha256": "recipe-a"}}],
               "beta": [{"recipe": {"family_sha256": "recipe-b"}}]}
    return X, paired, spec, recipes


def test_given_planted_interaction_when_discovered_then_frozen_rule_recovers_it(planted):
    X, paired, spec, recipes = planted

    result = discover(X, paired, "classification", spec, recipes)

    assert result["status"] == "ok"
    rules = [r for r in result["rules"] if r["disagreement_rate"] == 1]
    assert len(rules) == 1 and rules[0]["label"] == "multi_feature"
    assert len(rules[0]["predicates"]) == 2
    assert rules[0]["quality_difference"] == 1
    assert sum(r["evaluation_rows"] for r in result["rules"]) == result["split"]["evaluation_rows"]
    assert "shared-model dependence" in result["warning"]
    assert "not independent confirmation" in result["warning"]
    encoded = transform(X, result["feature_schema"])
    np.testing.assert_array_equal(rule_mask(encoded, rules[0]["predicates"]), paired.disputed)


def test_given_identical_inputs_when_discovered_twice_then_deterministic(planted):
    X, paired, spec, recipes = planted

    first = discover(X, paired, "classification", spec, recipes)
    second = discover(X, paired.sample(frac=1, random_state=8), "classification", spec, recipes)

    assert first == second
    assert len(first["rules"]) <= spec.max_rules
    assert all(len(r["predicates"]) <= 2 for r in first["rules"])


def test_given_no_disagreements_when_discovered_then_explicit_empty_export(planted, tmp_path):
    X, paired, spec, recipes = planted
    paired["disputed"] = False

    result = discover(X, paired, "classification", spec, recipes)
    write_exports(result, tmp_path)

    assert result["status"] == "no_disagreement_in_discovery" and result["rules"] == []
    assert pd.read_csv(tmp_path / CSV_FILE).empty
    assert json.loads((tmp_path / JSON_FILE).read_text())["status"] == "no_disagreement_in_discovery"


def test_given_insufficient_distinct_rows_when_repeated_then_support_not_inflated(planted):
    X, paired, spec, recipes = planted
    small = pd.concat([paired.iloc[:6]] * 100, ignore_index=True)

    result = discover(X, small, "classification", spec, recipes)

    assert result["status"] == "insufficient_support"
    assert result["split"]["discovery_rows"] + result["split"]["evaluation_rows"] == 6
    assert not result["rules"]


def test_given_unequal_repeat_counts_when_discovered_then_equal_row_weight(planted):
    X, paired, spec, recipes = planted
    repeated = pd.concat([paired, paired[paired.disputed]] * 3, ignore_index=True)

    base = discover(X, paired, "classification", spec, recipes)
    result = discover(X, repeated, "classification", spec, recipes)

    for before, after in zip(base["rules"], result["rules"]):
        assert before["predicates"] == after["predicates"]
        assert before["evaluation_rows"] == after["evaluation_rows"]
        assert before["quality_difference"] == after["quality_difference"]
        assert before["disagreement_rate"] == after["disagreement_rate"]
        assert after["evaluation_entries"] >= before["evaluation_entries"]


def test_given_varying_repeats_when_discovered_then_mean_rate_not_best_repeat(planted):
    X, paired, spec, recipes = planted
    other_repeat = paired.copy()
    other_repeat["disputed"] = False
    other_repeat["b_correct"] = True
    repeated = pd.concat([paired, other_repeat], ignore_index=True)

    result = discover(X, repeated, "classification", spec, recipes)

    assert max(r["disagreement_rate"] for r in result["rules"]) == 0.5
    assert max(r["quality_difference"] for r in result["rules"]) == 0.5


def test_given_declared_groups_when_partitioned_then_groups_and_rows_are_disjoint(planted):
    X, paired, spec, recipes = planted
    groups = np.repeat(np.arange(100), 8)
    X.attrs["quibble_groups"] = groups

    discovery, evaluation, _ = partition_rows(np.arange(len(X)), groups, spec)
    result = discover(X, paired, "classification", spec, recipes)

    assert not (discovery & evaluation).any()
    assert set(groups[discovery]).isdisjoint(groups[evaluation])
    assert result["split"]["unit"] == "declared_group"
    assert result["split"]["discovery_groups"] + result["split"]["evaluation_groups"] == 100
    assert all(r["discovery_groups"] >= spec.min_groups for r in result["rules"])


def test_given_insufficient_groups_when_discovered_then_no_rules(planted):
    X, paired, spec, recipes = planted
    X.attrs["quibble_groups"] = np.repeat([0, 1], len(X) // 2)

    result = discover(X, paired, "classification", spec, recipes)

    assert result["status"] == "insufficient_support" and not result["rules"]


def test_given_missing_and_unseen_values_when_transformed_then_discovery_only_semantics():
    discovery = pd.DataFrame({"amount": [1., 3., np.nan, np.inf],
                              "category": ["private-red", None, "private-blue", "private-red"]})
    evaluation = pd.DataFrame({"amount": [np.nan, 100000., -np.inf],
                               "category": ["never-seen", None, "private-red"]})

    schema = fit_schema(discovery)
    encoded = transform(evaluation, schema)

    assert schema[0]["impute"] == 2
    np.testing.assert_array_equal(encoded[:, 0], [2, 100000, 2])
    assert encoded[1, 1] == -1 and 0 <= encoded[0, 1] < 16
    assert encoded[2, 1] == transform(discovery, schema)[0, 1]
    assert "private-red" not in json.dumps(schema)
    assert "never-seen" not in json.dumps(schema)


def test_given_evaluation_only_values_when_discovered_then_preprocessing_unchanged(planted):
    X, paired, spec, recipes = planted
    discovery, evaluation, _ = partition_rows(np.arange(len(X)), None, spec)
    base = discover(X, paired, "classification", spec, recipes)
    X.loc[evaluation, ["left", "right"]] = 987654.

    changed = discover(X, paired, "classification", spec, recipes)

    assert base["feature_schema"] == changed["feature_schema"]
    assert [r["predicates"] for r in base["rules"]] == [r["predicates"] for r in changed["rules"]]
    assert discovery.sum() == changed["split"]["discovery_rows"]


def test_given_float32_boundary_when_frozen_then_double_threshold_not_rounded():
    # Arrange: midpoint rounds up to the upper value if cast to float32.
    encoded = np.array([[1048576.125], [1048576.25]], dtype=np.float32)
    predicates = [{"column": 0, "operator": "<=", "threshold": 1048576.1875}]

    mask = rule_mask(encoded, predicates)

    np.testing.assert_array_equal(mask, [True, False])


def test_given_transform_input_when_imputed_then_input_is_not_mutated():
    X = pd.DataFrame({"value": [1., np.nan, np.inf, 3.]})
    original = X.copy(deep=True)
    schema = fit_schema(X)

    transform(X, schema)

    pd.testing.assert_frame_equal(X, original)


def test_given_extreme_finite_values_when_fitted_then_median_and_json_are_finite():
    X = pd.DataFrame({"value": [1e308, 1e308, np.nan]})

    schema = fit_schema(X)

    assert schema[0]["impute"] == 1e30
    json.dumps(schema, allow_nan=False)


def test_given_missing_declared_group_metadata_when_discovered_then_no_silent_row_split(planted):
    X, paired, spec, recipes = planted
    X.attrs["quibble_split"] = {"kind": "group", "group_column": "customer"}

    with pytest.raises(ValueError, match="refusing a row-only split"):
        discover(X, paired, "classification", spec, recipes)


def test_given_export_cap_one_when_discovered_then_one_discovery_ranked_leaf(planted):
    X, paired, spec, recipes = planted

    result = discover(X, paired, "classification", replace(spec, max_rules=1), recipes)

    assert len(result["rules"]) == 1
    assert result["rules"][0]["disagreement_rate"] > 0


@pytest.mark.parametrize("name", ["target", "row", "customer_id", "fold", "post_outcome_score",
                                 "pred_probability", "secret", "outcome_at", "unsafe"])
def test_given_unsafe_allowlist_when_discovered_then_rejected_before_access(planted, name):
    X, paired, spec, recipes = planted
    # Only metadata is inspected: these unsafe columns do not even need to exist.
    X.attrs["quibble_source"] = {"target": "secret", "drop_columns": ["unsafe"]}
    X.attrs["quibble_split"] = {"time_column": "outcome_at"}

    with pytest.raises(ValueError, match="unsafe interaction feature"):
        discover(X, paired, "classification", replace(spec, features=(name,)), recipes)


def test_given_unused_private_columns_when_discovered_then_never_read_or_exported(planted, tmp_path):
    X, paired, spec, recipes = planted

    class PrivateValue:
        def __str__(self):
            raise AssertionError("private ID/target/post-outcome value was accessed")

    for name in ("target", "row_id", "post_outcome"):
        X[name] = [PrivateValue()] * len(X)
    X["category"] = "PRIVATE_CATEGORY_SENTINEL"
    spec = replace(spec, features=spec.features + ("category",))

    result = discover(X, paired, "classification", spec, recipes)
    write_exports(result, tmp_path)

    for name in (JSON_FILE, CSV_FILE):
        text = (tmp_path / name).read_text()
        assert "PRIVATE_CATEGORY_SENTINEL" not in text
        assert "row_id" not in text and "post_outcome" not in text
    assert result["feature_schema"][-1]["kind"] == "hash_bucket"


def test_given_one_feature_signal_when_discovered_then_not_labeled_interaction(planted):
    X, paired, spec, recipes = planted
    paired["disputed"] = X.left == 1

    result = discover(X, paired, "classification", spec, recipes)

    assert result["rules"]
    assert {r["label"] for r in result["rules"]} == {"single_feature"}


def test_given_regression_pair_when_discovered_then_quality_direction_favors_a(planted):
    X, paired, spec, recipes = planted
    paired["y"] = 0.
    paired["pred_a"] = 0.
    paired["pred_b"] = paired["disputed"].astype(float) * 2

    result = discover(X, paired, "regression", spec, recipes)

    assert result["quality_metric"] == "mae_b_minus_a"
    assert max(r["quality_difference"] for r in result["rules"]) == 2


def test_given_regression_evaluation_outcomes_when_changed_then_discovery_rules_unchanged(planted):
    X, paired, spec, recipes = planted
    paired["y"] = np.arange(len(X), dtype=float) / len(X)
    paired["pred_a"] = 0.
    paired["pred_b"] = paired["disputed"].astype(float) * 2
    base = discover(X, paired, "regression", spec, recipes)
    _, evaluation, _ = partition_rows(np.arange(len(X)), None, spec)
    paired.loc[evaluation, "y"] = 1e9

    changed = discover(X, paired, "regression", spec, recipes)

    assert changed["disagreement_definition"] == base["disagreement_definition"]
    assert [r["predicates"] for r in changed["rules"]] == [r["predicates"] for r in base["rules"]]


def test_given_wrong_recipe_pair_when_discovered_then_rejected(planted):
    X, paired, spec, recipes = planted

    with pytest.raises(ValueError, match="must match the named pair"):
        discover(X, paired, "classification", spec, {"other": []})
    with pytest.raises(ValueError, match="recipe mismatch for beta"):
        discover(X, paired, "classification",
                 replace(spec, recipe_ids=("recipe-a", "wrong")), recipes)
    result = discover(X, paired, "classification",
                      replace(spec, recipe_ids=("recipe-a", "recipe-b")), recipes)

    assert result["recipes"] == recipes


@pytest.mark.parametrize("changes", [
    {"features": ()}, {"max_features": 17}, {"max_rules": 5}, {"min_rows": 1},
    {"min_groups": 0}, {"evaluation_fraction": 1}, {"seed": -1}, {"pair": ("a", "a")},
])
def test_given_unbounded_configuration_when_created_then_rejected(planted, changes):
    _, _, spec, _ = planted

    with pytest.raises(ValueError):
        replace(spec, **changes)


def test_given_unsupported_evaluation_leaf_when_discovered_then_no_estimate(planted):
    X, paired, spec, recipes = planted
    discovery, evaluation, _ = partition_rows(np.arange(len(X)), None, spec)
    X.loc[evaluation, ["left", "right"]] = 0.

    result = discover(X, paired, "classification", spec, recipes)

    empty = [r for r in result["rules"] if r["evaluation_rows"] == 0]
    assert empty and all(r["evaluation_status"] == "insufficient_support" for r in empty)
    assert all(r["disagreement_rate"] is None and r["quality_difference"] is None for r in empty)


def test_given_real_saved_predictions_when_opted_in_then_report_archive_and_verdict_preserved(
        cv_results, monkeypatch):
    from quibble_ml import report
    from quibble_ml.analyze import AnalysisError, analyse, pair_frame
    from quibble_ml.datasets import load
    from quibble_ml.pool import align, load_pool
    from quibble_ml.slices import row_traits
    from quibble_ml.swap import evaluate

    # Arrange: existing local CV fixture, no downloads or model changes.
    X, _, task = load("breast_cancer")
    spec = InteractionSpec(("histgb", "weak"), tuple(X.columns[:2]), min_rows=8)
    before = evaluate("breast_cancer", "histgb", "weak", 0, cv_results)
    monkeypatch.setattr(report, "_plt", lambda: None)
    baseline = analyse("breast_cancer", cv_results, quiet=True)
    old_disagreements = (baseline["adir"] / "disagreements_histgb_vs_weak.csv").read_bytes()

    # Act: opt in, poison cached rules, and let report rediscover from the saved specification.
    result = analyse("breast_cancer", cv_results, quiet=True, interaction=spec)
    explanation = result["interaction"]
    adir = result["adir"]
    saved = json.loads((adir / "analysis.json").read_text())
    saved["interaction"]["rules"] = [{"text": "STALE_RULE_SENTINEL"}]
    (adir / "analysis.json").write_text(json.dumps(saved))
    (adir / JSON_FILE).write_text('{"rules": "STALE_RULE_SENTINEL"}')
    path = report.build(cv_results, ["breast_cancer"], swaps=[spec.pair])
    after = evaluate("breast_cancer", "histgb", "weak", 0, cv_results)

    # Assert: named pair evidence, unchanged existing outputs, fresh rules, and archived links.
    pool = align([load_pool(cv_results, "breast_cancer", name, 0)[0] for name in spec.pair])
    expected = discover(X, pair_frame(pool, 0, 1, row_traits(X), task), task, spec,
                        explanation["recipes"])
    assert expected["rules"] == explanation["rules"]
    assert before["overall"] == after["overall"] and before["checks"] == after["checks"]
    assert old_disagreements == (adir / "disagreements_histgb_vs_weak.csv").read_bytes()
    assert result["class_calibration_summary"] == baseline["class_calibration_summary"]
    assert result["class_calibration_bins"] == baseline["class_calibration_bins"]
    text = path.read_text()
    assert "Exploratory interaction rules" in text and "shared-model dependence" in text
    assert "histgb vs weak" in text and "STALE_RULE_SENTINEL" not in text
    assert json.loads((adir / JSON_FILE).read_text())["rules"] == explanation["rules"]
    archive = sorted(cv_results.glob("runs/*/*/breast_cancer/analysis/" + JSON_FILE))[-1]
    assert archive.exists() and archive.with_name(CSV_FILE).exists()
    archived_report = archive.parents[2] / "report" / "report.md"
    assert "../breast_cancer/analysis/interaction_rules.json" in archived_report.read_text()
    with pytest.raises(AnalysisError, match="two usable selected backends"):
        analyse("breast_cancer", cv_results, quiet=True,
                interaction=replace(spec, pair=("histgb", "absent")))


def test_given_cli_request_when_analyze_and_report_then_flags_and_saved_spec_work(cv_results):
    # Arrange
    features = "mean radius,mean texture"
    common = ["--out", str(cv_results), "--interaction-pair", "histgb:weak",
              "--interaction-features", features, "--interaction-min-rows", "8",
              "--interaction-seed", "17", "--interaction-max-rules", "3"]
    commands = [
        ["analyze", "--dataset", "breast_cancer", *common],
        ["report", "--datasets", "breast_cancer", *common],
    ]

    # Act
    for command in commands:
        completed = subprocess.run([sys.executable, "-B", "-m", "quibble_ml", *command],
                                   capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    # Assert
    exported = json.loads((cv_results / "breast_cancer" / "analysis" / JSON_FILE).read_text())
    assert exported["spec"]["seed"] == 17 and exported["spec"]["max_rules"] == 3
    assert exported["pair"] == ["histgb", "weak"]
