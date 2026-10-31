"""Offline relational contract proofs; all input data are generated synthetic."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quibble_ml import relational as rel
from quibble_ml.relational_synthetic import write_fixture


@pytest.fixture
def registered(tmp_path):
    """Provide one explicit synthetic registration with future/late traps."""
    return write_fixture(tmp_path / "input")


def read_spec(registry):
    return json.loads((registry.parent / "manifest.json").read_text(encoding="utf-8"))


def save_spec(registry, spec):
    (registry.parent / "manifest.json").write_text(json.dumps(spec), encoding="utf-8")


def mutate_table(registry, name, mutate):
    spec = read_spec(registry)
    path = registry.parent / spec["tables"][name]["path"]
    frame = pd.read_csv(path)
    frame = mutate(frame)
    frame.to_csv(path, index=False)
    spec["tables"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_spec(registry, spec)


def build(registry, cache=None):
    return rel.materialize_registered("relational_synthetic", registry, cache)


def test_given_temporal_traps_when_materialized_then_only_available_history_and_frozen_dimensions(registered):
    # Act
    result = build(registered)
    first = result.X.iloc[0]

    # Assert: answer-encoding future, late, tied rows and revised dimensions excluded.
    assert first["history_count"] == 1
    assert first["history_value"] == 0
    assert first["profile_score"] == 0
    assert first["history_recency"] == 8 * 86400
    assert result.X.iloc[5]["history_value"] == 1


def test_given_horizon_overlap_when_split_then_training_labels_are_purged(registered):
    # Act
    result = build(registered)
    tr, te, ytr, yte, positions = rel.split_registered(result.X, result.y)

    # Assert
    assert len(tr) == len(ytr) == 24
    assert len(te) == len(yte) == len(positions) == 12
    assert result.audit["purged_labels"] == {"train": 12, "val": 0}
    assert result.audit["split_counts"] == {"train": 24, "val": 12, "test": 12}


def test_given_repeated_entities_when_materialized_then_grouped_keys_are_distinct_and_private(registered):
    # Act
    result = build(registered)
    info = result.X.attrs["quibble_relational"]
    groups = result.X.attrs["quibble_groups"]
    public = json.dumps(result.audit) + json.dumps(info["example_keys"]) + json.dumps(groups.tolist())

    # Assert
    assert len(set(info["example_keys"])) == 60
    assert len(set(groups)) == 12
    assert len(set(groups[:5])) == 1
    assert "synthetic_person" not in public
    assert "identity_key" not in public
    assert result.audit["repeated_entities"]


def test_given_shuffled_tables_when_rebuilt_then_prediction_identity_and_features_are_stable(registered):
    # Arrange
    original = build(registered)
    for table in read_spec(registered)["tables"]:
        mutate_table(registered, table, lambda frame: frame.sample(frac=1, random_state=8))

    # Act
    shuffled = build(registered)

    # Assert
    np.testing.assert_array_equal(original.X.to_numpy(), shuffled.X.to_numpy())
    assert original.X.attrs["quibble_relational"]["example_keys"] == shuffled.X.attrs["quibble_relational"]["example_keys"]
    assert original.y.tolist() == shuffled.y.tolist()


def test_given_warm_cache_when_child_changes_then_stale_hash_is_refused_and_new_review_misses_cache(registered, tmp_path):
    # Arrange
    cache = tmp_path / "cache"
    cold = build(registered, cache)
    warm = build(registered, cache)
    spec = read_spec(registered)
    child = registered.parent / spec["tables"]["events"]["path"]
    child.write_text(child.read_text(encoding="utf-8").replace("20000", "20001"), encoding="utf-8")

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="declared hash mismatch"):
        build(registered, cache)
    spec["tables"]["events"]["sha256"] = hashlib.sha256(child.read_bytes()).hexdigest()
    save_spec(registered, spec)
    changed = build(registered, cache)
    assert cold.audit["costs"]["cache_state"] == "cold"
    assert warm.audit["costs"]["cache_state"] == "warm"
    assert warm.audit["costs"]["feature_build_s"] == 0
    assert warm.audit["costs"]["feature_cache_read_s"] >= 0
    assert changed.audit["costs"]["cache_state"] == "cold"
    assert cold.audit["fingerprint"] != changed.audit["fingerprint"]
    for stage in ("validation_s", "feature_total_s", "feature_build_s"):
        assert cold.audit["costs"][stage] >= 0


@pytest.mark.parametrize("section", ["recipe", "split", "task", "relations"])
def test_given_changed_contract_when_rebuilt_then_cache_identity_changes(registered, tmp_path, section):
    # Arrange
    before = build(registered, tmp_path / "cache")
    spec = read_spec(registered)
    if section == "recipe":
        spec["recipe"]["tie_policy"] = "include"
    elif section == "split":
        spec["split"]["embargo_seconds"] = 1
    elif section == "task":
        spec["task"]["horizon_seconds"] = 400000
    else:
        spec["relations"][1]["frozen_as_of"] = "2024-01-04T00:00:00Z"
    save_spec(registered, spec)

    # Act
    after = build(registered, tmp_path / "cache")

    # Assert
    assert before.audit["fingerprint"] != after.audit["fingerprint"]
    assert after.audit["costs"]["cache_state"] == "cold"


@pytest.mark.parametrize(("table", "mutation", "message"), [
    ("entities", lambda df: pd.concat([df, df.iloc[:1]], ignore_index=True), "duplicate primary key"),
    ("events", lambda df: df.assign(entity_id="orphan"), "orphan foreign key"),
    ("examples", lambda df: df.assign(entity_id=df["entity_id"].astype(str) + "_orphan"), "orphan foreign key"),
    ("events", lambda df: df.assign(event_time="not-a-timestamp"), "timezone offsets"),
    ("events", lambda df: df.assign(event_time="2024-99-99T00:00:00Z"), "invalid timestamp"),
    ("events", lambda df: df.assign(event_time="2024-01-02T00:00:00"), "timezone offsets"),
    ("events", lambda df: df.assign(available_time="2023-01-01T00:00:00Z"), "availability precedes event"),
    ("examples", lambda df: df.assign(label_available=df["cutoff"]), "outcome horizon"),
])
def test_given_invalid_table_when_materialized_then_explicit_failure(registered, table, mutation, message):
    # Arrange
    mutate_table(registered, table, mutation)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match=message):
        build(registered)


@pytest.mark.parametrize("path", ["../outside.csv", "C:/outside.csv", "/outside.csv", "tables\\events.csv", "missing.csv"])
def test_given_unsafe_or_unknown_path_when_materialized_then_refused(registered, path):
    # Arrange
    spec = read_spec(registered)
    spec["tables"]["events"]["path"] = path
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="path|missing"):
        build(registered)


@pytest.mark.parametrize("column", ["answer", "event_id", "entity_id", "event_time"])
def test_given_id_or_target_derived_predictor_when_requested_then_refused(registered, column):
    # Arrange
    spec = read_spec(registered)
    spec["recipe"]["aggregations"][1]["column"] = column
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="never IDs/targets"):
        build(registered)


@pytest.mark.parametrize(("limit", "value", "message"), [
    ("matches", 1, "cardinality"), ("examples", 2, "example row budget"),
    ("features", 1, "feature budget"), ("table_rows", 2, "row budget"),
    ("total_bytes", 100, "byte budget"),
])
def test_given_tight_budget_when_materialized_then_bounded_failure(registered, limit, value, message):
    # Arrange
    spec = read_spec(registered)
    spec["limits"] = {limit: value}
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match=message):
        build(registered)


def test_given_include_ties_when_materialized_then_exact_cutoff_rows_are_included(registered):
    # Arrange
    spec = read_spec(registered)
    spec["recipe"]["tie_policy"] = "include"
    save_spec(registered, spec)

    # Act
    result = build(registered)

    # Assert
    assert result.X.iloc[0]["history_count"] == 2
    assert result.X.iloc[0]["history_value"] == 30000


def test_given_lookback_when_materialized_then_missing_history_is_explicit(registered):
    # Arrange
    spec = read_spec(registered)
    spec["recipe"]["lookback_seconds"] = 86400
    save_spec(registered, spec)

    # Act
    result = build(registered)
    slices = result.X.attrs["quibble_relational"]["slices"]

    # Assert
    assert result.X.iloc[0]["history_count"] == 0
    assert np.isnan(result.X.iloc[0]["history_value"])
    assert slices["cold_start"][0]
    assert slices["history_length"][0] == 0
    assert slices["missing_relations"][0] == 1


def test_given_disjoint_policy_when_entities_repeat_then_refused(registered):
    # Arrange
    spec = read_spec(registered)
    spec["split"]["entity_overlap"] = "disjoint"
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="entity-disjoint"):
        build(registered)


def test_given_test_targets_changed_when_materialized_then_features_and_training_are_unchanged(registered):
    # Arrange
    original = build(registered)
    def replace_test(df):
        mask = df["split"] == "test"
        df.loc[mask, "target"] = 1 - df.loc[mask, "target"]
        return df
    mutate_table(registered, "examples", replace_test)

    # Act
    changed = build(registered)

    # Assert
    np.testing.assert_array_equal(original.X.to_numpy(), changed.X.to_numpy())
    assert rel.split_registered(original.X, original.y)[2].tolist() == rel.split_registered(changed.X, changed.y)[2].tolist()


def test_given_parquet_when_materialized_then_same_examples_and_features(registered):
    # Arrange
    original = build(registered)
    spec = read_spec(registered)
    for table in spec["tables"].values():
        old = registered.parent / table["path"]
        path = old.with_suffix(".parquet")
        pd.read_csv(old, dtype=str).to_parquet(path, index=False)
        table["path"] = path.relative_to(registered.parent).as_posix()
        table["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_spec(registered, spec)

    # Act
    parquet = build(registered)

    # Assert
    np.testing.assert_array_equal(original.X.to_numpy(), parquet.X.to_numpy())
    assert original.X.attrs["quibble_relational"]["example_keys"] == parquet.X.attrs["quibble_relational"]["example_keys"]


def test_given_unregistered_name_when_inspected_then_cli_fails_closed(registered, capsys):
    # Act
    code = rel.main(["inspect", "unregistered", "--registry", str(registered)])

    # Assert
    assert code == 2
    assert "not explicitly registered" in capsys.readouterr().err


def test_given_registered_fixture_when_cli_materializes_then_public_audit_is_returned(registered, capsys):
    # Act
    code = rel.main(["materialize", "relational_synthetic", "--registry", str(registered)])
    output = capsys.readouterr().out

    # Assert
    assert code == 0
    assert json.loads(output)["n_examples"] == 60
    assert "synthetic_person" not in output


def test_given_regression_fixture_when_materialized_then_targets_are_numeric(tmp_path):
    # Arrange
    registry = write_fixture(tmp_path / "regression", task_type="regression")

    # Act
    result = build(registry)

    # Assert
    assert result.task == "regression"
    assert result.y.dtype.kind == "f"


def test_given_cache_corruption_when_materialized_then_refused(registered, tmp_path):
    # Arrange
    cache = tmp_path / "cache"
    original = build(registered, cache)
    path = cache / f"{original.audit['fingerprint']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["data"]["values"][0][0] = 99999
    path.write_text(json.dumps(payload), encoding="utf-8")

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="checksum"):
        build(registered, cache)


def test_given_timezone_equivalent_example_keys_when_normalized_then_duplicates_fail(registered):
    # Arrange
    def duplicate_utc(df):
        extra = df.iloc[[0]].copy()
        extra["cutoff"] = "2024-01-09T19:00:00-05:00"
        return pd.concat([df, extra], ignore_index=True)
    mutate_table(registered, "examples", duplicate_utc)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="duplicate primary key"):
        build(registered)


def test_given_dimension_revision_ties_when_frozen_then_ambiguity_fails(registered):
    # Arrange
    def tied_revision(df):
        extra = df.iloc[[0]].copy()
        extra["revision_id"] = "different_revision_same_time"
        return pd.concat([df, extra], ignore_index=True)
    mutate_table(registered, "dimensions", tied_revision)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="ambiguous revision ties"):
        build(registered)


def test_given_floating_child_rows_when_shuffled_then_reductions_use_canonical_order(registered):
    # Arrange
    def floating_events(df):
        df["value"] = np.resize([1e16, -1e16, 1.1], len(df))
        return df
    mutate_table(registered, "events", floating_events)
    original = build(registered)
    mutate_table(registered, "events", lambda df: df.sample(frac=1, random_state=5))

    # Act
    shuffled = build(registered)

    # Assert
    np.testing.assert_array_equal(original.X.to_numpy(), shuffled.X.to_numpy())


@pytest.mark.parametrize(("cv", "split_id"), [("2x1", 0), ("holdout", 1)])
def test_given_random_cv_or_repeated_split_when_official_split_requested_then_refused(registered, cv, split_id):
    # Arrange
    result = build(registered)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="official protocol"):
        rel.split_registered(result.X, result.y, cv=cv, split_id=split_id)


def test_given_subsets_when_split_then_example_metadata_remain_aligned(registered):
    # Arrange
    result = build(registered)

    # Act
    train, test, _, _, test_indices = rel.split_registered(result.X, result.y)
    again = rel.subset_registered(test, [2, 0])

    # Assert
    source = result.X.attrs["quibble_relational"]["example_keys"]
    assert test.attrs["quibble_relational"]["example_keys"] == [source[i] for i in test_indices]
    assert len(again.attrs["quibble_groups"]) == len(again) == 2
    assert len(again.attrs["quibble_relational"]["slices"]["history_length"]) == 2
    assert "positions" not in train.attrs["quibble_relational"]
    with pytest.raises(rel.RelationalError, match="cannot re-split"):
        rel.split_registered(train, result.y.iloc[:len(train)])


@pytest.mark.parametrize("kind", ["unknown-field", "unknown-version", "nonobject-limits", "bad-task-field", "bad-pk"])
def test_given_malformed_manifest_when_materialized_then_configuration_error(registered, kind):
    # Arrange
    spec = read_spec(registered)
    if kind == "unknown-field":
        spec["infer_relations"] = True
    elif kind == "unknown-version":
        spec["schema_version"] = 99
    elif kind == "nonobject-limits":
        spec["limits"] = []
    elif kind == "bad-task-field":
        spec["task"]["entity_key"] = ["entity_id"]
    else:
        spec["tables"]["entities"]["primary_key"] = [["entity_id"]]
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError):
        build(registered)


def test_given_unseen_test_label_when_materialized_then_training_vocabulary_is_not_expanded(registered):
    # Arrange
    def unseen_target(df):
        df.loc[df["split"] == "test", "target"] = 99
        return df
    mutate_table(registered, "examples", unseen_target)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="training label vocabulary"):
        build(registered)


def test_given_future_snapshot_when_requested_then_refused(registered):
    # Arrange
    spec = read_spec(registered)
    spec["relations"][1]["frozen_as_of"] = "2024-01-11T00:00:00Z"
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="frozen before all prediction"):
        build(registered)


def test_given_unsafe_registry_name_when_resolved_then_refused(registered):
    # Arrange
    spec = json.loads(registered.read_text(encoding="utf-8"))
    spec["datasets"]["../escape"] = spec["datasets"]["relational_synthetic"]
    registered.write_text(json.dumps(spec), encoding="utf-8")

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="registered dataset name"):
        rel.registered_manifest("../escape", registered)


def test_given_existing_output_when_run_requested_then_no_execution(registered, tmp_path):
    # Act & Assert
    with pytest.raises(rel.RelationalError, match="output already exists"):
        rel.run_registered("relational_synthetic", registered, tmp_path, ["histgb", "lightgbm"])


def test_given_remote_route_when_run_requested_then_no_execution(registered, tmp_path):
    # Act & Assert
    with pytest.raises(rel.RelationalError, match="distinct local tree routes"):
        rel.run_registered("relational_synthetic", registered, tmp_path / "out", ["histgb", "kumo_tabular"])


@pytest.mark.parametrize("embargo", [0, 60])
def test_given_label_available_at_evaluation_boundary_when_purged_then_ties_are_excluded(registered, embargo):
    # Arrange
    boundary = pd.Timestamp("2024-02-09T00:00:00Z") - pd.Timedelta(embargo, unit="s")
    def boundary_label(df):
        df.loc[df["cutoff"].str.startswith("2024-01-20"), "label_available"] = boundary.isoformat()
        return df
    mutate_table(registered, "examples", boundary_label)
    spec = read_spec(registered)
    spec["split"]["embargo_seconds"] = embargo
    save_spec(registered, spec)

    # Act
    result = build(registered)

    # Assert
    assert result.audit["purged_labels"]["train"] == 24
    assert result.audit["split_counts"]["train"] == 12


def test_given_duplicate_json_keys_when_read_then_ambiguity_fails(registered):
    # Arrange
    path = registered.parent / "manifest.json"
    text = path.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path.write_text(text, encoding="utf-8")

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="duplicate JSON key"):
        build(registered)


def test_given_duplicate_csv_header_when_read_then_ambiguity_fails(registered):
    # Arrange
    spec = read_spec(registered)
    table = spec["tables"]["events"]
    path = registered.parent / table["path"]
    text = path.read_text(encoding="utf-8").replace("value,answer", "value,value", 1)
    path.write_text(text, encoding="utf-8")
    table["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="columns differ"):
        build(registered)


def test_given_reversed_official_split_when_read_then_no_random_fallback(registered):
    # Arrange
    mutate_table(registered, "examples", lambda df: df.assign(
        split=df["split"].map({"train": "test", "val": "val", "test": "train"})))

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="cutoff ranges overlap"):
        build(registered)


def test_given_label_table_join_when_requested_then_refused(registered):
    # Arrange
    spec = read_spec(registered)
    spec["relations"][0]["child"] = "examples"
    save_spec(registered, spec)

    # Act & Assert
    with pytest.raises(rel.RelationalError, match="only direct entity-to-child"):
        build(registered)


def test_given_relational_examples_when_diagnosed_then_fixed_masks_are_aligned_and_label_free(registered):
    # Arrange
    result = build(registered)
    test = rel.split_registered(result.X, result.y)[1]

    # Act
    masks = rel.diagnostic_masks(test)
    contract = rel.analysis_contract(test)

    # Assert
    assert set(masks) == set(rel.slice_masks(test))
    assert all(mask.dtype == bool and len(mask) == len(test) for mask in masks.values())
    assert masks["relational/history=1-4"].all()
    assert masks["relational/recency=30_days_or_more"].all()
    assert len(contract["entity_groups"]) == len(contract["example_keys"]) == len(test)
    assert "synthetic_person" not in json.dumps(contract)
    assert contract["uncertainty_unit"] == "entity cluster"


def test_given_no_history_when_diagnosed_then_missing_state_is_explicit(registered):
    # Arrange
    spec = read_spec(registered)
    spec["recipe"]["lookback_seconds"] = 1
    save_spec(registered, spec)

    # Act
    masks = rel.diagnostic_masks(build(registered).X)

    # Assert
    assert masks["relational/history=none"].all()
    assert masks["relational/recency=missing"].all()
    assert masks["relational/missing_relation"].all()
    assert masks["relational/cold_start"].all()


def test_given_repeated_test_cutoffs_when_bootstrapped_then_entity_pairs_share_resampling_unit(tmp_path):
    # Arrange
    from quibble_ml.metrics import Boot
    registry = write_fixture(tmp_path / "repeated", repeated_test=True)
    result = build(registry)
    test = rel.split_registered(result.X, result.y)[1]
    groups = test.attrs["quibble_groups"]

    # Act
    boot = Boot(np.arange(len(test)), n_boot=50, cluster=groups)

    # Assert: two distinct predictions stay together in every entity bootstrap.
    assert len(test) == 24
    assert len(set(test.attrs["quibble_relational"]["example_keys"])) == 24
    assert boot.counts.shape == (50, 12)
    for group in set(groups):
        pair = np.flatnonzero(groups == group)
        assert len(pair) == 2
        assert boot.unit_of_pred[pair[0]] == boot.unit_of_pred[pair[1]]
        weights = boot.counts[:, boot.unit_of_pred]
        np.testing.assert_array_equal(weights[:, pair[0]], weights[:, pair[1]])
