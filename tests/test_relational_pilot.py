"""Network-free source-pinning and deterministic sampled pilot tests."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quibble_ml import relational as rel
from quibble_ml import relational_pilot as pilot


def test_given_pinned_download_inventory_when_checked_then_budget_is_under_ten_megabytes():
    # Assert
    assert sum(size for size, _ in pilot.FILES.values()) == 9_237_686
    assert not any(name.endswith("/papers.parquet") for name in pilot.FILES)


def test_given_missing_source_when_pilot_prepared_then_no_network_or_output(tmp_path, monkeypatch):
    # Arrange
    def forbid_network(*args, **kwargs):
        raise AssertionError("routine tests must never download")
    monkeypatch.setattr(pilot.urllib.request, "urlopen", forbid_network)

    # Act & Assert
    with pytest.raises(ValueError, match="pinned pilot source"):
        pilot.prepare(tmp_path / "source", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_given_local_official_split_fixture_when_sampled_then_boundaries_and_sampling_are_stable(tmp_path, monkeypatch):
    # Arrange: synthetic stand-in source files, never public/remote data in CI.
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "tasks" / "author-publication").mkdir(parents=True)
    pd.DataFrame({"Author_ID": range(50)}).to_parquet(source / "db/authors.parquet", index=False)
    pd.DataFrame({"Author_ID": list(range(50)) * 2, "Paper_ID": range(100),
                  "Submission_Date": [pd.Timestamp("2019-01-01")] * 50
                  + [pd.Timestamp("2021-06-01")] * 50}).to_parquet(
                      source / "db/paperAuthors.parquet", index=False)
    for part, cutoff in (("train", "2020-01-01"), ("val", "2022-01-01"), ("test", "2023-01-01")):
        pd.DataFrame({"Author_ID": range(50), "date": [pd.Timestamp(cutoff)] * 50,
                      "publication_count": np.arange(50) % 4}).to_parquet(
                          source / f"tasks/author-publication/{part}.parquet", index=False)
    inventory = {name: ((source / name).stat().st_size,
                        hashlib.sha256((source / name).read_bytes()).hexdigest()) for name in pilot.FILES}
    monkeypatch.setattr(pilot, "FILES", inventory)

    # Act
    first = pilot.prepare(source, tmp_path / "first", per_split=16)
    second = pilot.prepare(source, tmp_path / "second", per_split=16)
    a = rel.materialize_registered("rel_arxiv_sample", first)
    b = rel.materialize_registered("rel_arxiv_sample", second)
    evidence = json.loads((first.parent / "pilot-evidence.json").read_text(encoding="utf-8"))

    # Assert
    np.testing.assert_array_equal(a.X.to_numpy(), b.X.to_numpy())
    assert a.y.tolist() == b.y.tolist()
    assert a.audit["split_counts"] == {"train": 16, "val": 16, "test": 16}
    assert evidence["warning"] == "NOT official full-corpus benchmark"
    assert evidence["inventory"]["val"]["min_cutoff"].startswith("2022-01-01")
    assert a.audit["source"]["availability_assumption"].startswith("Submission calendar date")
    assert set(a.X.columns) == {"past_year_publications", "publication_recency"}

    # A local mutation cannot masquerade as the pinned source.
    mutated = source / "db/authors.parquet"
    mutated.write_bytes(mutated.read_bytes() + b"x")
    with pytest.raises(ValueError, match="source size/hash mismatch"):
        pilot.prepare(source, tmp_path / "tampered", per_split=16)
