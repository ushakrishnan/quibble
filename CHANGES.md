# Changes in this revision

Review fixes applied to Quibble 0.0.1. Finding IDs (A, B, C, D, E, F) refer to the evaluation and
change plan. Nothing here is released; the version number is unchanged.

## Verdict statistics
- **Retraining variability counted (A1, A2).** Swap and gate intervals are the wider of a cluster
  bootstrap and a Nadeau-Bengio corrected interval over splits. It needs 5 or more splits; with fewer
  the note says retraining variability is not counted. `summary.csv` gains `cv_lo` and `cv_hi`.
- **Worst slice is family-wise (A3, F2).** The maximum over all slices is recomputed in every
  bootstrap resample, so picking the worst slice no longer biases the interval.
- **Placeholders are labeled (A4).** Swap prints and saves whether default limits were used.
  `--require-thresholds` refuses to produce a verdict from them. New tunable keys:
  `confident_prob`, `regression_gap_std`, `near_margin`.
- **No interval, no PASS (A5, F3).** A check without a usable interval is INCONCLUSIVE. Dropped
  bootstrap resamples are counted and reported.
- **Disagreement noise floor (A6, F4).** With 2 or more model seeds, swap subtracts the incumbent's
  own seed-to-seed disagreement. `summary.csv` gains `n_seeds`, `seed_std`, `reseed_disagreement`.
- **ECE check has an interval (A7).** Rate intervals are kept inside 0 to 100%.
- **Memory and latency (A8, F7, F11).** Memory compares the model's increase over the process
  baseline (floored at 50 MB). Latency adds warm-up calls and compares medians.

## Evaluation design
- **Group and temporal splits (B1)** for local datasets: `group_column`, `time_column`, `split` in
  `dataset.json`. Group-clustered bootstrap. The scheme is part of the dataset fingerprint.
- **Data flags (B2):** single-feature leakage, ID-like columns, duplicate rows, time-like columns on
  a random split. They never change a verdict.
- **Baselines and gate (B3):** `dummy` and `linear` backends, and `quibble gate`.
- **Seeds (B5):** AutoGluon is recorded as `seed_controlled: false`.

## Result reuse and correctness
- **Reuse check (C5, E2):** also compares AutoGluon preset and time limit, TabPFN row limit, latency
  sample size, and installed library versions.
- **`--retry-failed` (F1)** reruns only saved errors (run, demo, benchmark).
- **Atomic split files (F8).** Parallel workers no longer race on shared `.npy` files.
- **Fit time excludes weight loading (F6):** `load_s`, `fit_s`, `fit_total_s`.
- **One metrics implementation (F12).** Run records use `quibble.metrics.score`.
- **pandas 3 and 4 compatibility (F9):** unseen categories and string column detection.
- **AutoGluon column order (F10)** is explicit.
- **Slices (F5):** distinct-row counting, mutual-information selection, exploratory warning.

## Packaging, CI, docs
- **CI (E1):** `.[dev]` now installs everything the tests import. Matrix is Python 3.10 to 3.13 on
  Linux plus 3.13 on Windows, with the demo and swap exit-code checks.
- **Dependencies (E2, F13):** lower bounds added, `scipy` explicit, unused `xgboost`, `seaborn`,
  `jupyter` removed from `requirements.txt`.
- **README (E3 to E5, E11):** Linux and macOS setup, no personal notes, one tagline.
- **Doc files renamed (E8):** `docs/rationale.md`, `docs/recommend-abstain.md`.
- **`quibble doctor` (E7, C1):** prints backend verification labels, warns about a conflicting
  PyPI package named `quibble`.
- **Windows-path test (F14):** manifests reject backslashes in `file` on every platform.

## Tests
`tests/test_review_fixes.py` covers each fix. The full suite passes on Linux with Python 3.12.

## Not done
- Tuned tier, per-class calibration, interaction slices, relational or multi-table benchmarking.
- Mitra-v2 and TabICLv2 adapters, a license column, and a live TabPFN run (no network or weights here).
- Verifying the OpenML registry, remote CI, Docker, and the GitHub release tag.
- GitHub repository topics and description (set in the GitHub UI).
- Renaming the import and command to avoid the `quibble` PyPI collision (doctor warns instead).
