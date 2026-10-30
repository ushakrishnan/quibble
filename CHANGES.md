# Changes in this revision

Review fixes applied to Quibble 0.0.1. Finding IDs (A, B, C, D, E, F) refer to the evaluation and
change plan. Nothing here is released; the version number is unchanged.

## Revision handoff follow-up

- Gate coverage now includes baseline records and matching requested invocation cells.
  Missing, refused, errored or incomplete required results prevent PASS; common-split quality
  remains a separate diagnostic.
- Group/time classification splits missing any training class are refused before fitting.
  New records retain probability-column class order; legacy group/time classification evidence
  without matching metadata must be regenerated in a new output directory.
- Policy provenance identifies default, partial and fully explicit settings in CLI output,
  saved swap JSON, reports and workbench jobs. Partial policies remain supported with
  `--require-thresholds`; inherited defaults are listed instead of being called fully explicit.
- Reports and their archives preserve interval notes, split design, data flags and saved-setting
  mismatch warnings. Statistical and memory claims are qualified below and in the user guides.
- Validation: 458 tests passed on Windows. Fresh offline demo, explicit-policy swap,
  complete/incomplete gate exit checks, and grouped/temporal benchmark reports passed.
  Live optional backends, remote CI and empirical verdict calibration remain unverified.

## Workbench appearance

- Shared color tokens, lighter cards, clearer form hierarchy and responsive navigation refresh
  the workbench, public pages and documentation reader.
- System, Light and Dark appearance choices persist in browser storage and work in exported
  static documentation. Theme assets and fonts need no external service.
- Neutral white/gray and charcoal palettes replace tinted surfaces and the gradient header;
  one blue accent replaces teal and purple navigation colors. Semantic result colors stay distinct.
- Keyboard skip links, visible focus, reduced-motion support and text-labeled verdict badges
  improve navigation and report scanning without changing decisions or hiding caveats.
- Browser smoke and full comparisons retain reports, history, threshold provenance and notes.

## Verdict statistics
- **Approximate retraining interval (A1, A2).** Swap and gate use the outer bounds of a bootstrap
  and a Nadeau-Bengio corrected split interval under covariance assumptions. It needs 5 or more
  splits; with fewer the note says retraining variability is not counted. `summary.csv` gains
  `cv_lo` and `cv_hi`. The combined verdict's false-PASS/false-FAIL rates are not calibrated.
- **Worst-slice maximum (A3, F2).** The maximum over fixed selected slices is recomputed in every
  bootstrap resample. Target-driven discovery is not repeated; family-wise coverage is unverified.
- **Placeholders are labeled (A4).** Swap prints and saves whether default limits were used.
  `--require-thresholds` requires a non-empty explicit policy; partial policies still inherit
  listed defaults. New tunable keys:
  `confident_prob`, `regression_gap_std`, `near_margin`.
- **No interval, no PASS (A5, F3).** A check without a usable interval is INCONCLUSIVE. Dropped
  bootstrap resamples are counted and reported.
- **Disagreement noise floor (A6, F4).** With 2 or more model seeds, swap subtracts the incumbent's
  own seed-to-seed disagreement. `summary.csv` gains `n_seeds`, `seed_std`, `reseed_disagreement`.
  Raw disagreement remains visible; uncertainty in the noise estimate is not propagated.
- **ECE check has an interval (A7).** Rate intervals are kept inside 0 to 100%.
- **Memory and latency (A8, F7, F11).** Memory compares a baseline-adjusted process peak
  (floored at 50 MB), not model-only footprint. Imports, inference and earlier high-water
  marks can contribute. Latency adds warm-up calls and compares medians.

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
