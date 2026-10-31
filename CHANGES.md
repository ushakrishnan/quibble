---
title: Changes in this revision
description: Revision history and unreleased namespace migration notes.
---

Review fixes applied to Quibble 0.0.1. Finding IDs (A, B, C, D, E, F) refer to the evaluation and
change plan. Nothing here is released; the version number is unchanged.

## Namespace migration

* Move the Python import to `quibble_ml` and the command to `quibble-ml`; no
  colliding `quibble` import or executable aliases are installed.
* Update worker/web subprocesses, resources, packaging, automation, and executable
  documentation together. Product branding and `QUIBBLE_*` settings remain unchanged.
* Keep old JSON/prediction evidence and generated-bundle identifiers readable.
  Retained Python models may need their trusted old environment or retraining.
* Preserve the historical notes below as descriptions of earlier revisions.

## Dependency and recipe consolidation

* Use `pyproject.toml` as the only maintained dependency manifest and `uv.lock`
  as its generated resolution; remove the parallel `requirements.txt` list.
* Select web/data/model/CUDA extras explicitly and use a default development
  dependency group. CI, Pages, Docker, error messages and setup guides use uv.
  Exact sync removes unselected extras; keep the intended selection on later syncs.
* Snapshot code versus weight terms and requested recipes for all backend outcomes.
  Reports, doctor and method help expose license distinctions and unresolved
  checkpoint identities; old records remain explicitly incomplete.
* Reject incompatible recipe reuse and pooling; retain split/model-seed filename
  stems and require separate output directories for different recipes.
* Preserve provenance JSON and participation CSV alongside archived reports.
  No new live foundation-model fits, container builds or remote CI runs are implied.

## Scientific continuation

* Verify TabPFN9.2.0/v3.5 offline Wine and diabetes holdouts on CPU/CUDA after
  the provider's licence check, with existing checkpoint hash, row/class/shape
  validation, CPU HistGB comparison and empty-cache network-denied failure.
  Keep broad coverage and general checkpoint identity explicitly unknown.
* Verify a standalone direct native Kumo Relational SDM0.2.1 CPU/CUDA
  synthetic three-table probe using checked task artifacts and time-filtered
  history. No endpoint or data transfer is required by this local API;
  registered-manifest/worker/browser integration remains outstanding.
* Add pinned experimental Mitra-v2 CUDA task recipes, verifying real offline
  Wine classification and diabetes distributional regression. Keep zero-shot
  classification separate from prediction-time fine-tuning and inclusive costs.
* Stop timed-out workers and their descendants before removing temporary models,
  including Windows virtualenv launcher children. Mark unavailable adaptation
  latency budgets INCONCLUSIVE instead of silently passing them.
* Verify real pinned TabICLv2 Wine classification and diabetes regression holdouts
  on CPU and CUDA, including CPU HistGB comparisons and offline cache use.
  Keep the adapter experimental beyond those small datasets and weight downloads
  disabled by default.
* Add descriptive one-vs-rest class calibration without changing top-label ECE
  or swap policy. Export all classes/bins, equal-row repeated-prediction weights,
  explicit empty/rare/one-sided support and unknown legacy labels.
* Include bounded class tables/figures and complete machine-readable calibration
  outputs in archived reports.
* Add opt-in bounded interaction explanations with row/group-disjoint discovery
  and evaluation, reproducible predicates, aggregate exports and explicit
  exploratory/shared-model-dependence warnings; do not change swap gates.
* Add versioned relational manifests, point-in-time aggregates, frozen dimensions,
  horizon purge, pseudonymous example/entity pairing and shared feature costs.
  Verify synthetic future-row/cache/grouping cases and a permissively licensed
  sampled `rel-arxiv/author-publication` pilot, not an official benchmark.
* Add separate multi-table browser intake with hash-bound relationship/task
  review, edit invalidation, atomic confirmation and a real two-tree report flow.
* Add explicit bounded tree tuning, independent approval, persisted budgets,
  complete search/refit/full-cost disclosure and paired same-backend
  defaults-versus-tuned reports from separate evidence roots.
* Complete the question-led [What Quibble does](docs/what-quibble-does.md)
  guide for original/new capabilities, with everyday examples and honest limits.
  Update related guides and browser/static documentation navigation.
* Preserve the generated lock and revision notes in source distributions;
  verify isolated wheel/source imports, bundled assets, static exports and real
  CPU comparisons without changing the model-rich working environment.

## Browser data intake

* Add local CSV/Parquet upload, schema/sample inspection, target and column-role
  review, split validation, and explicit save confirmation without provider calls.
* Share the dataset reader and validator with CLI loading. Preserve original bytes,
  numeric categorical labels, manifest roles, parser settings and source hashes.
* Bound request bytes, dimensions, decoded size, child-process memory/time and
  staged storage; reject malformed tables and clean up owned failed/cancelled stages.
* Publish the manifest atomically after source files are ready, never overwrite
  existing datasets, and recheck reviewed source identity before fitting.

## Routed workbench shell

* Replace competing navigation cards with Home, a grouped sidebar, breadcrumbs
  and a keyboard-accessible mobile drawer, using existing API contracts.
* Separate reviewed data, methods/settings and explicit Start/monitoring. Share
  top/bottom action handlers and invalidate changed full-run approvals.
* Preserve legacy job bookmarks, browser history, draft inputs and full reports.
  Keep active execution independent of viewed historical evidence and avoid
  overlapping status requests or per-heartbeat dataset rescans.
* Refresh real UI screenshots. Add six accessible accents, high contrast, a
  desktop icon rail, synchronized settings and pre-paint theme restoration.
* Complete report-section navigation, full printing/downloads, explicit
  allowlisted diagnostics, saved-evidence swap/baseline checks with separate
  result artifacts, and descriptive two-job compatibility without paired verdicts.

## OpenML loader verification

* Load all fourteen registered entries and prove network-disabled cache reuse.
* Validate ID/name, row alignment and declared targets during normal loading,
  not only the verification command; use the configured cache consistently.
* Record upstream version/file IDs, terms/citation fields and loaded-content
  fingerprints in run evidence, reports and source-reuse checks.
* Run full CPU HistGB/Weak holdout comparisons on `credit-g` and `abalone`.
  Document exact coverage, task/leakage caveats and unresolved licence fields;
  this does not establish all-model coverage or deployment suitability.

## Experimental TabICLv2 adapter

* Add the optional pinned `tabicl_v2` backend without changing default presets.
* Pin task-specific checkpoint files, immutable revision and expected SHA-256;
  refuse missing cached weights unless download is explicitly enabled.
* Record ensemble/batch/cache/offload/device settings, checked artifacts and
  load timing. Reorder class probabilities and validate both task output shapes.
* Keep real CPU/CUDA inference unverified: this checkpoint uses mocked contracts,
  actual constructor checks and offline missing-weight refusals only.
* Leave live TabPFN validation blocked on explicit weight-term authorization.

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
