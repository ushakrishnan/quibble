---
title: Limitations
description: Statistical, platform, measurement, and validation boundaries.
---

## Limitations

The following boundaries determine which conclusions the current evaluation supports.

## Scope
- Tabular classification and regression with **one target column**. No multi-label, ranking, time
  series, or survival problems.
- Datasets come from OpenML (by id), the three built-in scikit-learn sets, or named local
  CSV/Parquet folders. Local inputs need explicit target/task metadata; see
  [Testing your own data](testing-your-data.md). Local data can declare a group or time column for
  group and temporal splits; OpenML and built-in sets use random splits. No Excel,
  unlabeled-prediction, streaming, relational or multi-table, or separate train/test-file interface
  is implemented.
- Whole datasets are held in memory.
- The [synthetic workbench](experiment-workbench.md) implements bounded independent-row
  generation and optional provider-configured specification drafting. It does not simulate arbitrary
  business rules, correlated features, time/group structure, or validate domain realism.
  Externally generated tables can also use the local-data interface.
- Backends use their configured presets/defaults; quibble does not tune hyperparameters, so a comparison reflects
  out-of-the-box behaviour.
- No hosted-service prediction backend. The optional configured planner sends explicitly
  consented scenario text for specification drafting, not dataset rows for evaluation.
- The browser workbench is localhost-only, single-user, and not a secure deployment
  platform. It runs one experiment at a time and has no cancellation interface.
- The core demo runs on native Windows. Optional model libraries can have additional platform
  requirements; local CUDA validation is limited to selected classification fits.

## Statistics
- Bootstrap intervals reflect test-set noise only. Approximate corrected split intervals need
  5 or more splits; with fewer, swap says retraining variability is not counted. The cutoff is an
  implementation rule, not a precision guarantee.
- The corrected interval uses Nadeau-Bengio covariance assumptions, including exchangeable
  splits, which temporal forward-chaining folds are not. Row bootstrap does not model serial
  dependence. The combined verdict is not calibrated for false-PASS or false-FAIL rates.
- Slices cover only missingness, rare categories and three numeric features; no interaction slices.
  They are chosen using the target and ranked by spread, so they are exploratory. `slices.csv` has no
  intervals. Swap bootstraps the maximum over fixed selected masks, counts test-set noise only,
  and does not repeat discovery. Family-wise coverage has not been established.
- Multiclass calibration is top-label only; there is no per-class reliability analysis. The swap ECE
  check has an interval, but ECE depends on its 10 equal-width bins.
- Disagreement thresholds (regression gap, confidence, rare category, margin) are configurable
  defaults, not tuned. The reseed noise floor needs 2 or more model seeds; deterministic backends
  show zero seed noise even though retraining on different data would change their predictions.
  Noise-adjusted disagreement retains the raw rate but does not propagate uncertainty in the
  estimated noise floor.
- The [packaged swap defaults](../quibble/swap_thresholds.json) need review against what a swap
  really costs you; they are not universally safe acceptance criteria. Verdicts
  use default model settings; quibble does not tune hyperparameters, so a swap verdict says nothing
  about a tuned candidate.
- Data flags are heuristics: a single-feature check, ID-like columns, duplicate rows and column
  names that look like dates. They do not detect leakage through combinations of features or
  through the way the data was collected.
- Group and temporal splits exist for local datasets only, and only one time or group column.
  Nested hierarchies, gaps between train and test periods, and expanding versus sliding windows are
  not configurable.
- Grouped or temporal classification splits missing any dataset class in training are explicitly
  refused before backend fitting. Quibble does not guess probability columns for unseen classes.
- Prediction agreement does not identify a predictor's internal algorithm, and a report's
  interpretation is not an explanation of its internal computation.
- Familiar public datasets may have been encountered during pretraining. Quibble does
  not currently test or rule out that exposure.
- Freshly sampled synthetic rows reduce direct reuse of familiar benchmark examples,
  but do not prove an unseen task. Similar generating rules may already be familiar, the
  planner can propose familiar assumptions, and the chosen process can favor a method.
- Synthetic-task findings apply to their generating assumptions; they do not establish
  real-world suitability or causal relationships. Repeated splits of one generated table
  do not measure variation across independently generated datasets.

## Measurement
- Latency and memory are single-machine, and meaningful only for `--jobs 1`. Latency uses 3 warm-up
  calls and a median; it includes a small pandas copy per call that applies to every backend.
- `peak_rss_mb` is a per-process high-water mark that includes the interpreter and imports.
  `peak_rss_increase_mb` subtracts the process size just before fitting, and is what swap compares.
  It is a baseline-adjusted process peak, not model-only memory: imports, inference, temporary
  allocations and earlier process peaks can contribute. Swap floors each estimate at 50 MB for
  the ratio and displays the unfloored estimates.
  It needs `psutil` or Linux `/proc`; on other systems only the peak is recorded.
  `peak_torch_vram_mb` counts PyTorch allocations only (including PyTorch components used by
  AutoGluon, but not CatBoost or other non-Torch allocations).
- `fit_s` excludes pretrained-weight loading, which is `load_s`; `fit_total_s` is both. Weight loading
  time is measured where the adapter builds the model, so lazily loaded weights can still land in `fit_s`.
- AutoGluon's seed is not controlled (`seed_controlled: false`) and its time budget dominates `fit_s`.
- HistGB uses random validation splits above about 10,000 training rows, so its seed matters there
  even though it does not on small data.
- Finished runs are identified by file name. Changing settings without changing `--out` reuses old
  results with a warning (they are skipped), and `quibble benchmark` then exits non-zero. The check
  compares device, preprocessing, CV scheme, run type, model-specific settings (TabFM, Kumo,
  AutoGluon preset and time limit, TabPFN row limit), latency sample size, the dataset fingerprint
  and the installed versions of numpy, pandas and the backend's library. Settings it does not
  compare, such as thread count, can still differ.
  Invocation records retain detected differences; reports warn when saved-setting mismatches
  exist. Detection is not automatic invalidation or a new evaluation under the requested settings.
- Error results count as finished. Rerun only the failed cells with `--retry-failed`; refusals are
  deterministic and are kept.
- Local data/manifest fingerprints prevent changed inputs from being analyzed against old
  predictions. Use a new output root for changed tables or column rules.
- Mixed CPU/GPU reports are explicitly labeled. AutoGluon's recorded device denotes its
  historical resource budget, not proof that every fitted component used GPU.
  New CUDA sweeps exclude CPU-only adapters and AutoGluon rather than mix model devices.
  This narrows the comparison; CPU runs are needed to include all baselines.
- Memory/worker failures are recorded without silent CPU fallback. Configuration, archive,
  or output-storage failures can still abort a command.
- Archived reports preserve generated evidence, not full source feature tables. Analysis/report
  regeneration can need original OpenML data and network access if its cache is missing.

## Not yet verified on real hardware or services
- TabPFN end to end. Its row/feature/class refusal limits are harness policy, not validated
  maximum capabilities of the installed library. `quibble doctor` labels it experimental.
- Comprehensive validation of the OpenML registry and its download paths.
- TabPFN on CUDA, CatBoost GPU regression/large-data coverage, and historical AutoGluon
  GPU-budget behavior. CatBoost has passed a small local CUDA classification smoke;
  new GPU-only runs exclude AutoGluon.
- TabFM and KumoTabular have passed small CPU classification and regression smoke tests with
  cached pretrained weights. Local CUDA Wine classification fits have also succeeded;
  GPU regression and large-dataset behavior remain unverified.
- The Docker files and the GitHub Actions workflow have not run remotely. The test suite and the
  demo have passed locally on Linux with Python 3.12 with the `dev` extra installed.

## Known rough edges
- The OpenML registry is hand-written.
- `quibble compare` looks at a single split id; use `analyze` for pooled results.
- The bootstrap's memory grows with `QUIBBLE_BOOT x rows`.

See the [project plan](project-plan.md) for what is intended next.
