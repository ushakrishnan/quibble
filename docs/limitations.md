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
  [Testing your own data](testing-your-data.md). No Excel, unlabeled-prediction, streaming,
  group/time split, or separate train/test-file interface is implemented.
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
- Intervals reflect test-set noise only, not retraining variability. `split_std` is a rough guide.
- Repeated CV splits overlap in their training sets, so split-to-split variation understates true
  variance.
- The worst-slice check is a maximum over several slices, without multiple-comparison correction.
- Slices cover only missingness, rare categories and three numeric features; no interaction slices.
- Multiclass calibration is top-label only; there is no per-class reliability analysis.
- The disagreement thresholds (0.25 target standard deviations, 90% confidence, 1% rare category,
  0.2 margin) are fixed defaults, not tuned.
- The swap limits are placeholders and need tuning to what a swap really costs you.
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
- Latency and memory are single-machine, and meaningful only for `--jobs 1`.
- `peak_rss_mb` is a per-process high-water mark; `peak_torch_vram_mb` counts PyTorch allocations
  only (including PyTorch components used by AutoGluon, but not CatBoost or other non-Torch allocations).
- AutoGluon's seed is not controlled and its time budget dominates `fit_s`.
- HistGB uses random validation splits above about 10,000 training rows, so its seed matters there
  even though it does not on small data.
- Finished runs are identified by file name; changing settings without changing `--out` reuses old
  results (a warning is printed, but they are still skipped).
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
  maximum capabilities of the installed library.
- Comprehensive validation of the OpenML registry and its download paths.
- TabPFN on CUDA, CatBoost GPU regression/large-data coverage, and historical AutoGluon
  GPU-budget behavior. CatBoost has passed a small local CUDA classification smoke;
  new GPU-only runs exclude AutoGluon.
- TabFM and KumoTabular have passed small CPU classification and regression smoke tests with
  cached pretrained weights. Local CUDA Wine classification fits have also succeeded;
  GPU regression and large-dataset behavior remain unverified.
- The Docker files and the GitHub Actions workflow.

## Known rough edges
- The OpenML registry is hand-written.
- `quibble compare` looks at a single split id; use `analyze` for pooled results.
- The bootstrap's memory grows with `QUIBBLE_BOOT x rows`.

See the [project plan](project-plan.md) for what is intended next.
