---
title: Results format
description: Run records, predictions, analysis files, and swap verdicts produced by quibble.
---

## Results format

Saved results are an experiment's receipts: what ran, which examples it saw,
what it predicted, how long it took, and what failed. Keep the matching receipts
together. A score copied without its recipe or test-row identities cannot
establish that two models took the same test.

Everything quibble produces is a plain file, so results can be inspected, diffed, archived and
re-analyzed without retraining. Slice/trait analysis reloads the original dataset; saved
predictions alone are not a complete replacement for its source/cache.

Browser experiments also save `results/web/<job-id>/job.json` and, for full runs,
`thresholds.json`. The job record's `thresholds` contains the complete effective policy;
`thresholds_default` identifies default versus custom selection. Both are null for smoke runs.
Saved policies remain available when reopening an experiment after a restart.

## Layout

```text
<out>/                                  default: results/
  <dataset>/
    <backend>_ss<split>_ms<mseed>.json   run outcome, including model-level failures
    <backend>_ss<split>_ms<mseed>.pred.npy   predictions; check matching record for success
    y_test_ss<split>.npy                 true test targets for that split
    idx_test_ss<split>.npy               test row positions in the loaded dataset
    analysis/                            written by `quibble-ml analyze` / `swap`
      summary.csv  calibration.csv  slices.csv  analysis.json   (analysis.json carries data_flags)
      disagreements_<a>_vs_<b>.csv
      swap_<incumbent>_to_<candidate>.json
  report/
    report.md  fig_<dataset>_*.png        stable latest report
    participation.csv                    participation and code/weight terms
    backend_provenance.json               complete runtime metadata and recipes
    history.md                          index of dated snapshots
    legacy/                             earlier outputs moved here during migration
  invocation.json                       latest sweep selection and cell execution status
  .quibble-tier.json                     root tier/search guard, including failed campaigns
  artifacts/<dataset>/<cell>/<run-id>/   optional retained models, not copied into history
  smoke/<unique-id>/                    isolated setup checks with their own reports
  runs/
    YYYY-MM-DD/
      HHMMSS-microseconds-id/
        snapshot.json                   snapshot provenance and source run IDs
        <dataset>/                      copied records, predictions, targets, analysis
        report/                         archived report and figures, when applicable
```
`ss` is the split id (the split seed in holdout mode, the fold id in CV mode); `ms` is the model seed.
The `.pred.npy`, `y_test` and `idx_test` files are NumPy arrays (`numpy.load`). Predictions are
`(n_test, n_classes)` probabilities for classification and `(n_test,)` for regression. Test arrays
are in the same row order for every backend of a split, which is what makes pairing possible.

## Recipe identity

New records snapshot the actual recipe tier, explicit adapter parameters, library
versions, requested preprocessing/device, CV assignment, split/model seeds, and
checkpoint identity (including unknowns). `sha256` covers the normalized request;
`family_sha256` excludes the individual split/model seed so normal repeated
evaluations can be pooled. `algorithm_sha256` identifies the algorithm-level
configuration for cached reuse. After fitting, `resolved` captures estimator
parameters where exposed, effective backend settings, source fingerprint and
train/test-row hashes. `resolved_sha256` and `full_sha256` cover that evidence.

Changed tiers, search recipes, dependency versions, ensemble settings or declared
checkpoints require a separate `--out`; `--force` cannot override those checks.
Writing new records with a different preprocessing/device/CV family is also
rejected. Cached runs with only those historical setting differences retain
their existing warning behavior: they are saved evidence, not a new evaluation.
Do not treat that warning as proof that the old predictions were invalidated.

Pooling rejects inconsistent digests or mixed recipe families, including across
model seeds of the same backend. Pure legacy evidence remains readable with
incomplete metadata; new fits require a separate output directory rather than
mixing versioned records into legacy evidence. A failed fit can still be retried
after installing a missing library because it contributed no usable predictions.

Foundation checkpoint revisions/checksums and some library-internal parameter
details are not exposed by the current adapters. Those fields remain unknown and
`identity_complete` stays false. These digests detect recorded configuration
differences; they cannot prove unknown downloaded weights were identical.
The metadata is not a signature or protection against deliberate file tampering.
Reports, analysis JSON, participation CSV, workbench jobs and dated snapshots
preserve the recorded terms and recipes instead of inventing historical terms.

## Tuned tree evidence and cost

`campaign_tier` identifies the requested comparison tier. `tier` and
`recipe.requested.tier` identify the actual model recipe. In a tuned campaign,
HistGB, LightGBM and CatBoost are `tuned`; unchanged foundation adapters remain
`defaults` (with their existing adaptation metadata), not implicitly fine-tuned.
Unsupported non-foundation families are refused explicitly.

Successful tuned trees add a `tuning` object with:

* `protocol`: frozen domains, search version/seed, metric rule, trial/fold budget,
  candidate ordering, aggregation, tie-breaker and deadline semantics
* `split_spec`, `outer_training_indices_sha256`, `inner_splits`: declared split
  structure and hashes of each inner training/validation row identity
* `attempts`: each attempted candidate's index/parameters, model/search seeds,
  metric and fold-level split IDs, values, durations, statuses and failures
* `attempted_fits`, `successful_fits`, `requested_candidate_fits`,
  `requested_refit_fits`: actual and requested counts, including failed fits
* `selected_trial`, `selected_params`, `refit`: the winner and its final
  outer-training refit status, seed, parameters, row count and duration
* `search_s`, `total_s`, `deadline_overrun`: measured selection and total
  search/refit wall time and cooperative overrun disclosure

Top-level `search_s`, `refit_s` and `search_refit_s` expose selection, final refit,
and their measured combined cost. `fit_total_s` also includes the full search
call; for tuned trees, legacy `fit_s` therefore includes inner fitting,
validation and selection overhead, not just winning-model training.
`load_s` measures final-refit model construction only; inner construction is
already included in search cost. Do not add `search_refit_s` to `fit_total_s`:
they overlap. Search time is not backend-independent compute or a fairness claim.

A search with no valid complete candidate writes `error` and
`tuning.status=failed`, retains completed failure evidence, and never falls back
to defaults. An interrupted worker records `error_kind=timeout` or `worker_exit`
and `tuning.status=incomplete`; attempt details may be unrecoverable and no
partial results qualify as success. Check the JSON outcome even if an old
prediction file remains after a failed forced rerun.

`invocation.json` carries `campaign_tier`, `tuning_protocol` and
`worker_timeout_s`. Recipe digests include the search version, exact finite
domains, metric, budgets, seeds and cooperative deadline. Changing any of these
cannot reuse successful tuned evidence. The root guard additionally prevents
mixing tiers/protocols even across different backends or failed records.
Reports export `training_protocol.csv` and `stage_costs.csv` alongside full
attempt evidence in `backend_provenance.json`. The report distinguishes requested
campaign from actual adapter tier and shows search/refit/full model costs with
explicit overlap warnings. `model_pipeline_s = fit_total_s + batch_prediction_s`
includes prediction-time adaptation but excludes latency probes and shared
feature construction.

Browser `job.json` saves `tier`, `campaign_tier`, all `tune_*` controls,
`tuning_confirmed` and `tuning_protocol`; full-run approval is separate.
Defaults are pinned independently of inherited tuning environment settings.
Changing the browser's data/method/settings request invalidates approval.
For explicit separate-root contrasts, `quibble_ml.tuning.load_tier_pair` returns
`defaults` and `tuned` entries containing `record` and `predictions`, plus shared
`y_test` and `idx_test` arrays. It rejects mismatched protocols, sources, row
identities, targets or failed/incomplete cells; it does not pool, copy or alter
either campaign.
The report CLI offers `--compare-tiers DEFAULTS_ROOT TUNED_ROOT --tier-backend
histgb`, with `--datasets` and `--report-dir` as needed. It emits tier-qualified
same-backend paired cell point estimates in `tier_comparison.csv` and complete
evidence in `tier_comparison.json`, without mixing roots or issuing swap approval.

Relational records do not require flat-file `path`, `target` or `drop_columns`.
Their `relational_audit` is shown separately. Worker `feature_costs.feature_total_s`
is displayed beside each model only as a shared reference, never summed across
backends. When present, root `relational-feature-stage.json` preserves the
explicit front-stage feature time and cache state once, separately from warm
worker reloads; it is linked and included in archived provenance.
When available, reports display at most twenty `relational_slices` rows and link
the complete `analysis/relational_slices.csv`. Limited and empty support remain
explicit. These fixed label-free masks provide descriptive point estimates,
not intervals, causal claims or gate criteria.

## Run record (`*.json`)

Adapters with `prediction_adapts=true` run the outer-test batch prediction once
and skip repeated warm-up and single-row latency probes.
`adaptation_and_prediction_s` measures that complete prediction call, including
fused prediction-time loading, adaptation and inference.
`predict_ms_per_row_batch` is its inclusive per-row amortization, not isolated
inference latency. `single_row_p50_ms` and `single_row_p95_ms` are null,
`single_row_latency_available` is false, and `weight_load_s`, `adaptation_s` and
`prediction_only_s` remain null when those components cannot be measured
separately. Fit-stage context registration costs remain in the existing fit/load
fields; they must not be presented as isolated prediction-time weight loading.

OpenML records now include `dataset_source` with dataset/version/file IDs,
the upstream ARFF `md5_checksum`, target, licence/citation fields and encoded
`class_labels` for classification. Its `sha256` hashes loaded content under the
recorded `hash_scheme`, not raw download bytes. The source `fingerprint` combines
that content with upstream identity and task/target. These fields survive
analysis, reports and snapshots; missing legacy fingerprints are not fabricated.
See [dataset source hashes](datasets.md#source-hashes-and-reuse).

A real example (environment block omitted):
```json
{
 "backend": "histgb", "dataset": "wine",
 "split_seed": 0, "model_seed": 0, "cv": "3x1",
 "device_requested": "auto", "device": "cpu", "threads": 0, "jobs": 1,
 "preprocess_requested": "ordinal",
 "task": "classification", "n_train": 118, "n_test": 60,
 "fit_s": 0.107, "predict_ms_per_row_batch": 0.040,
 "single_row_p50_ms": 1.70, "single_row_p95_ms": 1.87, "peak_rss_mb": 193.3,
 "preprocess_effective": "ordinal", "device_effective": "cpu",
 "accuracy": 0.9333, "log_loss": 0.1447, "brier": 0.0866, "auc": 0.9972
}
```

| Field | Meaning |
|---|---|
| `backend`, `dataset`, `split_seed`, `model_seed` | identity of the run (`split_seed` is the split id) |
| `run_id`, `timestamp` | new records: invocation ID and local ISO timestamp with UTC offset; sweep children share an ID |
| `run_type`, `cv_seed`, `cache_dir` | benchmark/smoke designation, CV assignment seed, and central cache root |
| `backend_metadata` | runtime library/version, code license/source, weight reference terms, review date, access caveats, verification scope, and explicit unknown checkpoint revision/checksum |
| `recipe` | versioned requested recipe and digests; fitted estimator parameters/context where exposed; explicit incomplete identity for unresolved checkpoints |
| `dataset_source` | local inputs: source path, file SHA-256, data/manifest fingerprint, target/task, excluded/categorical columns, and encoded class-label order; generated inputs also carry `synthetic` metadata (generator, version, seed, metadata hash) |
| `dataset_source.uploaded` | browser imports: original filename/hash, parser format/encoding/missing-value rules and pandas version, reviewed manifest/schema/rows, and import-metadata hash; changes invalidate the reviewed identity |
| `device_requested`, `device`, `device_attempted`, `device_effective` | requested policy, resolved sweep device, attempted backend device, and completed backend device where available |
| `error_kind`, `worker_exit_code`, `cleanup_error` | optional failure classification, crashed worker exit code, and cleanup failure detail; failures do not silently retry on another device |
| `n_train_available`, `n_test_available` | rows before optional smoke sampling; `n_train`/`n_test` record actual evaluated sizes |
| `model_retention`, `model_path`, `leaderboard` | disabled/retained/unsupported/failed retention status; optional artifact/leaderboard paths relative to the output root |
| `cv` | split scheme (`holdout` or `KxR`); absent in results from very early versions |
| `device_requested` / `device` | what was asked for (`cpu`/`cuda`/`auto`) and the resolved device |
| `device_effective` | device this backend actually used (LightGBM and HistGB are always `cpu`) |
| `preprocess_requested` / `preprocess_effective` | categorical rule asked for vs. applied |
| `threads`, `jobs` | threading and sweep parallelism; `jobs > 1` means timings are indicative only |
| `task`, `n_train`, `n_test` | task type and split sizes |
| `prediction_classes`, `split` | new classification records: encoded probability-column order; new records: declared evaluation scheme |
| `fit_s` | wall-clock seconds spent fitting, excluding pretrained-weight loading |
| `load_s`, `fit_total_s` | seconds spent building the model (loading pretrained weights) and the total of load plus fit |
| `predict_ms_per_row_batch` | milliseconds per row when predicting the whole test set |
| `single_row_p50_ms`, `single_row_p95_ms` | latency percentiles over `QUIBBLE_SINGLE_N` single-row predictions |
| `peak_rss_mb` | process peak resident memory in decimal MB (high-water mark for that process, including interpreter and imports) |
| `rss_before_fit_mb`, `peak_rss_increase_mb` | resident memory before fitting and lifetime process peak minus that baseline; an estimate including imports, inference and earlier peaks, not model-only footprint; absent when current memory is unavailable |
| `single_n`, `seed_controlled`, `backend_verification` | latency sample size used; whether the model seed makes training reproducible (`false` for AutoGluon); the adapter's verification label at run time |
| `tabpfn_max_rows` | TabPFN only: the configured row limit, so a changed limit is detected on reuse |
| `peak_torch_vram_mb` | CUDA-attempted backend only: peak PyTorch allocation in that process (not non-Torch libraries) |
| `accuracy`, `log_loss`, `brier`, `auc` | classification metrics |
| `rmse`, `mae`, `r2` | regression metrics |
| `ag_preset`, `ag_time_limit` | AutoGluon only |
| `tabfm_n_estimators` | TabFM ensemble count |
| `kumo_size`, `kumo_n_estimators` | KumoTabular pretrained model size and ensemble count |
| `refused` | backend declined the task; value is the reason |
| `error` | run failed or timed out (`ExceptionType: message`, or `timeout after Ns`); may coexist with metrics if diagnostics, retention, saving, or cleanup failed after prediction |
| `env` | reproducibility block (below) |

Treat any record with `error` or `refused` as unsuccessful even if it also has metrics or
an older prediction array still exists after a forced retry. The pooled analysis loader
excludes such records. Worker timeout/exit/missing-result records can contain fewer fields
than normal fit records. Configuration, archive, and output-storage failures can still abort
the command before a normal result is written.

Peak resident memory uses `resource.getrusage` on Unix and the peak working set from
`psutil` on Windows. Its platform marker in `pyproject.toml` installs it
automatically on Windows; the result field remains `peak_rss_mb`.

### `env` block
`python`, `platform`, `cpu`, `cores`, `ram_gb`, `git` (short commit hash, or null outside a git
repository), `packages` (versions of numpy, pandas, scikit-learn, lightgbm, catboost,
autogluon.tabular, tabpfn, tabfm, structured-data-models, torch; null if not installed) and, on `cuda`, `gpu` with `name`, `cuda`
version, `vram_total_gb`, `vram_free_gb`. Unavailable hardware/software values can be null.

## `analysis.json`
`dataset`, `task`, `metric`, `model_seed`, `splits` (ids used), `summary` (list of per-backend
rows as in `summary.csv`), optional local `dataset_source`, and `pairs`, keyed `<a>_vs_<b>`:

| Key | Meaning |
|---|---|
| `entries`, `disputed`, `rate` | pooled prediction pairs, how many disputed, and the share |
| `confident_both` | classification: disputed and both at least 90% sure |
| `a_right_only`, `b_right_only`, `both_wrong` | classification: shares of disputed rows |
| `a_closer`, `median_gap_std` | regression: share of disputed rows where `a` was closer; median gap |
| `trait_lift` | per trait: `share_disputed`, `share_all`, `lift` |
| `csv` | file name of the disputed-rows CSV |

## `swap_*.json`
`overall` (`PASS`/`FAIL`/`INCONCLUSIVE`), `thresholds` (the complete effective limits),
`thresholds_default` (true for implicit packaged defaults, false for a non-empty supplied policy),
and `checks`, a list of
`{check, observed, limit, verdict, note}`.
`threshold_provenance` records `kind` (`defaults`, `partial`, or `explicit`), the `provided`
and `inherited` setting names, and the packaged source. `flags` and `split` accompany the
verdict. Reports preserve these qualifications rather than using the legacy boolean alone.

## Sizes and version control
JSON files are small. `.pred.npy` files scale with test rows x classes x splits x backends x seeds.
Training preserves before-write and after-training snapshots; report regeneration also
creates a snapshot. Repeated commands still reuse cached cells unless forced. Snapshots contain
generated dataset files and optional reports, never previous snapshots, legacy folders,
downloaded weights, or virtual environments. `snapshot.json` is written only after copying
completes, and the history index lists completed snapshots. Archive failures stop the command
instead of silently overwriting unprotected results.

Snapshots preserve generated report evidence: images and disputed-row CSV links resolve inside
that snapshot. They do not include original source feature tables; retained-model links still
point to the separate artifacts root. Older reports and prediction arrays remain unchanged when current results are
retrained or re-analysed. Snapshot storage grows over time; there is no automatic pruning.

The default `.gitignore` excludes `results/`; to publish results, commit the JSON files and the
report, and decide separately whether to include the arrays.
