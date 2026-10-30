---
title: Results format
description: Run records, predictions, analysis files, and swap verdicts produced by quibble.
---

## Results format

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
    analysis/                            written by `quibble analyze` / `swap`
      summary.csv  calibration.csv  slices.csv  analysis.json   (analysis.json carries data_flags)
      disagreements_<a>_vs_<b>.csv
      swap_<incumbent>_to_<candidate>.json
  report/
    report.md  fig_<dataset>_*.png        stable latest report
    history.md                          index of dated snapshots
    legacy/                             earlier outputs moved here during migration
  invocation.json                       latest sweep selection and cell execution status
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

## Run record (`*.json`)
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
| `dataset_source` | local inputs: source path, file SHA-256, data/manifest fingerprint, target/task, excluded/categorical columns, and encoded class-label order; generated inputs also carry `synthetic` metadata (generator, version, seed, metadata hash) |
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
`psutil` on Windows. The Windows-only dependency is installed automatically by both
the package installer and `requirements.txt`; the result field remains `peak_rss_mb`.

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
