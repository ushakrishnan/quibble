---
title: Backends
description: Supported model libraries, unified requirements, and backend behavior.
---

## Backends

A backend is a small class in `quibble/backends.py` with two methods: `fit(Xtr, ytr, task, seed, cfg,
device)` and `predict(X)`. `predict` returns class probabilities `(n, k)` for classification and a
vector `(n,)` for regression. The table below lists registered backends. `--backends all`
selects all public backends for a sweep; it is not a list-only command and excludes `weak`.
When CUDA is selected or Auto resolves to CUDA, incompatible methods are visibly
excluded. CPU runs keep all selected methods. No algorithm is silently substituted.

| Name | Library | Tasks | GPU | Native categoricals | Tested |
|---|---|---|---|---|---|
| `histgb` | scikit-learn HistGradientBoosting | both | no | yes | yes (CPU) |
| `lightgbm` | LightGBM | both | no (CPU adapter) | yes | yes (CPU) |
| `catboost` | CatBoost | both | yes | yes | CPU both tasks; local CUDA classification smoke |
| `tabpfn` | TabPFN | both | yes | no (ordinal) | refusal logic only |
| `tabfm` | Google Research TabFM (PyTorch) | both | yes | yes | CPU both tasks; local CUDA Wine classification fits |
| `kumo_tabular` | NVIDIA structured-data-models | both | yes | yes | CPU both tasks; local CUDA Wine classification smoke fits |
| `autogluon` | AutoGluon Tabular | both | excluded from GPU-only runs | own preprocessing | CPU, one dataset per task |
| `weak` | HistGB, handicapped | both | no | yes | yes (internal) |

All backends use library defaults except where noted. quibble is not a tuner.
Install `requirements.txt` into this repository's `.venv` using the
[local setup](getting-started.md#install). It includes the
libraries below; separate backend installations are not needed.

## `histgb`
`HistGradientBoostingClassifier` / `Regressor` with `random_state=seed`. In `native` mode it uses
`categorical_features="from_dtype"`. On training sets under about 10,000 rows the defaults involve no
randomness, so different model seeds give identical predictions; above that, scikit-learn's automatic
early stopping uses a random validation split and the seed starts to matter.

## `lightgbm`
`LGBMClassifier` / `Regressor` with `random_state=seed`, `verbose=-1`, `n_jobs` from the thread
setting. With default settings (no bagging or feature subsampling) predictions do not depend on the
seed. This adapter uses LightGBM's CPU backend and is excluded from CUDA runs.
LightGBM has GPU backends, but they need a compatible build and adapter configuration;
its CUDA backend is not supported on Windows. The model is not silently replaced.

## `catboost`
`CatBoostClassifier` / `Regressor` with `random_seed=seed`, `verbose=0`, `allow_writing_files=False`,
`thread_count` from the thread setting, and `task_type="GPU"` when the device is `cuda`. In `native`
mode categorical columns are passed as strings (missing values become the string `__nan__`). The GPU
path has passed a small local classification smoke run. This does not establish
GPU regression, large-data support, or general device coverage.

## `tabpfn`
`TabPFNClassifier` / `Regressor` (a pretrained tabular foundation model) with `device` and
`random_state=seed`. quibble **refuses** (records a refusal, not an error) when:
- training rows exceed the limit: `QUIBBLE_TABPFN_MAX_ROWS`, default 10,000 on `cuda` and 1,000 on
  `cpu` (Quibble's conservative policy limits, not verified maximum capabilities of the installed TabPFN release),
- there are more than 500 features,
- a classification target has more than 10 classes.

Notes:
- `tabpfn` is included in the requirements. Model weights are downloaded on first use and may need
  network access and a Hugging Face account or token, depending on the TabPFN release.
- It always receives ordinal-encoded categoricals.
- TabPFN predicts by conditioning on the training set, so single-row latency is expected to be much
  higher than for tree models. That is a genuine cost of the approach and the benchmark will show it.
- Only the refusal logic is covered by tests; the fit/predict path has not been run.

## `tabfm`

The requirements select `tabfm[pytorch]` on Python 3.11+, not the JAX backend.
The PyTorch loader receives the resolved `cpu` or
`cuda` device explicitly; the estimator receives the model seed as `random_state`.
`QUIBBLE_TABFM_N_ESTIMATORS` controls ensemble size (default 4).

Like the tree adapters, TabFM receives ordinal codes by default and pandas categoricals
when `QUIBBLE_PREPROCESS=native`. It then applies its own model preprocessing.
Ordinary Quibble runs do not downsample the training split; explicit smoke checks do.
Large datasets and repeated single-row inference can therefore be slow on CPU.

Pretrained weights download from Hugging Face on first use and are subject to their own
terms. Quibble sets the local Hugging Face cache automatically; use `QUIBBLE_HF_CACHE_DIR`
to override it rather than an inherited `HF_HOME`. See [sources and destinations](datasets.md#actual-download-sources-destinations-and-triggers).
The loader's in-process model cache is disabled; sweep processes remain isolated.

## `kumo_tabular`

Included in the requirements on Python 3.11+. The distribution is
`structured-data-models`; its import is `sdm`. `sdm.models.KumoTabular` receives the
requested device and `QUIBBLE_KUMO_SIZE` (`small`, `medium`, or `large`; default `small`).
`QUIBBLE_KUMO_N_ESTIMATORS` controls ensemble size (default 4). A device-local PyTorch
generator seeded with the model seed controls the fit's stochastic preprocessing.

Features use Quibble's training-fitted ordinal or native preprocessing, then convert to
`sdm.TableTensor`. The training feature schema is reused for every prediction. Classification
targets are explicitly categorical; regression targets are numerical. Class probability
columns are reordered to the encoded target's class order, including multiclass targets.
Missing, duplicate, nonfinite, or invalid probability outputs raise an error.

Regression returns the mean of all 999 equally spaced predicted quantiles, `q001` through
`q999`, as an approximation to the predictive mean. It does not use the first quantile or
the median. Missing, duplicate, unexpected, or nonfinite quantiles raise an error.
The model's inference cache is cleared after each run, including failed fits.

Weights download on first use and reuse the configured local Hugging Face cache.
Full training splits are retained outside explicit smoke checks. Local CUDA Wine classification
smoke records are available; GPU regression and large-dataset behavior remain unverified.

Example comparison after installing the requirements:

```powershell
quibble run --sweep --backends lightgbm,tabfm,kumo_tabular --datasets breast_cancer --cv 3x2 --model-seeds 0 --device cpu --jobs 1 --out results_cpu
quibble report --out results_cpu --swap lightgbm:tabfm --swap lightgbm:kumo_tabular
```

Changing the TabFM ensemble count or Kumo size/ensemble count warns before reusing saved
results. Use `--force` or a different `--out` for each configuration.
Missing optional packages are recorded as run errors; they do not break the core demo.

## `autogluon`
`TabularPredictor` with preset `QUIBBLE_AG_PRESET` (default `medium_quality`) and time limit
`QUIBBLE_AG_TIME_LIMIT` (default 120 s, held fixed across datasets), `num_gpus=0` on CPU, and
`num_cpus` from the thread setting. Regression uses `problem_type="regression"`; classification is
inferred. It does its own preprocessing, so `QUIBBLE_PREPROCESS` does not change it (recorded as
`preprocess_effective: native`).

Notes:
- **The seed is not controlled**, so repeated runs can differ.
- Training uses up to the time limit, so `fit_s` is dominated by that budget. A sweep of
  12 datasets x 10 splits at 120 s is about 4 hours for a single model seed.
- Each run trains into a temporary directory that is deleted afterwards. With `--keep-models`,
  its contents are first copied to a unique retained-fit directory. The internal validation
  leaderboard is saved even without retention; see [reports](reports.md#participation-and-model-artifacts).
- AutoGluon and its model families constrain their dependencies. Install the complete
  requirements together, then run `python -m pip check` and `python -m quibble doctor`.
- GPU-only runs exclude AutoGluon because a GPU budget does not prevent CPU model families.
- It is tested on CPU on one classification and one regression dataset.

## `weak` (internal)
HistGB limited to `max_iter=2, learning_rate=0.05`. It exists so that a swap to a clearly worse model
must fail; the test suite and the demo rely on that. It is hidden from `--backends all`.

## Adding a backend
Add a class to `quibble/backends.py` and include it in the class tuple used to construct
the `BACKENDS` dictionary at the bottom of the file. The shortest path subclasses `_SK`,
which handles categorical preprocessing for you:

```python
class MyModel(_SK):
    name, native_ok, gpu = "mymodel", False, False   # native_ok: supports QUIBBLE_PREPROCESS=native
    def make(self, task, seed, cfg, device, cat_idx):
        from mylib import Clf, Reg                    # import inside make(): optional dependency
        return (Clf if task == "classification" else Reg)(random_state=seed)
```
Rules:
- Import optional libraries inside `make()` / `fit()`. A missing package is recorded as an error
  result (`ModuleNotFoundError`) rather than breaking everything else.
- Raise `Refused("reason")` for unsupported sizes or tasks. Refusals appear in the report's failure
  matrix.
- Set `gpu = True` only if the backend really uses the device argument; this controls scheduling.
- For full control (as `AutoGluon` does), subclass `Backend` and implement `fit`, `predict`, and
  optionally `close()` for cleanup. Put effective settings you want recorded in `self.info`.
- Add a test in `tests/test_backends_run.py` and a row to the table above.

## Hosted services
A backend that sends data to a third-party API is **not implemented**. If you add one, make it
refuse non-public datasets by default, never log credentials, and record the service version in
`self.info`.
