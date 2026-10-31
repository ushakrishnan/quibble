---
title: Backends
description: Supported model libraries, optional dependency extras, and backend behavior.
---

## Backends

Think of backends as different ways of studying for the same exam. Quibble
gives them the same training examples and compares their answers to the same
unseen questions. An installed method is not necessarily a verified method;
the table distinguishes working checks from unknown coverage and usage rights.

A backend is a small class in `quibble_ml/backends.py` with two methods: `fit(Xtr, ytr, task, seed, cfg,
device)` and `predict(X)`. `predict` returns class probabilities `(n, k)` for classification and a
vector `(n,)` for regression. The table below lists registered backends. `--backends all`
selects all public backends for a sweep; it is not a list-only command and excludes `weak`.
When CUDA is selected or Auto resolves to CUDA, incompatible methods are visibly
excluded. CPU runs keep all selected methods. No algorithm is silently substituted.

| Name | Library | Tasks | GPU | Native categoricals | Tested | Code / weight terms |
|---|---|---|---|---|---|---|
| `histgb` | scikit-learn HistGradientBoosting | both | no | yes | yes (CPU) | BSD-3-Clause / no pretrained weights |
| `lightgbm` | LightGBM | both | no (CPU adapter) | yes | yes (CPU) | MIT / no pretrained weights |
| `catboost` | CatBoost | both | yes | yes | CPU both tasks; local CUDA classification smoke | Apache-2.0 / no pretrained weights |
| `tabpfn` | TabPFN | both | yes | no (ordinal) | real offline Wine classification and diabetes regression on CPU/CUDA with 9.2.0/v3.5; broader coverage unverified | Apache-2.0 / reference TABPFN-3.5 License v1.0 |
| `tabicl_v2` | TabICL 2.2.0 | both | CPU and CUDA holdouts | no (ordinal) | real Wine classification and diabetes regression on both devices; broader coverage unverified | BSD-3-Clause components; Apache-2.0 forecast code unused / BSD-3-Clause weights |
| `mitra_v2` | pinned AutoGluon / mitra-finetune | both, different recipes | CUDA only | no (ordinal) | real Wine zero-shot classification and diabetes distributional regression; broader coverage unverified | Apache-2.0 components / pinned model cards label Apache-2.0 |
| `tabfm` | Google Research TabFM (PyTorch) | both | yes | yes | CPU both tasks; local CUDA Wine classification fits | Apache-2.0 / reference TabFM Non-Commercial v1.0 |
| `kumo_tabular` | NVIDIA structured-data-models | both | yes | yes | CPU both tasks; local CUDA Wine classification smoke fits | Apache-2.0 / reference OpenMDW-1.1 |
| `autogluon` | AutoGluon Tabular | both | excluded from GPU-only runs | own preprocessing | CPU, one dataset per task | Apache-2.0 / selected model-family terms unknown |
| `dummy` | scikit-learn Dummy (class priors / training mean) | both | no | not used | yes (baseline) | BSD-3-Clause / no pretrained weights |
| `linear` | scikit-learn logistic regression (classification) or ridge (regression), one-hot categoricals | both | no | one-hot | yes (baseline) | BSD-3-Clause / no pretrained weights |
| `weak` | HistGB, handicapped | both | no | yes | yes (internal) | BSD-3-Clause / no pretrained weights |

Adapters use their documented recipes. Default campaigns preserve previous
behavior; an explicit tuned tier adds training-only bounded search for the three
tree methods. A default recipe does not mean every foundation method is zero-shot.
Use the [locked local setup](getting-started.md#install). Core dependencies include
the tree methods and baselines. Select foundation-model libraries explicitly with
`--extra tabpfn`, `--extra tabicl`, `--extra mitra-v2`, `--extra tabfm`, `--extra kumo`, or `--extra autogluon` on the same
`uv sync --locked --extra web --extra data` command. Keep those flags on later syncs.
This installs libraries, not pretrained weights. TabFM/Kumo require Python 3.11+;
AutoGluon requires Python below 3.14.

### Code licenses and model terms

Reference terms were reviewed on 2026-10-30. Code licenses do not grant rights
to every checkpoint, dataset, or transitive component. TabFM's
[reference weight terms](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)
restrict non-commercial/non-production use. The
[TabPFN-3.5 reference terms](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)
also impose non-commercial/non-production and professional-use eligibility
conditions. Read the full terms before using either model.
[Kumo-Tabular's reference terms](https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE)
are separate from its Apache-licensed library.

TabICLv2 pins its two expected artifacts at an immutable Hub revision and checks
SHA-256 before loading. Its primary
[code licence](https://github.com/soda-inria/tabicl/blob/main/LICENSE) applies
BSD-3-Clause to the adapter components and Apache-2.0 to the unused forecast
directory; the [pinned model card](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/README.md)
labels the weights BSD-3-Clause.

Except for the separately pinned TabICLv2 and Mitra-v2 adapters, foundation
adapters still let their libraries choose weight revisions. Quibble does not
claim the referenced repository's latest revision is the one actually loaded:
runtime revisions and checksums remain explicitly unknown. Library installation
does not download all weights or approve their use. Access/gating can change.
This is provenance, not legal advice or a commercial-use guarantee.

Doctor and method help use the shared registry in
[`provenance.py`](../quibble_ml/provenance.py). New run records, including errors
and refusals, snapshot the same metadata. Old evidence remains labeled unknown
rather than inheriting today's terms. See [recipe identity](results-format.md#recipe-identity).

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

## `mitra_v2`

An experimental, opt-in CUDA adapter for Windows/Linux with Python 3.11-3.13.
Install `--extra mitra-v2`; Windows CUDA also needs `--extra windows-cuda`.
It verifies AutoGluon Tabular 1.6.3, TabArena 0.1.0, mitra-finetune 0.3.0,
Hub 1.33.0, Transformers 5.22.1 and Torch 2.11.0. An unverifiable package
version is a refusal, not proof of compatibility. CPU, MPS and macOS are not
supported by this adapter.

The code source is pinned to commit
`b4701e8148dc33b00ed15d7086ff59816957cde4` of
[mitra-finetune](https://huggingface.co/autogluon/mitra-finetune).
Both configuration and safetensors hashes are verified before loading.
The [artifact registry](../quibble_ml/mitra.py) records expected hashes:

| Task | Model repository | Immutable revision | Recipe |
|---|---|---|---|
| Classification | `autogluon/mitra-classifier-2` | `edada0d20759c58ada8c8605c25f22f6e98ea5f0` | One estimator, zero-shot; no fine-tuning |
| Regression | `autogluon/mitra-regressor-2` | `246c7d310fff04296a2388df5f448e3ffe809283` | Distributional-head mean, two bags, baseline adaptation up to 50 steps per bag |

These are different task APIs, not a scalar regressor substituted for the
distributional checkpoint. The two-bag regression recipe is not the published
eight-bag benchmark. Upstream defaults and incomplete seed control remain
explicit; `identity_complete` stays false. `MITRA_*` runtime overrides refuse
rather than silently changing the recipe. Conservative harness limits are
512 training rows, 32 predictors and 2-10 classes, not verified model maxima.

Regression fitting happens inside `predict`. Quibble performs one batch call,
records inclusive `adaptation_and_prediction_s` and `model_pipeline_s`, and
does not repeat expensive fitting for latency probes. Pure weight-loading,
adaptation and inference timings, completed adaptation steps, and single-row
latency remain unavailable when not exposed. An unavailable latency budget is
INCONCLUSIVE in swap, never a free pass. Temporary AutoGluon models stay inside
a parent-owned worker workspace, removed after completion or worker timeout;
the foundation estimator is not retained with `--keep-models`.

Missing weights refuse by default. After reviewing the pinned model cards, an
initial download requires `QUIBBLE_MITRA_ALLOW_DOWNLOAD=1`. This permits a
download, not legal approval; both cards label Apache-2.0, but neither model
repository supplies a separate licence file at the inspected revision.
Installation alone neither downloads weights nor accepts terms.

Real offline CUDA holdouts checked all three Wine probability columns (124
training, 54 test rows; accuracy 1.0) and 133 diabetes regression predictions
(309 training rows; RMSE approximately 54.8-55.0 across two executions).
TabICLv2 ran on the same held-out rows under the combined dependency selection.
Both pinned tasks produced finite saved predictions and reports. The observed
regression variation reinforces the incomplete-seed-control warning. These
are bounded execution checks, not a general ranking, capacity study, complete
reproducibility guarantee or commercial deployment approval.

## `tabicl_v2`

An experimental, separately named TabICLv2 adapter using `TabICLClassifier` or
`TabICLRegressor` from `tabicl==2.2.0` and Torch 2.11.0. It is not added to the
default foundations or tree presets. `all` still means all registered backends.

The adapter pins revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
of `jingang/TabICL`, selecting `tabicl-classifier-v2-20260212.ckpt` or
`tabicl-regressor-v2-20260212.ckpt`. Expected SHA-256 values are defined in
[the artifact registry](../quibble_ml/tabicl.py). Both bytes and the task-specific
filename are checked; a failed check never falls back to another generation.

Install with `--extra tabicl`. By default, missing cached weights refuse before
inference. After reviewing terms, an explicit initial-download workflow is:

```powershell
$env:QUIBBLE_TABICL_ALLOW_DOWNLOAD = "1"
python -m quibble_ml benchmark --backends tabicl_v2 --datasets grocery_sample --device cpu --cv holdout --model-seeds 0 --out results_tabicl
Remove-Item Env:\QUIBBLE_TABICL_ALLOW_DOWNLOAD
```

That command requests real execution on your machine, not a preapproved deployment.
Local validation used both immutable, SHA-256-checked checkpoints, then disabled
network access and downloads for real holdouts on CPU and CUDA. Wine used
124 training and 54 test rows, with all three probability columns checked;
diabetes regression used 309 training and 133 test rows. Both devices produced
finite, correctly shaped saved predictions and reports. The CPU campaign also
compared HistGB on the same held-out rows. Wine accuracy was 1.0; regression
RMSE was approximately 55.43, versus HistGB's 60.71 in that single split.
These small holdouts are execution evidence, not a general quality or safety
ranking. Broader datasets, capacities and other accelerators remain unverified;
doctor retains the experimental label. Defaults still prohibit weight downloads.

The fixed recipe uses eight ensemble members, batch size eight, no KV caching,
no offload, no FlashAttention-3 and no fine-tuning. The effective device is
explicitly `cpu` or `cuda`; ordinal categorical preparation is learned only on
training data. Missing/unseen categories remain missing. Regression requests the
predictive mean. Probabilities are reordered to training-class order, and malformed
or nonfinite predictions fail explicitly. Policy limits are 1,000 CPU or 10,000
CUDA training rows and 500 predictors, not validated upstream capacity claims.

Checkpoint lookup/hash checking and weight deserialization count as loading.
The pinned package loads inside `fit`, so a version-checked `_load_model` timing
hook separates that cost from preparation; `fit_total_s` retains the total.
Batch and single-row prediction timings include repeated in-context inference.
Fitted estimators are not serialized; closing releases the owned model.

## `tabpfn`
`TabPFNClassifier` / `Regressor` (a pretrained tabular foundation model) with `device` and
`random_state=seed`. quibble **refuses** (records a refusal, not an error) when:
- training rows exceed the limit: `QUIBBLE_TABPFN_MAX_ROWS`, default 10,000 on `cuda` and 1,000 on
  `cpu` (Quibble's conservative policy limits, not verified maximum capabilities of the installed TabPFN release),
- there are more than 500 features,
- a classification target has more than 10 classes.

Notes:
- Install TabPFN with `--extra tabpfn`. Model weights are downloaded on first use and may need
  network access and a Hugging Face account or token, depending on the TabPFN release.
- It always receives ordinal-encoded categoricals.
- TabPFN predicts by conditioning on the training set, so single-row latency is expected to be much
  higher than for tree models. That is a genuine cost of the approach and the benchmark will show it.
- Real offline holdouts passed on Windows/Python 3.13.4 with TabPFN 9.2.0,
  Torch 2.11.0+cu128 and the default v3.5 checkpoint. Wine used 124 training
  and 54 test rows, with accuracy 1.0 on CPU and CUDA. Diabetes used 309
  training and 133 test rows: RMSE 55.6766 on CPU and 55.7389 on CUDA,
  versus 60.7144 for CPU HistGB on the same rows. These are small holdouts,
  not broad rankings, maximum-capacity tests or replacement approval.
- The provider's official licence-acceptance check succeeded. No TabPFN weights
  were downloaded for these runs; the existing 876,027,932-byte
  `tabpfn-v3.5-20260909.safetensors` matched SHA-256
  `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`
  before and after fitting. Content matches Hub revision
  `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`; this separate validation
  receipt does not fill unknown checkpoint fields in historical/general records.
- `load_s` measures constructor work, not an independently isolated weight-load
  stage. `fit_total_s` includes constructor and fitting. The live validation's
  three warm single-row probes do not establish a stable production p95 or
  cold-start/service latency.
- An empty-cache test with socket connections denied produced an error record
  and no predictions. The upstream loader attempted licence-server connections
  and reported an authentication error; offline mode does not make that message
  proof of missing licence acceptance. Do not bypass the provider's check.

### Native Kumo local probe

The standalone SDM API also exposes `sdm.models.KumoRelational`. A bounded
Windows/Python 3.13.4 probe with `structured-data-models==0.2.1` and the
`v1.0.1` checkpoint tag passed classification and regression on CPU and CUDA.
It supplied task rows, users and orders as related tables, not one flattened
feature table. It used 32 context entities, 16 held-out query entities, one
estimator and one graph hop; 48 future/unavailable child rows were excluded
before either native inference or the descriptive HistGB reference.

Two public checkpoints totaling 243,649,432 bytes were SHA-256 checked at
immutable revision `2bd603d3d8f25f67a7aaa8579908595e20567e22`, with
OpenMDW-1.1 and BSD-3-Clause third-party notices. Inference reused them offline
without a service endpoint or third-party dataset transfer. CUDA model execution
used an upstream CPU join fallback because cuDF was absent; the warning remains
visible. This is not an all-GPU graph-building claim.

This standalone probe is not a registered Quibble backend, reviewed-manifest
worker integration or browser comparison. Its tiny synthetic scores are not
evidence of a generally better model. The SDK/NIM route still needs a running
deployment; it is not required by the direct local API.

## `tabfm`

The `--extra tabfm` selection installs `tabfm[pytorch]` on Python 3.11+, not the JAX backend.
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

Select `--extra kumo` on Python 3.11+. The distribution is
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

Example comparison after syncing with `--extra tabfm --extra kumo`:

```powershell
quibble-ml run --sweep --backends lightgbm,tabfm,kumo_tabular --datasets breast_cancer --cv 3x2 --model-seeds 0 --device cpu --jobs 1 --out results_cpu
quibble-ml report --out results_cpu --swap lightgbm:tabfm --swap lightgbm:kumo_tabular
```

Changing the TabFM ensemble count or Kumo size/ensemble count requires a different
`--out`. Recipe mismatches are rejected, including when `--force` is supplied.
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
- AutoGluon and its model families constrain their dependencies. Sync selected extras
  from the lock together, then run `uv pip check` and `python -m quibble_ml doctor`.
- GPU-only runs exclude AutoGluon because a GPU budget does not prevent CPU model families.
- It is tested on CPU on one classification and one regression dataset.

## `dummy` and `linear` (baselines)
Reference models for `quibble-ml gate` and for judging whether a complex model is worth using.
`dummy` predicts class priors or the training mean. `linear` is logistic regression (classification,
`max_iter=2000`) or ridge regression (regression, `alpha=1.0`) with median-imputed, scaled numeric
columns and one-hot categoricals (unseen categories are ignored). Both are deterministic and run on
CPU. They are included in `--backends all`.

## Verification labels
`quibble-ml doctor` prints what has actually been run for each backend, and each run record stores the
label as `backend_verification`. Treat `experimental` as "the adapter exists", not "it has produced a
verified result".

## `weak` (internal)
HistGB limited to `max_iter=2, learning_rate=0.05`. It exists so that a swap to a clearly worse model
must fail; the test suite and the demo rely on that. It is hidden from `--backends all`.

## Adding a backend
Add a class to `quibble_ml/backends.py` and include it in the class tuple used to construct
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
