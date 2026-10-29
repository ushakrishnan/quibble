---
title: Datasets and data locations
description: Dataset sources, loading and download timing, physical locations, and experiment evidence.
---

## Datasets and data locations

Quibble supports three scikit-learn built-in datasets, fourteen OpenML entries, four
bundled synthetic examples, and named local CSV/Parquet datasets. Public source entries are defined in
[the dataset loader](../quibble/datasets.py). Named local CSV/Parquet datasets are also
supported; see [Testing your own data](testing-your-data.md) for folder/schema requirements
and complete commands. Registration means a name is supported,
not that it has been downloaded, verified, or evaluated.

## What is stored where, and when

Paths below assume the local Windows `.venv`, default cache root `.cache`, and results
root `results`. Cache paths resolve from the project root; relative results paths resolve
from the working directory. Run commands from Quibble's root. Settings can override these
locations; see [configuration](configuration.md).

| Kind | Source and purpose | Default location | When created or loaded |
|---|---|---|---|
| Built-in source data | Official scikit-learn sample datasets used for offline comparisons | `.venv\Lib\site-packages\sklearn\datasets\data\` | Installed with scikit-learn; loaded into memory when selected |
| Bundled synthetic examples | Reviewed fictional grocery/retail clothing tables with recipes and hashes | `quibble\sample_data\<name>\` | Shipped with Quibble; existing rows loaded without provider calls or generation |
| Newly generated source table | Locally generated rows, manifest, and generation provenance | `data\<name>\` or `QUIBBLE_DATA_DIR\<name>\` | Written only after preview/review and explicit save; never overwrites an existing folder |
| Your source table | Named local CSV/Parquet with `dataset.json` | `data\<name>\` or `QUIBBLE_DATA_DIR\<name>\` | Placed by you; read when selected for verification, fitting, analysis, or reporting |
| OpenML source data | External feature tables, targets, and metadata identified by dataset ID | `.cache\openml\` | Fetched on first selection or verification; cached files are reused |
| Pretrained model weights | Existing model checkpoints, not the benchmark's input rows | `.cache\huggingface\hub\` or `.cache\tabpfn\`, depending on the library | Requested when a foundation model first loads; cached versions are reused where supported |
| Other runtime caches | Torch weights/compiler files, scikit-learn data, and library assets | Other subfolders of `.cache\` | Created or populated only as libraries need them; an empty folder is not proof of a download |
| Current benchmark evidence | Run metadata, test predictions, test labels/row positions, analysis | `results\<dataset>\` | Written as fits complete and analysis runs |
| Smoke evidence | Bounded setup-check results, separate from full benchmarks | `results\smoke\<unique-id>\<dataset>\` | Written by `benchmark --smoke` |
| Report | Generated interpretations, participation, device labels, and figures | `results\report\report.md` | Written by demo, benchmark, or report generation |
| Historical evidence | Preserved generated results and reports, not a second dataset/weight cache | `results\runs\<date>\<snapshot-id>\` | Before/after training and when reports are regenerated |
| Optional fitted models | Models fitted using this benchmark's training rows | `results\artifacts\<dataset>\<cell>\<run-id>\` | Only for supported backends when model retention is enabled |
| Verification metadata | Dataset dimensions and OpenML name checks, not training results | `datasets_meta.json` in the working directory | Written by `datasets --verify` |

Installing requirements installs software and bundled sample data. It does not automatically
fetch every OpenML table or every pretrained checkpoint. Downloaded weights are distinct
from fitted-model artifacts and from the input datasets. Package-manager caches (pip/uv)
are outside this runtime-cache policy.

## Built in (offline)

| Name | Task | Loader | Size | Installed source files |
|---|---|---|---|---|
| `breast_cancer` | binary classification | `load_breast_cancer` | 569 rows, 30 numeric features | `breast_cancer.csv` |
| `wine` | 3-class classification | `load_wine` | 178 rows, 13 numeric features | `wine_data.csv` |
| `diabetes_reg` | regression | `load_diabetes` | 442 rows, 10 numeric features | `diabetes_data_raw.csv.gz`, `diabetes_target.csv.gz` |

These are official scikit-learn datasets, not files manually added by Quibble. They need
no network after scikit-learn is installed and are used by the default demo and tests.
The benchmark presets default to `wine`; select other datasets with `--datasets`.
Wine contains chemical measurements and cultivar labels. It is not the separate OpenML
`wine_quality` entry, and `diabetes_reg` is not the OpenML classification entry `diabetes`.

For a Windows checkout, the installed data directory is:

```text
<checkout>\.venv\Lib\site-packages\sklearn\datasets\data\
```

For example, the original Wine table is `wine_data.csv` in that directory. Descriptions,
provenance, and citation information are in the adjacent `descr` directory:
`breast_cancer.rst`, `wine_data.rst`, and `diabetes.rst`.
Source filenames describe the installed scikit-learn layout, not a stable Quibble storage API.
Use the loaders rather than parsing the installed CSVs: headers and target layouts differ,
and `load_diabetes` applies its default feature scaling.

Locate the installed directories and inspect the complete Wine table without downloading:

```powershell
python -c "from pathlib import Path; import sklearn.datasets as d; p=Path(d.__file__).parent; print('Data:', p/'data'); print('Descriptions:', p/'descr')"
python -c "from sklearn.datasets import load_wine; d=load_wine(as_frame=True); print(d.frame.to_string(index=False))"
```

## Bundled synthetic examples

| Name | Task | Shape |
|---|---|---|
| `grocery_sample` | binary classification | 1,000 rows, 8 predictors plus target |
| `grocery_regression` | regression | 1,000 rows, 8 predictors plus target |
| `retail_clothing_binary` | binary classification | 1,000 rows, 8 predictors plus target |
| `retail_clothing_regression` | regression | 1,000 rows, 8 predictors plus target |

Each package folder contains `data.csv`, `dataset.json`, and `generation.json`.
These are already-created rows from Quibble's bounded generator, not live provider
output. CLI commands and the browser can reuse them offline. A local dataset folder
with the same name takes precedence; it is never overwritten or silently replaced by
the bundled copy. Changing a recipe requires a newly named, reviewed variant.
See the [workbench guide](experiment-workbench.md) for generation assumptions and limits.
The examples are fictional, not evidence of real grocery or clothing performance.

The browser uses one task-filtered selector with groups for built-in benchmarks,
bundled examples, and saved datasets. Classification includes binary and multiclass;
regression is separate. All entries are usable existing data, not readiness levels.

## OpenML registry
Defined as `name: (OpenML id, task, expected OpenML name)` in the loader.
The IDs are the requested source identifiers, not a claim that the entries were validated.

| Name | Id | Task | Expected OpenML name | Requested source |
|---|---|---|---|---|
| `credit-g` | 31 | classification | credit-g | [OpenML 31](https://www.openml.org/d/31) |
| `diabetes` | 37 | classification | diabetes | [OpenML 37](https://www.openml.org/d/37) |
| `adult` | 1590 | classification | adult | [OpenML 1590](https://www.openml.org/d/1590) |
| `bank-marketing` | 1461 | classification | bank-marketing | [OpenML 1461](https://www.openml.org/d/1461) |
| `phoneme` | 1489 | classification | phoneme | [OpenML 1489](https://www.openml.org/d/1489) |
| `blood-transfusion` | 1464 | classification | blood-transfusion-service-center | [OpenML 1464](https://www.openml.org/d/1464) |
| `kc1` | 1067 | classification | kc1 | [OpenML 1067](https://www.openml.org/d/1067) |
| `vehicle` | 54 | classification | vehicle | [OpenML 54](https://www.openml.org/d/54) |
| `abalone` | 183 | regression | abalone | [OpenML 183](https://www.openml.org/d/183) |
| `cpu_act` | 197 | regression | cpu_act | [OpenML 197](https://www.openml.org/d/197) |
| `wine_quality` | 287 | regression | wine_quality | [OpenML 287](https://www.openml.org/d/287) |
| `house_sales` | 42092 | regression | house_sales | [OpenML 42092](https://www.openml.org/d/42092) |
| `diamonds` | 42225 | regression | diamonds | [OpenML 42225](https://www.openml.org/d/42225) |
| `bike_sharing` | 42712 | regression | Bike_Sharing_Demand | [OpenML 42712](https://www.openml.org/d/42712) |

> [!WARNING]
> The registry has not been comprehensively validated against current OpenML metadata.
> Run `quibble datasets --verify --names ...` before using an OpenML entry. `NAME MISMATCH`
> means the returned name differs from the registry; `ERROR` means loading failed.
> Inspect the output as well as the exit code: verification reports failures per dataset
> and returns 1 for any load failure or name mismatch.

Datasets are downloaded on first use with scikit-learn's `fetch_openml` and cached in
`QUIBBLE_DATA_HOME`. When blank, that follows `QUIBBLE_CACHE_DIR/openml`, normally
`.cache\openml`. The current `.env` may explicitly set the same location.
Scikit-learn manages nested metadata and data files; do not assume there will be one
friendly `<name>.csv` per dataset. Quibble does not redistribute these source tables.
Only selected OpenML entries are loaded; registering fourteen entries does not download all fourteen.

## Actual download sources, destinations, and triggers

The following paths are relative to the source checkout's root for the default local
setup. Cache overrides in `.env` change the destinations, not the upstream model identity.

### Bundled datasets: installed with scikit-learn

| Dataset | Upstream source documentation | Local destination | When obtained |
|---|---|---|---|
| `breast_cancer` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | `.venv\Lib\site-packages\sklearn\datasets\data\breast_cancer.csv` | Installing scikit-learn via the requirements; no dataset download at run time |
| `wine` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) | `.venv\Lib\site-packages\sklearn\datasets\data\wine_data.csv` | Installing scikit-learn via the requirements; no dataset download at run time |
| `diabetes_reg` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | `.venv\Lib\site-packages\sklearn\datasets\data\diabetes_data_raw.csv.gz` and `diabetes_target.csv.gz` | Installing scikit-learn via the requirements; no dataset download at run time |

These pages identify the original dataset and its references. Quibble obtains the bundled
copy from the installed scikit-learn package, not by visiting those pages during a run.
The installer gets that package from your configured package index, normally
[PyPI's scikit-learn distribution](https://pypi.org/project/scikit-learn/), or your trusted mirror.

### OpenML tables: downloaded when a selected entry is needed

Every OpenML entry above has its own source link. Quibble calls `fetch_openml(data_id=...)`;
scikit-learn requests metadata from OpenML, then follows the returned data-file URL.
For example, `adult` requests [dataset 1590](https://www.openml.org/d/1590), whose metadata
is available at [the OpenML API](https://www.openml.org/api/v1/json/data/1590).
The metadata's file URL determines the actual download; it is not a Hugging Face model.

All fourteen entries use `.cache\openml\` by default, with scikit-learn-managed subdirectories.
The trigger is the first run, smoke run, verification, or analysis/report that needs that
entry and lacks the cached data. A later request reuses available cached files.
`datasets --verify` without `--names` requests all registered entries, so it can download
all fourteen, plus verify discovered local datasets. Source links identify requested IDs; verification is still required to
confirm they match the intended dataset names.

### Foundation checkpoints: downloaded when the selected model loads

Sources below were checked against the installed `tabfm 1.0.1`,
`structured-data-models 0.2.1`, and `tabpfn 9.2.0` loaders. Package upgrades, task selection,
or upstream version settings can change checkpoint filenames or revisions.

| Backend | Actual checkpoint repository | Default local destination | When downloaded |
|---|---|---|---|
| TabFM | [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch/tree/main) | `.cache\huggingface\hub\models--google--tabfm-1.0.0-pytorch\` | First selected TabFM fit loads the requested classification/regression model and needs uncached files |
| KumoTabular | [nvidia/Kumo-Tabular, revision v1.0.1](https://huggingface.co/nvidia/Kumo-Tabular/tree/v1.0.1) | `.cache\huggingface\hub\models--nvidia--Kumo-Tabular\` | First selected Kumo fit for an uncached task/size/revision |
| TabPFN | [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/main) for the installed default model version | `.cache\tabpfn\` for checkpoint files and loader-managed download metadata | First selected TabPFN fit requiring an uncached checkpoint; access/authentication conditions may apply |

For Kumo's default `small` size, the installed loader requests
[`small/classifier.pt`](https://huggingface.co/nvidia/Kumo-Tabular/blob/v1.0.1/small/classifier.pt)
or [`small/regressor.pt`](https://huggingface.co/nvidia/Kumo-Tabular/blob/v1.0.1/small/regressor.pt),
plus `config.json`. Changing size selects the corresponding `medium/` or `large/` files.
TabFM selects a task-specific subfolder from its repository; Quibble uses its PyTorch
loader, not Google's separate JAX model repository.

The installed TabPFN default is version 3.5, with default checkpoint
`tabpfn-v3.5-20260909.safetensors`; multiclass handling may select a variant.
TabPFN supports other model versions and may use different repositories when configured.
Its older v2 loader also contains a legacy Google Cloud Storage fallback at
<https://storage.googleapis.com/tabpfn-v2-model-files/05152025>;
that is not the documented default v3.5 source.

Hugging Face model directories contain `blobs`, `refs`, and `snapshots\<revision>` rather
than a single flat weights file. The hub resolves model-page URLs to download endpoints
and may use redirects or content-addressed transfer services. The repository links above
identify the upstream model; they are not a promise of one permanent CDN URL.
Cached weights can be reused, but the library may still check remote metadata unless
configured for offline use. A newly selected task, size, revision, or missing file can
require another download. Availability, model licences, and authentication are controlled
by the upstream providers.

HistGB, LightGBM, CatBoost, and Weak learn local models and do not request these foundation
checkpoints. AutoGluon trains local model families; if a chosen preset/model family uses
pretrained dependencies, its own library can request additional assets. Quibble does not
maintain a fixed exhaustive checkpoint URL list for every AutoGluon preset.

### What a first invocation actually downloads

```powershell
# Only uses bundled datasets and tree models; no OpenML tables or foundation weights.
python -m quibble demo

# Loads bundled Wine; downloads uncached TabFM/Kumo weights, then writes isolated smoke evidence.
python -m quibble benchmark --preset foundations --smoke

# Requests only these OpenML tables; does not train models or load foundation checkpoints.
python -m quibble datasets --verify --names credit-g,adult
```

In the second example, checkpoints stay in `.cache`, while metrics/predictions/report go to
`results\smoke\<unique-id>\`. Full benchmark results go to `results\<dataset>\`.
Existing successful or failed run records may cause fits to be skipped; `--force` retries
the fit but does not inherently force a new weight download.

To see effective destinations and files already present:

```powershell
python -c "from quibble.cache import prepare_caches; print('\n'.join(str(k)+': '+str(v) for k,v in prepare_caches().items()))"
Get-ChildItem .\.cache\huggingface\hub -Directory -Filter 'models--*'
Get-ChildItem .\.cache\tabpfn -Recurse -File
Get-ChildItem .\.cache\openml -Recurse -File
```

The listing examples assume default cache paths; use the printed paths for overrides.
Empty cache directories do not establish that any model or dataset was downloaded.

## How and when a command uses data

1. A run selects a registered dataset and loads its features and target. Built-ins come
   from the installed package; OpenML may need network access if cache files are absent.
2. Holdout uses approximately 70% training and 30% test rows. `KxR` uses K-fold splitting
   repeated R times. Classification is stratified when class counts permit.
3. Each backend receives the same split's training and test rows. Model seeds vary model
   randomness; CV seed and split ID determine row assignments.
4. Ordinary runs keep the full split. Smoke checks select at most 128 training and 32 test
   rows after splitting, deterministically, preserving classification labels where possible.
   The same sampled rows are used across backends. These are setup checks, not full-data evidence.
5. Models fit or condition on training rows and predict held-out test rows. Tree models
   learn trees; foundation models load pretrained weights and use the labelled training context.
   Quibble does not pretrain those foundation checkpoints from scratch.
6. Results save predictions, true test labels, and original test-row positions. The complete
   feature table and a separate training-feature CSV are not copied into each result directory.
7. Analysis and reporting read saved predictions, but also reload the original dataset
   for row traits and slices. They do not retrain models; an absent OpenML cache can still
   require network access during report generation.

Saved run records, including errors, are reused unless `--force` is supplied. Download caches
and completed-run caches are separate: reusing downloaded weights does not skip a new fit,
and skipping a completed fit does not prove its original source data is still available.
`--force` reruns selected cells after archiving existing results; it does not mean every
source dataset or model weight is downloaded again.

## Registered, downloaded, verified, and actually evaluated

These are separate states:

| State | Evidence |
|---|---|
| Registered | Name appears in the built-in/OpenML inventory above |
| Available locally | Built-in package data exists, or OpenML cache contains the requested source |
| Verified | A verification result reports successful loading and, for OpenML, the expected name |
| Attempted | A run JSON exists; it can still contain `error` or `refused` |
| Completed | The run has no error/refusal and usable prediction, target, and index arrays |
| Used in a comparison | The report participation table shows aligned fits used for that dataset |

This guide is the supported inventory, not a live list of successful experiments. Check
the current report's participation table and each record's `run_type`, row counts,
timestamp, settings, and device fields to establish what was actually evaluated.
`invocation.json` describes the latest sweep request, not the entire project's run history.
Use dated snapshots for older evidence. Smoke success does not establish full-benchmark success.

List current Wine records and their recorded outcomes from the repository root:

```powershell
Get-ChildItem .\results\wine\*_ss*_ms*.json | ForEach-Object {
    $r = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
    [pscustomobject]@{
        Backend = $r.backend
        Split = $r.split_seed
        Type = $r.run_type
        TrainRows = $r.n_train
        TestRows = $r.n_test
        Error = $r.error
        Refused = $r.refused
    }
}
```

For a smoke experiment, inspect the corresponding directory under `results\smoke\<id>\wine`
instead. Missing folders are not evidence of failure unless that dataset was actually requested.

## Verification

```powershell
python -m quibble datasets --verify --names breast_cancer,wine,diabetes_reg  # offline
python -m quibble datasets --verify --names credit-g,adult                 # may download
python -m quibble datasets --verify                                       # all registered; may download all OpenML entries
```
For each dataset it prints and saves (to `datasets_meta.json` in the current directory):

| Field | Meaning |
|---|---|
| `rows`, `features`, `task` | size and task |
| `cat_share` | share of feature columns that are categorical (object or category dtype) |
| `missing_share` | share of cells that are missing |
| `tier` | `small` (< 5,000 rows), `medium` (< 50,000), `large` |
| `classes`, `majority_share` | classification only: number of classes and the largest class's share |
| `openml_name`, `name_ok` | the name OpenML reports and whether it matches the registry |

Use the metadata to describe your benchmark (and to make sure it spans sizes, categorical share,
missingness and imbalance).

## Reading the saved Wine predictions

Use the row-index array to connect predictions back to the original source features:

```python
from pathlib import Path
import numpy as np
from quibble.datasets import load

directory = Path("results") / "wine"  # Or the wine directory inside one smoke run.
X, y, task = load("wine")
idx = np.load(directory / "idx_test_ss0.npy")
truth = np.load(directory / "y_test_ss0.npy")
probabilities = np.load(directory / "histgb_ss0_ms0.pred.npy")
rows = X.iloc[idx].copy()
rows["target"] = truth
rows["predicted_class"] = probabilities.argmax(axis=1)
print(rows)
```

Check the matching run JSON for errors before reading an array: an unsuccessful forced retry
can leave an older prediction file on disk. The analysis loader ignores failed records.
For classification, each probability column corresponds to the encoded class label.
For regression, the prediction array is a one-dimensional vector.
See [results format](results-format.md) for all array and JSON fields, and
[reports](reports.md) for interpretation and model-artifact locations.

## Loading rules
- Classification targets are label-encoded (sorted order of the original labels) to `0..k-1`.
- Regression targets are converted to floats.
- Features are passed through untouched except for the index reset; categorical columns keep their
  object/category dtype and are handled by each backend's preprocessing rule.
- Row identity is the 0-based position in the loaded table; that is what the `row` column in
  disagreement CSVs refers to.

## Choosing datasets for a benchmark
Aim for variety rather than convenience: some small (under 5,000 rows), some medium, at least one
imbalanced, one with many missing values, one categorical-heavy, and two or three regression sets.
Beware of datasets that every model solves perfectly: they hide differences.

## Adding a dataset
For your own file, create `data\<name>\dataset.json` plus CSV/Parquet as described in
[Testing your own data](testing-your-data.md); no code changes are required.
For another public OpenML entry, add its id, task, and exact expected name to `OPENML`,
then run `quibble datasets --verify --names yourname`.

## Licences
Each OpenML dataset has its own licence and citation requirements. Check them before publishing
results derived from a dataset, and cite sources in your write-up.
