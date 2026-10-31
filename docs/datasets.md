---
title: Datasets and data locations
description: Dataset sources, loading and download timing, physical locations, and experiment evidence.
---

## Datasets and data locations

Quibble supports three scikit-learn built-in datasets, fourteen OpenML entries, four
bundled synthetic examples, and named local CSV/Parquet datasets. Public source entries are defined in
[the dataset loader](../quibble_ml/datasets.py). Named local CSV/Parquet datasets are also
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
| Bundled synthetic examples | Reviewed fictional grocery/retail clothing tables with recipes and hashes | `quibble_ml\sample_data\<name>\` | Shipped with Quibble; existing rows loaded without provider calls or generation |
| Newly generated source table | Locally generated rows, manifest, and generation provenance | `data\<name>\` or `QUIBBLE_DATA_DIR\<name>\` | Written only after preview/review and explicit save; never overwrites an existing folder |
| Your source table | Named local CSV/Parquet with `dataset.json` | `data\<name>\` or `QUIBBLE_DATA_DIR\<name>\` | Placed by you; read when selected for verification, fitting, analysis, or reporting |
| OpenML source data | External feature tables, targets, and metadata identified by dataset ID | `.cache\openml\` | Fetched on first selection or verification; cached files are reused |
| Pretrained model weights | Existing model checkpoints, not the benchmark's input rows | `.cache\huggingface\hub\` or `.cache\tabpfn\`, depending on the library | Library-specific initial downloads and cache reuse; TabICLv2 defaults to cached-only |
| Other runtime caches | Torch weights/compiler files, scikit-learn data, and library assets | Other subfolders of `.cache\` | Created or populated only as libraries need them; an empty folder is not proof of a download |
| Current benchmark evidence | Run metadata, test predictions, test labels/row positions, analysis | `results\<dataset>\` | Written as fits complete and analysis runs |
| Smoke evidence | Bounded setup-check results, separate from full benchmarks | `results\smoke\<unique-id>\<dataset>\` | Written by `benchmark --smoke` |
| Report | Generated interpretations, participation, device labels, and figures | `results\report\report.md` | Written by demo, benchmark, or report generation |
| Historical evidence | Preserved generated results and reports, not a second dataset/weight cache | `results\runs\<date>\<snapshot-id>\` | Before/after training and when reports are regenerated |
| Optional fitted models | Models fitted using this benchmark's training rows | `results\artifacts\<dataset>\<cell>\<run-id>\` | Only for supported backends when model retention is enabled |
| Verification metadata | Dataset dimensions and OpenML name checks, not training results | `datasets_meta.json` in the working directory | Written by `datasets --verify` |

Syncing the selected extras installs software and bundled sample data. It does not automatically
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

All fourteen entries passed real loading on 2026-10-30, not just metadata lookup.
Ordinary loading now checks the returned ID/name, one declared target, row
alignment, labels and finite numeric regression targets. Verification uses that
same loader and returns 1 for any failed entry, including an ID/name mismatch.
Rerun `quibble-ml datasets --verify --names ...` in your own environment; the
snapshot below does not guarantee future network availability.

Datasets are downloaded on first use with scikit-learn's `fetch_openml` and cached in
`QUIBBLE_DATA_HOME`. When blank, that follows `QUIBBLE_CACHE_DIR/openml`, normally
`.cache\openml`. The current `.env` may explicitly set the same location.
Scikit-learn manages nested metadata and data files; do not assume there will be one
friendly `<name>.csv` per dataset. Quibble does not redistribute these source tables.
Only selected OpenML entries are loaded; registering fourteen entries does not download all fourteen.

### Verification on 2026-10-30

The following actual loads passed on Windows/Python 3.13.4 with scikit-learn
1.9.1 and pandas 2.3.3. All returned names matched and all upstream statuses were `active`.
Shape excludes the target; missing share measures predictor cells, not sentinel
values that a source uses instead of missing data.

| Name | Version / file ID | Rows x predictors | Actual target | Missing share |
|---|---|---|---|---|
| `credit-g` | 1 / 31 | 1,000 x 20 | `class` | 0.0000 |
| `diabetes` | 1 / 37 | 768 x 8 | `class` | 0.0000 |
| `adult` | 2 / 1595261 | 48,842 x 14 | `class` | 0.0095 |
| `bank-marketing` | 1 / 1586218 | 45,211 x 16 | `Class` | 0.0000 |
| `phoneme` | 1 / 1592281 | 5,404 x 5 | `Class` | 0.0000 |
| `blood-transfusion` | 1 / 1586225 | 748 x 4 | `Class` | 0.0000 |
| `kc1` | 1 / 53950 | 2,109 x 21 | `defects` | 0.0000 |
| `vehicle` | 1 / 54 | 846 x 18 | `Class` | 0.0000 |
| `abalone` | 1 / 3620 | 4,177 x 8 | `Class_number_of_rings` | 0.0000 |
| `cpu_act` | 1 / 3634 | 8,192 x 21 | `usr` | 0.0000 |
| `wine_quality` | 1 / 49817 | 6,497 x 11 | `quality` | 0.0000 |
| `house_sales` | 2 / 21578898 | 21,613 x 19 | `price` | 0.0000 |
| `diamonds` | 1 / 21792853 | 53,940 x 9 | `price` | 0.0000 |
| `bike_sharing` | 2 / 22044627 | 17,379 x 12 | `count` | 0.0000 |

All fourteen reloaded with network connections disabled, identical fingerprints,
and unchanged cached files. An empty cache failed explicitly with networking
disabled. The isolated cache used 3,710,183 bytes against a 512 MiB validation
budget. This budget describes this check, not a configurable production quota.

Full CPU holdout comparisons ran HistGB and Weak on `credit-g` and `abalone`,
including worker subprocesses, predictions, analysis, source guards and a report.
These four fits do not establish model coverage on the other twelve datasets,
other backends, other platforms, or production suitability. Routine tests mock
network loading; live checks are manual and opt-in.

### Source hashes and reuse

New records retain dataset ID, version, file ID, upstream ARFF MD5, target,
source URLs, licence/citation fields and a loaded-content SHA-256. The latter uses
`pandas-csv17-nullmask-v1`: ordered schema/dtypes/categories, then 10,000-row
blocks of missing-value masks and UTF-8 CSV with 17-digit numeric formatting.
It is not the SHA-256 of an ARFF, Parquet or compressed download. Its value may
change with a parser or dtype change even when upstream bytes are unchanged.

| Name | Upstream ARFF MD5 | Loaded-content SHA-256 |
|---|---|---|
| `credit-g` | `9a475053fed0c26ee95cd4525e50074c` | `6a68a40f8cdada1b6e2af5fb9b6c2ffd3a59e2438fc4635e36eea9bcd75e53ac` |
| `diabetes` | `3cbaa3e54586aa88cf6aacb4033e4470` | `c4fa5c538959947bf146197f27a5db796d89c75b13ee8dda5a4f879c80c3c39d` |
| `adult` | `bb6510925e5d4b23d136715febb2cdf5` | `3627ae281ea5d9ff5175bb9cc5e0193ef64c3c7ee212514081997526739d6873` |
| `bank-marketing` | `b29d2eb78e508569fd41172c97bda00e` | `9cbfeba0c87d9dfacd176a08d3f58fb55a552477955827808f25ae2a0317d033` |
| `phoneme` | `902ff27649dbfb512f3fb01fc05c6b24` | `8d1b23cf92f82627c630da1cb30920d582700cc05d14a8507e05b3754e5ab310` |
| `blood-transfusion` | `c3242468edab8c2e7a907674122dc851` | `82c5190900cec80194bd466c7dd24976de07afe3cae377eafd1741a3ada510af` |
| `kc1` | `bcc004464934075ecaf54d84e45c427a` | `b6c7314c2585c59aae1837320fb4b8ec7a32019ed941cc91876901b651607da5` |
| `vehicle` | `fbba18157b188f309d772f9ca4e578f5` | `6221e4d2110f49b6244882d1015f554cd19bb80498b22d4a9e04c577b687089a` |
| `abalone` | `a9555b1cfe39eb55df2c520f09172a96` | `dc5eedefb11b326f6179493973e689d1d318b12611dec53bb3117e6159aee2fa` |
| `cpu_act` | `c48cbb08c17633f2db3a78633301c81b` | `ae63ca08ed3b354817da95db288d663e511ac6acd345d7cf0a6d8ef7b68e6323` |
| `wine_quality` | `166e5bf23c0004b07d27985816bdb27d` | `5de90b641357e8934fe0f1bfd1a19fa59c26b7e8f21752b809fa123378b1d97e` |
| `house_sales` | `0208b93d87877d917afdc43987e7b4b4` | `8f4cb02df630cb286a9df81256129a195ddbab732222efb6aa4deccc1a81daaf` |
| `diamonds` | `ad048e0eed26e539e61d209f4f99cb14` | `40d50761f7f3bc64e4f9c2d1bd1ddbc0dcd2f24ab5c399eebbb168cdb37def8c` |
| `bike_sharing` | `d1f84b9ec74c54c1dfbca165452a5154` | `9fcd39ad05e265ed7693c35497402372b160c4c820ac8b5cacc40a327b212b81` |

Changed content or identity cannot be mixed with saved predictions. Historical
OpenML records without these fingerprints need a new output root for new fits
or regenerated analysis; historical saved reports remain readable. No source
identity is retroactively invented for old evidence.

### Task, licence and citation caveats

The upstream licence field was `Public` for twelve entries, `CC0 Public Domain`
for `house_sales`, and `public domain` for `diamonds`. These are recorded fields,
not independent legal verification or a blanket commercial-use permission.
Review each linked source and original terms before use. Several entries have
no structured citation or original-source field; missing information remains
unknown. In particular, `kc1`'s structured source URL ends in `kc2.html`, while
its description identifies KC1; check the original repository when citing it.

| Entries | Target meaning and suitability notes |
|---|---|
| `credit-g` | Historical credit class, not an operational lending decision. Encoded class 1 is `good`, not `bad`; read `class_labels` before interpreting positive-class metrics. |
| `diabetes` | Historical diagnostic class, not clinical validation. Zero-valued measurements can represent missingness without being recorded as NaN. |
| `adult` | Income-threshold class from historical census data; demographic predictors require fairness and use-context review. |
| `bank-marketing` | Subscription outcome. `V12` is completed-call duration, unavailable for pre-call prediction; the unchanged benchmark is not a valid pre-call deployment design. |
| `phoneme`, `blood-transfusion`, `kc1`, `vehicle` | Respectively phoneme class, donation outcome, software-defect label and vehicle class. Historical benchmark success does not validate current audio, medical, software or recognition use. |
| `abalone` | Numeric ring count, an age proxy; the target's `Class_` prefix does not make this a classification task. |
| `cpu_act` | Numeric user-mode CPU utilization percentage (`usr`), not a category. |
| `wine_quality` | Ordinal quality score treated explicitly as regression, not silently converted to classification. Regression error assumes meaningful numeric distances between scores. |
| `house_sales` | Historical house price with a date predictor. Random splits do not establish future-price performance. |
| `diamonds` | Historical price regression, not a current valuation service. |
| `bike_sharing` | Hourly rental count with calendar predictors. `casual`/`registered` count components are absent, but random splits still do not test forecasting. |

The loader preserves these registered versions and predictors. It does not
silently remove leakage-prone fields, change targets or switch evaluation tasks.
Use a separately named, explicitly reviewed local dataset for a different
prediction-time design or temporal split.

## Actual download sources, destinations, and triggers

The following paths are relative to the source checkout's root for the default local
setup. Cache overrides in `.env` change the destinations, not the upstream model identity.

### Bundled datasets: installed with scikit-learn

| Dataset | Upstream source documentation | Local destination | When obtained |
|---|---|---|---|
| `breast_cancer` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | `.venv\Lib\site-packages\sklearn\datasets\data\breast_cancer.csv` | Installing scikit-learn via uv; no dataset download at run time |
| `wine` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) | `.venv\Lib\site-packages\sklearn\datasets\data\wine_data.csv` | Installing scikit-learn via uv; no dataset download at run time |
| `diabetes_reg` | [scikit-learn loader and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | `.venv\Lib\site-packages\sklearn\datasets\data\diabetes_data_raw.csv.gz` and `diabetes_target.csv.gz` | Installing scikit-learn via uv; no dataset download at run time |

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
| TabICLv2 (experimental) | [jingang/TabICL, pinned revision](https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19) | `.cache\huggingface\hub\models--jingang--TabICL\` | Only when selected with `QUIBBLE_TABICL_ALLOW_DOWNLOAD=1`; missing cached weights otherwise refuse |

For Kumo's default `small` size, the installed loader requests
[`small/classifier.pt`](https://huggingface.co/nvidia/Kumo-Tabular/blob/v1.0.1/small/classifier.pt)
or [`small/regressor.pt`](https://huggingface.co/nvidia/Kumo-Tabular/blob/v1.0.1/small/regressor.pt),
plus `config.json`. Changing size selects the corresponding `medium/` or `large/` files.
TabFM selects a task-specific subfolder from its repository; Quibble uses its PyTorch
loader, not Google's separate JAX model repository.

TabICLv2 selects `tabicl-classifier-v2-20260212.ckpt` or
`tabicl-regressor-v2-20260212.ckpt` at the pinned revision and checks its SHA-256
before loading. Downloads are disabled by default, including in the `all` preset.
Its actual model predictions remain unverified; see the
[adapter limits and terms](backends.md#tabicl_v2).

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
python -m quibble_ml demo

# Loads bundled Wine; downloads uncached TabFM/Kumo weights, then writes isolated smoke evidence.
python -m quibble_ml benchmark --preset foundations --smoke

# Requests only these OpenML tables; does not train models or load foundation checkpoints.
python -m quibble_ml datasets --verify --names credit-g,adult
```

In the second example, checkpoints stay in `.cache`, while metrics/predictions/report go to
`results\smoke\<unique-id>\`. Full benchmark results go to `results\<dataset>\`.
Existing successful or failed run records may cause fits to be skipped; `--force` retries
the fit but does not inherently force a new weight download.

To see effective destinations and files already present:

```powershell
python -c "from quibble_ml.cache import prepare_caches; print('\n'.join(str(k)+': '+str(v) for k,v in prepare_caches().items()))"
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
python -m quibble_ml datasets --verify --names breast_cancer,wine,diabetes_reg  # offline
python -m quibble_ml datasets --verify --names credit-g,adult                 # may download
python -m quibble_ml datasets --verify                                       # all registered; may download all OpenML entries
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
from quibble_ml.datasets import load

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
then run `quibble-ml datasets --verify --names yourname`.

## Licences
Each OpenML dataset has its own licence and citation requirements. Check them before publishing
results derived from a dataset, and cite sources in your write-up.

## Explicit relational datasets (P7 backend contract)

Relational input is separate from the single-table `dataset.json` format. The
opt-in [relational module](../quibble_ml/relational.py) reads a schema-versioned
manifest with named CSV/Parquet tables, declared SHA-256 hashes, primary and
foreign keys, timestamp roles, an aggregation recipe, and official temporal
train/validation/test assignments. It never infers relationships or uploads data.
See [the relational workflow](testing-your-data.md#relational-manifests) for
executable synthetic examples and integration requirements.

Features use direct entity-to-child one-to-many relations only. Dimensions must
be frozen at one declared snapshot before every prediction cutoff. There are no
recursive joins, arbitrary SQL, target-table joins, automatic ID predictors, or
relational-native model claims. `kumo_tabular` remains a flattened tabular adapter,
not a relational-native model.

Prediction identity is a canonical `(entity, UTC prediction timestamp)` pair,
not the source file's row number. Private per-registry HMAC keys generate
pseudonymous example/group tokens for pairing and entity-clustered uncertainty.
Keep the registry and its secret out of public evidence. Pseudonyms and feature
aggregates are not a guarantee of anonymity.

### RelBench pilot research: rejected F1 candidate

The small candidate is **`rel-f1 / driver-position`**, a regression forecast.
Read-only metadata was reviewed on 2026-10-30:

* RelBench code release 3.1.0, commit
  `25cca45dd7a7cd2e17e116bd44a7951ebbdc06d0`.
* Hub dataset `stanford-star/relbench-v1`, revision
  `21d7284df2a9621c71f21bc4a1a30b7e8effb8dd`.
* The pinned task directory contains `train.parquet` (26,979 bytes),
  `val.parquet` (4,168 bytes), `test.parquet` (4,795 bytes), and the task
  manifest (752 bytes). All nine database Parquet files total about 1.32 MB;
  database metadata declares 97,606 rows.
* Official database boundaries are validation `2005-01-01` and test
  `2010-01-01`. The task manifest declares a **60-day** horizon, entity
  `drivers.driverId`, target `position`, and time column `date`. Official split
  tables, not guessed date partitions, must determine prediction examples.
* The actual pinned task directory inventory is `driver-dnf`,
  `driver-position`, and `driver-top3`; the broader dataset card additionally
  advertises tasks not present in this v1 directory. Use the pinned inventory.
* **Licensing blocks the pilot under the permissive-only download rule.**
  The database-specific metadata declares **CC BY-NC-SA 3.0 (Ergast database
  images)**, not MIT or CDLA-Permissive. RelBench's converted schemas, tasks,
  splits and statistics have CDLA-Permissive-2.0 terms, which do not replace the
  database's upstream terms. No F1 database/task data were downloaded and no
  restricted terms were accepted. Only bounded public metadata was read.

Sources: [pinned repository][relbench-code],
[database manifest][relbench-f1], [task manifest][relbench-task],
[task inventory][relbench-inventory], [per-database license policy][relbench-license],
and [database statistics containing source-specific licenses][relbench-stats].

The verified modern API is `relbench.load_dataset(...)`, then
`dataset.get_task_names()`, `dataset.load_task("driver-position")`,
`task.get_table("train" | "val" | "test")`, and `task.evaluate(predictions)`.
Test targets are hidden by the public task API. An eventual approved pilot must
use a local copy fetched at the pinned Hub revision, preserve official table
order for the evaluator, and explicitly review availability timestamps and
timezone assumptions. Do not copy obsolete `get_dataset/get_task` examples or
invent label visibility rules.

The v1 alternatives were also license-screened through bounded metadata:
Avito/Event/H&M carry competition or redistribution restrictions; Amazon has
no explicit license; Stack and Trial are substantially larger. The v2 extras
have large full corpora; SALT/RateBeer are non-commercial. The alternate arXiv
projection below meets the revised bounded-sampling scope without downloading
the large papers table or any restricted data.

### Bounded public pilot: arXiv sampled comparison

The permitted alternative is **`rel-arxiv / author-publication`**, pinned to
Hub revision `6b02dae348f68ecf5b08d8587ccdc839c2c33e3e` in
`stanford-star/relbench-v2-extra`. Its own upstream
[license text](https://github.com/PKUTHM/arxiv-physics/blob/main/LICENSE.txt)
explicitly grants **CC BY 4.0**, including commercial adaptation.
The reviewed license Git blob is `b2c3ce4e61335a1ea45b5adcf571f669acb6c49c`.
Credit: Tang, Haiming; He, Sirui; Li, Mengjie; Guo, Zhimao (2024),
*arXiv-physics: A Large-Scale Physics Citation and Authorship Dataset*.
RelBench conversions retain CDLA-Permissive-2.0 terms.

This is a **deterministic sampled pilot, NOT an official full-corpus benchmark**:

* Reviewed alternatives included F1 (small, non-commercial), Stack (CC BY-SA,
  large full corpus), Trial (government-data terms, large full corpus), and
  arXiv (explicit CC BY 4.0, small usable projection).
* Downloaded only five pinned, size/hash-verified files, **9,237,686 bytes total**:
  authors, paperAuthors, and the task's official train/validation/test tables.
  The 158 MB papers table and the rest of the full corpus were not downloaded.
* The pinned task inventory is `author-category`, `author-publication`,
  `paper-citation`, and `paper-paper-cocitation`. The selected forecast target
  is publication count over **182 days**.
* Select 64 examples per official split by ascending SHA-256 of split name,
  author ID and UTC cutoff, without reading labels for selection. Retain the
  original train/val/test assignments and official chronological boundaries:
  validation `2022-01-01`, test `2023-01-01`. This is not a new random split.
* Project authors to IDs only and retain the sampled authors' dated authorship
  history: **192 examples, 192 entities and 1,653 child rows**. No author name,
  identifier, paper ID or label becomes a predictor.
* Explicit pilot limitation: source submission calendar dates are interpreted
  as UTC and availability is assumed equal to event time. Actual ingestion
  timestamps are not supplied upstream. This does not establish real-world
  late-arrival safety for this public source; synthetic late-arrival proofs
  validate the builder's behavior when availability is supplied.
* Horizon purge removes 17 sampled training labels: **47 train / 64 validation /
  64 test** examples remain. HistGB and LightGBM both completed through the
  existing normal sweep on the same canonical test rows, RMSE approximately
  **0.9902**. This tiny comparison is execution evidence, not superiority or
  representative benchmark evidence.

The opt-in [pilot module](../quibble_ml/relational_pilot.py) records source-file
hashes, sample selection, transformations, attribution and official inventories
in `pilot-evidence.json`. Its import and routine tests perform no network calls.
To reproduce:

```powershell
.venv/Scripts/python.exe -B -m quibble_ml.relational_pilot --source .cache/p7-arxiv-source --out data/p7-arxiv-sample --download
.venv/Scripts/python.exe -B -m quibble_ml.relational run rel_arxiv_sample --registry data/p7-arxiv-sample/registry.json --cache-dir .cache/p7-public-features --backends histgb lightgbm --out results/p7-arxiv-sampled-comparison
```

Output directories must be new. Omitting `--download` requires the exact pinned
source files already present locally; their hashes are checked again.
Do not treat this successful sampled pilot as P7.5 relational-native completion.

[relbench-code]: https://github.com/stanford-star/relbench/tree/25cca45dd7a7cd2e17e116bd44a7951ebbdc06d0
[relbench-f1]: https://huggingface.co/datasets/stanford-star/relbench-v1/blob/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/rel-f1/manifest.yaml
[relbench-task]: https://huggingface.co/datasets/stanford-star/relbench-v1/blob/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/rel-f1/tasks/driver-position/manifest.yaml
[relbench-inventory]: https://huggingface.co/api/datasets/stanford-star/relbench-v1/tree/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/rel-f1/tasks
[relbench-license]: https://huggingface.co/datasets/stanford-star/relbench-v1/blob/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/LICENSE.md
[relbench-stats]: https://huggingface.co/datasets/stanford-star/relbench-v1/blob/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/STATS/databases.parquet
