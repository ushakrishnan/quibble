---
title: Testing your own data
description: Prepare a named CSV or Parquet dataset, validate it, run smoke and full comparisons, and interpret or reproduce the results.
---

## Testing your own data

Yes: the intended workflow is to provide a labeled table, select models, and get a paired
comparison plus an explanation of the measured results. You do not need to modify Python
code or register an OpenML ID.

Quibble evaluates classification or regression with one target column. The localhost
workbench accepts reviewed CSV/Parquet uploads and saved local datasets. It is not
a prediction-serving API. The public static site cannot import data or run models.

### Browser upload

Start the [local workbench](experiment-workbench.md#start-locally), choose
**Upload CSV / Parquet**, and upload a labeled file. Inspect its actual sample,
suggested types and missing counts. Choose the outcome column and task explicitly,
then select dropped/categorical columns and optional group/time roles. Validate
the evaluation design, review the final summary, and confirm before saving.
The original file is copied unchanged into a new managed dataset folder.
No provider request or model fit occurs during import.

Limits are 50 MiB, 250,000 rows and 1,000 columns; decoded table memory is bounded
to 128 MiB. Parsing runs in a separate process capped at 2 GiB and 45 seconds.
Preview cells show at most 160 characters and 20 rows; training data is not
truncated. Column names may contain at most 256 characters. UTF-8 comma-separated
CSV, including BOM, and Parquet are accepted; archives and pickle are not.
Compressed Parquet must also satisfy decoded-size and dimension checks.

At most four uploads may be staged. Cancel removes only that staged upload;
stages expire after one hour and are cleaned on subsequent imports or startup.
An interrupted or malformed upload is removed. Invalid roles remain staged for
correction, but cannot reuse an earlier confirmation. Existing datasets are never
overwritten. File/schema changes require another review; changed committed import
files must be imported under a new name. Reopened experiments retain their
reviewed source fingerprint. After a server restart, review the saved dataset
again before starting another run.

The folder-based workflow below remains available for scripts and larger inputs.
It does not impose the browser's upload limits.

Already have a generated dataset? Use the same workflow below, with the target included
for scoring but excluded from predictive inputs. Keep its generation code, assumptions,
seed, and synthetic label alongside your experiment records, outside the feature table.
Plausible column names do not establish realistic predictive relationships.

The runnable synthetic example later in this guide is a demonstration, not an
application benchmark. The [experiment workbench](experiment-workbench.md) also supports
reviewed, bounded synthetic specifications and optional provider-configured scenario drafting.
Real store/customer histories still need a suitable evaluation design.
Use declared group or temporal splits rather than random rows when the task requires them.

## 1. Prepare the environment

Follow [Getting started](getting-started.md#install) first.
From the project root:

```powershell
& .\.venv\Scripts\Activate.ps1
uv pip check
python -m quibble_ml doctor
```

The `--extra data` selection in the setup command provides `pyarrow` for Parquet.
CSV needs no additional package beyond the core dependencies.

## 2. Create a named dataset folder

```powershell
New-Item -ItemType Directory -Force .\data\payments | Out-Null
```

Place your table and its declaration together:

```text
data/
  payments/
    data.csv                  Your labeled table, or data.parquet
    dataset.json              Target, task, and optional column rules
```

Use the folder name, `payments`, in commands, not the file path.

- Names may contain letters, digits, hyphens, and underscores, beginning with a letter
  or digit. Do not use built-in/OpenML names such as `wine` or `adult`.
- `report`, `runs`, `smoke`, and `artifacts` are reserved output names.
- Names should be unique, meaningful, and stable while you compare models.
- Local folders default to `<project-root>\data`, not `.cache`: these are your inputs,
  not downloaded datasets.
- Set `QUIBBLE_DATA_DIR` in `.env` to relocate the folder collection. Relative paths
  resolve from the project root; absolute paths can point to another local drive.
- Each dataset needs its own `dataset.json`. A CSV alone is not discovered.

For example, to use your existing file, replace the source path before running:

```powershell
Copy-Item 'C:\path\to\your-file.csv' .\data\payments\data.csv
```

No separate upload step, service account, or network dataset registration is required.

## 3. Check the table format

Each row must be one observation. Each column must have one scalar value per observation.
The target must already be known for every row.

Example classification CSV, showing structure only:

```csv
customer_id,amount,days_open,region,paid_late
C001,120.50,45,east,yes
C002,80.00,10,west,no
C003,230.00,38,east,yes
C004,60.00,8,west,no
```

Four rows illustrate the schema; they are not enough for meaningful model comparison.
Use representative data, preferably hundreds or more labeled rows when available.

| Requirement | Behavior |
|---|---|
| Format | UTF-8 comma-separated CSV, optionally with a UTF-8 BOM, or `.parquet` |
| Header | First CSV row contains unique, nonempty column names without surrounding spaces |
| CSV records | Every nonblank row has the same number of fields as the header; use standard quoting for commas |
| One target | Exact case-sensitive column name declared in the manifest |
| Target completeness | Missing/empty target values are rejected, not imputed or silently dropped |
| Classification | At least two classes and at least two rows per class for basic loading |
| Regression | Target must be finite numeric values |
| Features | At least one feature remains after removing the target and excluded columns |
| Numeric features | Numeric values or missing values; infinity is rejected |
| Categorical features | Strings, booleans, or declared categorical columns; train-fitted adapters handle encoding |
| Unsupported types | Nested lists/objects and undeclared typed datetimes are rejected; engineer or exclude them first |
| Minimum size | At least four labeled rows to load; usable split sizes and model limits still apply |

Additional preparation rules:

- Exclude row/customer IDs, names, email addresses, addresses, free-form sensitive text,
  and columns that are not valid predictive inputs.
- Exclude information available only after the outcome. For example, a final collection
  status may reveal whether an invoice was late.
- Do not assume numeric category codes are continuous measurements. Declare postal codes,
  product codes, and similar columns categorical. Declared CSV categories are read as
  text, preserving leading zeros such as `001`.
- CSV uses pandas' standard missing-value tokens, including empty cells and common tokens
  such as `NA`. If those are legitimate category names, recode them before importing.
- Missing feature values are retained. Quibble does not promise that every backend can
  handle every missingness pattern or mixed schema.
- CSV date-looking strings are treated as categories unless you engineer numeric features
  or exclude them. Parquet datetime columns need exclusion or an explicit categorical
  declaration. A categorical timestamp does not make a random split temporally valid.
- Group/time splits are available through the manifest below. Leakage flags are heuristics,
  not comprehensive detection. No automatic deduplication, train/test-file pairing, or schema
  repair is implemented. Repeated observations or future-dependent features can invalidate
  random-split results.

The file is loaded into memory, not streamed. Size RAM/VRAM expectations accordingly.

## 4. Declare the target, task, and column rules

Create `data\payments\dataset.json`:

```json
{
  "file": "data.csv",
  "target": "paid_late",
  "task": "classification",
  "drop_columns": ["customer_id"],
  "categorical_columns": ["region"]
}
```

Or create that declaration from PowerShell:

```powershell
@'
{
  "file": "data.csv",
  "target": "paid_late",
  "task": "classification",
  "drop_columns": ["customer_id"],
  "categorical_columns": ["region"]
}
'@ | Set-Content -Encoding utf8 .\data\payments\dataset.json
```

| Key | Required | Meaning |
|---|---|---|
| `file` | yes | Relative path to `.csv` or `.parquet`, inside this dataset folder |
| `target` | yes | Column to predict; it is automatically removed from features |
| `task` | yes | Exactly `classification` or `regression`; never guessed from numeric labels |
| `drop_columns` | no | Columns excluded before splitting/training; defaults to `[]` |
| `categorical_columns` | no | Extra categorical declarations, including numeric codes; defaults to `[]` |
| `time_column` | no | Numeric or date column that orders rows in time; excluded from features. Enables temporal splits |
| `group_column` | no | Column identifying the customer, account or entity a row belongs to; excluded from features. Enables group splits and group-clustered intervals |
| `split` | no | `random`, `group` or `temporal`. Default: `temporal` if `time_column` is set, else `group` if `group_column` is set, else `random` |

All declared columns must exist. Column lists must be unique and must not overlap each
other or the target. Unknown manifest keys are rejected so a typo cannot silently change
the experiment. Absolute file paths and paths escaping the dataset folder are rejected.

For regression, use a numeric target and change the declaration, for example:

```json
{
  "file": "data.parquet",
  "target": "days_until_payment",
  "task": "regression",
  "drop_columns": ["customer_id"],
  "categorical_columns": ["region"]
}
```

Do not include both `paid_late` and `days_until_payment` as mutually revealing predictors.
Drop unused outcome columns too.

### Optional: generate a runnable synthetic example

If you do not have a file ready, this creates 180 synthetic observations matching the
classification manifest above. It is a workflow exercise, not evidence about real payment data.
It replaces the example input file, so do not run it over your real table.

```powershell
python -c "from pathlib import Path; import numpy as np; import pandas as pd; rng=np.random.default_rng(42); n=180; days=rng.integers(1,61,n); df=pd.DataFrame({'customer_id':['C'+str(i) for i in range(n)], 'amount':rng.uniform(20,500,n).round(2), 'days_open':days, 'region':rng.choice(['east','west'],n), 'paid_late':np.where(days>30,'yes','no')}); df.to_csv(Path('data')/'payments'/'data.csv',index=False)"
```

### Group and temporal splits
Rows are rarely independent. If several rows belong to one customer, or the model will predict the
future from the past, a random split lets it train on information it would not have, and scores
are inflated.

- **Group split** (`group_column`): whole groups stay on one side of every split. Holdout uses about 30%
  of the groups as the test set; `--cv KxR` assigns shuffled, size-balanced groups to K folds per
  repeat. Bootstrap intervals resample whole groups, so correlated rows are not treated as independent.
- **Temporal split** (`time_column`): the future is always the test set. Holdout tests on the latest
  30% of rows; `--cv Kx1` is forward chaining (train on everything before each of K consecutive test
  blocks). Temporal splits have no repeats, so `--cv 5x2` is rejected; use `--cv 5x1`.
- Set `"split": "random"` to reproduce a naive baseline on the same data; quibble then flags
  time-like column names.

```json
{
  "file": "invoices.csv",
  "target": "paid_late",
  "task": "classification",
  "drop_columns": ["invoice_id"],
  "time_column": "issued_on",
  "group_column": "customer_id"
}
```
With both columns declared, `split` defaults to `temporal`; the group column still clusters the
bootstrap. The split scheme is part of the dataset fingerprint, so changing it requires a new
`--out`. Before training, quibble also prints [data flags](analysis.md#data-flags) for suspicious
columns.

For classification, every dataset class must occur in each training partition. A group or
temporal split that omits a class is explicitly refused before any backend fits, rather than
mislabeling probability columns. This includes training classes 1 and 2 when the complete
encoded class set is 0, 1, and 2. Choose a representative training window or collect more
training data; do not move future observations into training to make the check pass.
Legacy group/time classification records without matching class-order metadata are rejected
when reused or analyzed. Regenerate them with a new output directory. New successful
classification records store `prediction_classes` in encoded probability-column order.

## 5. Discover and validate before training

```powershell
python -m quibble_ml datasets --list
python -m quibble_ml datasets --verify --names payments
```

Expected outcomes:

- `payments: local` appears in the list.
- Verification prints rows, features, task, missingness, categorical share, class balance
  for classification, and local-source metadata.
- Metadata is saved in `datasets_meta.json` in the working directory.
- Verification returns 0 if all requested entries load and names match; 1 if any fail
  or an OpenML name mismatches. Inspect messages as well as `$LASTEXITCODE`.
- Invalid input produces an explicit `ERROR`, including missing target/columns,
  unsupported format, incomplete labels, or a malformed manifest.

Always use `--names payments` here. Omitting it verifies all registered and discovered
datasets, including OpenML entries that may download data.

Verification checks basic loading, not leakage, representative sampling, full model
capacity, or whether your chosen split is valid.

## 6. Run a bounded smoke check

Begin with inexpensive local models:

```powershell
python -m quibble_ml benchmark --datasets payments --backends histgb,lightgbm --device cpu --jobs 1 --smoke --out results_payments
```

This uses one holdout split, model seed 0, at most 128 training and 32 test rows.
All selected backends receive the same sampled rows. Outputs are isolated:

```text
results_payments/
  smoke/<unique-id>/
    payments/                 Records, arrays, and analysis
    report/report.md          Interpretation of this setup check
```

Read the report's Interpretation, Model participation, device labels, and failure details.
Smoke interpretation explains what completed, the observed scores/resources, and any
disagreements. It does not choose a deployment winner or issue a swap verdict.
Perfect accuracy on a tiny sample is not proof of equivalence or real-world quality.

To check foundation-model setup after the trees work:

```powershell
python -m quibble_ml benchmark --datasets payments --preset foundations --device auto --jobs 1 --smoke --out results_payments
```

This can download large pretrained checkpoints into `.cache`. Small smoke inputs do not
make model weights smaller. Check package/model licenses and access requirements first.

## 7. Run the full paired comparison

```powershell
python -m quibble_ml benchmark --datasets payments --backends histgb,lightgbm --device cpu --jobs 1 --cv 3x2 --model-seeds 0 --out results_payments
```

Full results go to `results_payments\payments`, not its `smoke` subtree.
Each model receives the full training split. CV `3x2` means six fits per model;
each row is held out once per repeat.

- For local classification, every class needs at least K rows for K-fold CV. Use fewer
  folds or more labeled data if the split is rejected. Holdout must also have enough
  training and test rows to represent all classes.
- `--device cpu` pins CPU. `cuda` requires CUDA; `auto` resolves once per sweep.
  CPU-only models stay on CPU, and mixed-device reports say so.
- Resource failures are recorded without a hidden CPU retry or downsampling. Other fits
  continue, but the benchmark returns 1 for failed/refused/missing/incompatible cells.
- More splits/models can be expensive. Start with trees, then select foundation models or
  `--preset all` deliberately. AutoGluon has a per-fit training budget.
- Optional `--keep-models` retains supported tree/AutoGluon fits; foundation fitted
  estimators are not serialized.

Benchmark also analyzes results, checks swaps from HistGB (or the first selected backend),
and writes the report. You do not have to run each stage separately.

## 8. Read the report and inspect evidence

Open `results_payments\report\report.md`.

| Section | What to check |
|---|---|
| Model participation | Were all requested fits completed, compatible, and actually used? |
| Device policy | Is this CPU-only or mixed CPU/GPU, and are any measurements cached? |
| Interpretation | Which measured quality/resource tradeoffs warrant further evaluation, and why? |
| Local data source | Exact input path, target, SHA-256, manifest/data fingerprint, and classification label mapping |
| Quality and cost | Accuracy by default for classification, RMSE for regression; intervals, other metrics, time/memory |
| Disagreements | Which rows differ, who is right, and links to disputed-row CSVs |
| Slices/calibration | Weak subgroups and probability reliability where available |
| Swap checks | PASS / FAIL / INCONCLUSIVE against your chosen limits |
| Failures | Full errors/refusals and any resource problems |

Primary classification accuracy can hide imbalance. Inspect AUC, Brier/ECE, class balance,
and row-level errors too. You can explicitly use log loss or another supported metric:

```powershell
python -m quibble_ml analyze --dataset payments --out results_payments --metric log_loss
```

Report regeneration uses its default primary metric again; it does not inherit the previous
analysis command's `--metric`. To change the swap quality metric, set `primary` in a thresholds file.

To customize replacement limits, export the
[packaged defaults](../quibble_ml/swap_thresholds.json) and edit the copy before evaluating:

```powershell
python -m quibble_ml swap --export-thresholds payments_limits.json
python -m quibble_ml swap --dataset payments --incumbent histgb --candidate lightgbm --out results_payments --thresholds payments_limits.json --require-thresholds
```

Swap returns 0 for PASS, 1 for FAIL, and 2 for INCONCLUSIVE. A policy FAIL does not mean
training failed. PASS is not a production-safety guarantee or proof that the data has no leakage.
Review limits against the actual cost of changing predictions.
The [workbench policy controls](swap.md#workbench) can edit/import the same JSON for full runs;
reports show effective limits, including defaults retained for omitted keys.

## 9. Reuse, retry, and reproduce

- Repeating the same benchmark reuses saved cells, including errors. `--retry-failed` reruns only
  the cells whose saved result is an error; `--force` reruns every selected cell. Neither forces
  another checkpoint download.
- Local records save input SHA-256, a fingerprint of file content plus manifest settings,
  task, target, column rules, and the original classification labels.
- Source changes are not silently accepted as the old experiment. If file content, target,
  task, or column rules change, choose a new output root, for example `results_payments_v2`.
  Changes to rows/features can invalidate predictions even if the labels are unchanged.
- Restoring the original file/manifest lets you analyze the old experiment. Moving an unchanged
  input collection using `QUIBBLE_DATA_DIR` does not itself alter its content fingerprint.
- Keep original inputs and manifests securely if you want to reproduce analysis. Reports
  reload features for slices/traits and reject source mismatches; fingerprints do not
  reconstruct the original table.
- Regenerate the named smoke report by passing the entire smoke-run root, not its `report` folder:

  ```powershell
  python -m quibble_ml report --out results_payments\smoke\<unique-id>
  ```

- Regenerate the full report without fitting:

  ```powershell
  python -m quibble_ml report --out results_payments --swap histgb:lightgbm --thresholds payments_limits.json
  ```

- Dated history preserves generated records, arrays, and report evidence. Original input
  feature files and downloaded weights are not copied into history. Retained models live
  separately under the output root's `artifacts`.
- Use `--jobs 1`, keep split/CV/model seeds fixed, record package versions, and avoid changing
  files or installing packages while workers run.

## Privacy and publication

The default `data/` folder and `results/` are ignored by git. An explicitly chosen root
such as `results_payments` or an external data directory needs its own ignore/protection
rules; `.gitignore` is not access control.

Quibble's current adapters fit locally, not through hosted prediction APIs. Model downloads
can require network access; review optional libraries' telemetry and policies separately.
Do not pass data you are not permitted to process.

Results are not anonymous: labels, row indices, label names, file paths, fitted models, and
disputed-row CSVs can disclose input information. Disagreement exports include up to the
first 40 feature columns for disputed rows. Before sharing reports, snapshots, metadata,
or model artifacts, inspect them and remove sensitive material under your data policy.

For temporal/grouped deployment problems, get an appropriate evaluation design before
treating these random-split comparisons as decision evidence.

## Relational manifests

This is an opt-in relational contract, not an extension of flat CSV inference.
Browser multi-table import is available as **Advanced: import named relational
tables** inside Upload. The existing single-table path remains unchanged.
Do not upload a flattened relational dataset and claim the original temporal
protocol was preserved automatically.

### Offline synthetic proof

From the repository root in PowerShell, create a **new** synthetic directory:

```powershell
.venv/Scripts/python.exe -B -m quibble_ml.relational_synthetic data/p7-proof
.venv/Scripts/python.exe -B -m quibble_ml.relational inspect relational_synthetic --registry data/p7-proof/registry.json
.venv/Scripts/python.exe -B -m quibble_ml.relational materialize relational_synthetic --registry data/p7-proof/registry.json --cache-dir .cache/p7-features
```

Repeat `materialize` for a warm-cache audit. The fixture contains future rows
encoding the answer, late-arriving rows, cutoff ties, revised dimensions,
overlapping label horizons, and repeated entities. The default strict-cutoff
policy excludes future/late/tied history. Twelve training examples are purged.
Use `--task regression` on the fixture generator for a regression proof.
Use `--repeat-test-entities` in a new fixture directory to include two distinct
test cutoffs per entity. Its 24 test predictions have 12 entity resampling units;
the test suite verifies that both predictions receive identical bootstrap
weights in every replicate, while keeping distinct prediction keys.

With the explicit normal-path registry and split integration, run two existing
tree routes, without a remote model or a relational-native claim:

```powershell
.venv/Scripts/python.exe -B -m quibble_ml.relational run relational_synthetic --registry data/p7-proof/registry.json --cache-dir .cache/p7-features --backends histgb lightgbm --out results/p7-proof
```

`run` refuses an existing output directory, invokes the normal sweep in a fresh
process, pins CPU/defaults/one model seed/holdout, and verifies successful source-
matched evidence for both backends. It explicitly fails before model workers
start if the normal loader cannot resolve the reviewed registry fingerprint.
It does not monkeypatch dataset loading, invent random splits, or treat a
refused/failed backend as a successful comparison. The normal pipeline's report
and analysis still require the integration described below.

### Registry and manifest schema

`registry.json` explicitly selects permitted datasets; discovery alone is not
registration. Paths are relative to the registry directory:

```json
{
  "schema_version": 1,
  "datasets": {
    "my_relational_task": {
      "manifest": "my_task/manifest.json",
      "identity_key": "REPLACE_WITH_64_HEX_CHARACTERS_FROM_32_RANDOM_BYTES"
    }
  }
}
```

Generate a private identity key with Python's `secrets.token_hex(32)`. The fixture
uses a deliberately public deterministic key only because its data are synthetic.
Never publish a private registry/key or reuse the fixture key for private data.
Changing this key changes example pseudonyms and invalidates caches.

The generated `manifest.json` is a complete executable schema example. Schema 1
has these top-level fields; unknown fields fail rather than being ignored:

* `tables`: named tables with `path`, byte-level `sha256`, `primary_key` column
  list, `role`, and an exact `columns` role map. Supported table roles are
  `entity`, `event`, `dimension`, and `examples`. Paths must remain inside the
  manifest directory; absolute paths, traversal, backslashes, escaping symlinks,
  missing files, duplicate headers, and undeclared columns are refused.
* Column roles: `id`, `timestamp`, `observed`, `target`, `target_derived`, and
  `split`. Only explicitly allowlisted numeric `observed` columns can become
  predictors. IDs, target-derived values, labels and timestamps are not
  automatically feature candidates. Class vocabulary is fitted on purged
  training labels only; unknown evaluation labels fail closed.
* Event/dimension tables require separate, explicit `event_time` and
  `available_time` column declarations. Every timestamp must have `Z` or a numeric
  UTC offset; naive timestamps and DST guessing are rejected. Values normalize
  to UTC. Availability cannot precede event time.
* `relations`: unique names, `parent`, `parent_key`, `child`, `foreign_key`, and
  `kind` (`events` or `frozen_dimension`). Only direct entity-to-child joins are
  supported. Parent keys must be unique, children must have unique declared
  primary keys, and every child/example foreign key must resolve. No target/
  examples table can be used as aggregate history.
* `frozen_dimension` requires `frozen_as_of` no later than the earliest
  prediction. Select the latest revision whose event **and availability**
  timestamps are at or before that snapshot. Future revisions are excluded;
  ambiguous equal-time revisions fail. This is a fixed snapshot, not an inferred
  as-of dimension join at each prediction.
* `task`: `type` (`classification` or `regression`), `entity_table`, `entity_key`,
  `examples_table`, `entity_column`, `prediction_time`, `target`,
  `label_available_time`, and positive `horizon_seconds`. Example primary key is
  exactly `[entity_column, prediction_time]`. Rows sort canonically by entity
  string, then UTC cutoff; duplicate normalized prediction keys fail.
* The examples table has an explicit `split` column containing all three of
  `train`, `val`, and `test`. Labels are evaluation inputs only; neither
  examples nor label columns enter feature aggregation. `label_available_time`
  must be at least `prediction_time + horizon_seconds`.
* `split`: `kind: "official_temporal"`, `entity_overlap: "allowed" | "disjoint"`,
  and nonnegative `embargo_seconds`. Cutoff ranges must be strictly ordered.
  Train labels must be available **strictly before** the earliest validation
  cutoff minus embargo; validation labels must likewise precede test. Outcome
  horizon is already included in label availability. Embargo is additional,
  not a substitute for horizon purge. Empty purged splits fail.
* `recipe`: explicit `tie_policy` (`exclude` for event/availability `< cutoff`,
  or `include` for `<= cutoff`), `lookback_seconds` (positive or `null`), and
  `aggregations`. Lookback includes the lower event-time bound. Each aggregation
  has unique `name`, `relation`, `op`, and, where needed, `column`. Events support
  `count`, `recency_seconds`, `sum`, `mean`, `min`, and `max`; frozen dimensions
  support `value`. Missing history yields count zero and missing numeric
  aggregate/recency, not an invented zero observation.
* Optional `limits` can only tighten built-in caps: 16 tables, 16 MB per file,
  64 MB total encoded bytes, 256 columns per table, 100,000 rows per table, 250,000 total rows, 20,000
  prediction examples, 128 features, 2 million scanned join matches, and 128 MB
  decoded data. Compressed Parquet metadata is checked before decoding. Both
  retained and rejected candidate join rows count against the scan budget.
* Optional `source` requires bounded string fields `kind`, `license`, `version`;
  additional supported metadata are `dataset`, `task`, `url`, `revision`,
  `official_val_timestamp`, `official_test_timestamp`, and
  `availability_assumption`. Do not put identifiers or secrets in metadata.

No inferred transforms are fitted by this module. Normal tabular preprocessing
must still fit on training examples only. Repeated-entity overlap can be valid
forecasting; it is not cold-start/entity-disjoint evaluation. HMAC entity groups
are supplied for clustered uncertainty; residual temporal dependence remains.

### Cache, audit, and integration contract

`materialize_registered(name, registry, cache_dir)` returns a `Materialized`
object with `X`, `y`, `task`, and a public-safe `audit`.
`load_registered(...)` returns the existing `(X, y, task)` loader tuple.
`registered_names(registry)` enumerates only the explicit registry.

The content-addressed cache covers **every table hash**, full task, relation,
recipe and split schemas, source metadata, identity namespace, limits, and
builder version. Every invocation, including warm-cache calls, rehashes and
validates inputs. A changed child file with an unchanged declared hash fails;
after explicit review updates its hash, the cache key changes. Cache JSON has
an integrity checksum and never uses pickle. Cache files contain aggregates and
slices, not raw IDs or labels, but still need private-data protection.

The audit includes hashes/counts, task/source/recipe/split contracts, purge
counts, repeated-entity status and temporal caveat. Costs separately expose
validation, cold feature build or warm cache read, total feature-stage time and
cache state. A warm call still pays for input validation. Feature costs are
shared dataset work: charge once explicitly, or show an explicit amortization,
never silently charge only one backend or drop them from end-to-end totals.

Integration/API map:

1. **Registry/load (implemented):** honor explicit `QUIBBLE_RELATIONAL_REGISTRY` and optional
   `QUIBBLE_RELATIONAL_CACHE`; use `registered_names`/`load_registered` in dataset
   discovery, `known`, `load`, `task_of`, and `source_info`. Reject collisions
   with flat/built-in names. No ambient folder inference or remote downloads.
2. **Split (implemented):** for `X.attrs["quibble_split"]["kind"] == "relational"`, require
   holdout and use `split_registered(X, y, evaluation="test")`. It uses purged
   official train, not train+validation. Explicit validation is available with
   `evaluation="val"`. Never route this kind through ordinary temporal or random
   splitting. Nonzero repeated split IDs and random CV require explicit refusal.
   Tuned execution needs a separately reviewed horizon-safe inner protocol;
   do not interpret a relational kind as random/group-only inner tuning.
3. **Subsetting/identity (implemented):** call `subset_registered` for relational frames.
   It slices groups, cutoffs, example keys and disagreement-slice attributes;
   it removes full-frame official split positions from subsets. Per-row metadata
   in `X.attrs["quibble_relational"]` are `example_keys`,
   `prediction_timestamps`, and `slices` (history length, recency, missing
   relations, cold start). Original `positions` hold train/val/test integer
   positions in the complete canonical frame. Export example tokens, not raw
   identifiers or original flat-file row numbers. Preserve the final test order.
4. **Provenance/run/cost:** persist `quibble_source.fingerprint`,
   `quibble_relational.audit`, feature costs/cache state, selected example keys
   and group tokens alongside normal model loading, training, prediction and
   end-to-end costs. Revalidate child hashes before reuse, including existing
   sweep skip paths. Do not let smoke subsampling misalign metadata.
5. **Analysis/report:** retain entity-clustered uncertainty using
   `quibble_groups`; display the temporal caveat and relational prediction unit.
   Add history/recency/missing-relation/cold-start slices, audit and shared
   feature-stage costs to existing analysis/archive/download evidence. No
   relational-native badge for flattened `kumo_tabular`.
   `analysis_traits(X)` supplies fixed, label-free history/recency buckets and
   missing/cold-start traits. `diagnostic_masks(X)` (also available as
   `slice_masks(X)`) returns full-frame Boolean masks for qualified descriptive
   analysis only, never additional acceptance gates. `analysis_contract(X)`
   returns JSON-safe keys, groups, semantics, traits, audit and feature costs.
6. **Browser (implemented):** stage multiple named tables, validate combined
   budgets, explicitly confirm relationships/timestamp/task/split roles, and
   save atomically. Review binds all table bytes, manifest, registry identity and
   dataset name. Failed re-review invalidates old approval; changed inputs and
   unsupported model routes fail closed. Saving never overwrites a dataset.

### Browser relational intake

Choose **Upload**, expand **Advanced: import named relational tables**, enter a
new dataset name, and choose all CSV/Parquet files. Each filename stem is its
logical table name (`events.csv` becomes `events`); the manifest must declare
that exact table identity, `tables/events.csv` path and byte hash. Load a
complete manifest JSON or edit the supplied text, then:

1. Stage all tables locally.
2. Validate the explicit schema and cutoffs.
3. Review the named table schemas/roles/primary keys, relationships, event and
   availability roles, frozen dimensions, task horizon, recipe, official splits
   and purge counts shown in the summary.
4. Confirm review, then atomically save and continue to Methods/settings.

Edits revoke browser approval. Server confirmation independently rehashes and
validates inputs; forged/replayed approval cannot authorize changed data.
Publication copies into a temporary directory within the data root, validates
the copied bytes, atomically renames it and registers it under a cross-workbench
writer lock. Registry-write failure rolls back the new directory. No original
or existing saved dataset is overwritten. Stages expire after one hour and can
be explicitly cancelled, including expired stages. At most four stages exist.

Browser saves use a private `relational-registry.json` inside `QUIBBLE_DATA_DIR`;
this explicit saved registry is reopened on workbench restart. An externally
configured registry must also reside in that root for browser publication.
CLI registration can still use any explicitly chosen registry. Keep the
registry secret private. Browser run review permits CPU local trees and official
holdout only, and the server repeats these checks before starting work.

The dedicated offline proof suite is:

```powershell
.venv/Scripts/python.exe -B -m pytest tests/test_relational.py tests/test_web_relational.py -q -p no:cacheprovider --basetemp=.cache/p7-test
```

The [public sampled pilot](datasets.md#bounded-public-pilot-arxiv-sampled-comparison)
and synthetic comparison both completed real HistGB/LightGBM sweeps. A real
browser check confirmed four-table staging, schema review, edit invalidation,
atomic save and progression to CPU/holdout run review. Shared front-stage costs
are saved in `relational-feature-stage.json` separately from worker reloads.
These are flattened-baseline proofs, not P7.5 native-model evidence or an
official full-corpus public benchmark.
