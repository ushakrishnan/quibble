---
title: Testing your own data
description: Prepare a named CSV or Parquet dataset, validate it, run smoke and full comparisons, and interpret or reproduce the results.
---

## Testing your own data

Yes: the intended workflow is to provide a labeled table, select models, and get a paired
comparison plus an explanation of the measured results. You do not need to modify Python
code or register an OpenML ID.

Quibble evaluates classification or regression with one target column. Its localhost
workbench can use datasets saved in local folders, but it does not upload arbitrary
files through the browser or provide a prediction API. "Upload" here means copying
a file into a local folder.

Already have a generated dataset? Use the same workflow below, with the target included
for scoring but excluded from predictive inputs. Keep its generation code, assumptions,
seed, and synthetic label alongside your experiment records, outside the feature table.
Plausible column names do not establish realistic predictive relationships.

The runnable synthetic example later in this guide is a demonstration, not an
application benchmark. The [experiment workbench](experiment-workbench.md) also supports
reviewed, bounded synthetic specifications and optional provider-configured scenario drafting.
Real store/customer histories still need a suitable evaluation design;
the current random splits do not provide temporal or grouped validation.

## 1. Prepare the environment

Follow [Getting started](getting-started.md#install) first.
From the project root:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip check
python -m quibble doctor
```

The unified requirements include `pyarrow` for Parquet. CSV needs no additional package
beyond the core dependencies. For a minimal package install, the optional `[data]` extra
provides Parquet support.

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
- This version has no group/time split, automatic leakage detection, deduplication,
  train/test-file pairing, or automatic schema repair. Repeated observations from one
  person/account or future-dependent features can invalidate random-split results.

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

## 5. Discover and validate before training

```powershell
python -m quibble datasets --list
python -m quibble datasets --verify --names payments
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
python -m quibble benchmark --datasets payments --backends histgb,lightgbm --device cpu --jobs 1 --smoke --out results_payments
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
python -m quibble benchmark --datasets payments --preset foundations --device auto --jobs 1 --smoke --out results_payments
```

This can download large pretrained checkpoints into `.cache`. Small smoke inputs do not
make model weights smaller. Check package/model licenses and access requirements first.

## 7. Run the full paired comparison

```powershell
python -m quibble benchmark --datasets payments --backends histgb,lightgbm --device cpu --jobs 1 --cv 3x2 --model-seeds 0 --out results_payments
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
python -m quibble analyze --dataset payments --out results_payments --metric log_loss
```

Report regeneration uses its default primary metric again; it does not inherit the previous
analysis command's `--metric`. To change the swap quality metric, set `primary` in a thresholds file.

To customize replacement limits, copy the example and edit it before evaluating:

```powershell
Copy-Item .\swap_thresholds.example.json .\payments_limits.json
python -m quibble swap --dataset payments --incumbent histgb --candidate lightgbm --out results_payments --thresholds payments_limits.json
```

Swap returns 0 for PASS, 1 for FAIL, and 2 for INCONCLUSIVE. A policy FAIL does not mean
training failed. PASS is not a production-safety guarantee or proof that the data has no leakage.
Review limits against the actual cost of changing predictions.

## 9. Reuse, retry, and reproduce

- Repeating the same benchmark reuses saved cells, including errors. `--force` retries
  selected cells; it does not force another checkpoint download.
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
  python -m quibble report --out results_payments\smoke\<unique-id>
  ```

- Regenerate the full report without fitting:

  ```powershell
  python -m quibble report --out results_payments --swap histgb:lightgbm --thresholds payments_limits.json
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
