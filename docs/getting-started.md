---
title: Getting started
description: Install Quibble, run the offline demo, and start a comparison.
---

## Getting started

## Requirements
- Python 3.10+ for core Quibble; use Python 3.13 for the documented setup.
- Linux, macOS, or Windows. Windows peak memory uses an automatically installed `psutil`
  dependency; Unix uses `resource`. Optional model libraries have their own platform requirements.
- Optional models can require several GB for PyTorch and dependencies, plus
  pretrained weights. The default setup does not install these model extras.

## Install
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and obtain the
Quibble source checkout from [GitHub](https://github.com/ushakrishnan/quibble).
These instructions use source, not a package-index install:

```powershell
git clone https://github.com/ushakrishnan/quibble.git
Set-Location quibble
```

Run the following from the checkout's root.
Deactivate any other virtual environment first. uv creates or synchronizes `.venv`:

```powershell
uv sync --locked --python 3.13 --extra web --extra data
& .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
uv pip check
python -m quibble_ml doctor
```

`pyproject.toml` declares dependencies; `uv.lock` records exact resolved versions
and hashes. There is no separate requirements file to maintain. uv installs the
project editably, so source changes are reflected without a second install.
The command above includes the web UI, report rendering, Parquet, and the default
development group. Use `--no-dev` when tests are not needed.

Optional model libraries are explicit extras. Keep `--extra web --extra data` and
append only the models you want: `--extra tabfm`, `--extra kumo`,
`--extra tabpfn`, `--extra tabicl`, `--extra mitra-v2`, or `--extra autogluon`. TabFM includes its PyTorch extra;
AutoGluon includes the separately declared FastAI recipe dependencies.
TabFM/Kumo require Python 3.11+, and AutoGluon requires Python below 3.14.
These markers do not raise core Quibble's Python minimum; unsupported optional
combinations do not install the corresponding model library.
TabICLv2 pins `tabicl==2.2.0` and Torch 2.11.0. Its weights are cached-only unless
`QUIBBLE_TABICL_ALLOW_DOWNLOAD=1` is explicitly set after terms review.
Small real Wine/diabetes holdouts have passed on CPU and CUDA; this is not
broad capacity or quality validation. Mitra-v2 requires Windows/Linux,
Python 3.11-3.13 and CUDA, with separately pinned classification/regression
recipes. Its weights are cached-only unless `QUIBBLE_MITRA_ALLOW_DOWNLOAD=1`
is explicitly enabled after reviewing terms. Installation is not authorization
to use a checkpoint. See [exact model scope](backends.md).

Sync is exact and removes unselected extras and unrelated packages. Before
migrating an existing environment, run the intended command with `--dry-run`;
keep every wanted extra in later sync commands. Do not sync during a running sweep.
For Windows GPU packages, additionally select `--extra windows-cuda`; this selects
the CUDA 12.8 Torch/torchvision builds without using that index for other packages.
See [GPU setup](gpu-and-docker.md) and
[cache configuration](configuration.md#centralized-download-caches).

For dependency changes, use `uv add`, `uv add --optional <extra>`, or `uv add --dev`,
then commit both the manifest and lock. Use `uv remove` to remove a dependency and
`uv lock --upgrade-package <name>` for an intentional upgrade. Check consistency
with `uv lock --check`; normal installation uses `uv sync --locked` and does not
silently rewrite the lock.

The PyPI name is `quibble-ml` (the plain name `quibble` is taken by an unrelated project), but the
package is **not published yet**, so install from source. The Python import is
`quibble_ml`, the command is `quibble-ml`, and module invocation is
`python -m quibble_ml`.

## Migrating from the old import and command

Earlier source revisions used `import quibble` and the `quibble` command. The new
names avoid the unrelated PyPI project's import and executable. There are no
legacy aliases: use `import quibble_ml`, `python -m quibble_ml`, or `quibble-ml`.
Run `uv sync --locked --extra web --extra data` for this checkout to register the
renamed command, adding any optional extras you want to retain.

Prefer a fresh checkout and clean virtual environment using the installation
steps above. Do not copy another environment or install the unrelated `quibble`
package to fix an import error. That package can coexist, but its command and
imports are not Quibble's. Check your interpreter and active environment if an
old executable remains on your PATH.

The product/repository name, `QUIBBLE_*` settings, configured cache/data/output
directories, backend names, and saved JSON/prediction formats are unchanged.
Existing generated bundles retain their version-1 `quibble.generate` provenance
identifier; that field is a format identifier, not a command to execute.
Old plain-file evidence can be inspected and reported without retraining.
Use a new output directory for fresh experiments and preserve the original files
and their recorded settings; never label a cached result as a new run.

Retained Python models are different from plain-file evidence. Joblib/pickle
files may refer to old `quibble.*` classes and are not promised to load under
the new import name. Keep the original artifacts and a trusted old environment
if needed, or retrain in a new output directory. Quibble does not add import
shims or automatically rewrite serialized models. Never load an untrusted
pickle to test migration.

## Choose a browser or command-line workflow

After installation, the localhost workbench provides interactive dataset design,
saved-data review, comparisons, and reports:

```powershell
python -m quibble_ml web --open
```

Choose **Use existing dataset** to review built-in, bundled, or saved rows without
a planning service. Four bundled grocery/retail clothing datasets cover binary and
regression tasks. Choose **Design new dataset** and configure an OpenAI-compatible
provider for a new scenario. The CLI's `quibble-ml generate --example` remains available
for an offline editable recipe.
Follow **Choose data -> Methods and settings -> Start and monitor**, or open **Experiment history**
to read saved evidence. **Save and continue** opens configuration after saving a new
dataset. **Discard changes and continue** uses a variant's original unchanged.
Neither button starts a run.
The shared Quibble header links Workbench, Blog, About, and locally readable Docs.
GitHub links to the source repository. On the public static site, **Workbench** explains
local installation; only the local server executes experiments. See [About Quibble](about.md).
The [workbench guide](experiment-workbench.md) covers provider/auth setup, naming,
review gates, running indicators, and reopening saved reports.

Fresh synthetic rows reduce reliance on familiar benchmark examples that pretrained
predictors may have encountered. They do not prove an unseen task or real-world
accuracy; see [why generate fresh data](experiment-workbench.md#why-generate-a-fresh-dataset).

The UI supplements rather than replaces the commands below. Both use the same evaluation
pipeline. The CLI remains the route for automation, broader cross-validation schemes,
multiple model seeds, and other settings not exposed in the browser.

## The offline demo

```powershell
quibble-ml demo
```
This runs entirely offline on three datasets that ship inside scikit-learn:

| Dataset | Task |
|---|---|
| `breast_cancer` | binary classification |
| `wine` | 3-class classification |
| `diabetes_reg` | regression |

and three backends: `histgb`, `lightgbm`, and `weak` (a deliberately handicapped model used as a
sanity check). It runs 3-fold CV repeated twice (6 splits), then the analysis, two swap checks
(`histgb -> lightgbm` and `histgb -> weak`), and writes everything under `results/`
(or `QUIBBLE_OUT_DIR`).

Open `results/report/report.md`. Check:

- The participation table, effective devices, quality intervals, and disputed-row links.
- The HistGB-to-Weak sanity check, expected to FAIL with the demo's default thresholds.
- The HistGB-to-LightGBM checks, which may PASS, FAIL, or remain INCONCLUSIVE. Verdicts
  depend on measured quality, disagreements, slices, timing, memory, and configured limits,
  not just the headline score. Interval overlap alone does not establish equivalence.

The default demo typically takes about a minute, but runtime depends on hardware and dependencies.

Options: `quibble-ml demo --out DIR --cv 5x2 --jobs 4`.
Foundation models are optional. First sync with `--extra tabfm --extra kumo`
alongside your other extras, then select those methods:

```powershell
python -m quibble_ml demo --backends histgb,lightgbm,tabfm,kumo_tabular --jobs 1
```

This still uses CPU and the three built-in datasets, but optional models may download
weights and run longer. See [configuration](configuration.md)
for weight-cache setup. Cached errors are skipped just
like successful runs; add `--force` after fixing a missing package to retry selected cells.
Previous outputs are preserved automatically in dated snapshots under `results/runs/`.
Open `results/report/history.md` to browse older reports and raw-run snapshots.

For a device-aware comparison rather than the CPU-only demo, use
`python -m quibble_ml benchmark --preset foundations --smoke` to verify setup, then
`python -m quibble_ml benchmark --preset foundations --force` for measured runs.
These respect your device setting. Smoke outputs are isolated and are not benchmark evidence.
See [running benchmarks](running-benchmarks.md).

## Your first real run
1. Create settings only if absent, then edit `QUIBBLE_DEVICE` to `cpu`, `cuda`, or `auto`:

   ```powershell
   if (-not (Test-Path .env)) { Copy-Item .env.example .env }
   quibble-ml doctor           # shows what torch/CUDA it can see
   ```
2. Verify the selected dataset IDs and returned names before interpreting results:
   ```powershell
   quibble-ml datasets --verify --names credit-g,phoneme
   ```
3. Run backends over repeated cross-validation:
   ```powershell
   quibble-ml run --sweep --backends histgb,lightgbm,catboost --datasets credit-g,phoneme --cv 5x2 --model-seeds 0
   ```
4. Look at the numbers:
   ```powershell
   quibble-ml analyze --dataset credit-g
   quibble-ml swap --incumbent histgb --candidate lightgbm --dataset credit-g
   quibble-ml report --swap histgb:lightgbm
   ```

Results are written to `results/` (override with `QUIBBLE_OUT_DIR` or `--out`). See
[Running benchmarks](running-benchmarks.md) for how to do this properly if you plan to publish.
For a local CSV/Parquet rather than OpenML, follow [Testing your own data](testing-your-data.md).

## What to do when something looks odd
Read [Troubleshooting](troubleshooting.md). The most common surprises are listed there: results
being reused after a settings change, backends refusing large datasets, and `INCONCLUSIVE` on small
data.
