---
title: Getting started
description: Install Quibble, run the offline demo, and start a comparison.
---

## Getting started

## Requirements
- Python 3.11+ for the full requirements; use Python 3.13 for the documented Windows setup.
- Linux, macOS, or Windows. Windows peak memory uses an automatically installed `psutil`
  dependency; Unix uses `resource`. Optional model libraries have their own platform requirements.
- Several GB of free disk for PyTorch, AutoGluon, notebooks, and other dependencies,
  plus space for pretrained model downloads and result history.

## Install
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and obtain the
Quibble source checkout from [GitHub](https://github.com/ushakrishnan/quibble).
These instructions use source, not a package-index install:

```powershell
git clone https://github.com/ushakrishnan/quibble.git
Set-Location quibble
```

Run the following from the checkout's root.
Deactivate any other virtual environment first. Reuse an existing local environment:

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    uv venv --python 3.13 --seed .venv
}
& .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python -m quibble doctor
```

`requirements.txt` is the dependency list for this setup. It includes all supported model
libraries, matplotlib for report figures, Jupyter, and pytest. TabFM explicitly selects its
PyTorch extra. No other project's environment or cache is required.
The interpreter should be the checkout's `.venv\Scripts\python.exe`. After requirements
change, repeat both install commands and `pip check`; do not update a running sweep's
environment. Windows requirements select CUDA-enabled Torch wheels from the official
CUDA 12.8 index; GPU use still needs a compatible NVIDIA driver. See
[GPU setup](gpu-and-docker.md) and [cache configuration](configuration.md#centralized-download-caches).

The PyPI name is `quibble-ml` (the plain name `quibble` is taken by an unrelated project), but the
package is **not published yet**, so install from source. The import name and the command are both
`quibble`.

## Choose a browser or command-line workflow

After installation, the localhost workbench provides interactive dataset design,
saved-data review, comparisons, and reports:

```powershell
python -m quibble web --open
```

Choose **Use existing dataset** to review built-in, bundled, or saved rows without
a planning service. Four bundled grocery/retail clothing datasets cover binary and
regression tasks. Choose **Design new dataset** and configure an OpenAI-compatible
provider for a new scenario. The CLI's `quibble generate --example` remains available
for an offline editable recipe.
Follow **Choose data -> Configure & run -> Results**, or open **Past experiments**
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
quibble demo
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

Options: `quibble demo --out DIR --cv 5x2 --jobs 4`.
The requirements already include foundation model packages. To select them, run:

```powershell
python -m quibble demo --backends histgb,lightgbm,tabfm,kumo_tabular --jobs 1
```

This still uses CPU and the three built-in datasets, but optional models may download
weights and run longer. See [configuration](configuration.md)
for weight-cache setup. Cached errors are skipped just
like successful runs; add `--force` after fixing a missing package to retry selected cells.
Previous outputs are preserved automatically in dated snapshots under `results/runs/`.
Open `results/report/history.md` to browse older reports and raw-run snapshots.

For a device-aware comparison rather than the CPU-only demo, use
`python -m quibble benchmark --preset foundations --smoke` to verify setup, then
`python -m quibble benchmark --preset foundations --force` for measured runs.
These respect your device setting. Smoke outputs are isolated and are not benchmark evidence.
See [running benchmarks](running-benchmarks.md).

## Your first real run
1. Create settings only if absent, then edit `QUIBBLE_DEVICE` to `cpu`, `cuda`, or `auto`:

   ```powershell
   if (-not (Test-Path .env)) { Copy-Item .env.example .env }
   quibble doctor           # shows what torch/CUDA it can see
   ```
2. Verify the selected dataset IDs and returned names before interpreting results:
   ```powershell
   quibble datasets --verify --names credit-g,phoneme
   ```
3. Run backends over repeated cross-validation:
   ```powershell
   quibble run --sweep --backends histgb,lightgbm,catboost --datasets credit-g,phoneme --cv 5x2 --model-seeds 0
   ```
4. Look at the numbers:
   ```powershell
   quibble analyze --dataset credit-g
   quibble swap --incumbent histgb --candidate lightgbm --dataset credit-g
   quibble report --swap histgb:lightgbm
   ```

Results are written to `results/` (override with `QUIBBLE_OUT_DIR` or `--out`). See
[Running benchmarks](running-benchmarks.md) for how to do this properly if you plan to publish.
For a local CSV/Parquet rather than OpenML, follow [Testing your own data](testing-your-data.md).

## What to do when something looks odd
Read [Troubleshooting](troubleshooting.md). The most common surprises are listed there: results
being reused after a settings change, backends refusing large datasets, and `INCONCLUSIVE` on small
data.
