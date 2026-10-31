---
title: Troubleshooting
description: Resolve local installation, model execution, and report problems.
---

## Troubleshooting

Find the first stage that failed: installation, data review, fitting, analysis,
or reporting. An explicit refusal is different from a crash, and missing evidence
is different from a bad score. Keep the failed record and log before retrying.

## Installation and environment
**`pip install quibble` installs the wrong thing.**
The name `quibble` on PyPI belongs to an unrelated project. This project's distribution name is
`quibble-ml`, and it is not published yet. From this checkout run
`uv sync --locked --extra web --extra data`, adding any wanted model extras,
then activate this repository's `.venv`. There is no separate requirements file.

**`quibble-ml: command not found`.** Activate this repository's `.venv` and use
`python -m quibble_ml ...` from the repository root. Check `python -c "import sys; print(sys.executable)"`
to confirm you are not running another project's interpreter.
The import is `quibble_ml`, not `quibble`. Re-register the checkout with
`uv sync --locked --extra web --extra data` after updating from an older revision.
See [Migration](getting-started.md#migrating-from-the-old-import-and-command);
this project no longer installs the old command or import name.

**`No module named 'resource'` (or errors on native Windows).** Current Quibble uses `psutil`
on Windows and imports `resource` only on Unix. Run from the current source with the local
`.venv`; sync from the lock to ensure Windows `psutil` is available.

**Installing AutoGluon or TabPFN broke CUDA or scikit-learn.** They pin their own dependency
versions. Sync all selected extras together from the lock, run `uv pip check`,
then `python -m quibble_ml doctor`. Retain `--extra windows-cuda` for native Windows
CUDA wheels; a compatible NVIDIA driver is still required. Use `--dry-run` first:
sync removes unselected libraries. A lock conflict must be resolved, not bypassed
with an untracked install.
Never install packages while a sweep is running: it changes libraries under the
running processes and produces spurious errors.

**Mitra refuses a version that appears in `torch.__version__`.** The runtime also
requires valid installed-distribution metadata and the pinned Git source.
A damaged Torch installation can import while lacking a verifiable package
version. Do not edit metadata or weaken the guard. Use a healthy isolated
environment resolved from the lock with all intended extras, then rerun
`uv pip check` and doctor before downloading or fitting.

## Configuration
**`config error: QUIBBLE_DEVICE=cuda but torch.cuda is not available`.** Run `quibble-ml doctor`. Either
install a CUDA build of PyTorch (see [GPU and Docker](gpu-and-docker.md)) or set `QUIBBLE_DEVICE=cpu`.
This strictness is deliberate.

**`config error: QUIBBLE_CV must be 'holdout' or like '5x2'`.** Use `holdout` or `<folds>x<repeats>`.

**`WARNING: N existing result(s) ... different settings ... being SKIPPED`.** Finished runs are
already present, but recorded execution settings differ. Use a separate `--out` per
configuration. `--force` can rerun records only when recipe compatibility checks
permit it; it does not bypass algorithm, preprocessing or protocol-family guards.

**My `.env` change had no effect.** A real environment variable of the same name wins over `.env`, and
a command-line flag wins over both. In PowerShell, check
`Get-ChildItem Env:QUIBBLE_*`. Keep only one occurrence of each key in `.env`:
the loader preserves the first value rather than allowing later duplicates to override it.

## Running
**A backend shows `error: ModuleNotFoundError`.** Check that the local `.venv` is active and
select its documented model extra with `uv sync --locked`, retaining other wanted
extras. Then check `uv pip check` and retry
with `--force`: saved import failures are cached too. Other backends are unaffected.

**A backend shows `refused`.** Expected: it declined the task. For TabPFN this means too many rows,
features or classes (see [Backends](backends.md)). It is a result, not a bug, and the report's
failure matrix lists it.

**`error: timeout after 900s`.** Raise `QUIBBLE_TIMEOUT_S` (or `--timeout`), or set it to 0. Remember
to `--force` the affected cells.

**TabPFN fails to download weights.** The first run needs network access and may need a Hugging Face
account or token, depending on the release. Pre-download once on a networked machine.

**CatBoost on GPU fails.** Try `--device cpu` for CatBoost to confirm the model itself works, then
check your CUDA and driver versions with `quibble-ml doctor`.

**A model runs out of RAM/VRAM, or its worker exits.** Read the report's detailed failure
table and the run record's `error_kind`, `device_attempted`, and `worker_exit_code`.
Quibble does not silently retry on CPU. Free resources, reduce model size/ensemble count,
or intentionally select CPU, then rerun the affected cells with `--force`.
Use an isolated `benchmark --smoke` to check setup, not as evidence of full-data capacity.

**`Existing test rows differ from this split`.** Saved prediction arrays share test targets
and indices across backends. Match the original CV/split settings or use an isolated `--out`.
`--force` does not bypass this guard.

**The report says mixed CPU/GPU, or retains an old device after `.env` changes.** New CUDA
sweeps exclude incompatible methods rather than run them on CPU. Historical and cached fits retain their original labels;
changing `.env` does not retrain them. Rerun selected fits or use separate roots for
controlled CPU-versus-CUDA experiments.

**Timings fluctuate between runs.** Check that `jobs` is 1 and nothing else is using the machine. The
report flags results from parallel sweeps.

**The sweep is slow.** Each run is a separate process, giving it a separate memory high-water mark. Use
`--jobs N` for exploratory CPU work, a smaller `--cv` (such as `3x1`) for a first pass, and remember
AutoGluon spends its whole time limit on every run.

## Datasets

**My CSV does not appear in `datasets --list`.** Put it in a named folder with `dataset.json`
under `QUIBBLE_DATA_DIR` (default project `data`). The manifest declares target/task.
See [Testing your own data](testing-your-data.md). Built-in/OpenML names are reserved.

**`Local dataset source differs from saved results`.** The input file or manifest changed.
Restore the exact original input or use a new `--out`; `--force` does not bypass the guard.
Reports need the original features to regenerate slices.

**`quibble-ml datasets --verify` prints `ERROR` for an OpenML dataset.** Usually network access or a wrong
id. Offline, use `breast_cancer`, `wine` and `diabetes_reg`.
Verification now returns 1 for any error or name mismatch.

**`NAME MISMATCH`.** The id points at a different dataset than the registry expects. Fix the id in
`quibble_ml/datasets.py`.

## Analysis
**`no results for <dataset>`.** The dataset folder has no run records under the output directory you
pointed at. Check `--out`.

**`no splits completed by all backends`.** Either no backend has usable predictions, or the
usable backends have no common successful split. Entirely failed backends are otherwise
excluded from pooled analysis. Run `quibble-ml report` and look at participation/failures, or restrict
with `--backends`.

**`(skipping <backend>: no usable runs)` / `incumbent has no usable runs`.** The backend has run
records but no usable predictions, possibly because every record failed/refused. This also
happens with results produced by an early version, before
test indices were saved. Re-run with `--force`.

**Everything is `INCONCLUSIVE`.** Paired intervals straddle your chosen limits. Check whether
the limits are meaningful and collect more representative held-out rows where possible.
Repeated CV probes split sensitivity but does not add independent rows; other datasets
test generality, not this dataset's uncertainty. Do not turn uncertainty into a PASS.

**A large dataset runs out of memory during analysis.** The bootstrap holds a `QUIBBLE_BOOT x n_rows`
matrix. Lower `QUIBBLE_BOOT` (for example `QUIBBLE_BOOT=200`).

**The report has no figures.** Select the `report` or `web` extra in your locked sync
command, retaining other wanted extras, then regenerate the report.

## Browser workbench

**The header or buttons still look old.** Restart the local server after code changes
and refresh the browser. Confirm you are using the intended port and environment.

**GitHub or Docs still appears disabled.** Restart and refresh to load the current
header. GitHub now links to the source repository; Blog, About, and Docs also work locally.
You can read `docs/README.md` directly in the checkout.

**The public Workbench button does not start experiments.** GitHub Pages serves static
documentation only. Follow its installation guide and start your local Python server.
The public site never uploads your data or executes models.

**Generate preview is disabled.** An unchanged saved recipe already has rows. Use
**Discard changes and continue** for the original, or change the name or recipe to
enable a variant preview. After a successful preview, generation stays disabled until
an edit invalidates that preview. A blank design first needs a proposed specification.

**Save and continue is disabled.** Preview the current specification, then confirm
the review checkbox. Every edit requires another preview. Saving uses a new unused
name and opens configuration; it does not start a run. Discarding a variant needs no
new preview because it uses the original saved rows unchanged.

**The provider returned HTTP 200 but the design failed.** Transport success is not
schema validation. Quibble can make one correction request, then reports the exact
remaining error. Review provider compatibility and the recipe rather than assuming
a successful draft; neither invalid response is silently saved.

**A dataset is missing from the selector.** Check the outcome filter: classification
includes binary and multiclass, while regression is separate. A local folder with the
same name overrides a bundled example. For a new file, choose **Upload CSV / Parquet**
in the local workbench, review its column roles, validate, and confirm saving.
The [local-data guide](testing-your-data.md) also supports manually prepared manifests.
Unavailable or changed sources appear as dataset errors. Correct the source or
import a new reviewed dataset; saved experiment reports remain accessible.

## Docker
**Permission denied writing `results/`.** The container runs as UID 1000. `sudo chown -R 1000:1000
results .cache` on the host.

**`could not select device driver "nvidia"`.** Install the NVIDIA Container Toolkit (or enable GPU
support in Docker Desktop's WSL2 backend), then retry the `gpu` service.

## Still stuck
Run `quibble-ml doctor`, re-run the failing command with a single backend and dataset, and look at the
run record in `results/<dataset>/` (the `error` field has the exception). The test suite
(`pytest`) is a quick way to tell whether the installation itself is healthy.
