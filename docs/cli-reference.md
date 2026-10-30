---
title: CLI reference
description: Commands and options for running models, analyzing results, and evaluating swaps.
---

## CLI reference

All commands are `quibble <command> [options]`. `python -m quibble <command>` is equivalent. Run
`quibble <command> --help` for generated options, except `doctor`, which prints diagnostics.
Square-bracketed options and placeholder names below are syntax illustrations, not commands
to paste verbatim. See [getting started](getting-started.md) for executable examples.

Settings can come from `.env`, environment variables, or flags; flags win. See
[Configuration](configuration.md).

## `quibble run`
Run one backend on one dataset, or a sweep of many.

Single run:
```bash
quibble run --backend lightgbm --dataset credit-g --model-seed 0 --split-seed 0
```
Sweep:
```bash
quibble run --sweep --backends histgb,lightgbm --datasets credit-g,phoneme --cv 5x2 --model-seeds 0,1,2
quibble run --sweep --backends all --datasets breast_cancer       # 'all' excludes the internal 'weak'
```

| Flag | Meaning |
|---|---|
| `--backend`, `--dataset` | Single-run target. |
| `--model-seed N` | Model seed for a single run (default 0). |
| `--sweep` | Run the cross product of `--backends` x `--datasets` x model seeds x split ids. |
| `--backends`, `--datasets` | Comma lists; `--backends all` means every non-internal backend. |
| `--model-seeds 0,1,2` | Model seeds for a sweep (env `QUIBBLE_MODEL_SEEDS`). |
| `--cv holdout\|KxR` | Split scheme. Sweeps run every split id the scheme defines. |
| `--split-seed N` | In holdout mode: the split seed. In CV mode (single run): the split id. |
| `--cv-seed N` | Seed for the CV fold assignment (default 0). |
| `--device cpu\|cuda\|auto` | See [Configuration](configuration.md). `cuda` fails if CUDA is unavailable. |
| `--preprocess ordinal\|native` | Categorical handling rule. |
| `--jobs N` | Parallel CPU runs in a sweep. Use 1 for timings you will publish. |
| `--threads N` | Threads per run (0 = all, or cores/jobs when jobs > 1). |
| `--timeout S` | Per-run timeout in sweeps (0 = none). A timeout is recorded as an error result. |
| `--force` | Re-run cells that already have results. |
| `--retry-failed` | Re-run only cells whose saved result is an error (timeouts, out-of-memory, missing packages). Refusals are kept. |
| `--keep-models` | Retain fitted tree/AutoGluon models outside timed measurements; foundation estimators are not serialized. |
| `--out DIR` | Output directory (default `results`). |

Behaviour worth knowing:
- Finished runs are skipped, **including saved errors** (use `--retry-failed` to rerun only those). If
  their saved device, preprocessing mode, CV scheme, model-specific settings, dataset fingerprint or
  installed library versions differ from the current settings, quibble prints a **WARNING** (they
  are still skipped; use `--force` or a different `--out`).
- On `cuda`, GPU-capable backends run one at a time even when `--jobs` > 1.
- A backend refusing a dataset, or crashing, produces a result file with `refused` or `error`; the
  sweep carries on.

## `quibble analyze`
Confidence intervals, calibration, slices and disagreement for one dataset. Needs results from at
least two backends for the disagreement part, and CV results for meaningful intervals.

```bash
quibble analyze --dataset credit-g [--backends a,b,c] [--model-seed 0] [--metric auc] [--min-slice-n 30]
```
Writes to `<out>/<dataset>/analysis/`. Backends are aligned on the splits they all completed; splits
missing for any usable backend are dropped and listed; entirely unusable backends are
excluded. See [Analysis](analysis.md).

## `quibble swap`
```bash
quibble swap --incumbent histgb --candidate lightgbm --dataset credit-g [--model-seed 0] [--thresholds limits.json] [--require-thresholds]
```
Prints a table of checks and an overall verdict. **Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE**
(also 1 if the incumbent has no usable runs or `--require-thresholds` is set without `--thresholds`).
Without `--thresholds` the output identifies the
[packaged defaults](../quibble/swap_thresholds.json); every effective setting is printed.
These limits need application-specific review. Export an editable copy without training:

```powershell
quibble swap --export-thresholds limits.json
```

Export refuses to overwrite an existing file. `--require-thresholds` rejects missing or empty
policies, not just missing filenames. See [Swap checks](swap.md).
Partial policies remain allowed; their inherited packaged defaults are listed explicitly.

## `quibble gate`
```bash
quibble gate --candidate lightgbm --dataset credit-g [--baseline linear] [--min-gain 0.01] [--model-seed 0] [--metric auc] [--out DIR]
```
Does the candidate beat a simple baseline (`linear`, or `dummy`) by at least `--min-gain`, with the
whole interval above that margin? Run both models first (for example
`quibble run --sweep --backends linear,lightgbm --datasets credit-g --cv 5x2`). **Exit code: 0 = PASS,
1 = FAIL (not confidently better), 2 = INCONCLUSIVE.** "Nothing here is worth using" is a valid
outcome. Incomplete candidate or baseline coverage produces FAIL even when quality on common
successful splits passes. Required split IDs include saved baseline records and matching
invocation cells, not guessed contiguous IDs. The CLI lists missing/error/refusal/artifact
reasons; the Python result separates `quality_verdict` from `overall` and includes `coverage`.
See [Swap checks](swap.md) for how the interval is built.

## `quibble report`
```bash
quibble report [--datasets a,b] [--swap histgb:lightgbm]... [--thresholds limits.json] [--require-thresholds] [--report-dir DIR]
```
Writes `<out>/report/report.md` and figures. `--swap` can be repeated and is evaluated for every
dataset. Without `--datasets`, every dataset folder with results is included.
Reports show the effective policy and whether defaults or custom settings were used.
`--require-thresholds` requires a non-empty explicit policy before report generation.
See [Reports](reports.md).

## `quibble compare`
Quick single-split summary: mean metrics per backend, refusals and errors, and pairwise disagreement
for one split id. Useful for a fast look; use `analyze` for anything statistical.
```bash
quibble compare --dataset credit-g [--split-seed 0]
```

## `quibble datasets`
```bash
quibble datasets --list
quibble datasets --verify [--names credit-g,wine]
```
`--list` lists built-in, OpenML, and discovered local names without loading tables.
Verification loads each selected dataset (downloading OpenML entries if needed), prints rows, features, task, categorical share, missing share, majority-class
share and whether the OpenML name matches what the registry expects (`NAME MISMATCH` means the id
is wrong). Writes `datasets_meta.json` in the **current directory**. Network errors are reported per
dataset. Verification exits 1 if any dataset fails or a name mismatches, otherwise 0.
Without `--names`, all registered and discovered local datasets are verified.
See [Datasets](datasets.md) and [Testing your own data](testing-your-data.md).

Local names work with run, benchmark, analyze, swap, compare, and report:

```powershell
python -m quibble benchmark --datasets payments --backends histgb,lightgbm --device cpu --smoke
```

Create `data\payments\dataset.json` and the input table first. No direct file-path/target
CLI syntax is implemented; the manifest declares target/task and column rules.

## `quibble doctor`
Prints Python, platform, CPU, RAM, git hash, installed package versions, and torch/CUDA/VRAM. Warns
when free VRAM is low (another process may be holding the GPU).

## `quibble demo`
```bash
quibble demo [--out results] [--cv 3x2] [--jobs 2] [--backends histgb,lightgbm,weak] [--force] [--retry-failed]
```
End-to-end demonstration on `breast_cancer`, `wine`, and `diabetes_reg`; forces
`QUIBBLE_DEVICE=cpu` and model seed 0. The default backends run offline without PyTorch.
`--backends` accepts a nonempty comma-separated list of registered names; optional backends
need their packages in the chosen interpreter and may download weights.

```powershell
python -m quibble demo --backends histgb,lightgbm,tabfm,kumo_tabular --jobs 1
```

The report checks swaps from HistGB to the other selected backends, or from the first
selected backend if HistGB is absent. The weak-candidate sanity message only appears when
both HistGB and Weak are selected. A single backend produces a report without swap checks.

Saved results, including errors, are skipped. `--retry-failed` reruns only the saved errors (for
example cached import failures after switching environments); `--force` reruns every selected cell. Prefer a separate
`--out` only when intentionally isolating a comparison. By default, demo uses the same
`QUIBBLE_OUT_DIR` (or `results`) as run and report. Existing backend results in that directory
also appear in its report. Dated snapshots preserve overwritten outputs and regenerated
reports; `<out>/report/history.md` links to them.

## `quibble benchmark`

Device-aware preset comparison with report generation:

```powershell
python -m quibble benchmark --preset foundations --smoke
python -m quibble benchmark --preset foundations --device cuda --force
python -m quibble benchmark --backends histgb,autogluon --cv holdout --keep-models
```

`--retry-failed` reruns only cells whose saved result is an error.

Presets: `trees` selects HistGB/LightGBM/CatBoost; `foundations` (default) selects
HistGB/LightGBM/TabFM/KumoTabular; `all` selects all public backends. `--backends` overrides
the preset. `--datasets` defaults to `wine`, `--cv` to `3x2`, and `--model-seeds` to `0`.
Device and parallelism follow `.env` unless overridden with `--device`, `--jobs`, or `--threads`.
`--out`, `--force`, and `--keep-models` work as with run.

Full runs accept `--thresholds limits.json` for custom swap limits and
`--require-thresholds` to require a non-empty explicit policy. Invalid policies are rejected
before training. The generated report lists all effective settings and their source.
Without a custom policy, the [packaged defaults](../quibble/swap_thresholds.json) apply.

`--smoke` fixes holdout/seed 0, caps training/test rows at 128/32, and creates a unique
`<out>/smoke/<id>/` directory. Smoke reports interpret the bounded setup measurements but
do not provide deployment recommendations or swap verdicts. Downloads and authentication may still be needed.
Smoke rejects threshold options because no swap checks are run.
Preset reports exclude incompatible cached backends, show participation and failure
status, and include AutoGluon validation leaderboard links when available.

## `quibble web`

Start the localhost experiment workbench for scenario planning, reviewed synthetic data,
existing datasets, benchmark jobs, and reports:

```powershell
quibble web --open
quibble web --login --device-code --port 8876
```

Flags: `--port` (default 8765), `--open` (open browser), `--login` (explicit Azure CLI
login before startup), `--device-code` (use device-code login).
Only `127.0.0.1` is bound. See the [complete workbench guide](experiment-workbench.md)
for configuration, consent, generation assumptions, stored files, and local-only limits.
The header's About link opens a separate page at `/about`, including the running
version in its content and privacy/validation boundaries. All pages share navigation;
Blog and Docs work locally, and GitHub links to the source repository.
Browser saving continues directly to configuration;
starting the experiment remains a separate explicit action.

## `quibble generate`

Print an editable local example or generate a dataset from a reviewed specification:

```powershell
quibble generate --example
quibble generate --spec reviewed-spec.json
```

Choose exactly one of `--example` or `--spec`. Generation is deterministic for the
specification and software environment, never executes supplied code, and never
overwrites a dataset folder. See the [specification guide](experiment-workbench.md#specification-and-generation-mathematics).

## `quibble auth`

Check the optional planner's configuration or authenticate using Entra ID:

```powershell
quibble auth
quibble auth --device-code
quibble auth --check
```

`--check` acquires an AAD token without login; `--device-code` selects explicit device-code
login for Azure CLI credentials. Key mode checks only configuration presence.
Neither establishes live provider access. See [provider setup](experiment-workbench.md#configure-your-planning-provider).

## Exit codes
| Command | Codes |
|---|---|
| `swap` | 0 PASS, 1 FAIL or unusable incumbent, 2 INCONCLUSIVE |
| `gate` | 0 PASS, 1 FAIL, 2 INCONCLUSIVE |
| `benchmark` | 0 all requested fits completed; 1 failed, refused, missing or incompatible cells; 2 configuration/usage error |
| `datasets --verify` | 0 all selected entries loaded and names matched; 1 any load failure or name mismatch |
| others | 0 success; non-zero on configuration or usage errors |
| unknown command | 2 |

`run` and `demo` can return 0 after recording model-level errors; inspect records/report
participation. Use `benchmark` when failed or missing requested fits must produce a non-zero
aggregate status. Argparse usage errors also return 2.
