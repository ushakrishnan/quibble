---
title: Development
description: Local environment setup, tests, and Quibble implementation conventions.
---

## Development

## Layout

```text
quibble/                 the package
  benchmark.py           device-aware presets and isolated smoke checks
  cache.py               project-local download cache configuration
  config.py              settings: .env, environment, validation, device resolution
  env_info.py            hardware/software capture; `quibble doctor`
  datasets.py            registry, loading, splitting (holdout and repeated CV), metadata
  backends.py            Backend classes and the BACKENDS registry
  run.py                 one-run executor, sweep scheduler, saved artefacts
  compare.py             quick single-split summary
  metrics.py             weighted metrics, calibration, cluster bootstrap
  pool.py                load saved predictions; align backends on common splits
  slices.py              row traits and slice definitions
  analyze.py             analysis (intervals, calibration, slices, disagreement)
  swap.py                swap checks and verdicts
  report.py              Markdown report and figures
  generate.py            bounded synthetic tables, schema validation, saved provenance
  planner.py             explicit provider configuration and specification drafting/correction
  web.py                 loopback HTTP server, review gates, persistent benchmark jobs
  web_assets/            browser workflow, product header, and independent About page
  sample_data/            four bundled synthetic datasets, manifests, and recipes
  cli.py                 the `quibble` command and `demo`
tests/                   pytest suite
docs/                    this documentation
.github/workflows/ci.yml continuous integration
```

## Data flow

```text
config (.env/env/flags)
        |
   quibble run --sweep --> [process per run] --> run record .json + predictions .npy
                                                          |
                                       pool.py (align backends on common splits)
                                          |                       |
                                   analyze.py                 swap.py
                         (CIs, calibration, slices,      (PASS/FAIL/INCONCLUSIVE)
                          disagreement CSVs)                      |
                                          \                      /
                                           report.py --> report.md + figures
```
Analysis never retrains models. It reads saved predictions and reloads the original dataset
for traits/slices, so missing OpenML caches can require downloads. Re-analysis changes
thresholds without another training cost, but still uses compute and memory.

## Setup

Use the [local environment setup](getting-started.md#install).
The unified requirements already include pytest and matplotlib.

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
```

## Tests
The regression suite isolates environment settings and uses bundled data/mocked optional
adapters rather than downloading checkpoints. Enumerate current tests with
`python -m pytest --collect-only`; runtime depends on the machine and installed optional packages.

| File | Covers |
|---|---|
| `test_config.py` | device toggle (strict `cuda`, `auto`), validation, CV split ids, thread division, `.env` precedence |
| `test_data_and_metrics.py` | fixed splits, CV folds partition the data, metrics vs scikit-learn, integer weights equal duplicated rows, ECE bounds, degradation signs, bootstrap clustering |
| `test_backends_run.py` | determinism (HistGB, LightGBM, CatBoost), run-record contents, regression, resume, TabPFN refusal, missing package recorded as error, native categoricals with unseen values, stale-settings warning |
| `test_analysis_swap_report.py` | analysis outputs, disputed-row CSV, self-swap passes, the weak candidate fails (classification and regression), threshold inside/outside the interval, missing splits fail coverage, swap exit codes, report sections and figures, refusals in the failure matrix |
| `test_cli.py` | `--version`, help, unknown command, strict `cuda`, end-to-end sweep/resume/analyze/report |
| `test_benchmark_workflow.py` | presets, isolated bounded smoke runs, retention/leaderboard links, and participation |
| `test_config.py`, `test_history.py` | centralized cache settings, dated snapshots, evidence/link preservation |
| `test_foundation_backends.py`, `test_device_resources.py` | adapter contracts, device pinning, resource/worker failures, and mixed-device labels |
| `test_report_interpretation.py` | conditional shortlists, swap explanations, and resource-quality tradeoffs |
| `test_docs.py` | documentation links/anchors, settings, dataset inventory, CLI flags, and local setup |
| `test_local_datasets.py` | CSV/Parquet contracts, local CLI workflows, label mapping, and changed-source rejection |
| `test_generate.py`, `test_planner.py`, `test_web.py` | generator/schema/provenance, provider correction, task-filtered saved data, loopback HTTP, product/About page, and job persistence |

`tests/conftest.py` pins every `QUIBBLE_*` setting before the package is imported, so a developer's
`.env` cannot change test behaviour, and builds a shared set of CV results once per session.

Add tests with new behavior and regression tests with fixes. For example, metric-direction
tests verify that quality degradation is positive when a candidate is worse.

## Continuous integration
`.github/workflows/ci.yml` defines `pytest` and `quibble demo --cv 3x1` runs on Python
3.10, 3.11, and 3.12. Remote execution has not been verified. CI installs `.[dev]`, not the complete local
requirements; full model-library and Windows/CUDA setup validation are separate.

## Extending
**New backend:** see [Backends](backends.md#adding-a-backend).

**New dataset:** local CSV/Parquet needs only a folder and manifest
([Testing your own data](testing-your-data.md)); public OpenML entries can be added to
`OPENML` in `datasets.py` and verified ([Datasets](datasets.md)).

**New metric:** add it to `score()` in `metrics.py` and to the `HIGHER` table (and to `RELATIVE` if a
change should be reported relatively). It then works as `--metric` and as `primary` in swap
thresholds.

**New swap check:** add a step in `swap.evaluate()` that calls `add(name, observed, limit, verdict,
note)`. Use `tri()` for checks with an interval so verdicts stay three-state.

**New report section:** add it in `report.build()`; reuse saved evidence and documented
dataset loading, without fitting models during report generation.

## Conventions
- Optional dependencies are imported lazily, inside the function that needs them.
- Anything a backend cannot do raises `Refused`; unexpected model/worker failures become
  `error` records so other cells can continue. Configuration, archive, and output-storage
  failures remain explicit command failures.
- Results must remain readable by later versions: add fields, do not rename or repurpose them.
- Label point estimates, single-machine timings, and uncorrected exploratory maxima explicitly.

## Releasing (not done yet)
Source repository: [ushakrishnan/quibble](https://github.com/ushakrishnan/quibble).
Product navigation links to GitHub. Blog and Docs are also served locally.
The single packaged `web_assets/header.html` template is used by the workbench, About,
blog, and documentation reader. Update links in this shared template, not individual
pages. Information links from the workbench
open separately so they do not replace unsaved workflow state.

The docs folder has a GitHub landing page (`docs/README.md`), an index, relative guide
links, and self-contained installation instructions. It needs no site generator for
GitHub browsing; the static export and Pages workflow use the same content and renderer.
The local HTML reader provides grouped sidebar navigation, heading anchors and an
on-page index, breadcrumbs, and sequential guide links. `DOC_NAV` in `quibble/web.py`
defines the reading order; tests require every public Markdown guide to appear once.
The layout is responsive and uses native collapsible navigation without a JavaScript
dependency. A static export alone is not a successful live deployment.

Before initializing/staging a release, review the actual staged files. `.gitignore`
excludes environments, environment variants, build/egg-info, caches, local data/results,
and private reference folders; `.env.example` and packaged synthetic samples remain
included. `.dockerignore` excludes local/private material from `COPY . .`. Neither file
removes anything already committed, and the container build remains unverified.

The distribution name is `quibble-ml`. A release would: bump `version` in `pyproject.toml` and
`quibble/__init__.py`, update the docs' trust-level table, build with `python -m build`, and upload with
`twine`. The MIT copyright attribution is Quibble contributors.

## GitHub Pages deployment

The workflow in `.github/workflows/pages.yml` builds and deploys Docs and Blog on pushes
to `main`, or through **Actions > Deploy documentation > Run workflow**. The build reuses
the local reader's shared header, separate blog/docs navigation, and allowlisted assets.
It rewrites Markdown links to HTML and applies the Pages base path, including `/quibble`.
Code references link to the source repository rather than exposing arbitrary local files.
No environment files, private references, user datasets, model weights, or experiment
results are exported. The curated blog evidence remains explicitly limited smoke evidence.

Before the first deployment:

1. Publish the reviewed source branch to `https://github.com/ushakrishnan/quibble.git`.
2. Set repository **Settings > Pages > Source** to **GitHub Actions**.
3. Run **Deploy documentation** after enabling Pages, or push a new commit to `main`.
4. Inspect the workflow result and deployment URL. The expected project URL is
   `https://ushakrishnan.github.io/quibble/`; only a successful deployment establishes
   that it is live.

The deploy job uses the `github-pages` environment and the standard Pages/OIDC
permissions; no personal access token or provider credentials belong in the repository.
Remote deployment and CI are unverified until their actual workflow runs succeed.

To build and preview locally from the checkout:

```powershell
python -m pip install -e ".[site]"
python -m quibble.site --out build\quibble --base-path /quibble
python -m http.server 8877 --bind 127.0.0.1 --directory build
```

Open `http://127.0.0.1:8877/quibble/`. The preview directory matches the URL prefix;
the deployment workflow instead uploads `build/pages` as the site's artifact root.
The exporter requires a new or empty output directory and never deletes existing
files. Use a fresh output name for another build.
For a root-domain site, build with `--base-path ""` and serve the output directory itself.

About is the landing page. **Workbench** on the static site is a local-installation
page, not an interactive model service. Users run `python -m quibble web --open` on
their own machine; GitHub Pages cannot host the Python server or receive user data.
