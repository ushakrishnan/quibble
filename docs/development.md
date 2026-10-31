---
title: Development
description: Local environment setup, tests, and Quibble implementation conventions.
---

## Development

Quibble's core contract is a fair, inspectable comparison, not a shared method
name. When adding a model or data path, preserve held-out answers, class-column
order, source identity, real costs and explicit failures. A mocked adapter test
does not replace a real fit.

## Layout

```text
quibble_ml/                 the package
  benchmark.py           device-aware presets and isolated smoke checks
  cache.py               project-local download cache configuration
  config.py              settings: .env, environment, validation, device resolution
  env_info.py            hardware/software capture; `quibble-ml doctor`
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
  cli.py                 the `quibble-ml` command and `demo`
tests/                   pytest suite
docs/                    this documentation
.github/workflows/ci.yml continuous integration
```

## Data flow

```text
config (.env/env/flags)
        |
   quibble-ml run --sweep --> [process per run] --> run record .json + predictions .npy
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
The default `dev` group includes pytest; the web/data extras provide the other
dependencies used by tests. Optional model adapters are mocked in routine CI.

```powershell
& .\.venv\Scripts\Activate.ps1
uv sync --locked --extra web --extra data
uv pip check
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
`.github/workflows/ci.yml` defines `pytest`, `quibble-ml demo --cv 3x1` and two swap exit-code checks
on Python 3.10 to 3.13 (Linux) and Python 3.13 (Windows). Remote execution has not been verified; the
same steps passed locally on Linux with Python 3.12 before the uv migration.
CI now uses `uv sync --locked --extra web --extra data`, which installs the default
dev group and everything tests import, but not heavy model libraries.
Full model-library and Windows/CUDA validation are separate.

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

## Workbench navigation checks

The existing Node-backed pytest tests exercise route parsing, review/approval
guards, shared actions, single-flight monitoring and stale asynchronous responses.
They require Node; skipped browser-asset tests do not establish working navigation.

```powershell
python -m pytest tests\test_web_assets.py tests\test_web.py tests\test_web_palette.py
```

Also exercise the actual browser against an unused localhost port with isolated
`QUIBBLE_OUT_DIR`, `QUIBBLE_DATA_DIR` and `QUIBBLE_CACHE_DIR` locations. Check a
reviewed bundled dataset, local upload, changed approvals, Back/Forward, legacy
job bookmarks and an old report while a new bounded CPU experiment runs.
Verify keyboard drawer containment/restoration and no page-wide overflow at
320, 375, 768 and 1280 CSS pixels. Do not send provider requests or download
foundation checkpoints as a side effect of navigation testing.

The [workbench guide](experiment-workbench.md#one-review-flow-three-experiment-steps)
describes the shipped routes and scientific safeguards.

### Appearance regression checks

`theme.js` validates independent `quibble-theme`, `quibble-accent`,
`quibble-contrast`, and `quibble-rail` preferences before the stylesheet loads.
Use native labeled selects in Settings and the shared header, not custom radio
widgets. Keep accent overrides limited to the seven primary/focus tokens;
semantic verdict tokens must not vary with accent. The six inline navigation
icons are original CC0-1.0 artwork; preserve their HTML notice and `currentColor`
styling. Do not introduce remote appearance assets.

The palette suite resolves base, accent, and contrast overrides across 24
combinations. It checks ordinary, muted, selected, link, and button text at
4.5:1, and boundaries/focus at 3:1. Focus uses a surface-colored separator plus
an accent outline so it remains distinguishable from a primary button.
Test OS forced colors separately using system colors rather than RGB ratios.

For a no-training UI check, use the existing environment without syncing it:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
# Create a unique, short temporary directory owned by this validation run first.
& .\.venv\Scripts\python.exe -B -m pytest tests\test_web_assets.py tests\test_web_palette.py tests\test_web.py tests\test_uploads.py tests\test_device_resources.py tests\test_site.py tests\test_docs.py -k "not real_generated_dataset_benchmark_and_report and not upload_review_confirm_compare_and_reopen" -p no:cacheprovider --basetemp=C:\owned-ui-check\pytest
```

In an isolated browser/server origin, check all 24 appearance combinations,
multiple controls, OS mode changes, reload, cross-tab updates, cleared/invalid
storage and denied reads/writes. Verify the desktop rail's focus/hover labels,
native disclosures and Escape restoration; retain the mobile drawer's focus
trap, inert background and focus restoration. Check 320/375/768/1280 widths,
200% zoom (or record a 640-by-450 zoom-equivalent limitation), forced colors,
reduced motion, and print. Check shared About/Docs/Blog and a `/quibble` static
export without live API or external appearance requests. Do not use the user's
server, run fits or download weights for this appearance validation. Stop only
owned process IDs and remove only the run's owned temporary directory.

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
on-page index, breadcrumbs, and sequential guide links. `DOC_NAV` in `quibble_ml/web.py`
defines the reading order; tests require every public Markdown guide to appear once.
The layout is responsive and uses native collapsible navigation without a JavaScript
dependency. A static export alone is not a successful live deployment.

Before initializing/staging a release, review the actual staged files. `.gitignore`
excludes environments, environment variants, build/egg-info, caches, local data/results,
and private reference folders; `.env.example` and packaged synthetic samples remain
included. `.dockerignore` excludes local/private material from `COPY . .`. Neither file
removes anything already committed, and the container build remains unverified.

The distribution name is `quibble-ml`. A release would: bump `version` in `pyproject.toml` and
`quibble_ml/__init__.py`, update the docs' trust-level table, build with `uv build`, and upload with
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
uv sync --locked --no-dev --extra site
python -m quibble_ml.site --out build\quibble --base-path /quibble
python -m http.server 8877 --bind 127.0.0.1 --directory build
```

Open `http://127.0.0.1:8877/quibble/`. The preview directory matches the URL prefix;
the deployment workflow instead uploads `build/pages` as the site's artifact root.
The exporter requires a new or empty output directory and never deletes existing
files. Use a fresh output name for another build.
For a root-domain site, build with `--base-path ""` and serve the output directory itself.

About is the landing page. **Workbench** on the static site is a local-installation
page, not an interactive model service. Users run `python -m quibble_ml web --open` on
their own machine; GitHub Pages cannot host the Python server or receive user data.
