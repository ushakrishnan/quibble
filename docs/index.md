---
title: Quibble documentation
description: Guides, supported workflows, validation status, and future design notes.
---

## Quibble documentation

*Before you swap a tabular model, see exactly what changes.*

For a self-contained starting point, use the [documentation home](README.md),
[About Quibble](about.md), and [installation guide](getting-started.md).
These guides are browsable Markdown in [the source repository](https://github.com/ushakrishnan/quibble)
and can be exported as a [static Pages site](development.md#github-pages-deployment).

quibble answers one question for tabular machine learning: **if I swap one backend for another,
what actually changes?** It runs several backends (AutoGluon, TabPFN, TabFM, KumoTabular, LightGBM, CatBoost,
scikit-learn's HistGradientBoosting, and any you add) on identical data and identical splits, then
reports quality with confidence intervals, cost, refusals, calibration, slices, and the individual
rows on which the models disagree. `quibble-ml swap` condenses that into a PASS / FAIL / INCONCLUSIVE
verdict that can gate a CI pipeline.

It is deliberately **not** a universal wrapper around tabular models. scikit-learn's
`fit`/`predict` is already the common interface. quibble's value is in the *verification* around a
swap: did anything break, where, and how sure are we?

## Where to start

The localhost browser workbench and CLI are complementary interfaces to the same
evaluation pipeline. Use the UI for guided review and the CLI for scripts and settings
outside the browser's bounded controls.

| If you want to... | Read |
|---|---|
| Understand each capability through everyday examples and honest limits | [What Quibble does](what-quibble-does.md) |
| Read the project story and engineering lessons | [Predicting from Tables](blog/README.md) |
| Understand the product, privacy, and browser navigation | [About Quibble](about.md) |
| See it work in one minute | [Getting started](getting-started.md) |
| Use device-aware presets or isolated smoke checks | [Benchmark command](cli-reference.md#quibble-ml-benchmark) |
| Compare your own CSV or Parquet table | [Testing your own data](testing-your-data.md) |
| Design synthetic data and run comparisons in a browser | [Experiment workbench](experiment-workbench.md) |
| Understand the vocabulary and the fairness rules | [Concepts](concepts.md) |
| Look up a command or flag | [CLI reference](cli-reference.md) |
| Set up `.env` and choose CPU or GPU | [Configuration](configuration.md) |
| Know what each backend does and its limits | [Backends](backends.md) |
| Find dataset/model sources, download triggers, cache paths, or add datasets | [Datasets](datasets.md) |
| Plan a benchmark you intend to publish | [Running benchmarks](running-benchmarks.md) |
| Interpret the analysis output | [Analysis](analysis.md) |
| Gate a model swap in CI | [Swap checks](swap.md) |
| Understand the report | [Reports](reports.md) |
| Find a file or a JSON field | [Results format](results-format.md) |
| Set up WSL, CUDA or Docker | [GPU and Docker](gpu-and-docker.md) |
| Fix something that broke | [Troubleshooting](troubleshooting.md) |
| Contribute a backend or run the tests | [Development](development.md) |
| Review validation boundaries and planned work | [Limitations](limitations.md), [Project plan](project-plan.md) |
| Understand the project scope and motivation | [Project rationale](rationale.md) |

## Trust levels (what has actually been run)

| Area | Status |
|---|---|
| HistGB, LightGBM, CatBoost on CPU | Automated fits, classification/regression checks, and retained-model reload tests; demo uses HistGB/LightGBM/Weak |
| AutoGluon on CPU | Manual classification/regression smoke checks; retained Wine predictor reloaded and matched saved predictions |
| TabFM and KumoTabular on CPU | Small real classification/regression smoke checks plus mocked adapter contracts |
| Analysis, swap, report, CLI, caches/history | Automated regression tests; run `python -m pytest` for current results |
| Local CSV/Parquet | Automated input/provenance checks plus real CSV classification and Parquet regression CLI workflows |
| TabPFN | Real offline Wine classification and diabetes regression holdouts on CPU/CUDA with TabPFN9.2.0/v3.5, checked checkpoint hash, rows, class order, finite predictions and reports; experimental beyond this scope. An empty-cache, network-denied run produced an error and no predictions. |
| TabICLv2 | Pinned experimental adapter; real offline Wine classification and diabetes regression holdouts on CPU and CUDA, with checked shapes/class order/checkpoint SHA-256 and reports. CPU comparison includes HistGB. Broader coverage remains unverified. |
| Mitra-v2 | Pinned experimental CUDA adapter; real offline Wine zero-shot classification and diabetes distributional regression with prediction-time adaptation. Hashes, task APIs, shapes and inclusive costs checked; seed control and broader coverage remain unverified. |
| Tuned tree tier | Bounded CPU HistGB, LightGBM and CatBoost campaigns; real defaults/tuned browser workflows and separate-root cross-tier reports. Tiny tuned campaigns did not establish improvement. |
| Relational baseline | Synthetic point-in-time/cache/leakage proofs, repeated-entity grouping and real four-table browser review/run; permissive sampled `rel-arxiv/author-publication` pilot with official chronological boundaries, not a full benchmark. |
| Browser appearance | Real light/dark, six-accent and high-contrast checks; keyboard/mobile navigation, persisted settings and static exports. Appearance does not change scientific verdicts. |
| Browser evidence/verification/comparison | Complete report navigation and printing, explicit allowlisted diagnostics, separate saved-policy checks, and descriptive two-job compatibility. Real swap FAIL, baseline INCONCLUSIVE and matched comparison were exercised; no automatic paired comparison verdict. |
| Native Kumo Relational | Standalone direct SDM0.2.1 CPU/CUDA synthetic three-table probe with 32 context/16 query entities, checked task checkpoint hashes and time-filtered history. No endpoint/data transfer; CUDA graph joins used a visible CPU fallback. Registered Quibble manifest/worker/browser integration remains unimplemented. |
| Linear and dummy baselines, group/temporal splits, data flags, `quibble-ml gate` | Automated tests, including planted-signal, noise, leakage and grouped-bootstrap cases |
| OpenML downloads | All 14 loaded and reloaded offline on Windows, 2026-10-30; full HistGB/Weak holdouts covered `credit-g` and `abalone` only. See [exact scope and caveats](datasets.md#verification-on-2026-10-30). |
| CUDA detection | Confirmed locally with CUDA-enabled PyTorch on an RTX 5070 Ti |
| TabFM and KumoTabular on CUDA | Successful local Wine classification records, including isolated smoke runs; not full GPU/task/dataset coverage |
| CatBoost on CUDA | Successful local classification smoke run with exclusion/report checks; not full GPU/task/dataset coverage |
| TabPFN and AutoGluon on CUDA | TabPFN Wine/diabetes holdouts passed with the specified v3.5 checkpoint; broader coverage unverified. AutoGluon is excluded from new GPU-only runs. |
| Docker files | **Not tested** |
| GitHub Actions workflow | Definition targets Python 3.10 to 3.13 (Linux) and 3.13 (Windows); remote execution remains unverified. The same steps pass locally on Linux, Python 3.12 |
| Hosted-service backend | Not implemented |
| Synthetic generator and localhost workbench | Automated deterministic/schema/provenance/HTTP checks, real generated classification smoke/regression full comparisons, and browser checks for data selection, variants, save/discard navigation |
| Configurable planning providers and AAD/key/local auth | Mocked contracts including one validation-guided correction; a live HTTP response alone does not establish a valid generated specification |

Unverified paths require targeted validation before their outputs support published claims.

Local success is not a guarantee of sufficient RAM/VRAM or support for every library version.
Model-level resource failures are recorded without silent device fallback; smoke success is
not a completed full-data benchmark.

## Design proposals, not shipped features

- [Recommend and abstain layer](recommend-abstain.md): baseline gates and data flags are
  shipped (`quibble-ml gate`); a Python `recommend()` API and licensing checks remain proposed.
