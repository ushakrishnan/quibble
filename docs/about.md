---
title: About Quibble
description: Product purpose, workflow, privacy boundaries, and publication status.
---

## About Quibble

## Why Quibble?

Two models can get the same score and make different mistakes. Quibble shows which
predictions change, who gets them right, and what each method costs, so you can judge
whether a change is worth making.

## What it does

Quibble is a local comparison and verification tool for tabular classification and
regression. It evaluates prediction methods on identical held-out rows, then shows
quality, uncertainty, calibration, slices, disagreements, execution costs, errors,
and refusals. Configurable swap checks return PASS, FAIL, or INCONCLUSIVE.

It is not a model tuner, deployment platform, or guarantee of real-world suitability.
The [validation matrix](index.md#trust-levels-what-has-actually-been-run) distinguishes
implemented adapters from paths verified on real hardware.

For the motivation and engineering decisions, read the
[Predicting from Tables blog series](blog/README.md):
[Same Accuracy, Different Mistakes](blog/01-same-accuracy-different-mistakes.md) and
[Making Prediction Comparisons Worth Trusting](blog/02-making-prediction-comparisons-worth-trusting.md).
The articles provide context; the guides remain the reference for current behavior.
In the running workbench, About links to a local blog reader at `/blog`, with both
articles, diagrams, and evidence downloads. Public GitHub hosting is not required.

## Browser workflow

The default server address opens About as the landing page. **What to do next**
recommends reviewing a ready-made dataset and running a smoke comparison, with an
**Open workbench** button. The header's **Workbench** link opens `/workbench` directly.
Existing root-address experiment bookmarks still forward to the workbench with their
job identifier preserved.

Quibble is the product heading; the smaller subtitle describes its purpose. The
shared product navigation contains Workbench, Blog, About, GitHub, and Docs, separate
from experiment navigation. The current page is highlighted. Blog, About, and Docs
open in a new tab from the workbench to preserve its draft. Docs and Blog work locally;
GitHub links to [the source repository](https://github.com/ushakrishnan/quibble).
Version, release status, and license
appear in About's project-status section, not the shared header.

1. Choose data: review existing rows, or design a recipe, preview, confirm review,
   and choose **Save and continue**.
2. Configure and run: choose methods, device policy, and a smoke or full comparison.
3. Read results. Use **Past experiments** to reopen saved evidence independently.

For a recipe variant, **Save and continue** uses the saved variant, while
**Discard changes and continue** uses the unchanged original. Both open configuration;
neither starts an experiment automatically. See the [workbench guide](experiment-workbench.md)
for filters, review gates, generation limits, and report history.

## Data, privacy, and compute

Existing data stays local in the evaluation pipeline. Optional planning sends the
approved scenario text and controls to your configured provider, not dataset rows.
Malformed output can cause one correction request; charges can apply to both.
Never put sensitive information in scenario text. Review reports and logs before
sharing because they can contain source paths and data-derived information.

Fresh synthetic rows test generating assumptions, not real business performance.
They do not prove an unseen task or eliminate pretraining exposure. Repeat across
generation seeds and confirm useful findings on representative real data.

CPU pins model execution to CPU. CUDA requires availability and excludes incompatible
methods; Auto resolves once. GPU failure does not silently retry on CPU. Preparation,
metrics, and reporting still use CPU. See [backends](backends.md) and
[limitations](limitations.md).

## Version, license, and status

The local About page displays the running package version. Quibble v0.0.1 is an
experimental pre-release;
the distribution name is `quibble-ml`, and its command/import name is `quibble`.
Install from source using [Getting started](getting-started.md).

Quibble uses the MIT license. Optional libraries, datasets, and pretrained checkpoints
have their own licenses and access conditions. Copyright attribution is
Quibble contributors.

Source and issues are at [ushakrishnan/quibble](https://github.com/ushakrishnan/quibble).
The [Pages deployment workflow](development.md#github-pages-deployment) publishes
the docs and blog after repository setup and a successful deployment. The static site's
Workbench page explains how to run the Python server locally; it does not execute models.
Read the [documentation index](index.md) for all guides.
For diagnostics, run `python -m quibble doctor` in the
server's environment and review its output before sharing.
