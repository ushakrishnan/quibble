---
title: Quibble documentation
description: Standalone entry point for installing Quibble, using the workbench, and interpreting comparison evidence.
---

## Quibble documentation

Quibble compares tabular prediction methods on the same held-out rows and reports
quality, disagreements, execution costs, and replacement checks. Version 0.0.1 is an
experimental pre-release of a local tool, not a hosted service.

Start with [About Quibble](about.md) for its purpose and limits, then choose a guide:

| Goal | Guide |
|---|---|
| Read the project story and engineering lessons | [Predicting from Tables blog series](blog/README.md) |
| Install from source and run the offline demo | [Getting started](getting-started.md) |
| Choose data, design a variant, and run in the browser | [Experiment workbench](experiment-workbench.md) |
| Compare your own CSV or Parquet | [Testing your data](testing-your-data.md) |
| Automate runs | [CLI reference](cli-reference.md) |
| Interpret saved evidence | [Reports](reports.md), [Swap checks](swap.md) |
| Find all guides and current validation boundaries | [Documentation index](index.md) |
| Diagnose a problem | [Troubleshooting](troubleshooting.md) |

These Markdown guides can be read directly in a checkout or browsed as a GitHub
documentation folder. Installation commands assume you are in the source checkout's
root, not inside this folder. Source: [ushakrishnan/quibble](https://github.com/ushakrishnan/quibble).
The repository includes a [GitHub Pages deployment workflow](development.md#github-pages-deployment).
A site URL is live only after deployment succeeds. The
[v0.0.1 GitHub pre-release](https://github.com/ushakrishnan/quibble/releases/tag/v0.0.1)
is separate from publishing the Python package to PyPI.

In the running workbench, choose **Docs** for a site-style reader with grouped sidebar
navigation, current-page highlighting, an on-page section index, breadcrumbs, and
Previous/Next links. Blog articles have their own series navigation, separate from the
documentation sidebar. Navigation collapses on
smaller screens; the original Markdown remains directly browsable on GitHub.
