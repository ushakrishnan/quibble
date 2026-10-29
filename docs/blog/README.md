---
title: Predicting from Tables
description: A two-part Quibble story about prediction-level differences and engineering trustworthy comparisons.
---

## Predicting from Tables

A two-part series for applied ML engineers and technical leaders: why a model
comparison needs more than a score, and how Quibble makes its evidence inspectable.

1. [Same Accuracy, Different Mistakes](01-same-accuracy-different-mistakes.md):
   the decision problem, changed predictions, and the limits of a small smoke check.
2. [Making Prediction Comparisons Worth Trusting](02-making-prediction-comparisons-worth-trusting.md):
   row identity, source fingerprints, failure/device accounting, and local-data workflows.

The articles explain motivation and engineering decisions. The
[documentation index](../index.md) remains the reference for current behavior and
[validation status](../index.md#trust-levels-what-has-actually-been-run).
Start with [Getting started](../getting-started.md) to try the workflow.

## Figures and curated evidence

The [asset directory](../assets/blog/) contains the original SVG diagrams, PNG exports,
a [portable Wine smoke table](../assets/blog/wine-smoke-summary.csv), and its
[provenance and verification notes](../assets/blog/wine-smoke-evidence.json).
The historical example mixes CPU and CUDA, uses only 32 distinct test rows, and is
not a full benchmark, speed ranking, or replacement approval. New GPU-only runs
exclude the incompatible CPU methods shown in that historical example.

The evidence files are a curated summary, not a complete reproduction bundle.
Original prediction arrays, model weights, environment lock, and local result
snapshot are not included. Hashes identify supplied files; they do not reconstruct
missing inputs or prove execution independently.

These are public source files in the docs tree, also exported by the static-site build.
Private references, editorial notes, and social-platform drafts remain outside this
public collection. Source: [ushakrishnan/quibble](https://github.com/ushakrishnan/quibble).
The site becomes live only after the [Pages workflow](../development.md#github-pages-deployment)
successfully deploys.
