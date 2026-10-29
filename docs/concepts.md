---
title: Concepts
description: Shared splits, model seeds, preprocessing, disagreement, and statistical interpretation.
---

## Concepts

## The question
Teams swap tabular models for reasons that have nothing to do with accuracy: licensing, latency,
memory, a new foundation model, a deployment constraint. The risk is not "is the new model good?"
but "did the swap change behaviour in ways nobody checked?" quibble measures that change.

## Vocabulary

| Term | Meaning |
|---|---|
| **Backend** | A tabular learner behind a common `fit` / `predict` interface (e.g. `lightgbm`). |
| **Dataset** | A table with one target column. Task is classification or regression. |
| **Split** | One train/test partition. Identified by an integer *split id*. |
| **Model seed** | Randomness inside the model (not the split). |
| **Run** | One backend on one dataset, one split id, one model seed. Each run is its own OS process. |
| **Incumbent / candidate** | In a swap, the model you have and the model you are considering. |
| **Disputed** | A prediction pair where two backends differ (see below). |
| **Refusal** | A backend declining a task it cannot handle (e.g. too many rows). Recorded as a result. |

## Fixed splits: why comparisons are fair
A dataset, split scheme, assignment seed, and split id determine the test rows,
**independent of backend and model seed**. Every backend therefore sees the same train and test rows, so differences in
predictions are due to the models, not the data. Model seeds then vary only model randomness, which
makes "same backend, different seeds" a real stability measure.

Two ways to define splits (`QUIBBLE_CV` / `--cv`):
- `holdout` (default): one 70/30 split, seeded by `QUIBBLE_SPLIT_SEED`. Stratified for classification
  when every class has at least two members.
- `KxR` such as `5x2`: K-fold cross-validation repeated R times, giving K*R splits (ids
  `0..K*R-1`). Stratified when every class has at least K members, otherwise plain K-fold.
  Use this for anything you analyse statistically.

## Preprocessing rule
Backends differ in how they handle categorical columns. To keep the comparison fair:
- `ordinal` (default): the tree/foundation adapters receive ordinal-encoded categoricals.
  Unseen or missing categories become NaN. AutoGluon is the exception: it always preprocesses its own inputs.
- `native`: HistGB, LightGBM, CatBoost, TabFM, and KumoTabular use native categorical handling. AutoGluon always uses
  its own preprocessing; TabPFN always receives ordinal codes.

Each result records the *effective* mode, so a backend that cannot honour `native` says so. Run both
modes and report the difference. Use a separate output directory per mode (see
[Running benchmarks](running-benchmarks.md)).

## One process per run
A run executes in its own process so that peak memory (`peak_rss_mb`, the process high-water mark)
is not polluted by earlier runs. The cost is Python start-up time per run, which is why sweeps are
slower than a single in-process loop.

## Predictions are saved, not just metrics
Every successful run saves its test-set predictions (`*.pred.npy`) together with the test row
positions (`idx_test_ss*.npy`) and true targets. That is what makes the analysis possible after the
fact: quibble can re-pool, re-bootstrap and compare any two backends row by row without re-running
anything.

## Disagreement
- **Classification:** a prediction pair is *disputed* when the two backends predict different
  classes. It is *confident-both* when they differ and each is at least 90% sure.
- **Regression:** disputed when the two predictions differ by more than **0.25 standard deviations
  of the target**.

For each pair quibble reports who was right when they differed, and which row traits (missing
values, rare categories, near the decision boundary, extreme target) are over-represented among the
disputes. The disputed rows are exported as CSV with their features so you can look at them.

## Statistics in one paragraph
Intervals come from a **cluster bootstrap over data rows**: rows are resampled, and a row that
appears in several CV repeats is resampled as one unit. This captures test-set sampling noise. It
does **not** capture variation from retraining on different training sets, which is why the analysis
also reports `split_std` (spread across splits). Marginal confidence-interval overlap alone
does not decide a paired difference or establish equivalence. Use paired swap checks with
meaningful limits. Details in [Analysis](analysis.md).

## What quibble is not
- Not a hyperparameter tuner or an AutoML system; backends run with fixed, mostly default settings.
- Not a model server or a deployment tool.
- Not a universal API. Adding a backend takes about 20 lines because the interface is deliberately
  tiny, not because quibble hides differences between libraries.
