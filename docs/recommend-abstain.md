---
title: "Quibble: recommend and abstain layer"
description: Baseline gates and data flags (shipped) and the proposed recommend() API, calibration and licensing checks (not implemented).
---

## Quibble: recommend and abstain layer

Proposed extension to the current evidence-based comparison workflow.

> [!IMPORTANT]
> Partly implemented. Shipped: [`quibble-ml gate`](cli-reference.md#quibble-ml-gate) (does a model beat a
> `dummy` or `linear` baseline by a chosen margin, with an interval-backed verdict and an
> INCONCLUSIVE outcome), [data flags](analysis.md#data-flags) (single-feature leakage, ID-like columns,
> duplicates, time-like columns on a random split), and group and temporal splits for local data.
> Still proposed, and described below: a Python `compare()` / `recommend()` interface that picks one
> backend or abstains across many candidates, a calibration-fix check, small-sample and class-imbalance
> caveats, and license-aware selection. See [current report behavior](reports.md) and
> [implementation status](project-plan.md).

## Why this exists

Quibble compares tabular backends on the same data and split. Current reports provide
conditional shortlists and replacement checks. This proposal would add a separate decision
about whether any candidate meets a predefined usefulness criterion relative to a baseline.

An explicit abstention outcome would avoid selecting a candidate when the evidence is
insufficient. Lack of measured improvement must not be presented as proof that the data
has no signal or that a specific data-quality problem has been diagnosed.

## Scope

In scope:

- A `recommend()` step that runs after the comparison and returns a decision plus reasons.
- An abstain outcome with a stated cause.
- A short report section the decision is rendered into.

Out of scope:

- Connectors to Salesforce, Dynamics, QuickBooks or any source system.
- A semantic or schema-mapping layer.
- An LLM agent or planner.
- A pattern-discovery engine.

Connectors and schema mapping would remain separate concerns; this layer would consume
the same validated tables as the existing comparison workflow.

## Illustrative interface, not implemented

The gate command covers rule 1 below for one candidate at a time.

```python
result = quibble_ml.compare(dataset, target="invoice_paid_late", task="classification")
rec = quibble_ml.recommend(result)

rec.decision      # "use" | "abstain"
rec.backend       # chosen backend name, or None on abstain
rec.reasons       # list of plain-language reasons
rec.caveats       # data-size, leakage, licensing and calibration flags
```

This is a proposed design, not existing code.

## Decision rules

Rules run in order. The first that fires decides.

1. **Baseline gate** (shipped as `quibble-ml gate`). A backend is eligible only if it beats the baseline on the primary metric by at least a minimum margin, and the paired bootstrap interval on the difference excludes zero. Primary metric: log loss or AUC for classification, MAE or RMSE for regression. Make the margin and interval level configurable, with conservative defaults.
2. **Abstain.** If no backend passes the baseline gate, return `abstain` with the observed reason. Distinguish insufficient evidence from no useful measured improvement; do not infer no signal or leakage from a failed significance gate alone.
3. **Tie-break by simplicity.** Among eligible backends meeting a defined practical-equivalence rule, prefer the simpler, faster and cheaper one. Failure to detect a difference alone does not establish equivalence.
4. **Calibration check.** For classification, if the winner's probabilities are poorly calibrated and a cheap recalibration fixes it, say so. If not, add a caveat.
5. **Licensing flag.** Attach each backend's license note as a caveat so the user sees commercial-use restrictions before shipping.

## Data-quality caveats to attach

- Small sample: warn below a configurable row count.
- Temporal leakage: warn if a time column exists and a random split was used. Offer a point-in-time split.
- Class imbalance: report it, and make sure the primary metric is not accuracy.
- Duplicate or near-duplicate rows across train and test.

## Report output

One section at the top of the report:

- Decision (use X, or abstain) in one sentence.
- Three or fewer reasons, each tied to a number from the comparison.
- Caveats list.
- Link to the full comparison table below it.

## Build order

1. Baseline gate with the paired bootstrap, on one classification task.
2. Abstain outcome with causes.
3. Tie-break by simplicity and cost.
4. Caveats: leakage, small sample, imbalance, licensing.
5. Regression support.
6. Calibration check.
7. Report section and docs.

## Tests

- **Planted-signal data:** a synthetic table where one backend should clearly win. The decision must pick it.
- **Pure-noise data:** random labels. The decision must be `abstain`.
- **Leakage trap:** a feature that encodes the target. Must raise the leakage caveat.
- **Tie case:** two backends within noise. Must pick the simpler one.
- **Small data:** a few dozen rows. Must warn and usually abstain.

The noise case must demonstrate that the decision procedure can withhold selection rather
than convert a relative ranking into a claim of useful predictive performance.

## Open questions

- Default minimum margin over baseline: needs a quick empirical check on a few public datasets.
- Whether to bootstrap over rows or use cross-validation folds when the dataset is small.
- How to present abstain so it reads as a useful result, not a failure.

Related: [project rationale](rationale.md).
