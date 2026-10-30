---
title: Project plan and status
description: Implemented evaluation capabilities, validation priorities, and release milestones.
---

## Project plan and status

Quibble develops a focused workflow for evaluating tabular model replacements:
controlled execution, prediction-level evidence, and reviewable policy checks.
See the [project rationale](rationale.md) for the scope.

## Implemented foundations

| Area | Capabilities |
|---|---|
| Execution | Backend adapters, holdout/repeated CV, independent model seeds, recorded preprocessing/device policy, worker processes, timeouts, and failure records |
| Analysis | Row-cluster bootstrap intervals, prediction disagreements, diagnostic slices, calibration, and paired swap checks |
| Reporting | Evidence-based interpretation, participation/device/failure tables, figures, dated snapshots, and history |
| Inputs and operation | Built-in/OpenML inputs, named local CSV/Parquet, source fingerprints, centralized caches, benchmark presets, and isolated smoke checks |
| Interactive experiments | Task-filtered existing data, four bundled synthetic examples, provider-assisted recipes, reviewed variants, save/discard-and-continue, separate history, and product/About navigation |
| Reuse and development | Optional tree/AutoGluon fit retention, validation leaderboards, packaging, regression tests, CI definition, and container definitions |

Implemented capability and validation coverage are separate. The
[validation matrix](index.md#trust-levels-what-has-actually-been-run) identifies which
model/platform paths have been exercised.

## Evaluation priorities

1. Verify selected public datasets, their identities, licenses, and suitability for evaluation.
2. Run a broader campaign across classification/regression, dataset sizes, missingness,
   categorical features, and imbalance. A working target is 10 to 12 verified datasets.
3. Use repeated splits and multiple model seeds where model randomness is relevant;
   five seeds are a campaign target, not a requirement for deterministic backends.
4. Use `--jobs 1` for published resource measurements and document device differences.
5. Separate smoke success, successful full-data fits, and evidence supporting a replacement.
6. Publish configurations and failure coverage alongside findings, rather than only successful scores.
7. Include credible simple baselines alongside complex methods. `dummy` and `linear` (logistic or
   ridge regression) are implemented and `quibble gate` tests against them; nearest-neighbor
   predictors remain a proposed addition. The internal `weak` backend is a sanity check, not
   a substitute.
8. Use controlled synthetic tasks to investigate sensitivity to training size, feature
   count, irrelevant inputs, missingness, and noise. Repeat across generated datasets,
   not only repeated splits of one table.

## Proposed workflow: describe, generate, evaluate

Status: the bounded independent-row workflow is implemented through `quibble web` and
`quibble generate`. A user-configured OpenAI-compatible service can draft specifications
using API-key, Entra ID, or loopback no-auth configuration. Valid end-to-end live drafts
still need targeted validation; an HTTP 200 response does not establish schema correctness.
See the [workbench guide](experiment-workbench.md)
for exact supported mathematics, limits, and tested paths. The broader design below
also includes requirements beyond this first version.

The intended experience is to describe a scenario, review a concrete specification,
generate a repeatable dataset, and run a comparison through the existing workflow.

### Start with the decision

For example:

> Create a fictional grocery-chain dataset with 12,000 rows and 18 predictor fields.
> Each row represents an independent store-item replenishment decision. Predict whether
> demand during the next delivery window exceeds available inventory, using information
> available when the decision is made.

This particular request is illustrative; it is not a measured grocery application. The independent-row
version deliberately excludes repeated store-item histories. Real panel data needs group or
temporal evaluation; local datasets can declare `group_column` and `time_column` for that, but the
synthetic generator still produces independent rows.

Rows and fields alone are insufficient. The specification must make explicit:

- The unit represented by one row, the prediction time, and the outcome horizon.
- Whether the requested field count means predictor fields or total columns. Prefer
  an explicit predictor count, listing the target and excluded metadata separately.
- Field names, types, units, ranges, category frequencies, and permitted relationships.
- Which inputs carry signal, which are irrelevant, and which are unavailable at prediction time.
- Target rules, outcome noise, missingness, and expected class balance or numeric range.
- Independent-row versus repeated-entity/time structure, and a compatible split strategy.
- Generation seed, generator version, resource limits, and intended use.

Show this specification to the user before generating. Resolve incompatible constraints
explicitly rather than silently changing the requested size or task.

### Separate scenario design from outcome generation

A language model may help draft field descriptions and plausible scenario assumptions.
It should not independently invent every row and label and then treat its own narrative
as ground truth.

Use reviewed, versioned generation code to sample inputs and produce outcomes from an
explicit stochastic process. Preserve the final specification and executable generator;
a seed alone does not reproduce an unspecified procedure.

The outcome can contain irreducible noise and unobserved influences. Knowing the
generating process does not mean every observed label is deterministically recoverable
from the features.

Keep the generation recipe, latent variables, noiseless targets, and any oracle
probabilities outside the predictor feature table and outside predictor prompts.
The evaluation runner may access outcome labels for scoring, not expose them during
prediction. The implemented generator excludes latent/oracle columns by construction.
Cheap heuristic data flags exist, but general semantic leakage detection remains unimplemented.

Do not execute arbitrary code supplied by a user or returned by a language model.
Start with bounded, reviewed generation templates. Natural-language assistance and any
external data transfer should be explicit opt-ins; the deterministic generation path
should be usable locally without an LLM service.

### Produce usable inputs and inspectable provenance

The generated table and `dataset.json` should use the current named local-data interface.
Generation metadata should remain separate from that loader manifest, with its own
versioned specification, generator identity, seed, file hash, and synthetic-data label.
Do not imply new manifest keys are supported by the current loader.

Before any fitting, verify the exact requested row/predictor counts, valid types/ranges,
target completeness, feasible class counts, and usable split sizes. Check constraints,
duplicates, dependencies, and unavailable-at-prediction-time columns. If stochastic
generation makes a requested balance or size constraint infeasible, report it.

The user should be able to inspect the table, accept it, and run the same validation,
smoke, and full-comparison commands used for their own data. Keep each generated
variant and its experiment outputs identifiable; changing a generation seed creates
a different source table, not a model seed.

### Build experiments rather than one convincing-looking sample

Begin with a small independent-row template and create controlled variants:

- Change training-example count while holding the task and evaluation design fixed.
- Add irrelevant fields separately from changing the number of informative fields.
- Vary label noise, missingness, and interaction strength independently where possible.
- Include a no-signal negative control and a simple learnable positive control.
- Repeat across fresh generation seeds, reserve confirmatory variants, and state how
  any shared test rows are reused across conditions.

Keep split, model, and generation seeds distinct. Use the same evaluation rows across
methods within a condition. Avoid choosing a generator solely because it produces an
appealing ranking, and use multiple task structures rather than assuming one generator
is neutral among all prediction methods.

Reports must say "synthetic" and distinguish observed scores from known generation
assumptions. Synthetic results establish behavior under the chosen process, not grocery
forecasting accuracy, causal effects, or readiness for deployment. Validate promising
findings on suitable real data before drawing application conclusions.

## Engineering priorities

Implemented since the last plan revision: configuration identity checks on result reuse (settings,
model-specific settings and library versions), group and temporal splits and group-clustered
intervals for local data, data flags, `dummy` and `linear` baselines, `quibble gate`, retraining-aware
swap intervals under covariance assumptions, an exploratory maximum-slice bootstrap,
seed-noise subtraction, and `--retry-failed`.

1. Verify what is claimed: TabPFN end to end, the OpenML registry, remote CI, and Docker. Publish a
   simulation study of the swap verdict's false-PASS and false-FAIL rates on data with known truth.
2. Add backends that are usable commercially, with a license row in reports, and a tuned tier that
   gives boosted trees an equal-budget search so defaults-only comparisons are not the only option.
3. Add relational or multi-table benchmarking on top of temporal splits: point-in-time-correct
   feature construction, entity-level disagreement, and feature-building cost reported separately.
4. Expand calibration (per-class reliability) and slice diagnostics, including interaction slices
   from a disagreement explainer.
5. Explore a controlled LLM prediction comparison, distinguishing direct predictions
   from tool-assisted predictor building. Define held-out-label protection, prompts,
   output validation, tool/data access, retry budgets, provider/version records, and
   API cost/latency accounting before adding an adapter. Hosted execution also needs
   explicit data-handling permission. See the
   [comparison rationale](rationale.md#could-chatgpt-predict-from-the-same-csv).
6. Extend the implemented bounded generator only with explicit specification,
   provenance, validation, and evaluation contracts. Consider additional distributions
   and controlled campaign tooling separately from the local generation engine.

These are planned improvements, not claims that the additional interfaces are available.

## Publication and release gates

- Confirm dataset/model licenses, citations, and permission to publish derived artifacts.
- Keep claims consistent with the validation matrix and measured workload.
- Provide a reproducible configuration, environment details, and an appropriately reviewed evidence bundle.
- Finalize copyright attribution and repository metadata before public release.
- Confirm version/package metadata and exercise the build and release process.
- Validate the GitHub Actions and container definitions on their intended platforms.
- Invite relevant model maintainers to review evaluation configurations and interpretation.

The distribution name is `quibble-ml`; the import and command are `quibble`.
The package has not been published.

## Scope discipline

Prioritize correct paired comparisons, source identity, disagreement evidence, and explicit
uncertainty over additional adapters or platform features. The
[local-data guide](testing-your-data.md) and [benchmark protocol](running-benchmarks.md)
define the current user workflow.
