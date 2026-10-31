---
title: Project rationale
description: Why Quibble compares ways to predict outcomes from tables, what the evidence establishes, and how an LLM comparison could fit.
---

## Project rationale

Suppose a table contains historical invoices, the information available when each invoice
was issued, and whether it was paid late. The task is to predict late payment for new
invoices, not merely explain the historical table. This is an illustrative task, not a
claim that Quibble has been evaluated on invoice data.

Quibble compares ways to make such predictions: categories (classification) or numbers
(regression). It withholds some known outcomes, runs the selected methods on common
evaluation rows, then checks the answers against those outcomes.

Changing the prediction method changes more than a validation score. It can change individual
decisions, probability calibration, behavior on important subgroups, resource requirements,
and which datasets the system can process.

Quibble makes those changes inspectable through controlled comparisons, saved prediction
evidence, and explicit replacement checks.

## Why focus on changing prediction methods?

The project emerged while exploring broader machine-learning and AI systems, then narrowed
to tabular prediction. A focused engineering problem stood out: selecting a different
prediction method is straightforward; explaining and verifying the consequences requires
a consistent evaluation process.

The scope was narrowed to that process. Rather than building another orchestration platform,
Quibble connects prediction experiments to a reviewable decision:

1. Evaluate prediction methods on the same rows and split assignments.
2. Preserve predictions, provenance, execution outcomes, and resource measurements.
3. Examine quality, disagreements, calibration, and diagnostic slices.
4. Assess a proposed replacement against explicit limits.
5. Explain what the evidence supports and where it is insufficient.

## Value for users

| Reader or user | Practical question |
|---|---|
| Applied ML engineer | Which predictions change if I replace the incumbent? |
| Data scientist | Is the observed difference meaningful under a paired evaluation? |
| Platform engineer | Did both backends complete comparable work, on which device, and at what measured cost? |
| Technical reviewer | Can I trace the recommendation to records, predictions, and configured thresholds? |

This supports decisions on a selected dataset and workload. It does not establish a universal
ranking of model families.

## What exists today

- Classification and regression comparisons with built-in, OpenML, or named local CSV/Parquet inputs.
- Fixed holdout or repeated-CV assignments, separate model seeds, and recorded effective settings.
- Quality metrics, row-cluster bootstrap intervals, disagreement exports, calibration, and diagnostic slices.
- Measured fit time, prediction latency, process memory, and PyTorch allocation peaks where applicable.
- Recorded refusals, execution/resource failures, model participation, and mixed-device labels.
- PASS / FAIL / INCONCLUSIVE swap checks under configurable limits, with intervals that also count
  approximate retraining variation (5+ splits), an exploratory worst-slice bootstrap, and a seed-noise floor for
  disagreement.
- Simple baselines (`dummy`, `linear`) and `quibble-ml gate`, which asks whether a model beats a baseline.
- Group and temporal splits for local data, group-clustered intervals, and cheap data flags for
  suspicious columns.
- Interpreted reports, bounded smoke checks, dated result history, and optional supported-model retention.

See [reports](reports.md), [testing your own data](testing-your-data.md), and
[validation coverage](index.md#trust-levels-what-has-actually-been-run).

## Relationship to model libraries and benchmarks

Quibble uses existing model libraries; it does not replace their training algorithms or
AutoML capabilities. Public benchmarks help characterize models across shared tasks.
Quibble focuses on the evidence surrounding a particular comparison or proposed replacement.

Its comparisons include tree-based methods, AutoML systems that select or combine
predictors, and pretrained tabular predictors. The last category is also called tabular
foundation models: prior learning across tables is used for a new prediction task.

The project does not claim that disagreement analysis, paired evaluation, or replacement
gates are individually new. Its contribution is a documented workflow that brings them
together with execution provenance and inspectable artifacts.

## Learn where an approach works, not only which score is highest

A useful experiment can ask how the result changes with fewer training examples, more
irrelevant inputs, missing values, or a noisier outcome. Those conditions need separate
controls; changing several at once makes attribution difficult.

Saved predictions support investigation of observable behavior, not identification of
an internal algorithm. Likewise, a generated explanation is a claim to check against
observations, not a measurement of how the predictor computed its answer.

The [experiment workbench](experiment-workbench.md) lets users describe a prediction
task, review the assumptions, generate a repeatable labeled table, and compare methods.
A user-configured language-model service can assist specification drafting; reviewed local code creates the data.
Its purpose is experimental control, not manufacturing proof of real-world accuracy.
Generation is limited to explicit independent-row processes. Automated controlled
campaigns remain future work.

## Could ChatGPT predict from the same CSV?

Yes. A large language model (LLM) can analyze a table, attempt to predict missing outcomes,
or use tools to fit and run a statistical predictor. These are different workflows, and
none is ruled out because the input is a CSV.

The distinction is the evaluation question:

| Request | What would establish success? |
|---|---|
| Explain this table or investigate a pattern | Evidence supporting the explanation and correct use of data/tools |
| Predict an outcome for each unseen case | Predictions checked against held-out outcomes, with defined access rules and measurements |
| Use tools to build a predictor | Evaluation of the resulting predictor and disclosure of the tool-assisted workflow |

Quibble implements the second kind of evaluation for its current supported methods.
It does not evaluate arbitrary explanations or reasoning traces, and it has no general
chat-LLM adapter today. That is an implementation boundary, not a claim that LLMs cannot
predict from tables.

A future controlled comparison could include a direct LLM predictor or a separately
declared tool-assisted workflow. It would need:

- The same test rows, with test outcomes unavailable to prompts, examples, tools, or tuning.
- Defined training-data access and budgets, with any differences in external/pretraining
  information disclosed rather than assumed away.
- Recorded prompts, table serialization, provider/version identifiers where available,
  tool permissions, retries, and sampling settings.
- A fixed output schema, including how invalid answers, missing rows, and refusals count.
- Suitable quality measurements; verbal confidence must not be treated as calibrated
  class probabilities.
- End-to-end latency and API/token cost, reported separately from local process memory.
- Permission to send the data to a provider, or an approved local execution path.

Public benchmark rows may have appeared in pretraining; withholding their labels in a
prompt does not establish that the task is unseen. Private or newly collected data can
reduce that concern but requires appropriate handling and permission.

This is a proposed evaluation direction, not an available feature or a measured claim
that any approach wins. See the [project plan](project-plan.md).

## Scope boundaries

Quibble is not a prediction service, a general data platform, or a production-safety
certification system. Its leakage checks are cheap heuristics, not detection, and it does not
implement license-aware model selection, relational or multi-table evaluation, or hyperparameter tuning.

The [recommend and abstain proposal](recommend-abstain.md) records what is shipped (`quibble-ml gate`,
data flags) and what is still proposed (a Python `recommend()` API across many candidates and
license-aware selection).

Statistical, measurement, and platform boundaries remain explicit in
[limitations](limitations.md). Further work is tracked in the [project plan](project-plan.md).
