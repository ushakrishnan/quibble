---
title: What Quibble does
description: Plain-English explanations of Quibble's existing and new capabilities, everyday examples, inputs, outputs, and honest limits.
---

## What Quibble does

Imagine two programs predicting whether a shop will run out of a product.
Both get 80 answers right out of 100. That does not mean they make the same
mistakes, cost the same amount, or are equally useful to the shop.

Quibble gives the programs the same test and shows what changes. It is a local
comparison tool, not a program that automatically approves a model for real use.
The explanations below cover the original workflow and the newer additions.
Unavailable capabilities are identified rather than presented as working features.

## A few words first

| Word | Everyday meaning |
|---|---|
| Table | Rows of examples, such as purchases, with columns describing them |
| Target | The answer you want to predict, such as late payment |
| Predictor or feature | Information the program may use to make its prediction |
| Classification | Predicting a category, such as paid on time or paid late |
| Regression | Predicting a number, such as days until payment |
| Training | Studying examples whose answers are already known |
| Held-out test | Questions kept out of that training, used to check the answers |
| Backend | One of the prediction methods Quibble can run |
| Recipe | The recorded method, settings, dependencies and data rules for a run |

## Choose and review the data

### Can I bring my own spreadsheet-like data?

Yes, as a CSV or Parquet table. Think of handing a teacher an answer sheet and
identifying which column contains the answers. For invoices, you might choose
`paid_late` as the target and exclude an invoice identifier.

You supply the file and review the task, column roles and split settings.
The browser inspects a sample, validates the choices, and saves only after
confirmation; the CLI supports named local datasets too. Parsing does not call
a planning service or train a model. This is useful when comparing methods on
your own problem. It is not an Excel importer or an automatic judge of whether
every column was genuinely available at prediction time. Size and parser limits
apply. See [testing your data](testing-your-data.md).

### Can I start without finding a dataset?

You can choose bundled fictional examples, built-in datasets, or registered
public OpenML datasets. That is like using a practice paper before preparing
your own exam. You choose the dataset and methods and receive the same kind
of comparison report as for local data.

All 14 registered OpenML entries were loaded and reloaded offline in the local
verification; only selected entries received model comparisons. Downloading
successfully does not prove suitability, usage rights, or that a pretrained
model has never seen the examples before. Public downloads can need network
access on first use. See [datasets](datasets.md).

### Can I describe a fictional problem and generate examples?

Yes, within a bounded generator. For example, you can design independent
store-item examples where demand, inventory and noise affect stockouts.
The generator is like a controlled science experiment: you know the rules
because you chose them.

You provide or review a specification, inspect sample rows and confirm saving.
An optional configured language-model provider can help draft that specification.
Only explicitly approved scenario text is sent; local parsing does not send
your table. Provider requests can cost money. A good result on invented data
tests those invented assumptions, not a real shop. Arbitrary business processes,
correlated histories and unrestricted generated code are not supported.
See the [workbench guide](experiment-workbench.md).

### Can several related tables be used together?

Yes, through a separate bounded relational workflow. Suppose you have customers,
orders and deliveries. For a prediction on January 1, Quibble can summarize
orders known before January 1, not orders that arrive in March. It turns
permitted past information into features that ordinary tree models can use.

You supply tables plus explicit keys, relationships, time/availability rules,
target and prediction dates. Browser review checks the relationships and task;
editing them invalidates the previous review. Confirmation publishes the
hash-bound dataset atomically. You receive aligned examples, an audit and a
comparison report with shared feature-construction costs.

This is not an unrestricted database join tool or proof that event dates equal
real availability dates. Dimensions must obey the frozen-snapshot contract.
Training labels whose outcome windows cross the next split boundary are
purged. Synthetic leakage checks and one small public sampled pilot passed;
the pilot is not an official full-corpus benchmark.
See [multi-table input and evaluation](testing-your-data.md).

## Give the methods a fair test

### How do I know both methods took the same exam?

Quibble assigns the same training and held-out examples to each selected method.
Otherwise one student might receive easier questions. It also learns supported
categorical preparation from training rows, not from the hidden test answers.

You choose a holdout or supported repeated-fold scheme. The saved predictions
retain row identities so later comparisons can check the pairing. For example,
both methods must predict the same 30 withheld invoices, not two different sets
of 30. This controls the test assignment, not every difference between libraries
or the suitability of the dataset. See [concepts](concepts.md).

### What if rows belong to the same customer or different dates?

Randomly mixing them can make an exam too easy. A model might recognize a
customer from training, or learn from the future. Local group splits keep
declared groups apart; temporal splits keep training before testing.
Relational tasks use their declared chronological holdout and horizon rules.

You declare the relevant group or time column, or the relational task contract,
and receive split-aware evidence. Where declared, uncertainty resamples whole
entity groups rather than pretending their rows are independent. This matters
for repeated customers and dated decisions. It does not solve every form of
time dependence or verify that your declared dates are truthful.
See [data rules](testing-your-data.md) and [statistical limits](limitations.md).

### Which prediction methods and machines can I compare?

There are trees, simple baselines, AutoML and optional pretrained tabular
methods. A pretrained method has studied other tables before receiving your
examples; that does not make it automatically better for your table.

You select methods and CPU, CUDA or automatic device policy. Quibble records
what actually ran, and explicitly excludes incompatible CUDA methods instead
of quietly replacing them. An experimental adapter is not broad hardware
validation. Dependencies, task support, row limits and usage rights differ.
See [backends](backends.md) and [GPU setup](gpu-and-docker.md).

### What is special about the newer TabICLv2 and Mitra-v2 options?

Their expected task checkpoints and file hashes are pinned, like checking the
exact edition of a textbook rather than accepting whichever edition appears.
TabICLv2 passed small real classification and regression holdouts on CPU and
CUDA. Mitra-v2 passed small real CUDA holdouts.

You explicitly install and select them. Mitra classification uses a zero-shot
recipe without parameter fine-tuning; its distributional regression recipe
loads and adapts during prediction. Those are different kinds of work.
Mitra's two-bag regression recipe is not the published eight-bag benchmark.
Missing weights refuse unless initial downloads are explicitly enabled.
Broader capacities, full seed control and general mixed-type coverage remain
unverified. See [exact recipes and terms](backends.md).

### Why compare against a simple guess?

Before asking whether a complex program beats another complex program, ask
whether it beats something cheap. If 90 of 100 invoices are on time, always
answering "on time" already gets 90 right.

You choose `dummy` or `linear` as a baseline and use `gate` to test a chosen
improvement margin. You receive interval-backed PASS, FAIL or INCONCLUSIVE.
This helps avoid celebrating complexity without measured benefit. Failure to
show improvement is not proof that the data has no useful signal.
See [baseline gates](cli-reference.md#quibble-ml-gate).

### What do repeated splits and model seeds tell me?

Changing the split changes the exam questions. Changing a model seed changes
some methods' internal randomness while leaving the exam fixed. These answer
different questions, so Quibble records them separately.

You choose supported folds and seeds and receive split variation and, where
available, same-method reseeding disagreement. Identical predictions from a
deterministic tree do not establish that training on different data would give
the same result. Some adapters, including Mitra-v2, do not claim complete seed
control. More repeats do not create more independent customers.
See [concepts](concepts.md) and [analysis](analysis.md).

### Can trees try different settings before the final test?

Yes, in the explicit tuned tier. Think of trying study techniques on practice
questions, choosing one, then taking an unseen exam. Choosing settings after
seeing the final answers would be cheating.

You request and approve a bounded search. The default budget is 20 trials with
three training-only inner folds, followed by refitting on the outer training
data. Reports record attempts, valid fits, selected settings and costs.
Default and tuned campaigns use separate output roots; an explicit cross-tier
report can pair the same method's saved results. Equal trial budgets do not
mean equal compute or guaranteed improvement. Relational tuning currently
refuses pending a dedicated horizon-safe inner protocol.
See [running benchmarks](running-benchmarks.md).

## Read the evidence

### What does the headline score actually mean?

For categories, accuracy asks how many answers were right. For numbers, error
metrics ask how far the answers were from reality. Predicting 12 days when the
answer is 10 is different from predicting 100.

You supply a prediction task and receive the appropriate primary and secondary
metrics for common test rows. A single score can hide rare-class mistakes,
important subgroups or different operating costs. No metric is automatically
the right business objective. See [metric definitions](analysis.md#metrics).

### Where do the programs disagree?

Two students with the same mark may miss entirely different questions.
Quibble compares their predictions row by row and shows who was right when
their answers differed, plus traits common among disputed examples.

You select methods with compatible saved evidence and receive paired summaries
and disputed-row CSVs. For numbers, a configurable gap defines a dispute.
This is useful for reviewing changed decisions, not explaining the models'
internal reasoning. Existing row-level exports contain feature values and can
be sensitive; inspect them before sharing a report bundle.
See [disagreement analysis](analysis.md).

### Can a program be confident but wrong?

Yes. If a method says "70% likely" for 100 similar cases, roughly 70 should
happen over suitable repeated cases. Calibration checks whether stated
confidence behaves that way.

You receive overall reliability and separate one-versus-rest class summaries.
For red, blue and green answers, classwise checks ask whether confidence in
each color is reliable, rather than letting the most common color hide the
others. Empty bins and rare or absent classes stay explicit; old unrecorded
class names are not invented. These are descriptive diagnostics, not automatic
probability repairs or new classwise acceptance limits.
See [calibration](analysis.md).

### Does a method struggle on particular kinds of examples?

Slices are smaller scorecards, such as examples with missing fields. A new
optional interaction explanation can look for a short combination, such as
“missing income and short history,” that describes disagreements.

You opt into bounded, allowlisted interaction discovery and receive reproducible
predicates and aggregate evaluation summaries. Discovery and evaluation rows
or groups are separated, but both still involve shared trained predictors.
Treat the rules as leads to investigate, not causal explanations or independent
confirmation. They do not change swap gates. Ordinary diagnostic slices also
remain exploratory. See [analysis](analysis.md).

### What changes when the data comes from several tables?

Relational diagnostics compare examples with no history, short or long history,
recent or old activity, missing relations and cold starts. It is like checking
whether a pupil struggles because relevant background notes were absent.

For a relational comparison, the report and aggregate CSV show paired quality,
disagreement and separate example/entity support for fixed, label-free groups.
An empty slice gets no invented score; small support is labeled.
These summaries have no new confidence bounds or automatic acceptance gates,
and they do not prove why a method behaved differently.
See [relational diagnostics](analysis.md).

### How sure should I be about a small difference?

One extra correct answer in a small test may be luck. Quibble reports uncertainty
ranges by resampling test rows or declared groups. With at least five splits,
it also provides an approximate correction for split-to-split variation.

You receive intervals attached to the relevant metrics and paired checks.
They help distinguish an observed difference from a confident conclusion.
They do not cover every source of uncertainty, prove independence in time
series, or establish that a replacement is safe everywhere.
See [uncertainty](analysis.md#confidence-intervals) and [limitations](limitations.md).

### What does each method really cost?

Imagine comparing two delivery services: preparation, travel and unloading all
matter. Quibble records fit, loading, batch prediction, supported single-row
latency and memory rather than treating one stopwatch reading as everything.

You receive stage-aware costs and the recorded device. Tree search and refit
overlap total fit cost; do not add them twice. Relational construction is a
shared stage, not a full charge to every method. Mitra regression's inclusive
adaptation-and-prediction cost is not cheap pure inference; isolated timings
remain unavailable when they cannot be measured honestly.
Single-machine costs are not a production service-level guarantee.
See [reports](reports.md) and [measurement limits](limitations.md#measurement).

### What happens when something cannot run?

Quibble keeps a refusal or failure instead of deleting the competitor from the
story. A method declining too many rows is different from crashing or timing out.
Neither should appear as a successful zero-cost result.

You receive participation, exclusions, logs and recorded outcomes. Normal
worker deadlines stop the owned process tree, including Windows launcher
children, before temporary-model cleanup. This helps expose practical coverage
limits. A failed experiment is not evidence that the underlying algorithm is
always poor. See [troubleshooting](troubleshooting.md).

### Can Quibble spot suspicious data?

It flags clues such as a near-perfect single predictor, repeated feature rows,
identifier-like columns, and time-like columns with a random split. Think of a
teacher noticing that one column looks suspiciously like the answer key.

You receive warnings to investigate, not an automatic declaration of cheating
or a silent change to the verdict. The checks cannot detect every combination,
collection problem or unavailable-at-prediction-time field.
See [data flags](analysis.md#data-flags).

## Decide, preserve and operate

### What do PASS, FAIL and INCONCLUSIVE mean?

They are results under a rule sheet, not universal safety labels. For example,
you might allow at most a small accuracy drop and a twofold latency increase.
Quibble checks the paired evidence against those recorded limits.

You provide or review the policy and receive each check plus a combined verdict.
INCONCLUSIVE means the evidence cannot establish which side of a limit applies.
An unmeasurable adaptation-latency budget is not quietly passed.
Passing does not establish fairness, legal approval, business value or deployment
readiness. Compare the candidate to a baseline as well as an incumbent.
See [swap checks](swap.md).

### Can I revisit a result and understand what produced it?

Yes. Saved predictions, recipes, source fingerprints, split identities,
participation and archived reports are like a laboratory notebook. They make
it harder to confuse two different experiments.

You select a saved job or run report/analysis commands and receive readable
Markdown plus machine-readable evidence. Reports can need original source
data or caches to regenerate. Legacy evidence stays explicitly incomplete.
Not every library exposes a complete checkpoint or seed identity; a recipe
hash is not a guarantee of bit-for-bit reproduction.
See [results format](results-format.md) and [reports](reports.md).

### Can I check saved evidence without training again?

Yes. The browser's Verify page can run a swap or baseline check against a
completed full experiment. It is like marking the same exam against a recorded
rule sheet rather than asking the students to sit another exam.

You choose the saved job and methods. Quibble uses its original policy and
validates the source, recipes, coverage and matching predictions. A separate
verification artifact preserves the inputs and result; original reports and
model records are not rewritten. Smoke, incomplete or incompatible evidence
cannot produce an unsupported approval. Execution errors have no scientific
verdict. No model fitting, provider call or model download is needed.
See [saved-evidence checks](experiment-workbench.md).

### Can I put two saved experiments side by side?

Yes, but comparing their scorecards does not automatically make them the same
exam. The Compare page explains differences in sources, splits, recipes,
policies and coverage before showing descriptive evidence.

You select two saved jobs and receive a compatibility assessment and summaries.
Different sources or targets withhold shared quality metrics. Even matched
jobs do not receive an inferred winner, equivalence claim or paired scientific
verdict. Use this view to inspect what changed, not to replace a supported
paired check. See [comparison limits](experiment-workbench.md).

### Can I use the browser and keep my place?

The browser guides you through choosing data, methods/settings, and an explicit
start. History opens saved results separately from the active run. CLI commands
use the same evaluation pipeline for repeatable scripts.

You can choose light/dark appearance, six accents, high contrast and compact
desktop navigation; mobile navigation and keyboard controls remain available.
Settings are restored before painting and synchronized between tabs where
storage is available. These controls improve navigation and readability, not
the scientific strength of a result. Report section navigation includes
unfamiliar sections without discarding them from downloads or printing.
The server is a local, single-user tool, not a multi-user hosted platform.
See the [workbench](experiment-workbench.md).

### Can I inspect the installation without exposing my data?

The browser can show an explicitly requested package/backend inventory.
Think of checking the tools in a workshop before starting a project, not
proving that every tool will work on every material.

You request diagnostics and receive timestamped, allowlisted metadata without
credentials, environment variables, private rows or filesystem paths. This
does not automatically probe the GPU, call a network service, fit a model or
certify checkpoint access. Installed packages and working end-to-end recipes
are different levels of evidence. See [diagnostics](experiment-workbench.md).

### Do installation, licences and privacy need separate decisions?

Yes. Installing a library is like owning a player, not automatically having
permission to use every recording. Code licences, checkpoint terms and dataset
licences can differ. Download permission is not legal approval.

The maintained dependency manifest is `pyproject.toml`, with generated `uv.lock`.
The import is `quibble_ml` and command `quibble-ml`, avoiding an unrelated
package named `quibble`. Optional model downloads require their documented
controls. Local evaluation stays local; an explicitly configured planner sends
approved scenario text, and a future remote prediction endpoint would need
separate data-transfer approval. See [installation](getting-started.md),
[model terms](backends.md) and [privacy](about.md#data-privacy-and-compute).

### Is this already a native Kumo Relational comparison?

No. Summarizing related tables and giving the resulting features to local trees
is different from asking a native relational model to work over those tables.
The single-table Kumo adapter is not that native comparison either.

A standalone local Kumo Relational probe now works over synthetic related tables
on CPU and CUDA, without an endpoint or sending rows to a provider. That removes
the assumption that every native test needs a hosted service. However, this
probe is not yet wired into Quibble's reviewed manifests, workers and browser;
the existing multi-table workflow still uses local aggregated features.
The separate SDK/NIM route requires an actual deployment.

TabPFN now has real offline CPU/CUDA Wine and diabetes holdouts after successful
provider licence verification. Its use restrictions still apply; a successful
test does not grant commercial rights. See
[validation status](index.md#trust-levels-what-has-actually-been-run).

## What is verified, and what is not

The current local work includes real bounded tree/browser/relational campaigns,
small pinned TabICLv2 CPU/CUDA holdouts, small Mitra CUDA and TabPFN CPU/CUDA holdouts,
automated scientific/UI regressions and package/static-site checks.
The [validation matrix](index.md#trust-levels-what-has-actually-been-run) provides
the narrower scope; "implemented" does not mean every task and machine was tested.

The public relational check is a sampled `rel-arxiv/author-publication` pilot
with official chronological boundaries, not the full benchmark. Native Kumo and
authorized TabPFN live checks remain incomplete. Remote CI for these uncommitted
changes, Docker execution, publication and a fresh release are not established
by local success. Read the report's caveats before sharing conclusions.
