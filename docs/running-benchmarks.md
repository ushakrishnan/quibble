---
title: Running benchmarks you can publish
description: Dataset verification, controlled configurations, paired interpretation, and publication checks.
---

## Running benchmarks you can publish

This page is a protocol. Following it is what makes the numbers defensible.

You can explore datasets and run bounded comparisons in the
[browser workbench](experiment-workbench.md), or use the CLI below for automation and
the wider configuration space. Both use the same evaluation/report pipeline; a UI run
does not relax these controls. The UI currently fixes model seed 0 and offers holdout,
3-fold, or repeated 3-fold evaluation, so use the CLI for the broader settings below.

## What Full does and what the results establish

Full is a complete evaluation workflow without the smoke check's 128-training/32-test
row caps. It does not mean fitting and scoring on the same entire table. Method-specific
size limits, resource failures, and refusals still apply and remain visible.

| Stage | What happens | Why |
|---|---|---|
| Split | Holdout assigns about 70% of rows to training and 30% to testing. `3x1` trains on two thirds and tests on the remaining third in three fits. `3x2` repeats that process for six fits per method and model seed. | Held-out outcomes evaluate predictions on rows not supplied to that fit. Repeats show sensitivity to split assignments. |
| Prepare | Quibble learns categorical preprocessing from training rows and applies it to test rows. Methods may also perform their own preprocessing. Classification splits preserve class proportions when counts permit. | Test rows must not teach the fitted preprocessing; shared splits keep comparisons paired. |
| Fit | Every eligible selected method fits on each training split. Trees learn from those examples; pretrained tabular predictors prepare or condition on them without retraining foundation weights from scratch. | Each method receives the same training/test assignment under recorded settings. |
| Predict | Predict held-out rows as a batch, then measure separate single-row inference calls. Save predictions, test outcomes, and row positions. | Prediction is evaluated separately from fit; pretrained methods can do much of their computation here. |
| Analyze | Calculate quality, uncertainty, costs, disagreements, slices, and configured swap checks when comparable predictions exist. Record failures and exclusions. | Similar aggregate scores can hide different errors or operational costs. |
| Report | Write interpreted Markdown, figures, participation, and source/configuration records. | Conclusions stay attached to reproducible evidence and missing coverage remains visible. |

The CLI can reuse finished cells; inspect compatibility warnings and use a new output
root for changed conditions. Each browser experiment gets an isolated output directory.
Full is not automatic hyperparameter search, foundation-model pretraining, or a final
deployment fit on all rows. Internal method validation and early stopping can reserve
part of the supplied training split.

The evidence supports statements about held-out predictive quality and behavior for
this dataset, split scheme, settings, and hardware. It does not prove universal
superiority, deployment readiness, causality, or absence of pretraining exposure.
Repeated folds reuse one table; they are not independent newly generated datasets.
Do not repeatedly tune against final test outcomes. Use separate development and
confirmatory experiments, multiple generated tables when relevant, and representative
real data before generalizing. See [Limitations](limitations.md).

Device choice controls supported model execution, not every operation in the workflow.
`cpu` pins all selected models to CPU. `cuda`, or `auto` when CUDA is available,
visibly excludes methods that cannot meet GPU-only model execution. `auto` otherwise
uses CPU for all selected methods. It does not compare device speeds. Data preparation,
metrics, and reporting still use CPU; see [Configuration](configuration.md#choosing-cpu-or-gpu).

## 1. Prepare
1. Verify selected datasets with `quibble datasets --verify --names credit-g,phoneme,adult`.
   Inspect the output and fix or drop any `NAME MISMATCH` or error. Omitting `--names`
   requests every registered dataset and may download all OpenML entries.
2. Choose a device policy and keep it fixed. Use separate output roots for deliberate
   CPU-versus-CUDA experiments (`--out results_cpu`, `--out results_cuda`).
   CUDA sweeps exclude incompatible methods with reasons in the log and report participation.
   Use CPU to compare all baselines; CUDA-only subsets answer a narrower question.
3. Check the machine with `quibble doctor`. Close anything that competes for CPU, RAM, or VRAM (for
   example an inference server holding GPU memory).
4. Pin the environment. Results store available package versions and a git hash; commit your work first
   so the hash means something.

## 2. Configure
A sensible final-run configuration:

| Setting | Value | Why |
|---|---|---|
| `--cv` | `5x2` | 10 splits assess split sensitivity; intervals cluster pooled predictions by original row |
| `--model-seeds` | `0,1,2` (or just `0` for a first pass) | seeds only matter for stochastic backends |
| `--jobs` | `1` | latency and memory are meaningless when runs share the machine |
| `--preprocess` | run `ordinal` first, then `native` into a different `--out` | quantifies the preprocessing effect |
| `QUIBBLE_AG_TIME_LIMIT` | fixed (e.g. 120) | keeps AutoGluon's budget identical across datasets |
| `QUIBBLE_TIMEOUT_S` | generous (1800+) | a timeout is recorded as a failure, so do not let slow-but-valid runs die |

## 3. Estimate the cost
`runs = backends x datasets x splits x model seeds`. Each run costs a few seconds of process start-up
plus fit and predict time. Example: 5 backends x 12 datasets x 10 splits x 1 seed = 600 runs. AutoGluon
alone at 120 s is 12 x 10 x 120 s = 4 hours per model seed. Plan around that: for a first pass use
`--cv 3x1`, then scale up.

## 4. Run
```bash
quibble run --sweep --backends histgb,lightgbm,catboost,autogluon --datasets credit-g,phoneme,adult --cv 5x2 --model-seeds 0 --device cpu --jobs 1 --out results_cpu
```
- The sweep is **resumable**: re-running the same command skips finished cells. If you change device,
  preprocessing or CV scheme but keep the output directory, quibble warns that old results are being
  skipped. Use `--force` or a new `--out`.
  Changing the split scheme/seed can change shared test arrays; use a new root when
  that happens because `--force` does not bypass the test-row consistency guard.
- Failures do not stop the sweep. Check `quibble report` afterwards: the failure matrix shows which
  backend refused or crashed on which dataset.

## 5. Analyse
```bash
quibble analyze --dataset credit-g --out results_cpu
quibble swap --incumbent histgb --candidate lightgbm --dataset credit-g --out results_cpu
quibble report --out results_cpu --swap histgb:lightgbm
```
Read [Analysis](analysis.md) and the "How to read this" section of the report before drawing
conclusions. In particular:
- Marginal interval overlap alone does not determine the paired difference or equivalence.
- The worst-slice number is the maximum over several slices.
- If a backend refused or failed on a dataset, say so in your write-up. Those are findings.

## 6. Publish checklist
- [ ] Dataset ids verified; licences and citations noted.
- [ ] Hardware, package versions, and git hash inspected in the report; unavailable fields
      can be null, especially outside a git checkout.
- [ ] Timings from `--jobs 1` runs.
- [ ] Preprocessing rule stated, ideally both modes shown.
- [ ] Result JSON files and the report committed (the default `.gitignore` excludes `results/`;
      adjust it, and consider leaving out the large `*.npy` prediction files).
- [ ] Permission to process and publish the selected data and derived artifacts confirmed.
- [ ] Relevant model maintainers invited to review the evaluation configuration where appropriate.
- [ ] Copyright attribution and repository license metadata finalized.

## Design experiments that explain differences

A larger scoreboard is not automatically a more informative experiment. Define the
question before running: which property should change, which controls stay fixed,
and which outcome would contradict the working expectation?

- Use simple credible baselines as well as complex methods. The internal `weak` backend
  is deliberately limited; logistic/ridge regression and nearest-neighbor baselines
  are proposed, not currently available backend names.
- Separate increasing row count, adding useful information, and adding irrelevant fields.
  These are different changes, even when they produce the same table dimensions.
- Treat explanations of predictions as claims to investigate, not proof of the internal
  computation. Saved per-row answers are the behavioral evidence.
- Agreement does not establish a shared internal algorithm. Disagreement does not itself
  establish which method is preferable; inspect errors and task-specific costs.
- For pretrained methods, document possible prior exposure to public benchmark data.
  Successful setup on a familiar small dataset is not evidence that the learning task
  was new to the predictor.

Synthetic data can provide controlled tasks when the input sampling and outcome rules
are recorded. Preserve the generator/specification, generation seeds, and final input
hashes. Repeat across independently generated tables; one dataset's bootstrap interval
does not quantify variation over all possible generators or tasks.

Newly sampled rows reduce reuse of familiar public benchmark examples, but do not
establish that a pretrained predictor has never encountered similar task structures.
Changing only the name leaves the sampled data unchanged. Record the generating process
and distinguish new rows from verified absence of pretraining exposure.

Inspect synthetic inputs for label proxies and post-outcome features. Do not select the
generator, prompt, or method settings by repeatedly inspecting the final test outcomes.
Use separate development and confirmatory experiments when iterating.

Synthetic experiments supplement representative real-data evaluation. Their conclusions
are conditional on the generating process; a domain-themed schema is not validation
of real domain behavior. The [experiment workbench](experiment-workbench.md)
provides bounded generation and review; wider controlled campaigns still require
deliberate experimental design.

## What not to claim
- "X is equivalent to Y" from marginal interval overlap, or "X is better" without an
  appropriate paired comparison and a meaningful effect size.
- Anything about training-set variability beyond `split_std`.
- Latency or memory conclusions from a sweep that used `--jobs > 1`, or from a different machine.
