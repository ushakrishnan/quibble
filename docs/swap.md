---
title: Swap checks
description: Paired replacement checks, configurable limits, verdicts, and CI integration.
---

## Swap checks

`quibble-ml swap` asks whether a replacement meets your configured limits on
this dataset, given the available evidence. Think of it as a checklist, not a
permission slip for deployment: PASS means the recorded checks passed, FAIL
means a limit was exceeded, and INCONCLUSIVE means the evidence cannot decide.

```powershell
quibble-ml swap --incumbent histgb --candidate lightgbm --dataset credit-g
$LASTEXITCODE    # 0 PASS, 1 FAIL, 2 INCONCLUSIVE
```
It reads existing results (run a CV sweep first) and writes
`<out>/<dataset>/analysis/swap_<incumbent>_to_<candidate>.json`.

## Verdicts
Each check is one of:
- PASS: the whole 95% interval is within the limit.
- FAIL: the whole interval is beyond the limit.
- INCONCLUSIVE: the interval straddles the limit, **or no interval could be computed**. More
  representative held-out rows can help; extra CV repeats assess split sensitivity but do not add
  independent rows. Other datasets test generality, not this dataset's uncertainty.

Latency, memory and coverage have no interval, so they are PASS or FAIL from the point value
(latency and memory compare medians). If a recorded adapter performs fitting during
prediction and cannot supply isolated single-row latency, that budget is explicitly
INCONCLUSIVE. Its inclusive batch cost is not substituted for cheap inference or a
passing latency check. The **overall** verdict is FAIL if any check fails, else
INCONCLUSIVE if any is inconclusive, else PASS. A PASS is therefore always interval-backed for the
quality, disagreement, calibration and slice checks: if fewer than 20 valid bootstrap resamples are
available the check is INCONCLUSIVE and the note says how many resamples were unusable.

## Where the intervals come from
Quality, disagreement, and calibration checks combine two intervals and use their
outer bounds. The worst-slice check uses the bootstrap only:

1. A cluster bootstrap over rows, or over groups when the dataset declares `group_column`. It counts
   test-set noise.
2. An approximate Nadeau-Bengio corrected resampled interval over per-split values, intended
   to account for overlapping training sets and retraining variation. It needs **5 or more splits**
   (for example `--cv 5x1` or `--cv 3x2`). With fewer, a t interval would be so wide that it says
   almost nothing, so it is skipped and the note says "retraining variability not counted".

Verdicts from one holdout split or three folds therefore reflect test-set noise only. Prefer
`--cv 5x2` or more for decisions.

The five-split cutoff is an implementation rule, not a precision guarantee. The combined
decision rule has not been calibrated for false-PASS or false-FAIL rates. Temporal folds are
not exchangeable, and row resampling does not model serial dependence. Interpret these
intervals with those assumptions rather than as deployment certification.

## The checks
| Check | What is measured | Default limit |
|---|---|---|
| `coverage` | the candidate completed every split the incumbent did | all splits |
| `quality drop (<metric>)` | how much worse the candidate is on the primary metric (paired) | 0.01 |
| `disagreement rate` | share of prediction pairs that are disputed | 5% |
| `disagreement rate (above reseed noise)` | same, minus the incumbent's own disagreement across model seeds; used when the incumbent has 2+ seeds | 5% |
| `confident disagreements` | classification: disputed and both models at least 90% sure | 1% |
| `calibration (ECE increase)` | classification: candidate ECE minus incumbent ECE (paired, with an interval) | 0.02 |
| `worst slice` | largest quality drop over slices with enough rows; the interval covers the maximum over all slices | 0.05 |
| `latency ratio (single-row p50)` | candidate / incumbent median of `single_row_p50_ms` | 2.0x |
| `memory ratio (baseline-adjusted process peak)` | candidate / incumbent median of `peak_rss_increase_mb`, each floored at 50 MB | 1.5x |

Notes:
- Quality drop is positive when the candidate is worse. It is *absolute* for accuracy, AUC, R^2,
  log loss, Brier and ECE, and *relative* (a fraction) for RMSE and MAE. A candidate that is better
  shows a negative number.
- The maximum over fixed slice masks is recomputed inside every bootstrap resample. Target-driven
  slice discovery is not repeated, so family-wise or selection-adjusted coverage is not established.
- Noise-adjusted disagreement subtracts an estimated incumbent reseed rate. The note preserves raw
  disagreement and its interval; uncertainty in the noise estimate is not propagated.
- Latency and memory are single-machine medians over usable runs, not necessarily only the common
  quality-analysis splits. Use `--jobs 1` results. Memory subtracts resident memory before fitting
  from the lifetime process peak after inference. Imports, temporary allocations and earlier peaks
  can still contribute; this is not model-only footprint. Estimates below 50 MB are floored for
  the ratio, and the unfloored estimates remain visible beside it.
  Results from older versions without that field fall back to peak RSS and say so.
- Fit time reported in run records excludes pretrained-weight loading (`load_s`), which is recorded
  separately; `fit_total_s` is the sum.
- If the candidate has no usable runs, `coverage` fails and the remaining checks are skipped.
- If the incumbent has no usable runs, the command exits with code 1 and a message.
- Verdicts apply to the saved recipes, including explicit tuned-tree campaigns,
  not to different settings or an unevaluated tuning budget.
- Data flags (see [Analysis](analysis.md#data-flags)) are printed under the table and saved in the
  JSON. They never change the verdict, but a PASS should be read together with them.

## Changing the limits

The packaged [swap_thresholds.json](../quibble_ml/swap_thresholds.json) is the single source of
default limits for swap checks in the CLI and workbench. It ships with the Python package,
so defaults work outside a repository checkout. Analysis also uses its default confidence,
regression-gap, near-margin, and minimum-slice settings.

Think of this JSON as a rule sheet: measurements describe what changed between two models;
the limits describe how much change you will tolerate. For example, with accuracy as the
primary metric, `max_quality_drop: 0.01` permits a one-percentage-point drop, not a 1% relative
drop. With RMSE or MAE it permits a 1% relative increase.

These are the defaults Quibble uses, not universally safe acceptance criteria. Review them
against the cost of changed predictions and your deployment budgets. Keep the packaged file
unchanged for reproducibility; export a working copy, edit it, and supply it explicitly:

```powershell
quibble-ml swap --export-thresholds my_limits.json
# Edit my_limits.json in your editor before evaluating.
quibble-ml swap --incumbent histgb --candidate lightgbm --dataset wine --thresholds my_limits.json --require-thresholds
```

Run both models first; `swap` reads saved benchmark results and does not train models.
Export works with an installed package and refuses to overwrite an existing file.
`--thresholds` is supported by `swap`, `report`, and full `benchmark` runs. Only keys you list
are overridden; omitted keys retain packaged defaults. Changing limits does not change model
training. Report regeneration must receive your policy again; it does not automatically
discover a file in the working directory.

| Key | Default | Meaning |
|---|---|---|
| `primary` | `null` | primary metric; `null` means `accuracy` (classification) or `rmse` (regression). Any metric listed in [Analysis](analysis.md) works |
| `max_quality_drop` | 0.01 | limit for the quality drop |
| `max_disagreement` | 0.05 | limit for the disputed share |
| `max_confident_disagreement` | 0.01 | limit for confident disputes (classification) |
| `max_latency_ratio` | 2.0 | limit for candidate/incumbent latency |
| `max_memory_ratio` | 1.5 | limit for candidate/incumbent memory increase |
| `max_slice_drop` | 0.05 | limit for the worst slice |
| `min_slice_n` | 30 | minimum distinct rows for a slice to count |
| `max_ece_increase` | 0.02 | limit for the ECE increase |
| `confident_prob` | 0.9 | classification: both models at least this sure counts as confident |
| `regression_gap_std` | 0.25 | regression: a gap above this many target standard deviations is a dispute |
| `near_margin` | 0.2 | classification: top-two-class margin below this is near the boundary |

Without `--thresholds`, the command identifies the packaged default source and the saved
swap JSON has `"thresholds_default": true`. With a non-empty custom policy it is `false`,
even if the supplied values equal the defaults. The CLI prints the effective JSON, and
reports show the origin and every effective setting.
`threshold_provenance` distinguishes `defaults`, `partial`, and `explicit` policies and lists
provided and inherited keys. The legacy boolean alone does not mean all defaults were replaced.

For example, HistGB to LightGBM on the diabetes regression set can exceed the default 5% disagreement
limit even though LightGBM has a lower RMSE: with a continuous target, two good models often differ
by more than 0.25 standard deviations on a few percent of rows. Raise `regression_gap_std` or the
limit to what a swap actually costs you. A PASS is not a production-safety guarantee.

### Validation and explicit policies

Custom files must contain a JSON object with known keys and finite numeric values, not strings
or booleans. `primary` is null or a supported metric compatible with the dataset task.
`min_slice_n` is a positive integer; resource ratios are positive; confidence, margin, and
disagreement rates are between 0 and 1. Other settings are non-negative, except quality,
slice, and ECE degradation limits may be negative to require improvement.

Pass `--require-thresholds` to `swap`, `report`, or full `benchmark` to require a non-empty
explicit policy (recommended in CI). Missing or empty policies fail rather than falling back.
This checks that a policy was supplied, not that every inherited default has been reviewed.
Partial policies remain supported with this flag. CLI output and reports explicitly list their
inherited defaults; only supplying every setting produces `kind: explicit`.
Smoke benchmarks reject threshold options because they do not produce swap verdicts.

### Workbench

Choose **Full training splits**, then expand **Swap acceptance policy (full runs)**.
The default view shows the packaged JSON. Choose **Custom policy** to edit it or use
**Import custom JSON**. The effective-limits preview merges your overrides with defaults;
the server validates all values before training starts. **Reset to defaults** discards edits.

**Download effective policy** saves a JSON copy for later import or CLI use. Each full
experiment stores its complete effective policy in `results/web/<job-id>/thresholds.json`
and its job record, including whether defaults were used. Reopening an experiment shows
the saved policy, not today's defaults, with a download action. Reports display the policy
alongside swap results. Smoke runs disable policy controls and save no swap policy.

## Using it in CI

Example GitHub Actions steps using registered names:

```yaml
- run: quibble-ml run --sweep --backends histgb,lightgbm --datasets wine --cv 5x2 --model-seeds 0,1 --device cpu --out results_ci
- run: quibble-ml swap --incumbent histgb --candidate lightgbm --dataset wine --out results_ci --thresholds limits.json --require-thresholds
```
A non-zero exit fails the job. Decide explicitly whether INCONCLUSIVE (exit 2) should block.
In Bash, the following permits PASS and INCONCLUSIVE but preserves unexpected errors:

```bash
status=0
quibble-ml swap --incumbent histgb --candidate lightgbm --dataset wine --out results_ci || status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi
```

## The sanity candidate
`weak` is deliberately handicapped. The controlled demo and regression tests expect its
swap from HistGB to FAIL under default thresholds. That is a harness sanity check,
not a guarantee for every dataset, split, or user-supplied threshold.
