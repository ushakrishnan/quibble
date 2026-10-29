---
title: Swap checks
description: Paired replacement checks, configurable limits, verdicts, and CI integration.
---

## Swap checks

`quibble swap` answers: *is replacing the incumbent with the candidate safe on this dataset?*

```powershell
quibble swap --incumbent histgb --candidate lightgbm --dataset credit-g
$LASTEXITCODE    # 0 PASS, 1 FAIL, 2 INCONCLUSIVE
```
It reads existing results (run a CV sweep first) and writes
`<out>/<dataset>/analysis/swap_<incumbent>_to_<candidate>.json`.

## Verdicts
Each check is one of:
- PASS: the whole 95% interval is within the limit.
- FAIL: the whole interval is beyond the limit.
- INCONCLUSIVE: the interval straddles the limit. More representative held-out rows can
  help; extra CV repeats assess split sensitivity but do not add independent rows.
  Other datasets test generality, not this dataset's uncertainty.

For checks without an interval (calibration, latency, memory, coverage) the verdict is PASS or FAIL
from the point value. The **overall** verdict is FAIL if any check fails, else INCONCLUSIVE if any is
inconclusive, else PASS.
If fewer than 20 valid bootstrap resamples are available, interval checks fall back to
the point value. A PASS therefore is not always interval-backed.

## The checks
| Check | What is measured | Default limit |
|---|---|---|
| `coverage` | the candidate completed every split the incumbent did | all splits |
| `quality drop (<metric>)` | how much worse the candidate is on the primary metric (paired bootstrap) | 0.01 |
| `disagreement rate` | share of prediction pairs that are disputed | 5% |
| `confident disagreements` | classification: disputed and both models at least 90% sure | 1% |
| `calibration (ECE increase)` | classification: candidate ECE minus incumbent ECE (point value) | 0.02 |
| `worst slice` | largest quality drop over slices with enough rows (paired bootstrap) | 0.05 |
| `latency ratio (single-row p50)` | candidate / incumbent mean of `single_row_p50_ms` | 2.0x |
| `memory ratio (peak RSS)` | candidate / incumbent mean of `peak_rss_mb` | 1.5x |

Notes:
- Quality drop is positive when the candidate is worse. It is *absolute* for accuracy, AUC, R^2,
  log loss, Brier and ECE, and *relative* (a fraction) for RMSE and MAE. A candidate that is better
  shows a negative number.
- Worst slice takes the maximum over several slices and is **not corrected for multiple
  comparisons**. Treat it as a flag to investigate, not a verdict.
- Latency and memory are single-machine point comparisons averaged over usable runs,
  not necessarily only the common quality-analysis splits. Use `--jobs 1`
  results.
- If the candidate has no usable runs, `coverage` fails and the remaining checks are skipped.
- If the incumbent has no usable runs, the command exits with code 1 and a message.

## Changing the limits
Copy `swap_thresholds.example.json`, edit it, and pass `--thresholds my_limits.json` (to `swap` or
`report`). Only keys you list are overridden.

| Key | Default | Meaning |
|---|---|---|
| `primary` | `null` | primary metric; `null` means `accuracy` (classification) or `rmse` (regression). Any metric listed in [Analysis](analysis.md) works |
| `max_quality_drop` | 0.01 | limit for the quality drop |
| `max_disagreement` | 0.05 | limit for the disputed share |
| `max_confident_disagreement` | 0.01 | limit for confident disputes (classification) |
| `max_latency_ratio` | 2.0 | limit for candidate/incumbent latency |
| `max_memory_ratio` | 1.5 | limit for candidate/incumbent peak memory |
| `max_slice_drop` | 0.05 | limit for the worst slice |
| `min_slice_n` | 30 | minimum pooled predictions for a slice to count |
| `max_ece_increase` | 0.02 | limit for the ECE increase |

**The defaults are placeholders.** For example, HistGB to LightGBM on the diabetes regression set
fails the default 5% disagreement limit even though LightGBM has a lower RMSE: with a continuous
target, two good models often differ by more than 0.25 standard deviations on a few percent of
rows. Set limits to what a swap actually costs you.

## Using it in CI

Example GitHub Actions steps using registered names:

```yaml
- run: quibble run --sweep --backends histgb,lightgbm --datasets wine --cv 5x2 --model-seeds 0 --device cpu --out results_ci
- run: quibble swap --incumbent histgb --candidate lightgbm --dataset wine --out results_ci
```
A non-zero exit fails the job. Decide explicitly whether INCONCLUSIVE (exit 2) should block.
In Bash, the following permits PASS and INCONCLUSIVE but preserves unexpected errors:

```bash
status=0
quibble swap --incumbent histgb --candidate lightgbm --dataset wine --out results_ci || status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi
```

## The sanity candidate
`weak` is deliberately handicapped. The controlled demo and regression tests expect its
swap from HistGB to FAIL under default thresholds. That is a harness sanity check,
not a guarantee for every dataset, split, or user-supplied threshold.
