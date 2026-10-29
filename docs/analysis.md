---
title: Analysis
description: Paired prediction analysis, cluster-bootstrap uncertainty, calibration, slices, and disagreements.
---

## Analysis

`quibble analyze --dataset D` pools predictions across splits and writes results to
`<out>/D/analysis/`. All backends are **aligned**: only splits completed by every backend are used
(dropped splits are listed). Backends with no usable predictions are excluded and must be
reported as missing/failed, not treated as competitors. The model seed defaults to the smallest found; choose with
`--model-seed`.

## Output files
| File | Contents |
|---|---|
| `summary.csv` | One row per backend: primary metric, `ci_lo`, `ci_hi`, `split_std`, plus secondary metrics |
| `calibration.csv` | Classification only: reliability table (10 bins) per backend |
| `slices.csv` | Primary metric per slice and backend, with the spread between backends |
| `disagreements_<a>_vs_<b>.csv` | Every disputed prediction pair, with row traits and the row's features |
| `analysis.json` | Machine-readable summary of all of the above |
| `swap_<a>_to_<b>.json` | Written by `quibble swap` |

## Metrics
Primary metric: `accuracy` for classification, `rmse` for regression (override with `--metric`).

| Metric | Definition | Better |
|---|---|---|
| `accuracy` | share of rows whose argmax matches the label | higher |
| `auc` | ROC AUC; binary uses the positive-class probability, multiclass uses one-vs-rest | higher |
| `log_loss` | mean negative log-probability of the true class (probabilities clipped at 1e-7) | lower |
| `brier` | mean over rows of the sum over classes of squared probability error | lower |
| `ece` | expected calibration error on top-label confidence, 10 equal-width bins | lower |
| `rmse`, `mae` | root mean squared / mean absolute error | lower |
| `r2` | 1 - SSE / total sum of squares | higher |

## Confidence intervals
- Method: **cluster bootstrap over unique data rows.** Each resample draws row weights from a
  multinomial; a row that appears in several CV repeats keeps one weight. Metrics are recomputed with
  those weights. The interval is the 2.5th to 97.5th percentile of `QUIBBLE_BOOT` resamples
  (default 500).
- The resampling seed is fixed, so results are reproducible and *paired*: two backends are compared
  on identical resamples.
- With fewer than 20 valid resamples (e.g. an AUC that cannot be computed) the interval is `NaN`.
- **What it captures:** test-set sampling noise.
  **What it does not:** variability from retraining on different training sets. `split_std`, the
  standard deviation of the primary metric across splits, gives a rough sense of that.
- Marginal interval overlap alone neither decides a paired difference nor proves equality.
  Use the paired degradation interval from a swap check and application-specific limits.

## Calibration
`ece` is a single number; `calibration.csv` has the bins (`bin`, `n`, `mean_conf`, `accuracy`). A
well-calibrated model has `accuracy` close to `mean_conf` in each bin. Multiclass calibration uses
the top-label confidence only.

## Slices
Slices show whether an average hides a weak segment. Defined on the full dataset (diagnostic only):
- `missing=none` / `missing=some`: rows with no missing values vs at least one.
- `rare_category=yes`: rows containing a category value seen in under 1% of rows.
- For the three numeric features most correlated (absolute Pearson) with the target: `feature=low`,
  `=mid`, `=high` (terciles).

A slice is reported only if it has at least `--min-slice-n` pooled predictions (default 30; note that
CV repeats count a row once per repeat). `spread` is the best minus the worst backend on that slice.

## Disagreement
For every pair of backends:
- **Disputed:** classification = different predicted class; regression = gap above 0.25 target
  standard deviations.
- **confident_both** (classification): disputed, and both models at least 90% confident.
- **Who is right:** classification reports the share of disputed rows where only A, only B, or
  neither was correct; regression reports which was closer to the truth and the median gap.
- **Trait lift:** for each trait, the share among disputed rows divided by the share among all rows.
  A lift of 3x means disputes are three times as likely to involve that trait. Traits:
  `has_missing`, `rare_category`, `near_boundary` (classification: mean top-two-class margin of the two
  models below 0.2) and `extreme_target` (regression: target outside its 10th to 90th percentile).

### Columns of `disagreements_<a>_vs_<b>.csv`
Classification: `split, row, y, pred_a, pred_b, conf_a, conf_b, a_correct, b_correct, disputed,
confident_both, near_boundary, has_missing, rare_category`, then the row's features as `x_<name>`
(first 40 columns).
Regression: `split, row, y, pred_a, pred_b, gap_std, disputed, a_closer, extreme_target, has_missing,
rare_category`, then `x_<name>`.
`row` is the 0-based position in the loaded dataset. `a` and `b` are the backends in the file name.

## Interpreting a typical result
- *Intervals overlap, disputes are rare, and most disputes are near the decision boundary:*
  a candidate for further interchangeability checks, not proof of equivalence or ambiguity.
  Check paired quality limits, slices, and operational budgets before a swap.
- *Point estimates are similar but disputes are common:* similar observed average quality,
  different behavior. Inspect the paired quality difference and disputed-row CSV.
- *One model wins on a slice but loses overall:* the average is hiding a segment that matters.
