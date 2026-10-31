---
title: Analysis
description: Paired prediction analysis, cluster-bootstrap uncertainty, calibration, slices, and disagreements.
---

## Analysis

Two methods can both get 80 answers right out of 100 while disagreeing on
which answers are wrong. Analysis opens that scorecard: it checks changed
answers, confidence, difficult groups, and how much uncertainty remains.
It describes the saved test, not a guarantee about tomorrow's data.

`quibble-ml analyze --dataset D` pools predictions across splits and writes results to
`<out>/D/analysis/`. All backends are **aligned**: only splits completed by every backend are used
(dropped splits are listed). Backends with no usable predictions are excluded and must be
reported as missing/failed, not treated as competitors. The model seed defaults to the smallest found; choose with
`--model-seed`.

## Output files
| File | Contents |
|---|---|
| `summary.csv` | One row per backend: primary metric, `ci_lo`/`ci_hi` (test-set noise), `cv_lo`/`cv_hi` (approximate corrected split interval; 5+ splits), `split_std`, `n_seeds`, `seed_std`, `reseed_disagreement`, plus secondary metrics |
| `calibration.csv` | Classification only: reliability table (10 bins) per backend |
| `calibration_classwise_summary.csv` | One-vs-rest class ECE, class-column mapping and distinct positive/negative support |
| `calibration_classwise.csv` | Every class and all ten bins, including empty bins, counts, weights and observed frequencies |
| `slices.csv` | Primary metric per slice and backend, with the spread between backends |
| `disagreements_<a>_vs_<b>.csv` | Every disputed prediction pair, with row traits and the row's features |
| `interaction_rules.json`, `interaction_rules.csv` | Opt-in frozen exploratory rules and aggregate paired evaluation summaries |
| `relational_slices.csv` | Relational-only descriptive paired quality/disagreement by history, recency, missing relations and cold start; explicit distinct-row/group support |
| `analysis.json` | Machine-readable summary of all of the above, plus `data_flags` and the split scheme |
| `swap_<a>_to_<b>.json` | Written by `quibble-ml swap` |

## Metrics
Primary metric: `accuracy` for classification, `rmse` for regression (override with `--metric`).

Relational diagnostic boundaries are fixed and label-free. Empty slices retain
null measurements; small slices and one-group slices have limited support.
These summaries have no confidence bounds, causal interpretation, or additional
swap gates. Correlated prediction examples are counted separately from their
entity clusters. See [multi-table evaluation](testing-your-data.md).

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
- `ci_lo`, `ci_hi`: **cluster bootstrap.** Each resample draws weights from a multinomial over
  units; the unit is a unique data row, or a whole group when the dataset declares `group_column`, so
  correlated rows are never treated as independent. A row that appears in several CV repeats keeps
  one weight. The interval is the 2.5th to 97.5th percentile of `QUIBBLE_BOOT` resamples
  (default 500). It captures **test-set sampling noise only**.
- `cv_lo`, `cv_hi`: a Nadeau-Bengio corrected resampled interval over the per-split values. It
  inflates the variance by `1/J + n_test/n_train` to approximate overlapping training sets and
  retraining variation. It needs 5 or more splits; otherwise it is `NaN`. Swap uses the outer
  bounds of this interval and the bootstrap. This does not establish calibrated verdict error rates.
- The resampling seed is fixed, so results are reproducible and *paired*: two backends are compared
  on identical resamples.
- Resamples with an undefined metric (for example an AUC where a resample has one class) are dropped
  and counted. With fewer than 20 valid resamples the bootstrap interval is `NaN`, and swap treats
  that as INCONCLUSIVE.
- `split_std` is the standard deviation of the primary metric across splits.
- Marginal interval overlap alone neither decides a paired difference nor proves equality.
  Use the paired degradation interval from a swap check and application-specific limits.

## Seeds
With two or more model seeds saved for a backend, the summary adds `seed_std` (spread of the primary
metric across seeds) and `reseed_disagreement` (how often two seeds of the same backend disagree).
The latter is the noise floor for any candidate-versus-incumbent disagreement, and swap subtracts it.
Raw disagreement remains in the check note. The adjustment does not propagate noise-estimate
uncertainty and should not conceal concentrated harm in a small group.
The main analysis still uses one seed (`--model-seed`, default the smallest found). Backends that do
not control their seed (AutoGluon) are recorded with `seed_controlled: false`.

## Calibration
`ece` is a single number; `calibration.csv` has the bins (`bin`, `n`, `mean_conf`, `accuracy`). A
well-calibrated model has `accuracy` close to `mean_conf` in each bin. Multiclass calibration uses
the top-label confidence only. Its values and weighting are unchanged.

### Per-class reliability

The classwise exports ask a different question: when the model gives a particular
class a certain probability, how often does that class actually occur?
Each recorded probability column becomes a one-vs-rest diagnostic. Source labels
are shown when recorded; legacy unrecorded labels remain unknown, with an explicit
column index instead.

Bins have width 0.1: lower boundaries are inclusive, upper boundaries exclusive,
except the final bin includes probability 1. Every distinct source row has total
weight one. Repeated predictions divide that weight equally without averaging
their probabilities first. `count` is the number of prediction entries;
`unique_rows` and `weight` make repetition visible. The same row can contribute
fractional weight to several bins.

Empty bins have null mean probability and frequency. Classes with no positive or
no negative examples have null class ECE and an explicit support status. Fewer
than twenty distinct positive or negative rows are flagged as rare. This flag is
a descriptive warning, not a validated safety threshold. Class ECE otherwise sums
weighted absolute bin gaps, independently from top-label ECE.

These summaries have no confidence intervals. Existing grouped bootstrap behavior
is unchanged; no independence or temporal-calibration guarantee is added.
Classwise diagnostics do not change swap verdicts or apply post-hoc calibration.
Reports show bounded tables and figures; the CSV and JSON exports retain every
class and bin, including in archived reports.

## Slices
Slices show whether an average hides a weak segment. Defined on the full dataset (diagnostic only):
- `missing=none` / `missing=some`: rows with no missing values vs at least one.
- `rare_category=yes`: rows containing a category value seen in under 1% of rows.
- For the three numeric features with the highest **mutual information** with the target:
  `feature=low`, `=mid`, `=high` (terciles). Mutual information needs no ordering of class labels
  and does not assume a linear relationship.

A slice is reported only if it has at least `--min-slice-n` **distinct rows** (default 30; a row
predicted in several CV repeats counts once). `n` in `slices.csv` is that row count. `spread` is the
best minus the worst backend on that slice. Slices are **exploratory**: they are chosen using the
target and ranked by spread between backends, so a large spread in a small slice is often noise.
`slices.csv` has no intervals. Swap bootstraps the maximum over the fixed selected masks, without
repeating target-driven discovery. Family-wise and selection-adjusted coverage remain unverified.

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

## Data flags
Before reporting, quibble runs cheap checks and prints `DATA FLAG` lines, saved as `data_flags` in
`analysis.json` and shown with swap and gate results. They never block a run or change a verdict.

| Flag | Meaning |
|---|---|
| `single_feature_predicts_target` | a depth-3 tree on one feature scores about perfectly (AUC or accuracy at least 0.99, R^2 at least 0.98). Check the feature is known at prediction time and not derived from the target |
| `id_like_column` | almost every value is unique; an ID can let a model memorise rows |
| `duplicate_rows` | at least 5% of rows have identical features; duplicates can land on both sides of a split |
| `time_like_column_random_split` | the column name suggests dates or time, but the split is random. Declare `time_column` and use a temporal split |

These are heuristics: silence does not prove there is no leakage, and a flag is a prompt to
investigate. See [Testing your own data](testing-your-data.md#group-and-temporal-splits).

## Opt-in disagreement interaction rules

The bounded explainer supplements, rather than replaces, trait and single-feature
slices. Select an **ordered backend pair** and a **reviewed safe feature allowlist**:

```bash
quibble-ml analyze --dataset D --out results \
  --interaction-pair histgb:lightgbm \
  --interaction-features age,income,region \
  --interaction-exclude customer_id,event_time,final_outcome \
  --interaction-seed 17 --interaction-min-rows 20 --interaction-min-groups 2
```

Only allowlist features are read by the explainer. Targets, declared dropped/ID/time/group
columns, reserved target/ID/prediction names, and explicitly excluded columns are rejected.
This is not automatic leakage detection: review column roles and exclude arbitrary-named
post-outcome fields yourself. Optional `X.attrs["quibble_feature_roles"]` can declare unsafe
roles; only roles `feature` and `predictor` are eligible. Existing general analysis and
disagreement CSV generation retain their own behavior.

The implementation:

* Aligns the named pair's common completed splits, independently of other backends.
  Exports its recorded recipes; legacy recipes remain explicitly unknown (`null`).
  Optional `--interaction-recipes FAMILY_SHA_A:FAMILY_SHA_B` asserts the ordered
  `family_sha256` identities and rejects mismatches.
* Consolidates repeat entries to the mean disagreement rate per original source row.
  It never selects a favorable repeat or counts repeats as independent rows.
* Uses a fixed-seed permutation to split distinct rows equally into discovery and
  evaluation. Declared groups are kept wholly on one side. Support is checked in
  distinct rows **and**, when present, distinct groups, not prediction entries.
* Fits numeric medians and a depth-at-most-two squared-error regression tree on
  discovery only. `--interaction-max-features` caps the explicit allowlist at 16;
  `--interaction-max-rules` caps exports at 4 (defaults 16 and 4). Leaves are ranked
  by discovery disagreement, never by evaluation performance.
* Freezes structured `<=` / `>` predicates before evaluating them. A `multi_feature`
  leaf uses two different source features; repeated splits on one feature remain
  `single_feature`. No predicate is executable code.

Numeric nulls and nonfinite values use the median of discovery values clipped to
+/-1e30 (zero for all-missing discovery columns). Transformed numbers are clipped
to the same range, represented as float32, and compared to float64 tree thresholds.
Categorical values use SHA256 of `type-name:value`, first eight digest bytes modulo
16, with nulls encoded as -1. Unseen categories follow exactly the same stateless
hash mapping; collisions are possible and bucket intervals are not natural category
orderings. No private category vocabulary is exported. Hashing is not an anonymity
guarantee.

Classification disagreement means different predicted classes. For this explainer
only, regression disagreement uses an absolute prediction gap greater than the
configured regression multiplier times the distinct-discovery-row target standard
deviation (one if constant). That normalization is frozen before evaluation; the
existing disagreement CSV definition is unchanged.

`interaction_rules.json` records the specification, ordered pair and recipes,
seed/split method and counts, support limits, fitted schema, missing/unseen behavior,
software versions, structured predicates and human-readable rule text.
`interaction_rules.csv` contains per-rule discovery/evaluation support, paired
evaluation disagreement, and quality difference. Both statistics first average
within each original row, then equally across evaluation rows. Positive quality
difference favors A: accuracy A minus B, or MAE B minus A for regression.
Unsupported evaluation leaves have blank estimates, not zeroes. Empty discoveries
have an explicit JSON status and a header-only CSV.

These are **exploratory selected rules**, not independent confirmation. Subdividing
an existing CV/prediction pool leaves shared-model dependence even with disjoint
rows or groups. Confirmatory use needs a locked holdout unused by tuning, model
selection, or discovery. Rules never enter automatic swap PASS/FAIL checks.
New exports contain aggregates and learned thresholds/medians, not raw feature
records, category vocabularies, row IDs, or group IDs. Existing disagreement CSVs
still contain their original row-level evidence and features; review those before
sharing an entire archive.

Python callers can pass `interaction=InteractionSpec(...)` from
`quibble_ml.interactions` to `analyse()` or report `build()`. A normal analysis
without this opt-in removes prior interaction exports rather than leaving stale
rules. Report regeneration preserves a saved explicit specification and recomputes
rules from current evidence; see [Reports](reports.md#interaction-rule-evidence).
