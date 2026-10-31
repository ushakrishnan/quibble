---
title: Reports
description: Generated report structure, evidence-based interpretation, figures, and regeneration.
---

## Reports

```bash
quibble-ml report [--out results] [--datasets a,b] [--swap histgb:lightgbm] [--thresholds limits.json] [--require-thresholds]
```
Writes `<out>/report/report.md` plus PNG figures next to it. Datasets default to every folder under
`--out` that contains results. `--swap INCUMBENT:CANDIDATE` can be repeated and is evaluated for each
dataset.

Figures need matplotlib, included in the `report` and `web` extras. Without it the tables are still written and
a note says the figures were skipped.

## Structure
1. **Setup:** datasets, backends, split scheme(s), device(s), hardware, installed package versions,
   git hash. A warning appears if any result came from a parallel (`--jobs > 1`) sweep.
   A device summary calls out AUTO resolution and historical mixed CPU/GPU measurements.
   Participation records GPU-only exclusions and reasons separately from failures.
2. **Per dataset:**
   - a one-line description (task, rows, features, size tier, categorical and missing shares, number
     of splits, model seed) and a note if splits were dropped,
   - a participation table with selection, cached/executed/incompatible status, successful fits,
     errors, refusals, missing cells, and fits used,
   - an **Interpretation and recommended use** section with conditional quality-first,
     latency-sensitive, and memory-constrained shortlists, each justified by measured values,
   - **Quality and cost** table: primary metric with its 95% interval, secondary metrics, `split_std`,
     and mean `fit_s`, `single_row_p50_ms`, `peak_rss_mb` (and VRAM if available), plus the
     quality-versus-cost figure (`fig_<dataset>_quality_cost.png`),
   - **Where the backends disagree:** heatmap of disputed shares
     (`fig_<dataset>_disagreement.png`) and a table per pair: disputed count and share, who is right
     when they differ, the most over-represented trait (only shown if its lift is at least 1.5x,
     otherwise "none notable"), and a link to the CSV of disputed rows,
   - **Calibration** (classification): reliability curves (`fig_<dataset>_calibration.png`),
   - **Per-class calibration:** one-vs-rest ECE and support for up to twenty
     backend/class entries, reliability panels for the first six classes, and
     links to complete class-summary and bin CSVs. Empty and one-sided support,
     rare classes, repeated-row weighting, and unknown legacy labels stay visible.
     These descriptive diagnostics do not alter swap policy,
   - **Slices:** the five slices with the largest spread between backends,
   - **Swap check** sections, if requested, with the default/custom policy source and a table
     of all effective settings, including defaults inherited from omitted custom keys.
3. **What failed or refused:** a backend x dataset matrix. Cells read `ok n/n`, `refused k/n: <reason>`
   or `error k/n: <message>`, with `(m ok)` when some runs succeeded. This compact matrix
   prioritizes refusals over errors when both occur; the participation and detailed failure
   tables preserve the full counts and error text.
4. **Skipped datasets** that could not be analysed, with the reason.
5. **How to read this:** the caveats (intervals, disputes, slices, timing).

## Interpreting recommended uses

Every dataset gets interpretation, including smoke checks and datasets that cannot be
analyzed. Smoke interpretation describes completed inference, sample sizes, observed
scores/resources, equal scores or disagreements, partial failures, and next steps, without
deployment rankings or swap approval. Unanalyzable datasets explain why no comparison
is available rather than omitting interpretation entirely.

The generated prose follows the pattern "We should evaluate this backend for this use case,
since these measurements support it." It is deterministic and derived from the saved results,
not from general claims about a model family.

- Quality-first shortlists follow the observed primary metric: higher accuracy or AUC, lower
  RMSE, and so on. Tied point estimates are identified without claiming statistical equivalence.
- Latency and memory shortlists name the lowest valid measurements and also state the candidate's
  measured quality. A cheap model is not recommended unconditionally if its quality is unacceptable.
- Among tied quality leaders, a backend with both the lowest latency and peak RSS is preferred
  for further evaluation. Conflicting resource tradeoffs do not produce an invented winner.
- Parallel runs suppress resource recommendations and ask for a `--jobs 1` rerun. Missing,
  nonfinite, zero, or negative resource measurements do not produce resource winners.
- The internal `weak` sanity backend is never recommended for deployment.
- Requested swap checks explain whether to consider a replacement, keep the incumbent under
  the configured limits, or defer a decision. Failed and inconclusive checks include their
  observed values and limits. A swap FAIL can mean excessive latency or memory even when
  prediction quality improved; it does not mean model execution failed.

Quality and cost tables use the same selected model seed and common successful splits.
Swap latency/memory checks separately take medians of usable runs for each backend; with incomplete
coverage their contributing splits can differ from the quality pool.
Failure coverage remains visible in the interpretation and failure matrix. Without an evaluated
swap verdict, shortlists do not approve a replacement.

Recommendations apply only to the measured dataset and hardware. Marginal confidence-interval
overlap alone does not establish equivalence or decide the paired difference; use paired swap
checks. Bootstrap intervals capture test-row or group sampling; corrected split intervals,
available with 5 or more splits, approximate retraining variation under covariance assumptions.
The combined verdict is not calibrated for false-PASS or false-FAIL rates. Validate against
representative application data, deployment budgets, and meaningful swap thresholds before use.

Each dataset and swap section names its split design and displays informational data flags.
Swap tables preserve check notes, including unavailable intervals and missing retraining
variation. Policy tables identify explicit settings versus inherited packaged placeholders.
These qualifications survive report display and archival, including otherwise passing verdicts.
Detected saved-setting mismatches are listed separately: reused evidence is not newly
evaluated under the requested configuration.

## Regenerating
The report reuses saved predictions and records, so adding a `--swap` or changing thresholds
does not retrain models. Analysis also reloads the original dataset for traits and slices;
an absent OpenML cache can require network access even during report generation.
Pass the same `--thresholds` file when regenerating to preserve your policy; it is not
auto-discovered. Without it, reports use the [packaged defaults](../quibble_ml/swap_thresholds.json).
`--require-thresholds` rejects a missing or empty policy before writing a report.
Browser full runs save an effective policy beside their job record for reuse.
Custom runs also retain the original overrides so inherited defaults remain identifiable.
Demo and report share the same results root, defaulting to `results`. Every report generation
saves a dated snapshot under `<out>/runs/YYYY-MM-DD/<snapshot-id>/` containing the report,
figures, and generated dataset evidence (records, prediction/target/index arrays, and analysis).
It does not copy the full original feature table. The stable latest report remains `<out>/report/report.md`;
`<out>/report/history.md` links to completed snapshots. Raw-run snapshots also preserve older
records and predictions before retraining and after training completes.

## Participation and model artifacts

The model participation table separates selection, execution/cache status, completed fits,
errors, refusals, missing cells, and the number of fits used in the aligned comparison.
`invocation.json` records the latest sweep request. Preset comparisons exclude incompatible
cached backends rather than presenting them as fresh measurements. Older results without
invocation metadata show selection as unknown.
The device section distinguishes requested policy (`cpu`, `cuda`, `auto`), resolved sweep
device, attempted backend device, and actual completed execution. Mixed CPU/GPU results
carry an explicit hardware-comparison warning. Auto alone does not mean mixed: a CPU-only
auto run is labelled auto but not mixed. Cached measurements retain their original labels.
Execution/resource failure tables include full error text and failure kind. Failed fits are
not counted as successful comparisons and never trigger an unreported CPU retry.

AutoGluon's internal validation leaderboard is saved under the dataset's analysis directory
after timing completes. It does not evaluate every model on held-out test labels.
`--keep-models` optionally retains fitted tree and AutoGluon models under `<out>/artifacts/`;
foundation fitted estimators are explicitly unsupported. Retention and diagnostic I/O are
excluded from reported fit/prediction timings.

For a retained tree adapter, use its run record to find the artifact:

```python
import json
from pathlib import Path
import joblib

out = Path("results")
record = json.loads((out / "wine" / "histgb_ss0_ms0.json").read_text())
adapter = joblib.load(out / record["model_path"] / "model.joblib")
predictions = adapter.predict(X_new)
```

`X_new` must contain the original feature columns as a pandas DataFrame. Classification
returns probabilities; regression returns numeric predictions. Only load trusted model
artifacts, and retain the recorded package versions. For an AutoGluon artifact, use
`TabularPredictor.load(str(out / record["model_path"]))` instead.

Retained fits use unique paths and are not overwritten or copied into each snapshot.
When moving an archived report, copy retained model directories separately if needed:
report evidence/leaderboard CSVs are archived, but model links refer to the original artifact root.
Smoke reports are prominently labelled, contain no deployment recommendations or swap
verdicts, interpret bounded setup outcomes, and are separate from the main benchmark output.

Local inputs also show their source path, target, file hash, data/manifest fingerprint,
and encoded classification-label order. Reports reject changed local sources rather than
deriving slices from a different table. Keep the original table/manifest for regeneration;
see [Testing your own data](testing-your-data.md).

Datasets created by the [experiment workbench](experiment-workbench.md) also include
a synthetic-experiment warning, generation seed/version, and generation metadata hash.
Keep `generation.json` with the original CSV and manifest. Scores characterize the
specified simulation, not real-world operations.

## Editing for publication
`report.md` is plain Markdown with relative image and CSV links. Copy a whole dated snapshot,
including its dataset folders, when moving a report and its evidence. Review the generated
interpretation and add application-specific context before
publishing. Regeneration overwrites the report, including any manual edits; keep publication notes
in a separate document.

## Interaction rule evidence

Reports accept the same opt-in flags as analysis:

```bash
quibble-ml report --datasets D --out results \
  --interaction-pair histgb:lightgbm \
  --interaction-features age,income,region \
  --interaction-exclude customer_id,final_outcome \
  --interaction-seed 17 --interaction-min-rows 20 --interaction-min-groups 2
```

The interaction section names the ordered pair, discovery/evaluation split,
support status, rule type, paired disagreement and quality difference. It visibly
qualifies selection and shared-model dependence: splitting an existing CV pool is
not independent confirmation, and no interaction rule changes a swap verdict.
Insufficient evaluation support suppresses estimates; no rules is a valid result,
not evidence that the models are equivalent. Recipe identities, schema and exact
missing/unseen-category semantics are available in the linked JSON.

If analysis previously requested an explainer, report generation restores its
**specification**, not its learned rules, and recomputes everything against current
paired evidence. Explicit report flags override the saved specification. Invalid
pair/recipe requests produce diagnostics rather than silently displaying cached
rules. The Python `build(..., interaction=InteractionSpec(...))` API is also opt-in.

Reports link `interaction_rules.json` and `interaction_rules.csv` in the dataset's
analysis directory. Dated snapshots automatically include both files and rewrite
report links to the archived copies. Copy the whole snapshot for a self-contained
export. New rule exports contain no raw feature records, category vocabulary,
row IDs or group IDs, but learned thresholds and aggregate statistics still need
privacy review. Existing disagreement CSVs retain their original raw-feature and
row-level semantics; this opt-in does not make the entire archive public-safe.

See [Interaction analysis](analysis.md#opt-in-disagreement-interaction-rules) for
feature review, support, encoding, and reproducibility details.
