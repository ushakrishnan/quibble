"""One Markdown report (plus PNG figures) from a results directory.

  quibble-ml report [--datasets a,b] [--swap histgb:lightgbm] [--out results]
Writes <out>/report/report.md. Figures need matplotlib (uv sync --extra report);
without it the tables are still written.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import AnalysisError, analyse
from . import interactions
from .backends import BACKENDS
from .config import analysis_settings
from .history import snapshot
from .swap import SwapError, evaluate
from .thresholds import describe_policy, load_thresholds, resolve_thresholds

MORE_BETTER = {"accuracy", "auc", "r2"}


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def fmt(v, nd=4):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}g}"
    return str(v).replace("|", "/")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def load_runs(out, dataset):
    return [json.loads(p.read_text()) for p in sorted((Path(out) / dataset).glob("*_ss*_ms*.json"))]


def ok_run(r):
    return "refused" not in r and "error" not in r


def cost_table(runs):
    df = pd.DataFrame([r for r in runs if ok_run(r)])
    if df.empty:
        return pd.DataFrame(columns=["backend"])
    cols = [c for c in ("fit_s", "single_row_p50_ms", "peak_rss_mb", "peak_torch_vram_mb",
                        "model_pipeline_s", "adaptation_and_prediction_s") if c in df]
    for column in cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.groupby("backend")[cols].mean().reset_index()


def stage_cost_row(record):
    """Keep overlapping timing measurements separate; shared features are not model charges."""
    fit = record.get("fit_total_s")
    if fit is None and record.get("fit_s") is not None:
        fit = record["fit_s"] + (record.get("load_s") or 0)
    batch = record.get("batch_prediction_s", record.get("adaptation_and_prediction_s"))
    if batch is None and record.get("predict_ms_per_row_batch") is not None \
            and record.get("n_test") is not None:
        batch = record["predict_ms_per_row_batch"] * record["n_test"] / 1000
    pipeline = record.get("model_pipeline_s")
    if pipeline is None and fit is not None and batch is not None:
        pipeline = fit + batch
    tuning = record.get("tuning") or {}
    return {"backend": record["backend"], "split": record["split_seed"],
            "model seed": record["model_seed"],
            "shared feature_total_s": (record.get("feature_costs") or {}).get("feature_total_s"),
            "search_s": record.get("search_s", tuning.get("search_s")),
            "refit_s": record.get("refit_s", tuning.get("refit", {}).get("duration_s")),
            "search_refit_s": record.get("search_refit_s", tuning.get("total_s")),
            "fit_total_s": fit, "batch_prediction_s": batch,
            "adaptation_and_prediction_s": record.get("adaptation_and_prediction_s"),
            "prediction_only_s": record.get("prediction_only_s"),
            "model_pipeline_s": pipeline}


def training_protocol_row(record):
    request = (record.get("recipe") or {}).get("requested", {})
    tuning = record.get("tuning") or {}
    search = tuning.get("protocol") or request.get("search", {})
    adaptation = request.get("adaptation", "legacy/unknown")
    if isinstance(adaptation, dict):
        adaptation = adaptation.get(record.get("task"), "task-specific recipe; task unavailable")
    if record.get("fine_tuning") is True:
        label = "fine-tuned at prediction time" if record.get("prediction_adapts") \
            else "fine-tuned during fit"
        if isinstance(adaptation, dict):
            details = [f"{key}={adaptation[key]}" for key in ("num_bag_folds", "max_steps")
                       if key in adaptation]
            label += "; " + ", ".join(details) if details else ""
        adaptation = label
    elif isinstance(adaptation, dict):
        adaptation = json.dumps(adaptation, sort_keys=True)
    return {"backend": record["backend"], "split": record["split_seed"],
            "model seed": record["model_seed"], "task": record.get("task", "unknown"),
            "campaign tier": record.get("campaign_tier", request.get("campaign_tier",
                                                                    request.get("tier", "legacy/unknown"))),
            "actual recipe tier": record.get("tier", request.get("tier", "legacy/unknown")),
            "adaptation": adaptation,
            "prediction adapts": record.get("prediction_adapts", False),
            "search status": tuning.get("status", request.get("search", {}).get("status", "unknown")),
            "search version": search.get("version"), "trials": search.get("trials"),
            "inner folds": search.get("inner_folds"), "metric": tuning.get("metric", search.get("metric")),
            "attempted fits": tuning.get("attempted_fits"),
            "successful fits": tuning.get("successful_fits"),
            "valid trials": sum(a.get("status") == "ok" for a in tuning.get("attempts", []))
                            if "attempts" in tuning else None,
            "selected trial": tuning.get("selected_trial")}


def protocol_and_cost_section(runs, rdir):
    protocols = pd.DataFrame([{"dataset": r.get("dataset", "unknown"), **training_protocol_row(r)}
                              for r in runs])
    stages = pd.DataFrame([{"dataset": r.get("dataset", "unknown"), **stage_cost_row(r)} for r in runs])
    protocols.to_csv(rdir / "training_protocol.csv", index=False)
    stages.to_csv(rdir / "stage_costs.csv", index=False)
    return ("## Recipe tiers, search and full model cost\n\n"
            "Campaign tier is the requested comparison, not a claim that every adapter was tuned. "
            "Foundation defaults may include explicitly recorded fine-tuning; defaults is not a "
            "synonym for zero-shot. Search is equal trials/folds, not equal realized compute.\n\n"
            + md_table(protocols) + "\n\n"
            "Failed/incomplete searches are not usable gate evidence. Attempted/successful fits "
            "include the final refit; valid trials require all inner folds. Full attempts, "
            "parameters, split hashes, failures and seeds remain in provenance JSON.\n\n"
            + md_table(stages) + "\n\n"
            "`search_refit_s` overlaps `fit_total_s`: never add them. `model_pipeline_s` is "
            "`fit_total_s + batch_prediction_s`, excluding repeated latency probes and shared "
            "feature construction. Prediction-adapting models load/fit during prediction: "
            "`adaptation_and_prediction_s` is inclusive and overlaps batch time, not an extra "
            "charge. Null isolated inference/loading/adaptation or single-row timings are "
            "unavailable, not zero. Fit-only costs must not hide prediction-time fine-tuning.\n\n"
            "`shared feature_total_s` is shown beside each model as a shared feature-pipeline "
            "reference. Do not add it to each model or sum it across backends. Cost rows are "
            "per cell, not a claim of equal resource use. "
            "[Protocols](training_protocol.csv); [stage costs](stage_costs.csv).\n")


def status_cell(runs):
    n = len(runs)
    ok = [r for r in runs if ok_run(r)]
    if len(ok) == n:
        return f"ok {n}/{n}"
    bad = [r for r in runs if "refused" in r] or [r for r in runs if "error" in r]
    kind = "refused" if "refused" in bad[0] else "error"
    why = (bad[0].get("refused") or bad[0].get("error"))[:70]
    return f"{kind} {len(bad)}/{n}: {why}" + (f" ({len(ok)} ok)" if ok else "")


def participation(out, datasets, runs_by, analyses, invocation):
    """Distinguish model selection, execution, and actual aligned analysis participation."""
    from .provenance import backend_metadata, license_label
    rows = []
    for ds in datasets:
        runs = runs_by[ds]
        selected_dataset = ds in invocation.get("datasets", [])
        requested = invocation.get("requested_backends", invocation.get("backends", [])) \
            if selected_dataset else []
        excluded = invocation.get("excluded_backends", {}) if selected_dataset else {}
        names = sorted(set(requested) | {r["backend"] for r in runs} |
                       {n for n, b in BACKENDS.items() if not getattr(b, "internal", False)})
        res = analyses.get(ds)
        for name in names:
            records = [r for r in runs if r["backend"] == name]
            cells = [c for c in invocation.get("cells", [])
                     if c["dataset"] == ds and c["backend"] == name]
            if cells:
                keys = {(c["split_seed"], c["model_seed"]) for c in cells}
                records = [r for r in records if (r["split_seed"], r["model_seed"]) in keys]
            used = [r for r in records if res and ok_run(r) and name in res["backends"]
                    and r["model_seed"] == res["model_seed"] and r["split_seed"] in res["splits"]]
            errors = sum("error" in r for r in records)
            refused = sum("refused" in r for r in records)
            missing = max(0, len(cells) - len(records))
            executions = sorted({c["execution"] for c in cells})
            differences = sorted({d for c in cells for d in c["differences"]})
            reason = ", ".join(differences)
            if name in excluded:
                reason = "excluded by GPU-only policy: " + excluded[name]
                records, used = [], []
                executions = ["excluded"]
                errors = refused = missing = 0
            if not used and not reason:
                reason = "not selected" if requested and name not in requested else \
                    ("no aligned usable predictions" if records else "not run")
            snapshots = [r.get("backend_metadata") for r in records] if records else [
                invocation.get("backend_metadata", {}).get(name) or
                (backend_metadata(name) if name in BACKENDS else None)]
            licenses = "; ".join(sorted({license_label(s) for s in snapshots}))
            recipes = sorted({r["recipe"]["family_sha256"] for r in records if r.get("recipe")})
            rows.append({"dataset": ds, "backend": name,
                         "license": licenses,
                         "license source": "runtime snapshot" if records and all(snapshots) else
                                           ("legacy/incomplete" if records else "catalog only; not run"),
                         "recipe": ", ".join(r[:12] for r in recipes) or "unknown/not run",
                         "selected": "yes" if name in requested else ("no" if requested else "unknown"),
                         "execution": ", ".join(executions) or ("saved" if records else "not run"),
                         "completed": sum(ok_run(r) for r in records), "errors": errors,
                         "refused": refused, "missing": missing, "used in comparison": len(used),
                         "reason": reason or "-"})
    return pd.DataFrame(rows)


def artifact_table(out, rdir, runs):
    rows = []
    for r in runs:
        if not r.get("leaderboard") and r.get("model_retention", "disabled") == "disabled":
            continue
        row = {k: r[k] for k in ("dataset", "backend", "split_seed", "model_seed")}
        row["retention"] = r.get("model_retention", "disabled")
        for key in ("model_path", "leaderboard"):
            path = out / r[key] if r.get(key) else None
            row[key] = (f"[open]({Path(os.path.relpath(path, rdir)).as_posix()})"
                        if path and path.exists() else ("missing" if path else "-"))
        rows.append(row)
    return pd.DataFrame(rows)


def device_summary(runs):
    rows = []
    for r in runs:
        backend = BACKENDS.get(r["backend"])
        actual = r.get("device_effective")
        if actual is None and ok_run(r):
            actual = "cpu" if backend is not None and not backend.gpu else r.get("device", "unknown")
        rows.append({
            "backend": r["backend"], "requested": r.get("device_requested", "unknown"),
            "resolved": r.get("device", "unknown"),
            "attempted": r.get("device_attempted", actual or "unknown"),
            "actual (completed)": actual if ok_run(r) and actual else "not completed",
        })
    if not rows:
        return "## Device policy and actual execution\n\nNo device measurements are available.\n"
    df = pd.DataFrame(rows)
    table = df.groupby(list(df.columns), dropna=False).size().reset_index(name="runs")
    actuals = set(df["actual (completed)"]) - {"not completed", "unknown"}
    lines = ["## Device policy and actual execution\n",
             "`auto` selects a device once per sweep; it does not benchmark both devices. "
             "`cpu` stays on CPU; `cuda` requires CUDA. There is no out-of-memory CPU fallback. "
             "New CUDA sweeps exclude methods that cannot meet GPU-only model execution. "
             "Preprocessing, metrics, and reporting still use CPU. Historical and cached "
             "measurements retain their original device labels.\n"]
    if len(actuals) > 1:
        lines.append("> MIXED CPU/GPU MEASUREMENTS: completed models used different devices. "
                     "Latency and memory comparisons reflect both model and hardware differences, "
                     "not an equal-device speed comparison.\n")
    if "auto" in set(df.requested):
        lines.append("> AUTO DEVICE POLICY: see resolved and actual devices below. Cached runs "
                     "retain their original device records; they are not relabelled for this machine.\n")
    lines.append(md_table(table) + "\n")
    return "\n".join(lines)


def smoke_interpretation(res, cost, runs):
    lines = ["### Interpretation\n",
             f"This is a bounded setup check for `{res['dataset']}`, not a model-selection "
             "or deployment recommendation.\n"]
    for _, row in res["summary"].iterrows():
        records = [r for r in runs if ok_run(r) and r["backend"] == row.backend
                   and r.get("split_seed") in res["splits"]]
        sizes = sorted({(r["n_train"], r["n_test"]) for r in records
                        if "n_train" in r and "n_test" in r})
        sizes_text = ", ".join(f"{tr} train/{te} test rows" for tr, te in sizes) or "bounded rows"
        resource = cost[cost.backend == row.backend]
        measurements = [f"`{res['metric']}`={fmt(row[res['metric']])}"]
        if not resource.empty:
            for key, label, unit in (("single_row_p50_ms", "single-row p50", "ms"),
                                     ("peak_rss_mb", "process peak RSS", "MB")):
                value = resource.iloc[0].get(key)
                if value is not None and np.isfinite(value) and value > 0:
                    measurements.append(f"{label}={fmt(value)} {unit}")
        lines.append(f"- `{row.backend}` completed inference on {sizes_text}: "
                     + "; ".join(measurements) + ".")
    if len(res["summary"]) > 1 and res["summary"][res["metric"]].nunique() == 1:
        lines.append("- All evaluated backends have the same observed primary score on this "
                     "small sample. This does not establish equivalence or generalization.")
    if res["pairs"]:
        disputed = sum(pair[2]["disputed"] for pair in res["pairs"])
        entries = sum(pair[2]["entries"] for pair in res["pairs"])
        lines.append(f"- Across backend pairs, {disputed}/{entries} pooled prediction pairs "
                     "were disputed. These counts reuse test rows across model pairs, "
                     "not additional independent observations.")
    if any(not ok_run(r) for r in runs) or res["dropped"]:
        lines.append("- Some fits failed, refused, or lost common splits. Setup is only partially "
                     "verified; inspect participation and failure details before a full run.")
    if any(r.get("jobs", 1) > 1 for r in runs if ok_run(r)):
        lines.append("- Parallel jobs can distort latency and memory; use `--jobs 1` for measurements.")
    lines.append("\nSmall-sample intervals and one split do not validate full-data quality, "
                 "memory capacity, or production latency. Next, run the same selected models "
                 "and dataset without `--smoke`, using a separate full-benchmark output root. "
                 "No swap approval is issued for smoke results.\n")
    return "\n".join(lines)


def interpretation(res, cost, runs, swap_results, smoke=False):
    """Explain conditional use cases using only this dataset's measured evidence."""
    if smoke:
        return smoke_interpretation(res, cost, runs)
    m = res["metric"]
    s = res["summary"]
    s = s[s.backend.map(lambda n: not getattr(BACKENDS.get(n), "internal", False))]
    s = s[np.isfinite(s[m])]
    lines = ["### Interpretation and recommended use\n",
             f"These are conditional shortlists for `{res['dataset']}` on the measured hardware, "
             "not general recommendations about a model family.\n"]
    if any(r.get("prediction_adapts") for r in runs if ok_run(r)):
        lines.append("Prediction-time fine-tuning is present. Use full `model_pipeline_s`, "
                     "including inclusive `adaptation_and_prediction_s`, not fit-only cost. "
                     "Missing single-row or pure-inference timings are unavailable and cannot "
                     "support a latency recommendation.\n")
    parallel = any(r.get("jobs", 1) > 1 for r in runs if ok_run(r))
    if s.empty:
        lines.append("- No model recommendation is available: there are no valid primary-metric "
                     "estimates for non-internal backends.")
    else:
        best = s[m].max() if m in MORE_BETTER else s[m].min()
        leaders = s[np.isclose(s[m], best, rtol=0, atol=1e-12)]
        names = ", ".join(f"`{n}`" for n in leaders.backend)
        direction = "highest" if m in MORE_BETTER else "lowest"
        if len(s) == 1:
            lines.append(f"- We should treat {names} as an evaluation candidate, not a comparative "
                         f"winner, since it is the only non-internal backend with a valid "
                         f"`{m}` estimate ({fmt(best)}).")
        else:
            lines.append(f"- For quality-first work, we should shortlist {names}, since "
                         f"{'they share' if len(leaders) > 1 else 'it has'} the {direction} "
                         f"observed `{m}` ({fmt(best)}). This is a point-estimate ranking, "
                         "not proof of superiority or equivalence.")
        measured = s.merge(cost, on="backend", how="left")
        if parallel:
            lines.append("- We should rerun with `--jobs 1` before choosing a model for latency "
                         "or memory, since these runs used parallel jobs and their resource "
                         "measurements can be distorted by contention.")
        else:
            for column, purpose, label, unit in (
                    ("single_row_p50_ms", "latency-sensitive single-row prediction",
                     "mean single-row p50 latency", "ms"),
                    ("peak_rss_mb", "memory-constrained processes", "mean process peak RSS", "MB")):
                valid = measured[np.isfinite(measured[column]) & (measured[column] > 0)] \
                    if column in measured else measured.iloc[:0]
                if valid.empty:
                    lines.append(f"- No {purpose} shortlist is available, since valid positive "
                                 f"`{column}` measurements are missing.")
                    continue
                value = valid[column].min()
                candidates = valid[np.isclose(valid[column], value, rtol=0, atol=1e-12)]
                names = ", ".join(f"`{n}`" for n in candidates.backend)
                quality = ", ".join(f"`{r.backend}`: {m}={fmt(r[m])}"
                                    for _, r in candidates.iterrows())
                lines.append(f"- We should evaluate {names} for {purpose}, since the lowest "
                             f"{label} among backends with valid measurements is "
                             f"{fmt(value)} {unit}. Measured quality: {quality}. Use this "
                             "shortlist only if that quality meets the application's requirements.")
            if len(leaders) > 1 and {"single_row_p50_ms", "peak_rss_mb"} <= set(measured.columns):
                tied = measured[measured.backend.isin(leaders.backend)]
                costs = tied[["single_row_p50_ms", "peak_rss_mb"]]
                if np.isfinite(costs.to_numpy()).all() and (costs > 0).all().all():
                    preferred = tied[(tied.single_row_p50_ms == tied.single_row_p50_ms.min())
                                     & (tied.peak_rss_mb == tied.peak_rss_mb.min())]
                    if len(preferred) == 1:
                        r = preferred.iloc[0]
                        lines.append(f"- Among the tied quality point estimates, we should prefer "
                                     f"`{r.backend}` for further evaluation, since it also has the "
                                     f"lowest measured latency ({fmt(r.single_row_p50_ms)} ms) and "
                                     f"peak RSS ({fmt(r.peak_rss_mb)} MB) within that tied group.")
    for inc, cand, result in swap_results:
        verdict = result["overall"]
        if getattr(BACKENDS.get(cand), "internal", False):
            lines.append(f"- We should not use `{cand}` for deployment, since it is an internal "
                         f"sanity-check backend. Its swap verdict is {verdict}.")
            if verdict == "PASS":
                continue
        if verdict == "PASS":
            lines.append(f"- We should consider `{cand}` as a replacement for `{inc}` under "
                         "the configured limits, since every evaluated swap check passed. "
                         "This is not a production-safety guarantee.")
        else:
            unresolved = [c for c in result["checks"] if c["verdict"] == verdict]
            reasons = "; ".join(f"{c['check']}: {c['observed']} (limit {c['limit']})"
                                for c in unresolved)
            if verdict == "FAIL":
                lines.append(f"- We should keep `{inc}` rather than replace it with `{cand}` "
                             f"under the configured limits, since these checks failed: {reasons}. "
                             "A swap FAIL is a policy-limit failure, not a failed model run.")
            else:
                lines.append(f"- We should defer replacing `{inc}` with `{cand}`, since the "
                             f"swap is INCONCLUSIVE: {reasons}. Gather more paired evidence "
                             "before deciding.")
    if not swap_results:
        lines.append("- No evaluated swap verdict is available; these shortlists alone do not "
                     "approve replacing an existing model.")
    if any(not ok_run(r) for r in runs) or res["dropped"]:
        lines.append("- Some runs failed, refused, or lost splits during alignment. The quality "
                     "comparison uses common successful splits; partial coverage is not "
                     "deployment readiness. Check the failure matrix.")
    lines.append("\nQuality-table intervals reflect test-row or group sampling, not full retraining "
                 "variability. Swap checks may also use approximate corrected split intervals. "
                 "Marginal confidence-interval overlap alone does not decide a paired difference. "
                 "Validate on representative application data and tune swap limits to actual needs. "
                 "Resource figures include process overhead and are not production service budgets.\n")
    return "\n".join(lines)


def fig_quality_cost(plt, res, cost, path):
    m, s = res["metric"], res["summary"].merge(cost, on="backend", how="left")
    if "single_row_p50_ms" not in s or s["single_row_p50_ms"].isna().all():
        return False
    s = s[np.isfinite(s["single_row_p50_ms"]) & (s["single_row_p50_ms"] > 0)]
    if s.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in s.iterrows():
        ax.errorbar(r.single_row_p50_ms, r[m], yerr=[[r[m] - r.ci_lo], [r.ci_hi - r[m]]],
                    fmt="o", capsize=3)
        ax.annotate(r.backend, (r.single_row_p50_ms, r[m]), textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_xlabel("single-row predict latency, p50 (ms, log)")
    ax.set_ylabel(f"{m} ({'higher' if m in MORE_BETTER else 'lower'} is better, 95% CI)")
    ax.set_title(f"{res['dataset']}: quality vs cost")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_heatmap(plt, res, path):
    names = res["backends"]
    if len(names) < 2:
        return False
    k = {n: i for i, n in enumerate(names)}
    M = np.full((len(names), len(names)), np.nan)
    for a, b, s in res["pairs"]:
        M[k[a], k[b]] = M[k[b], k[a]] = s["rate"]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.0 * len(names) + 1.5))
    vmax = max(0.05, np.nanmax(M))
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.1%}", ha="center", va="center",
                        color="w" if M[i, j] > 0.6 * vmax else "k")
    ax.set_title(f"{res['dataset']}: share of predictions in dispute")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_calibration(plt, res, path):
    if not res["calibration"]:
        return False
    cal = pd.DataFrame(res["calibration"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for n, g in cal.groupby("backend"):
        ax.plot(g.mean_conf, g.accuracy, "o-", label=n)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{res['dataset']}: reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_class_calibration(plt, res, path):
    records = res.get("class_calibration_bins", [])
    if not records:
        return False
    table = pd.DataFrame(records)
    columns = sorted(table.class_column.unique())[:6]
    cols = min(3, len(columns))
    fig, axes = plt.subplots((len(columns) + cols - 1) // cols, cols,
                             figsize=(4 * cols, 3.5 * ((len(columns) + cols - 1) // cols)),
                             squeeze=False)
    for ax, column in zip(axes.flat, columns):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        subset = table[table.class_column == column]
        for backend, data in subset.groupby("backend"):
            occupied = data[data["weight"] > 0]
            ax.plot(occupied.mean_probability, occupied.observed_frequency, "o-", label=backend)
        label = subset.iloc[0].class_label
        title = f"column {column} (unrecorded label)" if pd.isna(label) else str(label)
        ax.set(title=title, xlabel="mean class probability", ylabel="observed class frequency",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("One-vs-rest reliability (descriptive; first six classes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def pair_rows(res, rdir):
    rows = []
    for a, b, s in res["pairs"]:
        if res["task"] == "classification":
            who = (f"only {a} right {s.get('a_right_only', 0):.0%}, only {b} right "
                   f"{s.get('b_right_only', 0):.0%}, both wrong {s.get('both_wrong', 0):.0%}"
                   if s["disputed"] else "-")
        else:
            who = f"{a} closer {s['a_closer']:.0%}; median gap {s['median_gap_std']:.2f} std" \
                if s["disputed"] else "-"
        lifts = s["trait_lift"]
        top = max(lifts, key=lambda t: lifts[t]["lift"]) if lifts else None
        if top and lifts[top]["lift"] < 1.5:  # only report traits that are clearly over-represented
            top = None
        rel = os.path.relpath(res["adir"] / s["csv"], rdir)
        rows.append({"pair": f"{a} vs {b}", "disputed": f"{s['disputed']}/{s['entries']} ({s['rate']:.1%})",
                     "who is right when they differ": who,
                     "most over-represented trait": f"{top} ({lifts[top]['lift']:.1f}x)" if top else "none notable",
                     "rows": f"[csv]({rel})"})
    return pd.DataFrame(rows)


def build(out=None, datasets=None, report_dir=None, model_seed=None, swaps=(), thresholds=None,
          require_thresholds=False, interaction=None):
    resolve_thresholds(thresholds, require_thresholds)
    S = analysis_settings()
    out = Path(out or S["out_dir"])
    rdir = Path(report_dir or out / "report")
    rdir.mkdir(parents=True, exist_ok=True)
    invocation_path = out / "invocation.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8")) \
        if invocation_path.exists() else {}
    feature_stage_path = out / "relational-feature-stage.json"
    feature_stage = json.loads(feature_stage_path.read_text(encoding="utf-8")) \
        if feature_stage_path.exists() else None
    if not datasets:
        datasets = sorted(p.name for p in out.iterdir()
                          if p.is_dir() and p.name != "report" and list(p.glob("*_ss*_ms*.json")))
    plt = _plt()
    L, skipped, matrix, envs, runs_by = [], [], {}, [], {}
    sections, analyses = [], {}
    smoke = invocation.get("run_type") == "smoke"
    for ds in datasets:
        runs_by[ds] = load_runs(out, ds)
        envs += [r["env"] for r in runs_by[ds] if r.get("env")]
        smoke |= any(r.get("run_type") == "smoke" for r in runs_by[ds])
        # Preserve an explicitly requested explainer, but rediscover from CURRENT evidence.
        # Never render cached rules as if they were newly validated.
        requested_interaction = interaction
        saved_analysis = out / ds / "analysis" / "analysis.json"
        if requested_interaction is None and saved_analysis.exists():
            saved = json.loads(saved_analysis.read_text(encoding="utf-8")).get("interaction")
            if saved is not None:
                requested_interaction = interactions.InteractionSpec(**saved["spec"])
        interaction_kw = {"interaction": requested_interaction} if requested_interaction is not None else {}
        try:
            if invocation.get("workflow") == "benchmark" and ds in invocation.get("datasets", []):
                eligible = [b for b in invocation["backends"]
                            if not any(c["execution"] in ("incompatible settings", "not run")
                                       for c in invocation["cells"]
                                       if c["dataset"] == ds and c["backend"] == b)]
                if not eligible:
                    raise AnalysisError("No requested backend has settings compatible with this invocation")
                res = analyse(ds, out, model_seed, quiet=True, backends=eligible, **interaction_kw)
            else:
                res = analyse(ds, out, model_seed, quiet=True, **interaction_kw)
        except AnalysisError as e:
            skipped.append((ds, str(e)))
            sections.append(f"## {ds}\n\n### Interpretation\n\n"
                            f"No usable aligned analysis is available: {e}.\n\n"
                            "No quality ranking or replacement recommendation can be made. "
                            "Inspect model participation and failure details, correct the input "
                            "or execution problem, and retry the affected fits. "
                            + ("This remains a smoke setup check, not benchmark evidence.\n"
                               if smoke else "\n"))
            continue
        analyses[ds] = res
        selected_runs = [r for r in runs_by[ds] if r.get("model_seed") == res["model_seed"]]
        aligned_runs = [r for r in selected_runs if r.get("split_seed") in res["splits"]
                        and r["backend"] in res["backends"]]
        cost = cost_table(aligned_runs)
        m, s, md = res["metric"], res["summary"], []
        t = pd.DataFrame({"backend": s.backend, m: [f"{r[m]:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]"
                                                    for _, r in s.iterrows()]})
        for c in ("auc", "brier", "ece", "mae", "r2", "split_std"):
            if c in s:
                t[c] = s[c]
        t = t.merge(cost, on="backend", how="left")
        mt = res["meta"]
        md.append(f"## {ds}\n\n*{res['task']}; {mt['rows']} rows, {mt['features']} features "
                  f"({mt['tier']}), categorical share {mt['cat_share']}, missing share "
                  f"{mt['missing_share']}. {len(res['splits'])} splits, model seed "
                  f"{res['model_seed']}.*\n")
        md.append("### Evaluation design and data flags\n\n"
                  f"Declared split: `{json.dumps(res['split'], sort_keys=True)}`.\n")
        if res["flags"]:
            md.append(md_table(pd.DataFrame(res["flags"])) + "\n")
        md.append("Data flags are informational and do not change the verdict; no flags does not "
                  "prove absence of leakage. Temporal splits do not make a row bootstrap account "
                  "for serial dependence.\n")
        source = res.get("dataset_source")
        if source:
            if source.get("synthetic"):
                synthetic = source["synthetic"]
                md.append("### Synthetic experiment\n\n"
                          "These rows were generated under an explicit simulation specification, not "
                          "collected from real operations. Results apply to that generating process; "
                          "they do not establish real-world prediction quality or causal effects.\n\n"
                          f"- Generator: `{synthetic['generator']}` version {synthetic['version']}; "
                          f"generation seed: {synthetic['seed']}.\n"
                          f"- Generation metadata SHA-256: `{synthetic['metadata_sha256']}`. "
                          "Keep the original `generation.json` with the source dataset.\n")
            if source.get("kind") == "openml":
                md.append("### OpenML data source\n\n"
                          f"- Dataset: {source['url']}; version: `{source['version']}`; "
                          f"file ID: `{source['file_id']}`; target: `{source['target']}`.\n"
                          f"- Upstream ARFF MD5: `{source['md5_checksum']}`.\n"
                          f"- Loaded-content SHA-256: `{source['sha256']}` "
                          f"(`{source['hash_scheme']}`, not a raw download hash).\n"
                          f"- Source fingerprint: `{source['fingerprint']}`.\n"
                          f"- Upstream licence field: `{source['licence']}`. "
                          "This field alone does not establish permission for your use.\n")
            elif source.get("kind") == "relational":
                md.append("### Relational data source\n\n"
                          f"- Builder version: `{source.get('builder_version', 'unknown')}`; "
                          f"source fingerprint: `{source.get('fingerprint', 'unknown')}`.\n"
                          "- Official prediction-example splits and label-horizon purging apply; "
                          "this is not a flat-file random-row benchmark.\n")
                audits = [r["relational_audit"] for r in selected_runs if r.get("relational_audit")]
                if audits:
                    md.append("Relational audit (source permissions, feature availability, split/purge "
                              "and entity-overlap evidence):\n\n```json\n"
                              + json.dumps(audits[0], indent=2, sort_keys=True) + "\n```\n")
                else:
                    md.append("Relational audit unavailable in saved evidence; no audit is fabricated.\n")
            else:
                md.append("### Local data source\n\n"
                          f"- File: `{source['path']}`; target: `{source['target']}`.\n"
                          f"- Input SHA-256: `{source['sha256']}`.\n"
                          f"- Data/manifest fingerprint: `{source['fingerprint']}`.\n"
                          f"- Excluded columns: {', '.join(source['drop_columns']) or 'none'}.\n")
            if source.get("class_labels"):
                md.append("- Probability columns follow encoded labels: "
                          + ", ".join(f"{i}={label}" for i, label in enumerate(source["class_labels"]))
                          + ".\n")
        if res["dropped"]:
            md.append(f"> Splits dropped because another backend failed on them: {res['dropped']}\n")
        md.append(f"### Quality and cost\n\n{md_table(t)}\n\n"
                  f"`{m}` shows a 95% bootstrap interval. Interval overlap alone does not establish "
                  f"equivalence or decide a paired difference. `split_std` is the spread across splits.\n")
        if plt:
            f1 = f"fig_{ds}_quality_cost.png"
            if fig_quality_cost(plt, res, cost, rdir / f1):
                md.append(f"![quality vs cost]({f1})\n")
        md.append("### Where the backends disagree\n")
        if plt:
            f2 = f"fig_{ds}_disagreement.png"
            if fig_heatmap(plt, res, rdir / f2):
                md.append(f"![disagreement]({f2})\n")
        if res["pairs"]:
            md.append(md_table(pair_rows(res, rdir)) + "\n")
        if res["task"] == "classification" and plt:
            f3 = f"fig_{ds}_calibration.png"
            if fig_calibration(plt, res, rdir / f3):
                md.append(f"### Calibration\n\n![reliability]({f3})\n")
        if res.get("class_calibration_summary"):
            classes = pd.DataFrame(res["class_calibration_summary"])
            display = classes[["backend", "class_column", "class_label", "class_ece",
                               "positive_rows", "negative_rows", "support_status", "class_mapping"]]
            links = []
            for name in ("calibration_classwise_summary.csv", "calibration_classwise.csv"):
                relative = Path(os.path.relpath(res["adir"] / name, rdir)).as_posix()
                links.append(f"[{name}]({relative})")
            md.append("### Per-class calibration\n\n"
                      + md_table(display.head(20))
                      + "\n\nOne-vs-rest, ten equal-width bins; each distinct source row has total "
                      "weight one, divided among its repeated predictions. Empty bins have no "
                      "estimate; one-sided classes have no class ECE. Fewer than twenty distinct "
                      "positive or negative rows are marked rare, not certified safe. Legacy "
                      "unrecorded class labels remain unknown. No intervals or new verdict "
                      "thresholds are implied. The table shows up to twenty backend/class entries; "
                      "complete summaries and all bins: " + ", ".join(links) + ".\n")
            if plt:
                figure = f"fig_{ds}_class_calibration.png"
                if fig_class_calibration(plt, res, rdir / figure):
                    md.append(f"![per-class reliability]({figure})\n")
        if len(res["slices"]):
            sl = res["slices"].head(5)
            md.append(f"### Slices ({m}, top 5 by spread between backends)\n\n{md_table(sl)}\n")
        if res.get("relational_slices") is not None:
            relational_slices = pd.DataFrame(res["relational_slices"])
            relative = Path(os.path.relpath(
                Path(res["adir"]) / "relational_slices.csv", rdir)).as_posix()
            md.append("### Descriptive relational slices\n\n"
                      "Fixed label-free history, recency and cold-start masks; these are "
                      "descriptive point estimates, not intervals, causal claims or swap gates. "
                      "Rows and entity groups are both reported. Limited or empty support must "
                      "not be interpreted as safety, equivalence or a reliable ranking; missing "
                      "values are unavailable, not zero.\n\n")
            if relational_slices.empty:
                md.append("No relational slice estimates are available: empty support.\n")
            else:
                columns = ["backend_a", "backend_b", "slice", "n_rows", "n_groups", "metric",
                           "status", "disagreement_rate", "quality_a", "quality_b"]
                md.append(md_table(relational_slices.reindex(columns=columns).head(20))
                          + "\n\nAt most twenty rows are displayed; support status is preserved.\n")
            md.append(f"[Complete descriptive relational slices CSV]({relative}).\n")
        explanation = res.get("interaction")
        if explanation is not None:
            md.append("### Exploratory interaction rules\n\n" + interactions.WARNING + "\n\n"
                      f"Pair: **{explanation['pair'][0]} vs {explanation['pair'][1]}**; "
                      f"status: `{explanation['status']}`. "
                      f"Quality difference: `{explanation['quality_metric']}` (positive favors A). "
                      "Disagreement and quality average repeats within each original row first. "
                      "Unsupported evaluation leaves have no rate or quality estimate.\n\n"
                      f"Discovery/evaluation design: `{json.dumps(explanation['split'], sort_keys=True)}`.\n")
            for name in explanation["pair"]:
                families = interactions.recipe_families(explanation["recipes"][name])
                md.append(f"Recipe family for `{name}`: `{families}`.\n")
            if explanation["rules"]:
                columns = ["rule_id", "label", "text", "discovery_rows", "discovery_groups",
                           "evaluation_rows", "evaluation_groups", "evaluation_status",
                           "disagreement_rate", "quality_difference"]
                table = pd.DataFrame(explanation["rules"]).reindex(columns=columns)
                md.append(md_table(table) + "\n")
            for name in (interactions.JSON_FILE, interactions.CSV_FILE):
                relative = Path(os.path.relpath(res["adir"] / name, rdir)).as_posix()
                md.append(f"[{name}]({relative})\n")
        swap_results = []
        for inc, cand in (() if smoke else swaps):
            try:
                if inc not in res["backends"] or cand not in res["backends"]:
                    raise SwapError("Both models must have compatible, usable predictions in this analysis")
                r = evaluate(ds, inc, cand, res["model_seed"], out, thresholds, require_thresholds)
                swap_results.append((inc, cand, r))
                ct = pd.DataFrame([{"verdict": c["verdict"], "check": c["check"],
                                    "observed": c["observed"], "limit": c["limit"], "note": c["note"]}
                                   for c in r["checks"]])
                md.append(f"### Swap check: {inc} -> {cand}: **{r['overall']}**\n\n{md_table(ct)}\n")
                source = describe_policy(r["threshold_provenance"])
                md.append(f"{source}. Review these limits for your application; "
                          "PASS is not a production-safety guarantee.\n")
                md.append(f"Swap evaluation design: `{json.dumps(r['split'], sort_keys=True)}`.\n")
                if r["flags"]:
                    md.append("Data flags accompanying this verdict:\n\n"
                              + md_table(pd.DataFrame(r["flags"])) + "\n")
                provided = set(r["threshold_provenance"]["provided"])
                policy = pd.DataFrame([{"setting": key, "value": json.dumps(value),
                                        "origin": "explicit" if key in provided else "packaged default"}
                                       for key, value in r["thresholds"].items()])
                md.append("Effective swap thresholds (`primary: null` selects accuracy for "
                          f"classification or RMSE for regression):\n\n{md_table(policy)}\n")
            except SwapError as e:
                md.append(f"### Swap check: {inc} -> {cand}\n\nnot possible: {e}\n")
        md.insert(1, interpretation(res, cost, selected_runs, swap_results, smoke=smoke))
        sections.append("\n".join(md))
    backends = sorted({r["backend"] for rs in runs_by.values() for r in rs})
    if backends:
        rows = [{"backend": b, **{ds: status_cell([r for r in runs_by[ds] if r["backend"] == b])
                                  if any(r["backend"] == b for r in runs_by[ds]) else "-"
                                  for ds in datasets}} for b in backends]
        matrix = pd.DataFrame(rows)
    env = envs[0] if envs else {}
    pk = {k: v for k, v in (env.get("packages") or {}).items() if v}
    runs_all = [r for rs in runs_by.values() for r in rs]
    cv = sorted({r.get("cv", "holdout") for r in runs_all})
    devs = sorted({r.get("device", "?") for r in runs_all})
    generated = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    L.append(f"# quibble report\n\n*Generated {generated}.*\n")
    if smoke:
        L.append("> SMOKE TEST ONLY: bounded setup check. Do not use these results for model "
                 "ranking, deployment recommendations, or swap approval.\n")
    L.append("## Setup\n")
    L.append(f"- Datasets: {', '.join(datasets)}\n- Backends: {', '.join(backends)}\n"
             f"- Splits: {', '.join(cv)}; device: {', '.join(devs)}\n"
             f"- Hardware: {env.get('cpu')}, {env.get('cores')} cores, {env.get('ram_gb')} GB RAM, "
             f"GPU: {(env.get('gpu') or {}).get('name', 'none')}\n"
             f"- Installed packages: Python {env.get('python')}; "
             + ", ".join(f"{k} {v}" for k, v in pk.items())
             + (f"; git {env['git']}" if env.get("git") else "") + "\n")
    participants = participation(out, datasets, runs_by, analyses, invocation)
    participants.to_csv(rdir / "participation.csv", index=False)
    provenance = {ds: [{"backend": r["backend"], "split_seed": r["split_seed"],
                        "model_seed": r["model_seed"], "backend_metadata": r.get("backend_metadata"),
                        "recipe": r.get("recipe"),
                        **{key: r.get(key) for key in (
                            "campaign_tier", "tier", "tuning", "prediction_adapts",
                            "adaptation_and_prediction_s", "batch_prediction_s", "model_pipeline_s",
                            "feature_costs", "relational_audit")},
                        "stage_costs": stage_cost_row(r)}
                       for r in runs] for ds, runs in runs_by.items()}
    (rdir / "backend_provenance.json").write_text(
        json.dumps({"runs": provenance, "invocation_metadata": invocation.get("backend_metadata"),
                    "relational_feature_stage": feature_stage},
                   indent=2), encoding="utf-8")
    L.append("## Model participation\n\n" + md_table(participants) + "\n\n"
             "Code and weight terms are separate; reference terms do not grant usage rights. "
             "Legacy records have unknown metadata, not today's terms. Checkpoint revisions and "
             "checksums remain unknown unless captured. See [participation CSV](participation.csv) "
             "and [full provenance JSON](backend_provenance.json).\n")
    mismatches = [c for c in invocation.get("cells", [])
                  if c.get("dataset") in datasets and c.get("execution") == "incompatible settings"]
    if mismatches:
        rows = [{"dataset": c["dataset"], "backend": c["backend"], "split": c["split_seed"],
                 "model seed": c["model_seed"], "mismatched settings": ", ".join(c["differences"])}
                for c in mismatches]
        L.append("## Saved-setting mismatches\n\n"
                 "These saved results were skipped, not newly evaluated under the requested "
                 "configuration. Any retained comparison describes the saved settings, not the "
                 "new request. Use `--force` or an isolated `--out` to evaluate changed settings.\n\n"
                 + md_table(pd.DataFrame(rows)) + "\n")
    L.append(device_summary(runs_all))
    if runs_all:
        L.append(protocol_and_cost_section(runs_all, rdir))
    if feature_stage:
        shared = (feature_stage.get("feature_audit") or {}).get("costs") or {}
        relative = Path(os.path.relpath(feature_stage_path, rdir)).as_posix()
        L.append("## Shared relational feature stage (once)\n\n"
                 + md_table(pd.DataFrame([{
                     **{key: shared.get(key) for key in ("feature_total_s", "cache_state",
                                                        "validation_s", "feature_build_s",
                                                        "feature_cache_read_s")},
                     "sweep_wall_s": feature_stage.get("sweep_wall_s")}]))
                 + "\n\nThis explicit front stage precedes model workers; a cold feature build "
                 "must not disappear behind warm worker reload timings. It is shared once, "
                 "not charged to every backend. Worker feature timings above are separate reload "
                 "references; sweep wall time overlaps model work and must not be added again. "
                 f"[Shared stage evidence]({relative}); also preserved in full provenance JSON.\n")
    failures = [{"dataset": r["dataset"], "backend": r["backend"],
                 "split": r["split_seed"], "model seed": r["model_seed"],
                 "kind": r.get("error_kind", "execution"),
                 "attempted device": r.get("device_attempted", "unknown"),
                 "error": r["error"]} for r in runs_all if r.get("error")]
    if failures:
        L.append("## Execution and resource failures\n\n" + md_table(pd.DataFrame(failures)) +
                 "\n\nFailed fits are excluded from quality comparisons. Resource limits do not "
                 "trigger device fallback or automatic downsampling. Reduce jobs/model size, "
                 "use an explicit CPU run, or adjust the timeout before retrying with `--force`.\n")
    artifacts = artifact_table(out, rdir, runs_all)
    if not artifacts.empty:
        L.append("## Retained models and AutoGluon diagnostics\n\n" + md_table(artifacts) + "\n"
                 "Leaderboards use AutoGluon's validation scores, not held-out test labels. "
                 "Retained models live outside snapshots and are not duplicated by archival. "
                 "Only load trusted model files.\n")
    if any(r.get("jobs", 1) > 1 for r in runs_all):
        L.append("> Some runs used parallel jobs, so latency and memory figures are indicative "
                 "only. Use `--jobs 1` for timings you want to publish.\n")
    L += sections
    if len(matrix):
        L.append("## What failed or refused\n\n" + md_table(matrix) + "\n")
    if skipped:
        L.append("## Skipped datasets\n\n" + "\n".join(f"- {d}: {e}" for d, e in skipped) + "\n")
    L.append("## How to read this\n\n"
             "- Quality-table intervals bootstrap rows or declared groups, counting test-set noise. "
             "Swap checks also use approximate corrected split intervals with 5+ usable splits; "
             "per-check notes identify when retraining variability is not counted.\n"
             "- The combined decision rule is not calibrated for false-PASS or false-FAIL rates. "
             "Serial dependence and target-driven slice selection need further validation.\n"
             "- \"Disputed\" for classification means the predicted classes differ; for regression "
             "it means a gap above 0.25 target standard deviations.\n"
             "- Slice sizes count distinct rows. Worst-slice bootstrap samples recompute the maximum "
             "over fixed, target-selected masks; this is exploratory, not established family-wise coverage.\n"
             "- Latency and memory are single-machine measurements. Baseline-adjusted process peak "
             "includes imports, inference and lifetime high-water marks; it is not model-only memory. "
             "Swap ratios floor estimates below 50 MB; raw estimates and limits remain visible.\n")
    if plt is None:
        L.append("\n*Figures skipped: matplotlib is not installed (select the report extra; "
                 "from a checkout: `uv sync --extra report`).*\n")
    path = rdir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    archive = snapshot(out, "report", report_dir=rdir)
    relative = Path(os.path.relpath(archive / "report" / "report.md", rdir)).as_posix()
    history = Path(os.path.relpath(out / "report" / "history.md", rdir)).as_posix()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"\nArchived copy: [{archive.name}]({relative}). "
                     f"[Run history]({history}).\n")
    return path


def build_tier_comparison(defaults_root, tuned_root, backend="histgb", datasets=None, report_dir=None):
    """Explicit read-only paired contrasts across roots, never a mixed analysis pool."""
    from .tuning import load_tier_pair
    from .metrics import score
    defaults_root, tuned_root = Path(defaults_root), Path(tuned_root)
    if not datasets:
        datasets = sorted({p.parent.name for root in (defaults_root, tuned_root)
                           for p in root.glob(f"*/{backend}_ss*_ms*.json")})
    rows, evidence = [], []
    for dataset in datasets:
        cells = []
        for root in (defaults_root, tuned_root):
            cells.append({(r["split_seed"], r["model_seed"])
                          for path in (root / dataset).glob(f"{backend}_ss*_ms*.json")
                          for r in [json.loads(path.read_text(encoding="utf-8"))]})
        if not cells[0] or cells[0] != cells[1]:
            raise ValueError(f"{dataset}: tier comparison needs exactly matching complete cell sets")
        for split_seed, model_seed in sorted(cells[0]):
            pair = load_tier_pair(defaults_root, tuned_root, dataset, backend, split_seed, model_seed)
            metric = "rmse" if pair["defaults"]["record"]["task"] == "regression" else "log_loss"
            values = {}
            for tier in ("defaults", "tuned"):
                record = pair[tier]["record"]
                value = score(metric, pair["y_test"], pair[tier]["predictions"])
                if value is None or not np.isfinite(value):
                    raise ValueError(f"{tier} cell has invalid {metric} evidence")
                values[tier] = float(value)
                rows.append({"dataset": dataset, "recipe": f"{backend}[{tier}]",
                             "split": split_seed, "model seed": model_seed, "metric": metric,
                             "value": values[tier], **{k: v for k, v in stage_cost_row(record).items()
                                                     if k not in ("backend", "split", "model seed")}})
                evidence.append({"root": str(defaults_root if tier == "defaults" else tuned_root),
                                 "record": record})
            rows[-1]["tuned minus defaults"] = values["tuned"] - values["defaults"]
    if not rows:
        raise ValueError("No matching tier evidence found")
    rdir = Path(report_dir or tuned_root / "report" / "tier-comparison")
    rdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(rdir / "tier_comparison.csv", index=False)
    (rdir / "tier_comparison.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    text = ("# Defaults versus tuned: explicit paired comparison\n\n"
            f"Backend: `{backend}`. Defaults root: `{defaults_root}`. Tuned root: `{tuned_root}`.\n\n"
            "Each tier-qualified recipe is evaluated on verified identical outer test rows/targets "
            "and matching training/protocol/source identities. Roots are read separately, never "
            "merged into a backend pool. All cells must match and be successful. "
            "Log loss/RMSE are lower-is-better; negative tuned-minus-defaults favors tuned. "
            "These are paired cell point estimates, not significance tests, swap approvals or "
            "independent repeated samples. Test results must not feed subsequent search.\n\n"
            + md_table(table) + "\n\n"
            "Full model pipeline is fit/search/refit plus batch prediction. Search/refit overlaps "
            "fit_total_s; inclusive prediction adaptation overlaps batch time. Shared feature cost "
            "is a reference, never charged once per model. Missing timings are unavailable.\n\n"
            "[Paired cells CSV](tier_comparison.csv); [full evidence JSON](tier_comparison.json).\n")
    path = rdir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--datasets")
    ap.add_argument("--report-dir")
    ap.add_argument("--model-seed", type=int)
    ap.add_argument("--compare-tiers", nargs=2, metavar=("DEFAULTS_ROOT", "TUNED_ROOT"),
                    help="explicit paired same-backend comparison; never merge output roots")
    ap.add_argument("--tier-backend", default="histgb", help="tree backend for --compare-tiers")
    ap.add_argument("--swap", action="append", default=[], help="INCUMBENT:CANDIDATE (repeatable)")
    ap.add_argument("--thresholds", help="JSON file for swap limits")
    ap.add_argument("--require-thresholds", action="store_true",
                    help="require a non-empty policy; partial overrides still inherit listed defaults")
    interactions.add_arguments(ap)
    a = ap.parse_args()
    swaps = [tuple(s.split(":", 1)) for s in a.swap]
    try:
        thr = load_thresholds(a.thresholds) if a.thresholds else None
        if a.compare_tiers:
            if a.swap or a.thresholds or a.require_thresholds:
                raise ValueError("Cross-tier point estimates do not issue swap approvals")
            if a.out is not None or a.model_seed is not None or interactions.spec_from_arguments(a):
                raise ValueError("Cross-tier reports use explicit roots and all matching cells; "
                                 "use --report-dir, not --out/--model-seed/explainer options")
            path = build_tier_comparison(*a.compare_tiers, backend=a.tier_backend,
                                        datasets=a.datasets.split(",") if a.datasets else None,
                                        report_dir=a.report_dir)
        else:
            path = build(a.out, a.datasets.split(",") if a.datasets else None, a.report_dir,
                         a.model_seed, swaps, thr, a.require_thresholds,
                         interaction=interactions.spec_from_arguments(a))
    except ValueError as exc:
        ap.error(str(exc))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
