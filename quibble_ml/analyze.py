"""Build 2 analysis for one dataset: confidence intervals, calibration, slices, disagreement.

  python -m quibble_ml.analyze --dataset breast_cancer [--backends a,b,c] [--model-seed 0]
Writes to <out>/<dataset>/analysis/: summary.csv, calibration.csv, slices.csv,
disagreements_<a>_vs_<b>.csv (the disputed rows, with features), analysis.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import analysis_settings
from .datasets import load, validate_sources
from . import leakage
from . import interactions
from .metrics import Boot, class_calibration, corrected_interval, ece, norm, score
from .pool import align, load_pool, seeds_available
from .slices import row_traits, slice_masks, unique_rows
from .thresholds import DEFAULTS

HI, BIG, NEAR = (DEFAULTS[key] for key in ("confident_prob", "regression_gap_std", "near_margin"))
MIN_SLICE_ROWS = DEFAULTS["min_slice_n"]


def primary_of(task):
    return "accuracy" if task == "classification" else "rmse"


def pair_frame(pool, ia, ib, traits, task, hi=HI, big=BIG, near=NEAR):
    """One row per pooled prediction pair, with disputed flag and row traits."""
    A, B, y, idx = pool["Ps"][ia], pool["Ps"][ib], pool["y"], pool["idx"]
    if task == "classification":
        A, B, yi = norm(A), norm(B), y.astype(int)
        pa, pb = A.argmax(1), B.argmax(1)
        disp = pa != pb
        mg = lambda P: (lambda s: s[:, -1] - s[:, -2])(np.sort(P, 1))
        f = pd.DataFrame({"split": pool["split"], "row": idx, "y": yi, "pred_a": pa, "pred_b": pb,
                          "conf_a": A.max(1), "conf_b": B.max(1), "a_correct": pa == yi,
                          "b_correct": pb == yi, "disputed": disp,
                          "confident_both": disp & (A.max(1) >= hi) & (B.max(1) >= hi)})
        f["near_boundary"] = 0.5 * (mg(A) + mg(B)) < near
    else:
        sd = np.std(y) or 1.0
        gap = np.abs(A - B) / sd
        f = pd.DataFrame({"split": pool["split"], "row": idx, "y": y, "pred_a": A, "pred_b": B,
                          "gap_std": gap, "disputed": gap > big,
                          "a_closer": np.abs(A - y) < np.abs(B - y)})
        f["extreme_target"] = (y < np.percentile(y, 10)) | (y > np.percentile(y, 90))
    f["has_missing"] = traits["has_missing"].to_numpy()[idx]
    f["rare_category"] = traits["rare_category"].to_numpy()[idx]
    return f


def summarise_pair(f, task):
    d = f[f["disputed"]]
    out = {"entries": len(f), "disputed": len(d), "rate": len(d) / len(f)}
    if task == "classification":
        out["confident_both"] = int(f["confident_both"].sum())
        if len(d):
            out["a_right_only"] = float((d.a_correct & ~d.b_correct).mean())
            out["b_right_only"] = float((~d.a_correct & d.b_correct).mean())
            out["both_wrong"] = float((~d.a_correct & ~d.b_correct).mean())
        traits = ["has_missing", "rare_category", "near_boundary"]
    else:
        if len(d):
            out["a_closer"] = float(d.a_closer.mean())
            out["median_gap_std"] = float(d.gap_std.median())
        traits = ["has_missing", "rare_category", "extreme_target"]
    lifts = {}
    for t in traits:
        allv = float(f[t].mean())
        if allv > 0 and len(d):
            lifts[t] = {"share_disputed": float(d[t].mean()), "share_all": allv,
                        "lift": float(d[t].mean() / allv)}
    out["trait_lift"] = lifts
    return out


def print_pair(a, b, s, task, say=print):
    say(f"\n{a} vs {b}: {s['disputed']}/{s['entries']} disputed ({s['rate']:.2%})"
        + (f", {s['confident_both']} confident-both" if task == "classification" else ""))
    if not s["disputed"]:
        return
    if task == "classification":
        say(f"  when they differ: only {a} right {s['a_right_only']:.0%} | "
            f"only {b} right {s['b_right_only']:.0%} | both wrong {s['both_wrong']:.0%}")
    else:
        say(f"  when they differ: {a} closer {s['a_closer']:.0%}; "
            f"median gap {s['median_gap_std']:.2f} std")
    for t, v in s["trait_lift"].items():
        say(f"  {t}: {v['share_disputed']:.1%} of disputed vs {v['share_all']:.1%} overall "
            f"(lift {v['lift']:.1f}x)")


class AnalysisError(Exception):
    pass


def make_boot(pool, X, n_boot):
    """Bootstrap whose resampling unit is a group (customer, account, ...) when the dataset
    declares one, otherwise a row. Correlated rows must not be treated as independent."""
    groups = X.attrs.get("quibble_groups")
    cluster = None if groups is None else np.asarray(groups)[np.unique(pool["idx"])]
    return Boot(pool["row_pos"], n_boot, cluster=cluster)


def split_info(X):
    return X.attrs.get("quibble_split") or {"kind": "random"}


def reseed_rate(out, dataset, backend, task, traits, **kw):
    """Disagreement between two training seeds of the same backend: the noise floor that any
    candidate-vs-incumbent disagreement should be read against. None with fewer than two seeds."""
    pools = [p for p in (load_pool(out, dataset, backend, s)[0] for s in seeds_available(out, dataset, backend)) if p]
    rates = []
    for a, b in itertools.combinations(range(len(pools)), 2):
        pool = align([pools[a], pools[b]])
        if pool:
            rates.append(float(pair_frame(pool, 0, 1, traits, task, **kw)["disputed"].mean()))
    return float(np.mean(rates)) if rates else None


def analyse(dataset, out=None, model_seed=None, metric=None, min_slice_n=MIN_SLICE_ROWS, backends=None,
            quiet=False, interaction=None):
    """Full Build 2 analysis for one dataset. Returns a dict of results (DataFrames etc.)."""
    say = (lambda *a, **k: None) if quiet else print
    S = analysis_settings()
    out = Path(out or S["out_dir"])
    d = out / dataset
    files = list(d.glob("*_ss*_ms*.json")) if d.exists() else []
    if not files:
        raise AnalysisError(f"no results for {dataset} in {out}")
    run_types = {json.loads(p.read_text()).get("run_type", "benchmark") for p in files}
    if len(run_types) > 1:
        raise AnalysisError("Smoke and benchmark records must not be mixed in one dataset directory")
    try:
        X, yall, task = load(dataset, S["data_home"])
        validate_sources(out, dataset, X.attrs.get("quibble_source"))
    except (ValueError, OSError, ImportError) as e:
        raise AnalysisError(f"Dataset {dataset}: {e}") from e
    names = backends or sorted({p.name.split("_ss")[0] for p in files})
    ms = model_seed if model_seed is not None else min(int(p.stem.split("_ms")[1]) for p in files)
    try:
        loaded = {n: load_pool(out, dataset, n, ms) for n in names}
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc
    provenance = {n: [{"split_seed": r["split_seed"], "model_seed": r["model_seed"],
                       "backend_metadata": r.get("backend_metadata"), "recipe": r.get("recipe")}
                      for r in loaded[n][1]] for n in names}
    pools = {n: v[0] for n, v in loaded.items()}
    for n in names:
        if not pools[n]:
            say(f"(skipping {n}: no usable runs)")
    keep = [n for n in names if pools[n]]
    pool = align([pools[n] for n in keep]) if keep else None
    if not pool:
        raise AnalysisError("no splits completed by all backends")
    if interaction is not None and not all(n in keep for n in interaction.pair):
        raise AnalysisError("interaction pair must name two usable selected backends")
    y, metric = pool["y"], metric or primary_of(task)
    adir = d / "analysis"
    adir.mkdir(exist_ok=True)
    say(f"{dataset} ({task}): {len(keep)} backends, {len(pool['splits'])} splits, "
        f"{len(np.unique(pool['idx']))} rows, model seed {ms}, {S['n_boot']} bootstrap resamples")
    lost = {n: sorted(set(pools[n]) - set(pool["splits"])) for n in keep
            if set(pools[n]) - set(pool["splits"])}
    if lost:
        say(f"note: splits dropped because another backend failed on them: {lost}")
    boot = make_boot(pool, X, S["n_boot"])
    clf = task == "classification"
    flags = leakage.check(X, yall, task, split_info(X).get("kind", "random"))
    for line in leakage.format_flags(flags):
        say(f"DATA FLAG {line}")
    traits_all = row_traits(X)

    rows, cal, class_summary, class_bins = [], [], [], []
    for n, P in zip(keep, pool["Ps"]):
        pt, lo, hi = boot.ci(lambda w: score(metric, y, P, w))
        per = [score(metric, y[pool["split"] == s], P[pool["split"] == s]) for s in pool["splits"]]
        runs = [q for q in loaded[n][1] if q.get("split_seed") in pool["splits"] and "error" not in q
                and "refused" not in q]
        n_tr = float(np.mean([q["n_train"] for q in runs])) if runs else 0
        n_te = float(np.mean([q["n_test"] for q in runs])) if runs else 0
        _, cv_lo, cv_hi = corrected_interval(per, n_tr, n_te)
        r = {"backend": n, metric: pt, "ci_lo": lo, "ci_hi": hi, "split_std": float(np.std(per)),
             "cv_lo": cv_lo, "cv_hi": cv_hi}
        seeds = seeds_available(out, dataset, n)
        by_seed = []
        for sd in seeds:
            sp_ = load_pool(out, dataset, n, sd)[0]
            al = align([sp_]) if sp_ else None
            if al and set(al["splits"]) >= set(pool["splits"]):
                keep_ = np.isin(al["split"], pool["splits"])
                by_seed.append(score(metric, al["y"][keep_], al["Ps"][0][keep_]))
        r["n_seeds"] = len(seeds)
        r["seed_std"] = float(np.std(by_seed, ddof=1)) if len(by_seed) >= 2 else np.nan
        rr = reseed_rate(out, dataset, n, task, traits_all) if len(seeds) >= 2 else None
        r["reseed_disagreement"] = np.nan if rr is None else rr
        if clf:
            r["auc"], r["brier"] = score("auc", y, P), score("brier", y, P)
            e, tab = ece(y.astype(int), P)
            r["ece"] = e
            cal += [{"backend": n, "bin": k, "n": nn, "mean_conf": mc, "accuracy": ac}
                    for k, nn, mc, ac in tab]
            recorded = [q.get("prediction_classes") for q in runs]
            columns = recorded[0] if recorded and all(c is not None for c in recorded) else None
            cs, cb = class_calibration(y, P, columns if columns is not None else range(P.shape[1]),
                                       row_ids=pool["idx"])
            labels = (X.attrs.get("quibble_source") or {}).get("class_labels")
            for record in cs + cb:
                column = record["class_column"]
                record["backend"] = n
                record["class_mapping"] = "recorded_encoded" if columns is not None else "legacy_unrecorded"
                if columns is None:
                    record["class_label"] = None
                elif labels is not None:
                    record["class_label"] = labels[columns[column]]
                    record["class_mapping"] = "recorded_source_label"
            class_summary.extend(cs)
            class_bins.extend(cb)
        else:
            r["mae"], r["r2"] = score("mae", y, P), score("r2", y, P)
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(adir / "summary.csv", index=False)
    say("\n== quality (ci = 95% bootstrap CI over data rows or groups, test-set noise only; "
        "cv = approximate corrected split interval, needs 5+ splits; "
        "split_std = spread across splits) ==")
    say(summ.round(4).to_string(index=False))
    if clf:
        pd.DataFrame(cal).to_csv(adir / "calibration.csv", index=False)
        pd.DataFrame(class_summary).to_csv(adir / "calibration_classwise_summary.csv", index=False)
        pd.DataFrame(class_bins).to_csv(adir / "calibration_classwise.csv", index=False)

    masks, traits = slice_masks(X, yall, task), row_traits(X)
    srows = []
    for name, mk in masks.items():
        sel = mk[pool["idx"]]
        rows_in = unique_rows(sel, pool["idx"])
        if rows_in < min_slice_n:
            continue
        r = {"slice": name, "n": rows_in}
        for n, P in zip(keep, pool["Ps"]):
            r[n] = score(metric, y[sel], P[sel])
        r["spread"] = max(r[n] for n in keep) - min(r[n] for n in keep)
        srows.append(r)
    sl = pd.DataFrame(srows).sort_values("spread", ascending=False) if srows else pd.DataFrame()
    if len(sl):
        sl.to_csv(adir / "slices.csv", index=False)
        say(f"\n== slices ({metric}; top 8 by spread between backends; n = distinct rows; exploratory: "
            "chosen using the target, no intervals) ==")
        say(sl.head(8).round(4).to_string(index=False))

    relational_masks, relational_rows = {}, []
    if X.attrs.get("quibble_relational"):
        from .relational import diagnostic_masks
        relational_masks = diagnostic_masks(X)
    groups = X.attrs.get("quibble_groups")
    pairs, pair_list = {}, []
    for i, j in itertools.combinations(range(len(keep)), 2):
        f = pair_frame(pool, i, j, traits, task)
        s = summarise_pair(f, task)
        pairs[f"{keep[i]}_vs_{keep[j]}"] = s
        pair_list.append((keep[i], keep[j], s))
        print_pair(keep[i], keep[j], s, task, say)
        csv = f"disagreements_{keep[i]}_vs_{keep[j]}.csv"
        disputed = f[f["disputed"]].copy()
        feats = X.iloc[disputed["row"].to_numpy()].iloc[:, :40].add_prefix("x_").reset_index(drop=True)
        pd.concat([disputed.reset_index(drop=True), feats], axis=1).to_csv(adir / csv, index=False)
        s["csv"] = csv
        for label, mask in relational_masks.items():
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (len(X),):
                raise AnalysisError("Relational diagnostic masks must align with loaded example rows.")
            selected = mask[pool["idx"]]
            count = unique_rows(selected, pool["idx"])
            group_count = None if groups is None else int(len(np.unique(
                np.asarray(groups)[pool["idx"]][selected])))
            row = {"backend_a": keep[i], "backend_b": keep[j], "slice": label,
                   "n_rows": count, "n_groups": group_count, "metric": metric,
                   "status": "empty" if not count else
                             "limited support" if count < min_slice_n or group_count == 1
                             else "descriptive",
                   "disagreement_rate": float(f.loc[selected, "disputed"].mean()) if count else None,
                   "quality_a": None, "quality_b": None}
            if count:
                for key, position in (("quality_a", i), ("quality_b", j)):
                    value = score(metric, y[selected], pool["Ps"][position][selected])
                    row[key] = float(value) if np.isfinite(value) else None
            relational_rows.append(row)
    relational_columns = ["backend_a", "backend_b", "slice", "n_rows", "n_groups", "metric",
                          "status", "disagreement_rate", "quality_a", "quality_b"]
    relational_slices = pd.DataFrame(relational_rows, columns=relational_columns)
    if relational_masks:
        relational_slices.to_csv(adir / "relational_slices.csv", index=False)
    else:
        (adir / "relational_slices.csv").unlink(missing_ok=True)
    interaction_result = None
    if interaction is not None:
        try:
            pair_pool = align([pools[n] for n in interaction.pair])
            if not pair_pool:
                raise ValueError("interaction pair has no common completed splits")
            recipes = {n: [r for r in provenance[n] if r["split_seed"] in pair_pool["splits"]]
                       for n in interaction.pair}
            interaction_result = interactions.discover(
                X, pair_frame(pair_pool, 0, 1, traits, task), task, interaction, recipes)
            interaction_result.update(dataset=dataset, model_seed=ms, splits=pair_pool["splits"])
            interactions.write_exports(interaction_result, adir)
        except ValueError as exc:
            raise AnalysisError(str(exc)) from exc
        say(f"\n== interaction rules: {interaction_result['status']} ==")
        say(interactions.WARNING)
    else:
        # A non-opted-in analysis must not leave apparently current rules from an older run.
        for name in (interactions.JSON_FILE, interactions.CSV_FILE):
            (adir / name).unlink(missing_ok=True)
    (adir / "analysis.json").write_text(json.dumps(
        {"dataset": dataset, "task": task, "metric": metric, "model_seed": ms,
         "splits": pool["splits"], "summary": rows, "pairs": pairs, "data_flags": flags,
         "split": split_info(X), "dataset_source": X.attrs.get("quibble_source"),
         "backend_provenance": provenance, "class_calibration_summary": class_summary,
         "class_calibration_bins": class_bins, "interaction": interaction_result,
         "relational_slices": relational_rows},
        indent=2, default=float))
    say(f"\nwritten to {adir}/")
    from .datasets import meta
    return {"dataset": dataset, "task": task, "metric": metric, "model_seed": ms, "backends": keep,
            "splits": pool["splits"], "n_rows": int(len(np.unique(pool["idx"]))), "summary": summ,
            "slices": sl, "calibration": cal, "pairs": pair_list, "dropped": lost, "adir": adir,
            "meta": meta(X, yall, task), "dataset_source": X.attrs.get("quibble_source"),
            "flags": flags, "split": split_info(X), "backend_provenance": provenance,
            "class_calibration_summary": class_summary, "class_calibration_bins": class_bins,
            "interaction": interaction_result,
            "relational_slices": relational_slices if relational_masks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backends")
    ap.add_argument("--model-seed", type=int)
    ap.add_argument("--metric")
    ap.add_argument("--out")
    ap.add_argument("--min-slice-n", type=int, default=MIN_SLICE_ROWS,
                    help="smallest slice, in distinct rows")
    interactions.add_arguments(ap)
    a = ap.parse_args()
    try:
        analyse(a.dataset, a.out, a.model_seed, a.metric, a.min_slice_n,
                a.backends.split(",") if a.backends else None,
                interaction=interactions.spec_from_arguments(a))
    except (AnalysisError, ValueError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
