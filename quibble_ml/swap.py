"""Swap report: is replacing INCUMBENT with CANDIDATE safe on this dataset?

  python -m quibble_ml.swap --incumbent histgb --candidate lightgbm --dataset breast_cancer
  python -m quibble_ml.swap ... --thresholds my_thresholds.json [--require-thresholds]
Each check is PASS (confidently within the limit), FAIL (confidently beyond it) or
INCONCLUSIVE (the 95% interval straddles the limit, or no interval could be computed: collect
more data). Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE.
Intervals combine two sources and use the wider: a bootstrap over rows (or groups) for test-set
noise, and an approximate Nadeau-Bengio corrected interval over splits (needs 5+ splits).
Their union is not a calibrated end-to-end decision rule.
Quality drop is absolute (accuracy etc.) or relative (rmse/mae). The worst-slice bootstrap
recomputes the maximum over fixed, target-selected slices; selection coverage is not established.
Default limits come from the packaged swap_thresholds.json. Review them for your application;
use --thresholds for overrides and --require-thresholds in CI to require an explicit policy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import leakage
from .analyze import make_boot, pair_frame, primary_of, reseed_rate, split_info
from .config import analysis_settings
from .datasets import load, validate_sources
from .metrics import corrected_interval, degradation, score, widen
from .pool import align, load_pool
from .slices import row_traits, slice_masks, unique_rows
from .thresholds import (describe_policy, export_thresholds, load_thresholds, policy_provenance,
                         resolve_thresholds, validate_primary)

MEMORY_FLOOR_MB = 50.0


def tri(pt, lo, hi, thr):
    """PASS / FAIL / INCONCLUSIVE. Without an interval there is no evidence either way."""
    if np.isnan(lo) or np.isnan(hi):
        return "INCONCLUSIVE"
    return "PASS" if hi <= thr else "FAIL" if lo > thr else "INCONCLUSIVE"


def center_of(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None and "error" not in r and "refused" not in r]
    return float(np.median(v)) if v else np.nan


class SwapError(Exception):
    pass


def interval(boot, fn, per, n_train, n_test, clip=None):
    """(point, lo, hi, notes): bootstrap widened by an approximate corrected split interval.
    ``clip=(0, 1)`` keeps the bounds of a rate inside its possible range."""
    b = boot.ci(fn)
    notes = []
    if boot.dropped:
        notes.append(f"{boot.dropped} of {boot.dropped + boot.valid} bootstrap resamples unusable")
    cv = corrected_interval(per, n_train, n_test) if per is not None else (np.nan,) * 3
    if np.isnan(cv[1]):
        notes.append("retraining variability not counted (needs 5+ splits)")
    else:
        notes.append("bounds include approximate corrected split interval (5+ usable splits)")
    pt, lo, hi = widen(b, cv)
    if clip is not None:
        lo, hi = max(lo, clip[0]), min(hi, clip[1])  # NaN stays NaN: max/min with nan return the number, so guard
        if np.isnan(b[1]) and np.isnan(cv[1]):
            lo = hi = np.nan
    if np.isnan(lo) or np.isnan(hi):
        notes.append("no interval available")
    return pt, lo, hi, notes


def evaluate(dataset, incumbent, candidate, model_seed=0, out=None, thresholds=None,
             require_thresholds=False):
    """-> {"overall", "checks", "splits", "task", "thresholds", "thresholds_default", "flags"}.
    Raises SwapError if unusable."""
    try:
        thr = resolve_thresholds(thresholds, require_thresholds)
    except ValueError as exc:
        raise SwapError(str(exc)) from exc
    using_defaults = not thresholds
    S = analysis_settings()
    out = Path(out or S["out_dir"])
    try:
        X, yall, task = load(dataset, S["data_home"])
        validate_sources(out, dataset, X.attrs.get("quibble_source"))
        validate_primary(thr, task)
    except (ValueError, OSError, ImportError) as e:
        raise SwapError(f"Dataset {dataset}: {e}") from e
    clf, m = task == "classification", thr["primary"] or primary_of(task)
    try:
        (pi, ri), (pc, rc) = (load_pool(out, dataset, n, model_seed) for n in (incumbent, candidate))
    except ValueError as exc:
        raise SwapError(str(exc)) from exc
    if not pi:
        raise SwapError(f"incumbent {incumbent} has no usable runs")
    pf = {"hi": thr["confident_prob"], "big": thr["regression_gap_std"], "near": thr["near_margin"]}
    checks = []

    def add(name, obs, limit, verdict, note=""):
        checks.append({"check": name, "observed": obs, "limit": limit, "verdict": verdict, "note": note})

    def fmt(pt, lo, hi, pct=False):
        return f"{pt:.2%} [{lo:.2%}, {hi:.2%}]" if pct else f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    missing = sorted(set(pi) - set(pc))
    add("coverage", f"{len(pc)}/{len(pi)} splits ran", "all", "PASS" if not missing else "FAIL",
        "" if not missing else f"candidate failed/refused on splits {missing}")
    pool = align([pi, pc]) if pc else None
    flags = leakage.check(X, yall, task, split_info(X).get("kind", "random"))
    if pool:
        y, A, B = pool["y"], pool["Ps"][0], pool["Ps"][1]
        boot = make_boot(pool, X, S["n_boot"])
        sp, splits = pool["split"], pool["splits"]
        good = [r for r in ri if "error" not in r and "refused" not in r and r.get("split_seed") in splits]
        n_tr = float(np.mean([r["n_train"] for r in good if "n_train" in r])) if good else 0.0
        n_te = float(np.mean([r["n_test"] for r in good if "n_test" in r])) if good else 0.0

        def deg(w, sel=slice(None)):
            ww = None if w is None else w[sel]
            return degradation(m, score(m, y[sel], A[sel], ww), score(m, y[sel], B[sel], ww))

        per_q = [deg(None, sp == s) for s in splits]
        pt, lo, hi, notes = interval(boot, deg, per_q, n_tr, n_te)
        add(f"quality drop ({m})", fmt(pt, lo, hi), f"<= {thr['max_quality_drop']}",
            tri(pt, lo, hi, thr["max_quality_drop"]), "; ".join(notes))

        f = pair_frame(pool, 0, 1, row_traits(X), task, **pf)
        disp = f["disputed"].to_numpy()
        per_d = [float(disp[sp == s].mean()) for s in splits]
        pt, lo, hi, notes = interval(boot, lambda w: float(np.average(disp, weights=w)), per_d, n_tr, n_te,
                                     clip=(0.0, 1.0))
        noise = reseed_rate(out, dataset, incumbent, task, row_traits(X), **pf)
        if noise is None:
            label = "disagreement rate"
            notes.append("no reseed baseline: run 2+ model seeds to subtract the incumbent's own noise")
        else:
            notes.append(f"raw disagreement: {fmt(pt, lo, hi, pct=True)}")
            label, pt, lo, hi = "disagreement rate (above reseed noise)", pt - noise, lo - noise, hi - noise
            notes.append(f"incumbent disagrees with itself across seeds by {noise:.2%}; "
                         "noise-estimate uncertainty is not propagated")
        add(label, fmt(pt, lo, hi, pct=True), f"<= {thr['max_disagreement']:.0%}",
            tri(pt, lo, hi, thr["max_disagreement"]), "; ".join(notes))
        if clf:
            cb = f["confident_both"].to_numpy()
            per_c = [float(cb[sp == s].mean()) for s in splits]
            pt, lo, hi, notes = interval(boot, lambda w: float(np.average(cb, weights=w)), per_c, n_tr, n_te,
                                         clip=(0.0, 1.0))
            add("confident disagreements", fmt(pt, lo, hi, pct=True),
                f"<= {thr['max_confident_disagreement']:.0%}",
                tri(pt, lo, hi, thr["max_confident_disagreement"]),
                "; ".join([f"both models >= {thr['confident_prob']:.0%} sure"] + notes))

            def ece_diff(w, sel=slice(None)):
                ww = None if w is None else w[sel]
                return score("ece", y[sel], B[sel], ww) - score("ece", y[sel], A[sel], ww)

            per_e = [ece_diff(None, sp == s) for s in splits]
            pt, lo, hi, notes = interval(boot, ece_diff, per_e, n_tr, n_te)
            add("calibration (ECE increase)", fmt(pt, lo, hi), f"<= {thr['max_ece_increase']}",
                tri(pt, lo, hi, thr["max_ece_increase"]), "; ".join(notes))
        slices = []
        for name, mk in slice_masks(X, yall, task).items():
            sel = mk[pool["idx"]]
            if unique_rows(sel, pool["idx"]) >= thr["min_slice_n"]:
                slices.append((name, sel))
        if slices:
            def worst(w):
                vals = [v for v in (deg(w, sel) for _, sel in slices) if np.isfinite(v)]
                return max(vals) if vals else np.nan

            points = [deg(None, sel) for _, sel in slices]
            name, sel = slices[int(np.nanargmax(points))] if np.isfinite(points).any() else slices[0]
            pt, lo, hi = boot.ci(worst)
            add(f"worst slice: {name} (n={unique_rows(sel, pool['idx'])} rows/{len(slices)} slices)",
                fmt(pt, lo, hi), f"<= {thr['max_slice_drop']}", tri(pt, lo, hi, thr["max_slice_drop"]),
                "bootstrap maximum over all slices with fixed, target-selected masks; "
                "test-set noise only; selection coverage is not calibrated"
                + (f"; {boot.dropped} unusable bootstrap resamples" if boot.dropped else "")
                + ("; no interval available" if not np.isfinite(lo) else ""))
    for key, lim, label in (("single_row_p50_ms", "max_latency_ratio", "latency ratio (single-row p50)"),):
        ra, rb = center_of(ri, key), center_of(rc, key)
        if np.isnan(ra) or np.isnan(rb) or ra == 0:
            if any(r.get("prediction_adapts") or r.get("single_row_latency_available") is False
                   for r in ri + rc):
                add(label, "unavailable", f"<= {thr[lim]}x", "INCONCLUSIVE",
                    "Prediction-time adaptation prevents an isolated single-row latency "
                    "measurement; inclusive batch cost cannot certify this budget.")
            continue
        add(label, f"{rb / ra:.2f}x ({ra:.2f} -> {rb:.2f})", f"<= {thr[lim]}x",
            "PASS" if rb / ra <= thr[lim] else "FAIL", "single-machine medians; use --jobs 1 runs")
    ia, ib = center_of(ri, "peak_rss_increase_mb"), center_of(rc, "peak_rss_increase_mb")
    if not (np.isnan(ia) or np.isnan(ib)):
        ra, rb = max(ia, MEMORY_FLOOR_MB), max(ib, MEMORY_FLOOR_MB)
        add("memory ratio (baseline-adjusted process peak)", f"{rb / ra:.2f}x ({ia:.0f} -> {ib:.0f} MB)",
            f"<= {thr['max_memory_ratio']}x", "PASS" if rb / ra <= thr["max_memory_ratio"] else "FAIL",
            f"increases under {MEMORY_FLOOR_MB:.0f} MB are treated as {MEMORY_FLOOR_MB:.0f} MB; "
            "includes imports, inference and process high-water marks; not model-only memory; use --jobs 1")
    else:
        ra, rb = center_of(ri, "peak_rss_mb"), center_of(rc, "peak_rss_mb")
        if not (np.isnan(ra) or np.isnan(rb) or ra == 0):
            add("memory ratio (peak RSS)", f"{rb / ra:.2f}x ({ra:.2f} -> {rb:.2f})",
                f"<= {thr['max_memory_ratio']}x", "PASS" if rb / ra <= thr["max_memory_ratio"] else "FAIL",
                "includes interpreter and imports; baseline-adjusted estimate unavailable; use --jobs 1")
    verdicts = [c["verdict"] for c in checks]
    overall = "FAIL" if "FAIL" in verdicts else "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "PASS"
    return {"overall": overall, "checks": checks, "task": task, "thresholds": thr,
            "thresholds_default": using_defaults, "threshold_provenance": policy_provenance(thresholds),
            "flags": flags, "split": split_info(X),
            "settings": "Recorded model recipes; inspect each backend's tier, search and adaptation provenance.",
            "splits": len(pool["splits"]) if pool else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent")
    ap.add_argument("--candidate")
    ap.add_argument("--dataset")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--thresholds", help="JSON file overriding defaults")
    ap.add_argument("--require-thresholds", action="store_true",
                    help="require a non-empty policy; partial overrides still inherit listed defaults")
    ap.add_argument("--export-thresholds", type=Path, metavar="PATH",
                    help="write the packaged defaults to a new JSON file and exit")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.export_thresholds is not None:
        if any((a.incumbent, a.candidate, a.dataset, a.thresholds, a.require_thresholds)):
            ap.error("--export-thresholds cannot be combined with evaluation options")
        try:
            export_thresholds(a.export_thresholds)
        except OSError as exc:
            ap.error(str(exc))
        print(f"Wrote default thresholds to {a.export_thresholds}. Review and edit before use.")
        return
    if not all((a.incumbent, a.candidate, a.dataset)):
        ap.error("--incumbent, --candidate, and --dataset are required for evaluation")
    try:
        thr = load_thresholds(a.thresholds) if a.thresholds else None
        r = evaluate(a.dataset, a.incumbent, a.candidate, a.model_seed, a.out, thr, a.require_thresholds)
    except (SwapError, ValueError) as e:
        sys.exit(str(e))
    print(f"SWAP REPORT: {a.incumbent} -> {a.candidate} on {a.dataset} ({r['task']}), "
          f"{r['splits']} splits, model seed {a.model_seed}; {r['settings']}")
    print("THRESHOLDS: " + describe_policy(r["threshold_provenance"]).lower())
    print(json.dumps(r["thresholds"], indent=2))
    print()
    for c in r["checks"]:
        print(f"{c['verdict']:12s} {c['check']:46s} {c['observed']:34s} limit {c['limit']}"
              + (f"   ({c['note']})" if c["note"] else ""))
    for line in leakage.format_flags(r["flags"]):
        print(f"DATA FLAG    {line}")
    print(f"\nOVERALL: {r['overall']}")
    adir = Path(a.out or analysis_settings()["out_dir"]) / a.dataset / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / f"swap_{a.incumbent}_to_{a.candidate}.json").write_text(
        json.dumps({k: r[k] for k in ("overall", "thresholds", "thresholds_default", "threshold_provenance",
                                      "checks", "flags",
                                      "split", "settings")}, indent=2, default=float))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[r["overall"]])


if __name__ == "__main__":
    main()
