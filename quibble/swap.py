"""Swap report: is replacing INCUMBENT with CANDIDATE safe on this dataset?

  python -m quibble.swap --incumbent histgb --candidate lightgbm --dataset breast_cancer
  python -m quibble.swap ... --thresholds my_thresholds.json
Each check is PASS (confidently within the limit), FAIL (confidently beyond it) or
INCONCLUSIVE (the 95% interval straddles the limit: collect more data). Exit code:
0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE.
Quality drop is absolute (accuracy etc.) or relative (rmse/mae). Slice check is the WORST of
several slices and is not multiple-comparison corrected; read it as a flag, not a verdict.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .analyze import HI, pair_frame, primary_of
from .config import analysis_settings
from .datasets import load, validate_sources
from .metrics import Boot, degradation, score
from .pool import align, load_pool
from .slices import row_traits, slice_masks

DEFAULTS = {"primary": None, "max_quality_drop": 0.01, "max_disagreement": 0.05,
            "max_confident_disagreement": 0.01, "max_latency_ratio": 2.0, "max_memory_ratio": 1.5,
            "max_slice_drop": 0.05, "min_slice_n": 30, "max_ece_increase": 0.02}


def tri(pt, lo, hi, thr):
    if np.isnan(lo) or np.isnan(hi):
        return "PASS" if pt <= thr else "FAIL"
    return "PASS" if hi <= thr else "FAIL" if lo > thr else "INCONCLUSIVE"


def mean_of(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None and "error" not in r and "refused" not in r]
    return float(np.mean(v)) if v else np.nan


class SwapError(Exception):
    pass


def evaluate(dataset, incumbent, candidate, model_seed=0, out=None, thresholds=None):
    """-> {"overall", "checks", "splits", "task", "thresholds"}. Raises SwapError if unusable."""
    thr = dict(DEFAULTS)
    thr.update(thresholds or {})
    S = analysis_settings()
    out = Path(out or S["out_dir"])
    try:
        X, yall, task = load(dataset, S["data_home"])
        validate_sources(out, dataset, X.attrs.get("quibble_source"))
    except (ValueError, OSError, ImportError) as e:
        raise SwapError(f"Dataset {dataset}: {e}") from e
    clf, m = task == "classification", thr["primary"] or primary_of(task)
    (pi, ri), (pc, rc) = (load_pool(out, dataset, n, model_seed) for n in (incumbent, candidate))
    if not pi:
        raise SwapError(f"incumbent {incumbent} has no usable runs")
    checks = []

    def add(name, obs, limit, verdict, note=""):
        checks.append({"check": name, "observed": obs, "limit": limit, "verdict": verdict, "note": note})

    missing = sorted(set(pi) - set(pc))
    add("coverage", f"{len(pc)}/{len(pi)} splits ran", "all", "PASS" if not missing else "FAIL",
        "" if not missing else f"candidate failed/refused on splits {missing}")
    pool = align([pi, pc]) if pc else None
    if pool:
        y, A, B = pool["y"], pool["Ps"][0], pool["Ps"][1]
        boot = Boot(pool["row_pos"], S["n_boot"])

        def deg(w, sel=slice(None)):
            ww = None if w is None else w[sel]
            return degradation(m, score(m, y[sel], A[sel], ww), score(m, y[sel], B[sel], ww))

        pt, lo, hi = boot.ci(deg)
        add(f"quality drop ({m})", f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]", f"<= {thr['max_quality_drop']}",
            tri(pt, lo, hi, thr["max_quality_drop"]))
        f = pair_frame(pool, 0, 1, row_traits(X), task)
        disp = f["disputed"].to_numpy()
        pt, lo, hi = boot.ci(lambda w: float(np.average(disp, weights=w)))
        add("disagreement rate", f"{pt:.2%} [{lo:.2%}, {hi:.2%}]", f"<= {thr['max_disagreement']:.0%}",
            tri(pt, lo, hi, thr["max_disagreement"]))
        if clf:
            cb = f["confident_both"].to_numpy()
            pt, lo, hi = boot.ci(lambda w: float(np.average(cb, weights=w)))
            add("confident disagreements", f"{pt:.2%} [{lo:.2%}, {hi:.2%}]",
                f"<= {thr['max_confident_disagreement']:.0%}",
                tri(pt, lo, hi, thr["max_confident_disagreement"]), f"both models >= {HI:.0%} sure")
            e = score("ece", y, B) - score("ece", y, A)
            add("calibration (ECE increase)", f"{e:+.4f}", f"<= {thr['max_ece_increase']}",
                "PASS" if e <= thr["max_ece_increase"] else "FAIL")
        worst, tested = None, 0
        for name, mk in slice_masks(X, yall, task).items():
            sel = mk[pool["idx"]]
            if sel.sum() < thr["min_slice_n"]:
                continue
            tested += 1
            p0 = deg(None, sel)
            if worst is None or p0 > worst[1]:
                worst = (name, p0, sel)
        if worst:
            name, _, sel = worst
            pt, lo, hi = boot.ci(lambda w: deg(w, sel))
            add(f"worst slice: {name} (n={int(sel.sum())}/{tested} slices)",
                f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]", f"<= {thr['max_slice_drop']}",
                tri(pt, lo, hi, thr["max_slice_drop"]), "not multiple-comparison corrected")
    for key, lim, label in (("single_row_p50_ms", "max_latency_ratio", "latency ratio (single-row p50)"),
                            ("peak_rss_mb", "max_memory_ratio", "memory ratio (peak RSS)")):
        ra, rb = mean_of(ri, key), mean_of(rc, key)
        if np.isnan(ra) or np.isnan(rb) or ra == 0:
            continue
        add(label, f"{rb / ra:.2f}x ({ra:.2f} -> {rb:.2f})", f"<= {thr[lim]}x",
            "PASS" if rb / ra <= thr[lim] else "FAIL", "single-machine timings; use --jobs 1 runs")
    verdicts = [c["verdict"] for c in checks]
    overall = "FAIL" if "FAIL" in verdicts else "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "PASS"
    return {"overall": overall, "checks": checks, "task": task, "thresholds": thr,
            "splits": len(pool["splits"]) if pool else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--thresholds", help="JSON file overriding defaults")
    ap.add_argument("--out")
    a = ap.parse_args()
    thr = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    try:
        r = evaluate(a.dataset, a.incumbent, a.candidate, a.model_seed, a.out, thr)
    except SwapError as e:
        sys.exit(str(e))
    print(f"SWAP REPORT: {a.incumbent} -> {a.candidate} on {a.dataset} ({r['task']}), "
          f"{r['splits']} splits, model seed {a.model_seed}\n")
    for c in r["checks"]:
        print(f"{c['verdict']:12s} {c['check']:46s} {c['observed']:34s} limit {c['limit']}"
              + (f"   ({c['note']})" if c["note"] else ""))
    print(f"\nOVERALL: {r['overall']}")
    adir = Path(a.out or analysis_settings()["out_dir"]) / a.dataset / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / f"swap_{a.incumbent}_to_{a.candidate}.json").write_text(
        json.dumps({k: r[k] for k in ("overall", "thresholds", "checks")}, indent=2, default=float))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[r["overall"]])


if __name__ == "__main__":
    main()
