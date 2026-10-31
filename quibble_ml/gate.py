"""Baseline gate: does a model beat a simple reference by a margin you chose?

  python -m quibble_ml.gate --candidate lightgbm --dataset credit-g [--baseline linear] [--min-gain 0.01]
Reuses the swap machinery with the reference as the incumbent. PASS means the candidate is better
than the baseline by at least --min-gain with the whole interval above it; FAIL means it is
confidently not that much better (or worse); INCONCLUSIVE means the interval straddles the margin.
Incomplete required candidate or baseline coverage overrides the quality verdict with FAIL.
Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE. "Nothing here is worth using" is a valid outcome.
Gain is absolute (accuracy etc.) or relative (rmse/mae), like the swap quality check.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import leakage
from .analyze import make_boot, primary_of, split_info
from .config import analysis_settings
from .datasets import load, validate_sources
from .metrics import degradation, score
from .pool import align, coverage_status, load_pool
from .swap import SwapError, interval, tri


def evaluate(dataset, candidate, baseline="linear", min_gain=0.0, model_seed=0, out=None, metric=None):
    S = analysis_settings()
    out = Path(out or S["out_dir"])
    try:
        X, yall, task = load(dataset, S["data_home"])
        validate_sources(out, dataset, X.attrs.get("quibble_source"))
    except (ValueError, OSError, ImportError) as e:
        raise SwapError(f"Dataset {dataset}: {e}") from e
    m = metric or primary_of(task)
    try:
        (pb, rb), (pc, rc) = (load_pool(out, dataset, n, model_seed) for n in (baseline, candidate))
    except ValueError as exc:
        raise SwapError(str(exc)) from exc
    if not pb:
        raise SwapError(f"baseline {baseline} has no usable runs; run it first "
                        f"(for example: quibble-ml run --sweep --backends {baseline},{candidate} --datasets {dataset})")
    if not pc:
        raise SwapError(f"candidate {candidate} has no usable runs")
    required = set(pb) | {r["split_seed"] for r in rb}
    invocation_path = out / "invocation.json"
    if invocation_path.exists():
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        required.update(c["split_seed"] for c in invocation.get("cells", [])
                        if c["dataset"] == dataset and c["backend"] in (baseline, candidate)
                        and c["model_seed"] == model_seed)
    coverage = coverage_status(required, pc, rc)
    baseline_coverage = coverage_status(required, pb, rb)
    coverage["baseline_missing"] = baseline_coverage["missing"]
    coverage["baseline_reasons"] = baseline_coverage["reasons"]
    pool = align([pb, pc])
    if not pool:
        raise SwapError("no split was completed by both models")
    y, A, B = pool["y"], pool["Ps"][0], pool["Ps"][1]
    sp, splits = pool["split"], pool["splits"]
    boot = make_boot(pool, X, S["n_boot"])

    def gain(w, sel=slice(None)):  # positive = the candidate is better than the baseline
        ww = None if w is None else w[sel]
        return -degradation(m, score(m, y[sel], A[sel], ww), score(m, y[sel], B[sel], ww))

    good = [r for r in rb if "error" not in r and "refused" not in r and r.get("split_seed") in splits]
    n_tr = float(np.mean([r["n_train"] for r in good])) if good else 0.0
    n_te = float(np.mean([r["n_test"] for r in good])) if good else 0.0
    pt, lo, hi, notes = interval(boot, gain, [gain(None, sp == s) for s in splits], n_tr, n_te)
    # PASS when the interval lies wholly above the margin; mirror the sign so tri() can be reused
    verdict = tri(-pt, -hi, -lo, -min_gain)
    incomplete = coverage["missing"] or coverage["baseline_missing"]
    return {"overall": "FAIL" if incomplete else verdict, "quality_verdict": verdict,
            "coverage": coverage, "notes": notes,
            "metric": m, "gain": pt, "lo": lo, "hi": hi, "min_gain": min_gain,
            "candidate": candidate, "baseline": baseline, "splits": len(splits), "task": task,
            "flags": leakage.check(X, yall, task, split_info(X).get("kind", "random")),
            "settings": "default model settings (no tuning)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="linear", help="reference backend (default: linear; dummy also works)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="smallest improvement over the baseline that counts (default 0)")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--metric")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        r = evaluate(a.dataset, a.candidate, a.baseline, a.min_gain, a.model_seed, a.out, a.metric)
    except SwapError as e:
        sys.exit(str(e))
    print(f"BASELINE GATE: {a.candidate} vs {a.baseline} on {a.dataset} ({r['task']}), {r['splits']} splits; "
          f"{r['settings']}")
    coverage = r["coverage"]
    print(f"COVERAGE: required {coverage['required']}; candidate completed {coverage['completed']}")
    for label, reasons in (("candidate", coverage["reasons"]), ("baseline", coverage["baseline_reasons"])):
        for ss, reason in reasons.items():
            print(f"  {label} split {ss}: {reason}")
    print(f"gain in {r['metric']}: {r['gain']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]   "
          f"required: >= {a.min_gain}")
    print(f"Quality verdict on common successful splits: {r['quality_verdict']}")
    for note in r["notes"]:
        print(f"NOTE: {note}")
    for line in leakage.format_flags(r["flags"]):
        print(f"DATA FLAG    {line}")
    print(f"\nOVERALL: {r['overall']}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[r["overall"]])


if __name__ == "__main__":
    main()
