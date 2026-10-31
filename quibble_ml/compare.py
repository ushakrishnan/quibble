"""Summarise one dataset and measure disagreement between backends and between seeds.

  python -m quibble_ml.compare --dataset credit-g
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import _s
from .datasets import source_info, validate_sources

HI = 0.9    # classification: both models at least this confident
BIG = 0.25  # regression: gap larger than this many target std devs


def gap(a, b, y, task):
    """(headline disagreement, count of 'serious' disagreements)."""
    if task == "classification":
        diff = a.argmax(1) != b.argmax(1)
        return diff.mean(), int((diff & (a.max(1) >= HI) & (b.max(1) >= HI)).sum())
    d = np.abs(a - b) / (np.std(y) or 1.0)
    return float(d.mean()), int((d > BIG).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=_s("QUIBBLE_OUT_DIR", "results"))
    ap.add_argument("--split-seed", type=int, default=int(_s("QUIBBLE_SPLIT_SEED", "0")))
    a = ap.parse_args()
    d, ss = Path(a.out) / a.dataset, a.split_seed
    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(d.glob(f"*_ss{ss}_ms*.json"))])
    if df.empty:
        return print("no results found")
    try:
        validate_sources(a.out, a.dataset, source_info(a.dataset))
    except (ValueError, OSError) as e:
        ap.error(str(e))
    ok = df[df["task"].notna() & (df.get("accuracy", df.get("rmse")).notna())] \
        if ("task" in df and ("accuracy" in df or "rmse" in df)) else df.iloc[0:0]
    if ok.empty:
        print("no successful runs")
    else:
        task = ok["task"].iloc[0]
        qual = ["accuracy", "auc", "brier"] if task == "classification" else ["rmse", "mae", "r2"]
        cols = [c for c in qual + ["fit_s", "fit_total_s", "search_s", "refit_s",
                                   "search_refit_s", "batch_prediction_s", "model_pipeline_s",
                                   "adaptation_and_prediction_s", "single_row_p50_ms", "peak_rss_mb",
                                   "peak_torch_vram_mb"] if c in ok]
        print(f"== {a.dataset} ({task}), split seed {ss}, mean over model seeds ==")
        if "tier" in ok:
            print("Actual recipe tiers: " + ", ".join(
                f"{name}={','.join(sorted(set(group['tier'].dropna())))}"
                for name, group in ok.groupby("backend")))
        if "search_refit_s" in ok:
            print("Search/refit timings overlap fit_total_s; do not add them together.")
        if "prediction_adapts" in ok and ok["prediction_adapts"].eq(True).any():
            print("Prediction-time adaptation is included in batch/model_pipeline cost; "
                  "pure-inference single-row latency is unavailable.")
        t = ok.groupby("backend")[cols].mean().round(4)
        t.insert(0, "runs", ok.groupby("backend").size())
        t.insert(1, "device", ok.groupby("backend")["device_effective"].first()
                 if "device_effective" in ok else "")
        print(t.to_string())
    for col in ("refused", "error"):
        if col in df and df[col].notna().any():
            print(f"\n== {col} ==")
            print(df[df[col].notna()][["backend", "model_seed", col]].to_string(index=False))
    if ok.empty:
        return
    y = np.load(d / f"y_test_ss{ss}.npy")
    P = {(r.backend, r.model_seed): np.load(d / f"{r.backend}_ss{ss}_ms{r.model_seed}.pred.npy")
         for r in ok.itertuples()}
    names = sorted(ok.backend.unique())
    unit = "rows differ" if task == "classification" else "mean |gap| in target std"
    ser = "confident-both disagreements" if task == "classification" else \
        f"rows with gap > {BIG} std"
    print(f"\n== backend vs backend ({unit}; {ser}/model seed) ==")
    for x, z in itertools.combinations(names, 2):
        r = [gap(P[(x, s)], P[(z, s)], y, task) for s in sorted(ok.model_seed.unique())
             if (x, s) in P and (z, s) in P]
        if r:
            v = np.mean([q[0] for q in r])
            print(f"{x} vs {z}: {v:.2%} {''}| {np.mean([q[1] for q in r]):.1f}"
                  if task == "classification" else
                  f"{x} vs {z}: {v:.3f} | {np.mean([q[1] for q in r]):.1f}")
    print("\n== same backend, different model seeds (split fixed) ==")
    for n in names:
        seeds = [s for (b, s) in P if b == n]
        r = [gap(P[(n, s1)], P[(n, s2)], y, task) for s1, s2 in itertools.combinations(seeds, 2)]
        if r:
            v = np.mean([q[0] for q in r])
            print(f"{n}: {v:.2%}" if task == "classification" else f"{n}: {v:.3f}")


if __name__ == "__main__":
    main()
