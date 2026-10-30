"""Device-aware comparison presets and isolated setup checks."""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .backends import BACKENDS
from .config import Config
from .datasets import known
from .history import new_run_id
from .run import FLAGS, stem, sweep
from .thresholds import load_thresholds, resolve_thresholds, validate_primary

PRESETS = {
    "trees": ("histgb", "lightgbm", "catboost"),
    "foundations": ("histgb", "lightgbm", "tabfm", "kumo_tabular"),
    "all": tuple(n for n, b in BACKENDS.items() if not getattr(b, "internal", False)),
}


def main() -> int:
    ap = argparse.ArgumentParser(prog="quibble benchmark")
    ap.add_argument("--preset", choices=PRESETS, default="foundations")
    ap.add_argument("--backends", help="override preset with comma-separated backend names")
    ap.add_argument("--datasets", default="wine", help="comma-separated names (default: wine)")
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), help="default: .env/environment")
    ap.add_argument("--cv", help="default: 3x2; smoke requires holdout")
    ap.add_argument("--model-seeds", help="default: 0; smoke requires 0")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--out")
    ap.add_argument("--thresholds", type=Path, help="JSON overrides for swap limits")
    ap.add_argument("--require-thresholds", action="store_true",
                    help="require a non-empty policy; partial overrides still inherit listed defaults")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="rerun only cells whose saved result is an error")
    ap.add_argument("--keep-models", action="store_const", const="1", default=None,
                    help="retain fitted tree/AutoGluon models; not foundation estimators")
    ap.add_argument("--smoke", action="store_true",
                    help="isolated setup check: holdout, seed 0, at most 128 train/32 test rows")
    a = ap.parse_args()
    backends = list(dict.fromkeys(n.strip() for n in a.backends.split(","))) \
        if a.backends is not None else list(PRESETS[a.preset])
    datasets = list(dict.fromkeys(n.strip() for n in a.datasets.split(",")))
    if not backends or any(n not in BACKENDS for n in backends):
        ap.error(f"--backends must contain known names: {', '.join(BACKENDS)}")
    if not datasets or any(not known(n) for n in datasets):
        ap.error("--datasets must contain registered or local dataset names; use datasets --list")
    if a.smoke and (a.cv not in (None, "holdout") or a.model_seeds not in (None, "0")):
        ap.error("--smoke requires holdout and model seed 0")
    if a.smoke and (a.thresholds or a.require_thresholds):
        ap.error("--smoke does not run swap checks; threshold options require a full benchmark")
    try:
        thresholds = load_thresholds(a.thresholds) if a.thresholds else None
        effective = resolve_thresholds(thresholds, a.require_thresholds)
        if effective["primary"] is not None:
            from .datasets import load
            for name in datasets:
                validate_primary(effective, load(name)[2])
    except ValueError as exc:
        ap.error(str(exc))
    a.cv = a.cv or ("holdout" if a.smoke else "3x2")
    a.model_seeds = a.model_seeds or "0"
    for dest, key in FLAGS.items():
        value = getattr(a, dest, None)
        if value is not None:
            os.environ[key] = str(value)
    os.environ["_QUIBBLE_WORKFLOW"] = "benchmark"
    os.environ["_QUIBBLE_RUN_TYPE"] = "smoke" if a.smoke else "benchmark"
    try:
        cfg = Config.from_env()
    except (ValueError, RuntimeError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if a.smoke:
        directory = Path(cfg.out_dir) / "smoke" / new_run_id()
        cfg = replace(cfg, out_dir=str(directory))
        os.environ["QUIBBLE_OUT_DIR"] = cfg.out_dir
        print("SMOKE TEST ONLY: bounded setup check, not benchmark or deployment evidence.")
    print(f"Benchmark preset: {a.preset}; backends: {', '.join(backends)}; device={cfg.device}")
    try:
        invocation = sweep(backends, datasets, cfg, a.force, retry_failed=a.retry_failed)
    except ValueError as e:
        print(f"selection error: {e}", file=sys.stderr)
        return 2
    from .backends import select_backends
    backends, _ = select_backends(backends, cfg.device)
    from .report import build
    incumbent = "histgb" if "histgb" in backends else backends[0]
    swaps = [] if a.smoke else [(incumbent, b) for b in backends if b != incumbent]
    path = build(cfg.out_dir, datasets, model_seed=cfg.model_seeds[0], swaps=swaps,
                 thresholds=thresholds, require_thresholds=a.require_thresholds)
    print(f"\n{'Smoke' if a.smoke else 'Benchmark'} report: {path}")
    failed = False
    for cell in invocation["cells"]:
        record_path = stem(cfg.out_dir, cell["dataset"], cell["backend"],
                           cell["split_seed"], cell["model_seed"]).with_suffix(".json")
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        failed |= (cell["execution"] in ("not run", "incompatible settings")
                   or not record or "error" in record or "refused" in record)
    if failed:
        print("Some requested cells failed, refused, are missing, or have incompatible settings; "
              "see report participation.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
