"""`quibble-ml <command>`: run | compare | analyze | swap | report | datasets | doctor | demo"""
import importlib
import os
import sys

from . import __version__

COMMANDS = {"run": "quibble_ml.run", "compare": "quibble_ml.compare", "analyze": "quibble_ml.analyze",
            "swap": "quibble_ml.swap", "gate": "quibble_ml.gate", "report": "quibble_ml.report", "datasets": "quibble_ml.datasets",
            "doctor": "quibble_ml.env_info", "benchmark": "quibble_ml.benchmark",
            "generate": "quibble_ml.generate", "web": "quibble_ml.web", "auth": "quibble_ml.planner"}
USAGE = """quibble-ml: what changes when you swap tabular backends?

  quibble-ml demo ...             end-to-end demo: offline defaults, optional --backends; see --help
  quibble-ml benchmark ...        device-aware presets, isolated --smoke, optional --keep-models
  quibble-ml run ...              run backends (--sweep for many); see --help
  quibble-ml compare --dataset D  quick summary of a single split
  quibble-ml analyze --dataset D  CIs, calibration, slices, disagreement CSVs
  quibble-ml swap --incumbent A --candidate B --dataset D   PASS/FAIL/INCONCLUSIVE (exit 0/1/2)
  quibble-ml gate --candidate B --dataset D   does B beat a simple baseline? (exit 0/1/2)
  quibble-ml report               Markdown report with figures
  quibble-ml datasets --verify    check the OpenML registry
  quibble-ml doctor               show device, torch/CUDA, free VRAM and backend verification status
  quibble-ml web --open           localhost scenario generation, review, experiments, and reports
  quibble-ml generate --spec S    generate a reviewed synthetic specification locally
  quibble-ml auth                explicit Azure sign-in and token check (--check for no login)
"""
DEMO_DATASETS = "breast_cancer,wine,diabetes_reg"
DEMO_BACKENDS = "histgb,lightgbm,weak"


def demo(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="quibble-ml demo")
    from .tuning import add_arguments
    add_arguments(ap)
    ap.add_argument("--out", help="results root (default: QUIBBLE_OUT_DIR or results)")
    ap.add_argument("--cv", default="3x2")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--backends", default=DEMO_BACKENDS,
                    help="comma-separated backend names (default: %(default)s); optional models need their packages")
    ap.add_argument("--force", action="store_true", help="rerun saved results, including cached errors")
    ap.add_argument("--retry-failed", action="store_true", help="rerun only cells whose saved result is an error")
    a = ap.parse_args(argv)
    from .backends import BACKENDS
    names = [name.strip() for name in a.backends.split(",")]
    if not all(names):
        ap.error("--backends must be a nonempty comma-separated list of backend names")
    unknown = sorted(set(names) - BACKENDS.keys())
    if unknown:
        ap.error(f"unknown backend(s): {', '.join(unknown)}; available: {', '.join(BACKENDS)}")
    backends = list(dict.fromkeys(names))
    incumbent = "histgb" if "histgb" in backends else backends[0]
    swaps = [(incumbent, name) for name in backends if name != incumbent]
    os.environ.update({"QUIBBLE_CV": a.cv, "QUIBBLE_JOBS": str(a.jobs),
                       "QUIBBLE_DEVICE": "cpu", "QUIBBLE_MODEL_SEEDS": "0"})
    if a.out is not None:
        os.environ["QUIBBLE_OUT_DIR"] = a.out
    from .run import FLAGS
    for dest in ("tier", "tune_trials", "tune_folds", "tune_seed", "tune_metric", "tune_timeout_s"):
        value = getattr(a, dest)
        if value is not None:
            os.environ[FLAGS[dest]] = str(value)
    from .config import Config
    from .report import build
    from .run import sweep
    cfg = Config.from_env()
    print(f"Demo backends: {', '.join(backends)} (CPU)")
    sweep(backends, DEMO_DATASETS.split(","), cfg, a.force, retry_failed=a.retry_failed)
    path = build(cfg.out_dir, DEMO_DATASETS.split(","), swaps=swaps)
    print(f"\nDemo report: {path}")
    if ("histgb", "weak") in swaps:
        print("(histgb -> weak is a deliberately bad candidate and must FAIL.)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"quibble-ml {__version__}")
        return 0
    cmd = argv[0]
    if cmd == "demo":
        return demo(argv[1:])
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    sys.argv = [f"quibble-ml {cmd}"] + argv[1:]
    return importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    sys.exit(main())
