"""`quibble <command>`: run | compare | analyze | swap | report | datasets | doctor | demo"""
import importlib
import os
import sys

from . import __version__

COMMANDS = {"run": "quibble.run", "compare": "quibble.compare", "analyze": "quibble.analyze",
            "swap": "quibble.swap", "report": "quibble.report", "datasets": "quibble.datasets",
            "doctor": "quibble.env_info", "benchmark": "quibble.benchmark",
            "generate": "quibble.generate", "web": "quibble.web", "auth": "quibble.planner"}
USAGE = """quibble: what changes when you swap tabular backends?

  quibble demo ...             end-to-end demo: offline defaults, optional --backends; see --help
  quibble benchmark ...        device-aware presets, isolated --smoke, optional --keep-models
  quibble run ...              run backends (--sweep for many); see --help
  quibble compare --dataset D  quick summary of a single split
  quibble analyze --dataset D  CIs, calibration, slices, disagreement CSVs
  quibble swap --incumbent A --candidate B --dataset D   PASS/FAIL/INCONCLUSIVE (exit 0/1/2)
  quibble report               Markdown report with figures
  quibble datasets --verify    check the OpenML registry
  quibble doctor               show device, torch/CUDA and free VRAM
  quibble web --open           localhost scenario generation, review, experiments, and reports
  quibble generate --spec S    generate a reviewed synthetic specification locally
  quibble auth                explicit Azure sign-in and token check (--check for no login)
"""
DEMO_DATASETS = "breast_cancer,wine,diabetes_reg"
DEMO_BACKENDS = "histgb,lightgbm,weak"


def demo(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="quibble demo")
    ap.add_argument("--out", help="results root (default: QUIBBLE_OUT_DIR or results)")
    ap.add_argument("--cv", default="3x2")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--backends", default=DEMO_BACKENDS,
                    help="comma-separated backend names (default: %(default)s); optional models need their packages")
    ap.add_argument("--force", action="store_true", help="rerun saved results, including cached errors")
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
    from .config import Config
    from .report import build
    from .run import sweep
    cfg = Config.from_env()
    print(f"Demo backends: {', '.join(backends)} (CPU)")
    sweep(backends, DEMO_DATASETS.split(","), cfg, a.force)
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
        print(f"quibble {__version__}")
        return 0
    cmd = argv[0]
    if cmd == "demo":
        return demo(argv[1:])
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    sys.argv = [f"quibble {cmd}"] + argv[1:]
    return importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    sys.exit(main())
