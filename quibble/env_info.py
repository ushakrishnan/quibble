"""Hardware/software capture for reproducibility, plus `python -m quibble.doctor`."""
import os
import platform
import subprocess
import sys
from importlib import metadata

PKGS = ["numpy", "pandas", "scikit-learn", "lightgbm", "catboost",
        "autogluon.tabular", "tabpfn", "tabfm", "structured-data-models", "torch"]


def _ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


def _cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except Exception:
        return None


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "vram_total_gb": round(total / 1e9, 1), "vram_free_gb": round(free / 1e9, 1)}
    except Exception:
        return None


def collect(with_gpu=False):
    from .cache import prepare_caches
    paths = prepare_caches()
    d = {"python": sys.version.split()[0], "platform": platform.platform(),
         "cache_dir": str(paths["root"]),
         "cpu": _cpu(), "cores": os.cpu_count(), "ram_gb": _ram_gb(), "git": _git(),
         "packages": {p: _ver(p) for p in PKGS}}
    if with_gpu:
        d["gpu"] = gpu_info()
    return d


def conflicting_distributions():
    """Another PyPI project named plain `quibble` installs the same import name."""
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name == "quibble":
            found.append(dist.version)
    return found


def main():
    info = collect(with_gpu=True)
    for k, v in info.items():
        print(f"{k}: {v}")
    from .backends import BACKENDS
    print("\nbackend verification (what has actually been run):")
    for name, B in BACKENDS.items():
        print(f"  {name:13s} {B.verification}")
    if conflicting_distributions():
        print("\nWARNING: another distribution named 'quibble' is installed. It provides the same "
              "import name and command as quibble-ml and can shadow it. Uninstall it, or use a "
              "separate environment.")
    g = info.get("gpu")
    print()
    if not g:
        print("CUDA not available to torch: QUIBBLE_DEVICE=cuda will refuse to run; auto uses CPU.")
        if not info["packages"]["torch"]:
            print("torch is not installed (fine for CPU-only runs).")
    else:
        print(f"CUDA OK: {g['name']}, {g['vram_free_gb']} of {g['vram_total_gb']} GB VRAM free.")
        if g["vram_free_gb"] < 0.4 * g["vram_total_gb"]:
            print("Low free VRAM. Another process (e.g. a vLLM server) may be holding it; "
                  "stop it or lower its gpu-memory-utilization before GPU benchmarks.")


if __name__ == "__main__":
    main()
