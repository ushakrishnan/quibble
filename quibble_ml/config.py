"""Settings come from .env / environment; CLI flags are mapped onto the environment."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].split("\t#")[0].strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)  # real environment wins over .env


load_dotenv()


def _s(key, default):
    v = os.environ.get(key, "").strip()
    return v if v else default


def parse_cv(spec):
    k, r = spec.lower().split("x")
    k, r = int(k), int(r)
    if k < 2 or r < 1:
        raise ValueError("CV requires at least two folds and one repeat")
    return k, r


def analysis_settings():
    """Settings for analysis tools: no device validation needed."""
    from .cache import prepare_caches
    paths = prepare_caches()
    return {"out_dir": _s("QUIBBLE_OUT_DIR", "results"),
            "data_home": str(paths["openml"]),
            "n_boot": int(_s("QUIBBLE_BOOT", "500"))}


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested, pinned=""):
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError(f"QUIBBLE_DEVICE must be cpu, cuda or auto (got {requested!r})")
    if pinned and pinned not in ("cpu", "cuda"):
        raise ValueError("Invalid sweep-resolved device")
    if pinned and requested != "auto" and pinned != requested:
        raise ValueError("Sweep-resolved device conflicts with requested device")
    if (requested == "cuda" or pinned == "cuda") and not cuda_available():
        raise RuntimeError(
            "QUIBBLE_DEVICE=cuda but torch.cuda is not available. "
            "Run `python -m quibble_ml.doctor`, or set QUIBBLE_DEVICE=cpu.")
    return pinned or (requested if requested != "auto" else ("cuda" if cuda_available() else "cpu"))


@dataclass(frozen=True)
class Config:
    device_requested: str
    device: str  # effective: cpu | cuda
    threads: int
    preprocess: str
    split_seed: int
    model_seeds: Tuple[int, ...]
    jobs: int
    timeout_s: int
    ag_preset: str
    ag_time_limit: int
    tabpfn_max_rows: Optional[int]
    single_n: int
    out_dir: str
    data_home: str
    cv: str = "holdout"
    cv_seed: int = 0
    n_boot: int = 500
    tabfm_n_estimators: int = 4
    kumo_size: str = "small"
    kumo_n_estimators: int = 4
    keep_models: bool = False
    run_type: str = "benchmark"
    cache_dir: str = ""
    tabicl_allow_download: bool = False
    mitra_allow_download: bool = False
    tier: str = "defaults"
    tune_trials: int = 20
    tune_folds: int = 3
    tune_seed: int = 0
    tune_metric: str = "auto"
    tune_timeout_s: float = 0

    @property
    def split_ids(self):
        if self.cv == "holdout":
            return (self.split_seed,)
        k, r = parse_cv(self.cv)
        return tuple(range(k * r))

    @classmethod
    def from_env(cls):
        from .cache import prepare_caches
        paths = prepare_caches()
        keep = _s("QUIBBLE_KEEP_MODELS", "0")
        if keep not in ("0", "1"):
            raise ValueError("QUIBBLE_KEEP_MODELS must be 0 or 1")
        tabicl_download = _s("QUIBBLE_TABICL_ALLOW_DOWNLOAD", "0")
        if tabicl_download not in ("0", "1"):
            raise ValueError("QUIBBLE_TABICL_ALLOW_DOWNLOAD must be 0 or 1")
        mitra_download = _s("QUIBBLE_MITRA_ALLOW_DOWNLOAD", "0")
        if mitra_download not in ("0", "1"):
            raise ValueError("QUIBBLE_MITRA_ALLOW_DOWNLOAD must be 0 or 1")
        run_type = _s("_QUIBBLE_RUN_TYPE", "benchmark")
        if run_type not in ("benchmark", "smoke"):
            raise ValueError("_QUIBBLE_RUN_TYPE must be benchmark or smoke")
        req = _s("QUIBBLE_DEVICE", "auto").lower()
        pinned = os.environ.get("_QUIBBLE_RESOLVED_DEVICE", "").strip()
        dev = resolve_device(req, pinned)
        pre = _s("QUIBBLE_PREPROCESS", "ordinal").lower()
        if pre not in ("ordinal", "native"):
            raise ValueError(f"QUIBBLE_PREPROCESS must be ordinal or native (got {pre!r})")
        jobs = max(1, int(_s("QUIBBLE_JOBS", "1")))
        threads = int(_s("QUIBBLE_THREADS", "0"))
        if threads == 0 and jobs > 1:
            threads = max(1, (os.cpu_count() or 1) // jobs)
        cv = _s("QUIBBLE_CV", "holdout").lower()
        if cv != "holdout":
            try:
                parse_cv(cv)
            except ValueError:
                raise ValueError(f"QUIBBLE_CV must be 'holdout' or like '5x2' (got {cv!r})")
        tp = _s("QUIBBLE_TABPFN_MAX_ROWS", "")
        tabfm_n = int(_s("QUIBBLE_TABFM_N_ESTIMATORS", "4"))
        kumo_n = int(_s("QUIBBLE_KUMO_N_ESTIMATORS", "4"))
        if tabfm_n < 1 or kumo_n < 1:
            raise ValueError("QUIBBLE_TABFM_N_ESTIMATORS and QUIBBLE_KUMO_N_ESTIMATORS must be positive")
        kumo_size = _s("QUIBBLE_KUMO_SIZE", "small").lower()
        if kumo_size not in ("small", "medium", "large"):
            raise ValueError("QUIBBLE_KUMO_SIZE must be small, medium or large")
        tier = _s("QUIBBLE_TIER", "defaults").lower()
        trials = int(_s("QUIBBLE_TUNE_TRIALS", "20"))
        folds = int(_s("QUIBBLE_TUNE_FOLDS", "3"))
        metric = _s("QUIBBLE_TUNE_METRIC", "auto").lower()
        deadline = float(_s("QUIBBLE_TUNE_TIMEOUT_S", "0"))
        if tier not in ("defaults", "tuned"):
            raise ValueError("QUIBBLE_TIER must be defaults or tuned")
        if not 1 <= trials <= 1000 or not 2 <= folds <= 20:
            raise ValueError("Tuning requires 1..1000 trials and 2..20 inner folds")
        if metric not in ("auto", "log_loss", "rmse"):
            raise ValueError("QUIBBLE_TUNE_METRIC must be auto, log_loss or rmse")
        import math
        if not math.isfinite(deadline) or deadline < 0:
            raise ValueError("QUIBBLE_TUNE_TIMEOUT_S must be finite and nonnegative")
        return cls(
            device_requested=req, device=dev, threads=threads, preprocess=pre,
            split_seed=int(_s("QUIBBLE_SPLIT_SEED", "0")),
            model_seeds=tuple(int(x) for x in _s("QUIBBLE_MODEL_SEEDS", "0,1,2").split(",")),
            jobs=jobs, timeout_s=int(_s("QUIBBLE_TIMEOUT_S", "900")),
            ag_preset=_s("QUIBBLE_AG_PRESET", "medium_quality"),
            ag_time_limit=int(_s("QUIBBLE_AG_TIME_LIMIT", "120")),
            tabpfn_max_rows=int(tp) if tp else None,
            single_n=int(_s("QUIBBLE_SINGLE_N", "30")),
            out_dir=_s("QUIBBLE_OUT_DIR", "results"),
            data_home=str(paths["openml"]), cache_dir=str(paths["root"]),
            keep_models=keep == "1", run_type=run_type,
            tabicl_allow_download=tabicl_download == "1",
            mitra_allow_download=mitra_download == "1",
            cv=cv, cv_seed=int(_s("QUIBBLE_CV_SEED", "0")), n_boot=int(_s("QUIBBLE_BOOT", "500")),
            tabfm_n_estimators=tabfm_n, kumo_size=kumo_size, kumo_n_estimators=kumo_n,
            tier=tier, tune_trials=trials, tune_folds=folds,
            tune_seed=int(_s("QUIBBLE_TUNE_SEED", "0")), tune_metric=metric,
            tune_timeout_s=deadline,
        )
