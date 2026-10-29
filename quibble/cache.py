"""Configure download caches before importing model libraries."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def prepare_caches() -> dict[str, Path]:
    from .config import load_dotenv
    load_dotenv()
    root = project_path(os.environ.get("QUIBBLE_CACHE_DIR", "").strip() or ".cache")
    hf = project_path(os.environ.get("QUIBBLE_HF_CACHE_DIR", "").strip() or root / "huggingface")
    sklearn = project_path(os.environ.get("QUIBBLE_SKLEARN_CACHE_DIR", "").strip() or root / "sklearn")
    transformers = project_path(
        os.environ.get("QUIBBLE_TRANSFORMERS_CACHE_DIR", "").strip() or hf / "hub")
    paths = {
        "root": root, "huggingface": hf, "torch": root / "torch",
        "tabpfn": root / "tabpfn", "sklearn": sklearn, "xdg": root / "xdg",
        "transformers": transformers,
        "openml": project_path(os.environ.get("QUIBBLE_DATA_HOME", "").strip() or root / "openml"),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in {
        "HF_HOME": hf, "HF_HUB_CACHE": hf / "hub", "HUGGINGFACE_HUB_CACHE": hf / "hub",
        "HF_DATASETS_CACHE": hf / "datasets", "HF_ASSETS_CACHE": hf / "assets",
        "TORCH_HOME": paths["torch"], "TABPFN_MODEL_CACHE_DIR": paths["tabpfn"],
        "TORCH_EXTENSIONS_DIR": paths["torch"] / "extensions",
        "TORCHINDUCTOR_CACHE_DIR": paths["torch"] / "inductor",
        "TRITON_CACHE_DIR": root / "triton",
        "SKLEARN_DATA": paths["sklearn"], "OPENML_CACHE_DIRECTORY": paths["openml"],
        "XDG_CACHE_HOME": paths["xdg"],
    }.items():
        os.environ[key] = str(path)
    # Older Transformers versions can otherwise retain an inherited external cache.
    if "TRANSFORMERS_CACHE" in os.environ or os.environ.get("QUIBBLE_TRANSFORMERS_CACHE_DIR", "").strip():
        os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    return paths
