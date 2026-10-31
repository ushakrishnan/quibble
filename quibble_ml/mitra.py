"""Immutable Mitra-v2 artifacts and the tested, task-specific runtime contract."""
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

SOURCE_REVISION = "b4701e8148dc33b00ed15d7086ff59816957cde4"
SOURCE_URL = "https://huggingface.co/autogluon/mitra-finetune"
VERSIONS = {"autogluon.tabular": "1.6.3", "tabarena": "0.1.0", "mitra-finetune": "0.3.0",
            "huggingface-hub": "1.33.0", "transformers": "5.22.1", "torch": "2.11.0"}
ARTIFACTS = {
    "classification": {
        "repository": "autogluon/mitra-classifier-2",
        "revision": "edada0d20759c58ada8c8605c25f22f6e98ea5f0",
        "files": {"config.json": "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df",
                  "model.safetensors": "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09"}},
    "regression": {
        "repository": "autogluon/mitra-regressor-2",
        "revision": "246c7d310fff04296a2388df5f448e3ffe809283",
        "files": {"config.json": "f760c38c8884757f6141c1fd2e841e93a470be052dc5a4b489790097a002f0e6",
                  "model.safetensors": "8c90b8f281a793f9feb19f8fc3c5529c58ac844dd7ffc690a0fdbb96fae0967c"}},
}
CLASSIFIER_PARAMETERS = {
    "model_type": "Tab2D", "n_estimators": 1, "fine_tune": False, "fine_tune_steps": 0,
    "metric": "log_loss", "patience": 40, "lr": 1e-4, "warmup_steps": 1000,
    "shuffle_classes": False, "shuffle_features": False, "use_random_transforms": False,
    "random_mirror_regression": True, "random_mirror_x": True, "verbose": False,
}
REGRESSION_PARAMETERS = {
    "num_bag_folds": 2, "in_process": True, "max_steps": 50, "lr": 1e-5,
    "warmup_steps": 10, "weight_decay": .3, "validation_interval": 1,
    "tune": "full", "quantile_transform": False, "category_frequency": False,
    "snapshot_k": 1,
}


@contextmanager
def adaptation_workspace():
    """Contain upstream AutoGluon files; the outer worker owns timeout cleanup."""
    from contextlib import ExitStack
    previous = Path.cwd()
    with ExitStack() as stack:
        directory = os.environ.get("_QUIBBLE_WORKSPACE") or stack.enter_context(
            tempfile.TemporaryDirectory(prefix="quibble-mitra-"))
        try:
            os.chdir(directory)
            yield
        finally:
            os.chdir(previous)


def verify_runtime():
    from .backends import Refused
    from .env_info import _ver
    if sys.platform not in ("win32", "linux") or not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise Refused("Mitra-v2 currently requires Windows/Linux, Python 3.11-3.13 and CUDA.")
    for package, expected in VERSIONS.items():
        actual = _ver(package)
        if actual is None or actual.split("+", 1)[0] != expected:
            raise Refused(f"Mitra-v2 requires verifiable {package}=={expected}; install the "
                          "locked mitra-v2 extra. Missing distribution metadata is not verification.")
    direct = metadata.distribution("mitra-finetune").read_text("direct_url.json")
    source = json.loads(direct) if direct is not None else {}
    if source.get("url") != SOURCE_URL \
            or source.get("vcs_info", {}).get("commit_id") != SOURCE_REVISION:
        raise Refused("Mitra-v2 requires the pinned mitra-finetune Git source revision.")
    if any(key.startswith("MITRA_") for key in os.environ):
        raise Refused("Unset MITRA_* upstream runtime overrides; this adapter uses a fixed recipe.")


def checkpoint(task, allow_download):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from .backends import Refused
    artifact = ARTIFACTS[task]
    directory = None
    for filename, expected in artifact["files"].items():
        try:
            path = Path(hf_hub_download(repo_id=artifact["repository"], filename=filename,
                                       revision=artifact["revision"],
                                       local_files_only=not allow_download))
        except LocalEntryNotFoundError as exc:
            raise Refused("Pinned Mitra-v2 files are not cached. Review their Apache-2.0 "
                          "model cards, then explicitly set QUIBBLE_MITRA_ALLOW_DOWNLOAD=1 "
                          "for the initial download; no other checkpoint is substituted.") from exc
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Mitra-v2 {filename} SHA-256 mismatch; refusing unverified files.")
        if directory is not None and directory != path.parent:
            raise ValueError("Mitra-v2 configuration and weights must share one pinned snapshot.")
        directory = path.parent
    return directory


@contextmanager
def cpu_budget(threads):
    """Use AutoGluon's supported CPU-discovery override inside an isolated worker."""
    import torch
    previous_env, previous_threads = os.environ.get("AG_CPU_COUNT"), torch.get_num_threads()
    try:
        if threads:
            os.environ["AG_CPU_COUNT"] = str(threads)
            torch.set_num_threads(threads)
        yield
    finally:
        torch.set_num_threads(previous_threads)
        if previous_env is None:
            os.environ.pop("AG_CPU_COUNT", None)
        else:
            os.environ["AG_CPU_COUNT"] = previous_env
