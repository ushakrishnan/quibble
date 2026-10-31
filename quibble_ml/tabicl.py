"""Pinned TabICLv2 artifacts; downloading requires an explicit opt-in."""
import hashlib
from pathlib import Path

VERSION = "2.2.0"
REPOSITORY = "jingang/TabICL"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
ARTIFACTS = {
    "classification": {
        "filename": "tabicl-classifier-v2-20260212.ckpt",
        "sha256": "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"},
    "regression": {
        "filename": "tabicl-regressor-v2-20260212.ckpt",
        "sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"},
}
PARAMETERS = {"n_estimators": 8, "batch_size": 8, "kv_cache": False,
              "offload_mode": False, "use_amp": "auto", "use_fa3": False,
              "norm_methods": None, "feat_shuffle_method": "latin", "outlier_threshold": 4.0}


def checkpoint(task, allow_download):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from .backends import Refused
    artifact = ARTIFACTS[task]
    try:
        path = Path(hf_hub_download(repo_id=REPOSITORY, filename=artifact["filename"],
                                   revision=REVISION, local_files_only=not allow_download))
    except LocalEntryNotFoundError as exc:
        if allow_download:
            raise Refused("Pinned TabICLv2 checkpoint is unavailable; no alternate weights "
                          "or model version will be substituted.") from exc
        raise Refused("TabICLv2 checkpoint is not cached. Review its terms, then explicitly set "
                      "QUIBBLE_TABICL_ALLOW_DOWNLOAD=1 for the initial download.") from exc
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != artifact["sha256"]:
        raise ValueError("TabICLv2 checkpoint SHA-256 mismatch; refusing to load unverified weights.")
    return path
