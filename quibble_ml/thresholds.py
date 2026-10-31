"""Shared, packaged swap defaults and validation for CLI, reports, and the workbench."""
import json
import math
from importlib.resources import files
from pathlib import Path

from .metrics import HIGHER

DEFAULTS = json.loads(files("quibble_ml").joinpath("swap_thresholds.json").read_text(encoding="utf-8"))
DEFAULT_SOURCE = "quibble_ml/swap_thresholds.json"


def resolve_thresholds(overrides=None, require=False):
    """Validate overrides and merge them with the packaged defaults."""
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Thresholds must be a JSON object.")
    if require and not overrides:
        raise ValueError("No thresholds supplied and --require-thresholds is set: "
                         f"supply a non-empty policy instead of defaults from {DEFAULT_SOURCE}.")
    unknown = set(overrides or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown threshold keys: {', '.join(sorted(map(str, unknown)))}")
    result = {**DEFAULTS, **(overrides or {})}
    primary = result["primary"]
    if primary is not None and (not isinstance(primary, str) or primary not in HIGHER):
        raise ValueError(f"primary must be null or one of: {', '.join(HIGHER)}")
    for key, value in result.items():
        if key == "primary":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{key} must be a finite number.")
        if value < 0 and key not in ("max_quality_drop", "max_slice_drop", "max_ece_increase"):
            raise ValueError(f"{key} must be non-negative.")
        if key == "min_slice_n" and (not isinstance(value, int) or value < 1):
            raise ValueError("min_slice_n must be a positive integer.")
        if key in ("max_latency_ratio", "max_memory_ratio") and value <= 0:
            raise ValueError(f"{key} must be greater than zero.")
        if key in ("max_disagreement", "max_confident_disagreement", "confident_prob",
                   "near_margin") and value > 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    return result


def validate_primary(thresholds, task):
    """Reject metrics that do not match the dataset's prediction shape."""
    primary = thresholds["primary"]
    regression = {"rmse", "mae", "r2"}
    if primary is not None and ((primary in regression) != (task == "regression")):
        raise ValueError(f"primary metric {primary!r} is not valid for {task}.")


def policy_provenance(overrides):
    """Keep explicit settings distinct from values inherited from packaged placeholders."""
    provided = sorted(overrides or {})
    inherited = sorted(set(DEFAULTS) - set(provided))
    kind = "defaults" if not provided else "partial" if inherited else "explicit"
    return {"kind": kind, "provided": provided, "inherited": inherited, "source": DEFAULT_SOURCE}


def describe_policy(provenance):
    if provenance["kind"] == "defaults":
        return f"Packaged defaults ({DEFAULT_SOURCE}); placeholders, not application-approved limits"
    if provenance["kind"] == "partial":
        return ("Custom policy (partial override); inherited packaged defaults: "
                + ", ".join(provenance["inherited"]))
    return "Custom policy (explicit values for all settings)"


def load_thresholds(path):
    """Read and validate a custom policy, retaining partial overrides."""
    try:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Thresholds must be a JSON object.")
        resolve_thresholds(overrides)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load thresholds from {path}: {exc}") from exc
    return overrides


def export_thresholds(path):
    """Export a working copy without overwriting an existing policy."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(DEFAULTS, indent=2) + "\n")
