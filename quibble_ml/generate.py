"""Validated synthetic prediction tasks; model output is data, never executable code."""
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets

VERSION = 1
MAX_CELLS = 2_000_000
LIMITS = "50-50,000 rows, 1-100 predictors, at most 2,000,000 predictor cells"


def _number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a finite number in [{low}, {high}]")
    return value


def _integer(value, label, low, high):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return _number(value, label, low, high)


def _text(value, label, maximum=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum \
            or any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise ValueError(f"{label} must be nonempty text, at most {maximum} characters")
    return value


def _keys(value, required, label):
    if not isinstance(value, dict) or set(value) != set(required.split()):
        raise ValueError(f"{label} requires exactly: {required}")


def validate_spec(spec):
    """Validate a bounded, independent-row generator specification without coercion."""
    _keys(spec, "version name scenario assumptions task rows seed target features outcome interactions", "spec")
    if type(spec["version"]) is not int or spec["version"] != VERSION:
        raise ValueError("Unsupported generator version")
    name = spec["name"]
    if not datasets.valid_name(name) or len(name) > 64 \
            or name.lower() in {n.lower() for n in (*datasets.BUILTIN, *datasets.OPENML)} \
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])", name):
        raise ValueError("Use a unique portable dataset name (1-64 letters/digits/_/-), not a registered name")
    _text(spec["scenario"], "scenario", 6000)
    if not isinstance(spec["assumptions"], list) or not 1 <= len(spec["assumptions"]) <= 20:
        raise ValueError("assumptions must contain 1-20 descriptions")
    for assumption in spec["assumptions"]:
        _text(assumption, "assumption", 1000)
    if spec["task"] not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    _integer(spec["rows"], "rows", 50, 50_000)
    _integer(spec["seed"], "seed", 0, 2**32 - 1)
    features = spec["features"]
    if not isinstance(features, list) or not 1 <= len(features) <= 100 \
            or len(features) * spec["rows"] > MAX_CELLS:
        raise ValueError(f"Generation limits: {LIMITS}")
    names = []
    for feature in features:
        _keys(feature, "name description distribution minimum maximum categories effect weight missing_rate",
              "feature")
        name = feature["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name):
            raise ValueError("Feature names must be lowercase identifiers up to 48 characters")
        names.append(name)
        _text(feature["description"], f"{name} description", 500)
        distribution = feature["distribution"]
        if distribution not in ("uniform", "integer", "normal", "categorical"):
            raise ValueError(f"{name}: unsupported distribution")
        lo = _number(feature["minimum"], f"{name} minimum", -1e6, 1e6)
        hi = _number(feature["maximum"], f"{name} maximum", -1e6, 1e6)
        if lo >= hi:
            raise ValueError(f"{name}: minimum must be less than maximum")
        if distribution == "integer" and (type(lo) is not int or type(hi) is not int):
            raise ValueError(f"{name}: integer bounds must be integers")
        categories = feature["categories"]
        if not isinstance(categories, list):
            raise ValueError(f"{name}: categories must be a list")
        if distribution == "categorical":
            if not 2 <= len(categories) <= 20:
                raise ValueError(f"{name}: provide 2-20 equally likely categories")
            for category in categories:
                _text(category, "category", 80)
                if category != category.strip() or category.lower() in ("nan", "na", "null", "none", "n/a"):
                    raise ValueError(f"{name}: category conflicts with CSV missing-value conventions")
            if len(categories) != len(set(categories)):
                raise ValueError(f"{name}: categories must be unique")
        elif categories:
            raise ValueError(f"{name}: numeric features must have no categories")
        if feature["effect"] not in ("linear", "quadratic", "threshold", "none"):
            raise ValueError(f"{name}: effect must be linear, quadratic, threshold, or none")
        _number(feature["weight"], f"{name} weight", -10, 10)
        if feature["effect"] == "none" and feature["weight"] != 0:
            raise ValueError(f"{name}: irrelevant fields must have weight 0")
        _number(feature["missing_rate"], f"{name} missing_rate", 0, 0.5)
    target = spec["target"]
    if not isinstance(target, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", target) \
            or target in names or len(set(names)) != len(names):
        raise ValueError("Target and predictor names must be unique lowercase identifiers")
    outcome = spec["outcome"]
    _keys(outcome, "intercept scale noise", "outcome")
    _number(outcome["intercept"], "outcome intercept", -10000, 10000)
    _number(outcome["scale"], "outcome scale", 0.001, 10000)
    _number(outcome["noise"], "outcome noise", 0, 10)
    interactions = spec["interactions"]
    if not isinstance(interactions, list) or len(interactions) > 20:
        raise ValueError("interactions must be a list of at most 20 pairs")
    for interaction in interactions:
        _keys(interaction, "left right weight", "interaction")
        if interaction["left"] not in names or interaction["right"] not in names \
                or interaction["left"] == interaction["right"]:
            raise ValueError("Interactions must name two distinct predictors")
        _number(interaction["weight"], "interaction weight", -10, 10)
    return spec


def recommendation(rows=1000, fields=8, seed=42, task="classification"):
    """An explicitly fictional grocery scenario, editable before generation."""
    _integer(fields, "fields", 1, 100)
    definitions = [
        ("inventory_coverage", "Inventory coverage at decision time, in days", 0, 10, "linear", -2.0),
        ("delivery_lead_days", "Expected days until delivery", 1, 7, "linear", 1.4),
        ("promotion_intensity", "Planned promotion intensity, 0 to 1", 0, 1, "linear", 0.8),
        ("recent_demand_index", "Demand index calculated only from earlier observations", 0, 3, "linear", 1.0),
    ]
    features = []
    for i in range(fields):
        name, description, lo, hi, effect, weight = definitions[i] if i < len(definitions) else (
            f"context_{i+1}", "Independent irrelevant numeric control, not a measured business variable",
            0, 1, "none", 0)
        features.append(dict(name=name, description=description, distribution="uniform",
                             minimum=lo, maximum=hi, categories=[], effect=effect,
                             weight=weight, missing_rate=0.0))
    spec = dict(version=VERSION, name="grocery_sample",
                scenario="Fictional independent grocery replenishment decisions; predict inventory pressure.",
                assumptions=[
                    "Independent rows, no repeated store/item histories or temporal forecasting.",
                    "Inputs are sampled independently; business relationships are illustrative, not calibrated.",
                    "Classification uses a stochastic logistic outcome; regression is a signed pressure index, not demand units.",
                    "Inputs exist before the outcome; no IDs, future sales, or post-outcome indicators are predictors.",
                ], task=task, rows=rows, seed=seed,
                target="stockout" if task == "classification" else "inventory_pressure",
                features=features, outcome=dict(intercept=0.0, scale=1.0, noise=0.5), interactions=[])
    return validate_spec(spec)


def generate(spec):
    """Generate full deterministic rows; no recipe or latent quantities enter the table."""
    validate_spec(spec)
    rng = np.random.default_rng(spec["seed"])
    n = spec["rows"]
    columns, encoded = {}, {}
    signal = np.zeros(n)
    for feature in spec["features"]:
        lo, hi = feature["minimum"], feature["maximum"]
        distribution = feature["distribution"]
        if distribution == "categorical":
            categories = feature["categories"]
            values = rng.integers(0, len(categories), n)
            columns[feature["name"]] = np.array(categories, dtype=object)[values]
            z = (values / (len(categories) - 1) - 0.5) * 2
        else:
            if distribution == "normal":
                values = np.clip(rng.normal((lo + hi) / 2, (hi - lo) / 6, n), lo, hi)
            elif distribution == "integer":
                values = rng.integers(lo, hi + 1, n)
            else:
                values = rng.uniform(lo, hi, n)
            columns[feature["name"]] = values
            z = (values - lo) / (hi - lo) * 2 - 1
        encoded[feature["name"]] = z
        effect = feature["effect"]
        contribution = z if effect == "linear" else z**2 if effect == "quadratic" \
            else (z > 0).astype(float) if effect == "threshold" else np.zeros(n)
        signal += feature["weight"] * contribution
    for interaction in spec["interactions"]:
        signal += interaction["weight"] * encoded[interaction["left"]] * encoded[interaction["right"]]
    outcome = spec["outcome"]
    latent = outcome["intercept"] + outcome["scale"] * signal + rng.normal(0, outcome["noise"], n)
    if spec["task"] == "classification":
        probabilities = 1 / (1 + np.exp(-np.clip(latent, -40, 40)))
        y = rng.binomial(1, probabilities)
        if min(np.bincount(y, minlength=2)) < 10:
            raise ValueError("Generated classes need at least 10 rows each. Review the intercept, weights, noise, or row count.")
    else:
        y = latent
        if np.ptp(y) == 0:
            raise ValueError("Generated regression target is constant; add signal or noise")
    frame = pd.DataFrame(columns)
    for feature in spec["features"]:
        missing = rng.random(n) < feature["missing_rate"]
        if missing.all():
            raise ValueError(f"{feature['name']}: all values are missing; review the specification")
        if missing.any():
            frame.loc[missing, feature["name"]] = np.nan
    frame[spec["target"]] = y
    assert frame.shape == (n, len(spec["features"]) + 1)
    return frame


def preview(spec):
    frame = generate(spec)
    target = frame[spec["target"]]
    return {"rows": len(frame), "predictors": len(spec["features"]),
            "columns": list(frame.columns),
            "sample": json.loads(frame.head(20).to_json(orient="records")),
            "missing": {name: int(frame[name].isna().sum()) for name in frame},
            "target_summary": target.value_counts().sort_index().to_dict()
            if spec["task"] == "classification" else
            {"min": float(target.min()), "max": float(target.max()), "mean": float(target.mean())},
            "warning": "SYNTHETIC: independent rows under reviewed assumptions, not real business evidence."}


def save(spec, planner=None):
    """Publish a validated local dataset without overwriting an existing folder."""
    frame = generate(spec)
    root = datasets.data_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / spec["name"]
    if destination.exists():
        raise ValueError("Dataset name already exists. Choose a new name; generated data is never overwritten.")
    staging = Path(tempfile.mkdtemp(prefix="generated-", dir=root))
    created = False
    try:
        csv_path = staging / "data.csv"
        frame.to_csv(csv_path, index=False, lineterminator="\n")
        manifest = {"file": "data.csv", "target": spec["target"], "task": spec["task"],
                    "categorical_columns": [f["name"] for f in spec["features"]
                                            if f["distribution"] == "categorical"]}
        (staging / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        datasets.load(staging.name)
        # Stable format identifier, not the current Python import path.
        provenance = {"synthetic": True, "generator": "quibble.generate", "version": VERSION,
                      "specification": spec, "planner": planner,
                      "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                      "numpy_version": np.__version__, "pandas_version": pd.__version__,
                      "limits": "Independent synthetic rows. No real-world or causal validity is claimed."}
        (staging / "generation.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        datasets.load(staging.name)
        destination.mkdir()
        created = True
        for name in ("data.csv", "generation.json", "dataset.json"):
            (staging / name).rename(destination / name)
    except (OSError, ValueError):
        if created:
            for name in ("data.csv", "generation.json", "dataset.json"):
                (destination / name).unlink(missing_ok=True)
            destination.rmdir()
        raise
    finally:
        shutil.rmtree(staging)
    return {"name": spec["name"], "directory": str(destination),
            "sha256": provenance["sha256"], "rows": len(frame), "predictors": len(spec["features"])}


def create_parser():
    parser = argparse.ArgumentParser(prog="quibble-ml generate", description="Generate a reviewed synthetic dataset locally")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="reviewed generator JSON specification")
    source.add_argument("--example", action="store_true", help="print a recommended specification; does not save data")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.example:
        print(json.dumps(recommendation(), indent=2))
        return 0
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        print(json.dumps(save(spec), indent=2))
    except (ValueError, OSError) as exc:
        print(f"generation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
