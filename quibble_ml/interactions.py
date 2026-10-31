"""Bounded, exploratory disagreement rules; never inputs to swap decisions."""
import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor

from .thresholds import DEFAULTS

WARNING = (
    "Exploratory selection: rules were selected using disagreement in discovery data. "
    "Evaluation subdivides an existing prediction/CV pool, with shared-model dependence; "
    "it is not independent confirmation, even with disjoint rows or groups. Confirmatory "
    "use requires a locked holdout unused by tuning, model selection, or rule discovery. "
    "These rules never change any swap verdict."
)
JSON_FILE = "interaction_rules.json"
CSV_FILE = "interaction_rules.csv"
SUMMARY_COLUMNS = [
    "rule_id", "backend_a", "backend_b", "recipe_family_a", "recipe_family_b",
    "seed", "min_rows", "min_groups", "split_unit", "label", "text", "discovery_rows",
    "discovery_groups", "evaluation_rows", "evaluation_groups", "evaluation_entries",
    "evaluation_status", "disagreement_rate", "quality_difference", "quality_metric",
]


@dataclass(frozen=True)
class InteractionSpec:
    """Features are a caller-reviewed allowlist, not all available columns."""

    pair: tuple[str, str]
    features: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    seed: int = 0
    min_rows: int = 20
    min_groups: int = 2
    max_features: int = 16
    max_rules: int = 4
    evaluation_fraction: float = 0.5
    recipe_ids: tuple[str, str] | None = None

    def __post_init__(self):
        if len(self.pair) != 2 or len(set(self.pair)) != 2 or not all(self.pair):
            raise ValueError("interaction pair must name two different backends")
        if not 1 <= self.max_features <= 16 or not 1 <= len(self.features) <= self.max_features:
            raise ValueError("interactions require 1..max_features explicit safe features (cap 16)")
        if len(set(self.features)) != len(self.features):
            raise ValueError("interaction features must be unique")
        if not 1 <= self.max_rules <= 4 or self.min_rows < 2 or self.min_groups < 1:
            raise ValueError("interaction rules cap is 1..4; min_rows >= 2 and min_groups >= 1")
        if not 0 < self.evaluation_fraction < 1 or not 0 <= self.seed < 2**32:
            raise ValueError("interaction fraction must be in (0,1); seed must be uint32")
        if self.recipe_ids is not None and len(self.recipe_ids) != 2:
            raise ValueError("interaction recipe IDs must match the ordered backend pair")


def add_arguments(parser):
    """Share opt-in CLI flags between analysis and reports."""
    parser.add_argument("--interaction-pair", help="ordered BACKEND_A:BACKEND_B")
    parser.add_argument("--interaction-features", help="comma-separated reviewed safe feature names")
    parser.add_argument("--interaction-exclude", default="", help="IDs/post-outcome/unsafe columns")
    parser.add_argument("--interaction-recipes", help="optional ordered family_sha256 A:B assertions")
    parser.add_argument("--interaction-seed", type=int, default=0)
    parser.add_argument("--interaction-min-rows", type=int, default=20)
    parser.add_argument("--interaction-min-groups", type=int, default=2)
    parser.add_argument("--interaction-max-features", type=int, default=16)
    parser.add_argument("--interaction-max-rules", type=int, default=4)


def spec_from_arguments(args):
    if not args.interaction_pair:
        if args.interaction_features or args.interaction_recipes or args.interaction_exclude:
            raise ValueError("--interaction-pair is required for interaction options")
        return None
    return InteractionSpec(
        pair=tuple(args.interaction_pair.split(":")),
        features=tuple(s.strip() for s in (args.interaction_features or "").split(",") if s.strip()),
        exclude=tuple(s.strip() for s in args.interaction_exclude.split(",") if s.strip()),
        recipe_ids=tuple(args.interaction_recipes.split(":")) if args.interaction_recipes else None,
        seed=args.interaction_seed, min_rows=args.interaction_min_rows,
        min_groups=args.interaction_min_groups, max_features=args.interaction_max_features,
        max_rules=args.interaction_max_rules,
    )


def _validate_features(X: pd.DataFrame, spec: InteractionSpec) -> None:
    source = X.attrs.get("quibble_source") or {}
    split = X.attrs.get("quibble_split") or {}
    protected = set(spec.exclude) | set(source.get("drop_columns", []))
    for metadata in (source, split):
        protected.update(metadata.get(k) for k in ("target", "time_column", "group_column",
                                                   "id_column", "row_id_column"))
    roles = X.attrs.get("quibble_feature_roles") or {}
    protected.update(k for k, role in roles.items() if role not in ("feature", "predictor"))
    for name in spec.features:
        normalized = re.sub(r"\W+", "_", name.lower())
        reserved = normalized in {"y", "id", "row", "index", "target", "label", "split", "fold",
                                  "row_id", "row_index", "split_seed", "model_seed", "outcome"}
        reserved |= normalized.endswith(("_id", "_target", "_outcome", "_label"))
        reserved |= normalized.startswith(("pred_", "prediction", "post_outcome", "a_correct",
                                           "b_correct", "disputed"))
        if name in protected or reserved:
            raise ValueError(f"unsafe interaction feature: {name}")
        if name not in X.columns or list(X.columns).count(name) != 1:
            raise ValueError(f"missing or ambiguous interaction feature: {name}")


def _hash_bucket(value) -> int:
    # Stateless, bounded encoding: no private category vocabulary in exported rules.
    raw = f"{type(value).__name__}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 16


def fit_schema(discovery: pd.DataFrame) -> list[dict]:
    """Learn numeric medians only from discovery; categorical hashing is stateless."""
    schema = []
    for name in discovery:
        col = discovery[name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
            finite = np.clip(values[np.isfinite(values)], -1e30, 1e30)
            schema.append({"feature": name, "kind": "numeric", "dtype": str(col.dtype),
                           "impute": float(np.median(finite)) if len(finite) else 0.0,
                           "missing": "null/nonfinite -> median of clipped discovery values; "
                                      "all missing -> 0"})
        else:
            schema.append({"feature": name, "kind": "hash_bucket", "dtype": str(col.dtype),
                           "buckets": 16, "missing": "null -> -1",
                           "unseen": "same SHA256(type-name:value) bucket mapping; collisions possible"})
    return schema


def transform(X: pd.DataFrame, schema: list[dict]) -> np.ndarray:
    """Apply a frozen schema without fitting, vocabulary lookup, or executable predicates."""
    columns = []
    for field in schema:
        col = X[field["feature"]]
        if field["kind"] == "numeric":
            values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = field["impute"]
        else:
            values = np.asarray([-1 if pd.isna(v) else _hash_bucket(v) for v in col], dtype=float)
        # sklearn trees compare float32 inputs; freeze that same comparison semantics.
        columns.append(np.clip(values, -1e30, 1e30).astype(np.float32))
    return np.column_stack(columns)


def rule_mask(encoded: np.ndarray, predicates: list[dict]) -> np.ndarray:
    """Interpret only structured <= and > predicates, never eval."""
    mask = np.ones(len(encoded), dtype=bool)
    for predicate in predicates:
        # Trees store double thresholds and compare them against float32 feature values.
        # NumPy's weak Python-scalar coercion must not round a threshold to float32.
        values = encoded[:, predicate["column"]].astype(np.float64, copy=False)
        if predicate["operator"] == "<=":
            mask &= values <= predicate["threshold"]
        elif predicate["operator"] == ">":
            mask &= values > predicate["threshold"]
        else:
            raise ValueError("unknown interaction predicate operator")
    return mask


def partition_rows(rows: np.ndarray, groups, spec: InteractionSpec):
    """Fixed-seed row/group split. No row or group identities are exported."""
    codes = np.arange(len(rows)) if groups is None else pd.factorize(np.asarray(groups)[rows])[0]
    if (codes < 0).any():
        raise ValueError("declared interaction groups must not be missing")
    units = np.unique(codes)
    shuffled = np.random.RandomState(spec.seed).permutation(units)
    n_eval = min(len(units) - 1, max(1, int(np.ceil(len(units) * spec.evaluation_fraction))))
    evaluation = np.isin(codes, shuffled[:n_eval]) if len(units) > 1 else np.zeros(len(rows), bool)
    return ~evaluation, evaluation, codes


def discover(X: pd.DataFrame, paired: pd.DataFrame, task: str, spec: InteractionSpec,
             recipes: dict) -> dict:
    """Discover on distinct rows, freeze, then summarize paired evaluation evidence.

    ``paired`` is the named pair's existing pair_frame, never another backend's evidence.
    ``recipes`` contains exactly that ordered pair's aligned recipe records.
    """
    _validate_features(X, spec)
    if set(recipes) != set(spec.pair):
        raise ValueError("interaction recipe records must match the named pair")
    for i, name in enumerate(spec.pair):
        if spec.recipe_ids is not None:
            actual = {r["recipe"].get("family_sha256") if r.get("recipe") else None
                      for r in recipes[name]}
            if actual != {spec.recipe_ids[i]}:
                raise ValueError(f"interaction recipe mismatch for {name}")
    rates = paired.groupby("row", sort=True)["disputed"].mean()
    rows = rates.index.to_numpy(dtype=int)
    groups = X.attrs.get("quibble_groups")
    split_metadata = X.attrs.get("quibble_split") or {}
    if groups is None and (split_metadata.get("group_column") or split_metadata.get("kind") == "group"):
        raise ValueError("declared interaction groups are unavailable; refusing a row-only split")
    if groups is not None and len(groups) != len(X):
        raise ValueError("interaction group metadata must align with source rows")
    discovery, evaluation, codes = partition_rows(rows, groups, spec)
    disagreement_definition = "predicted class A != predicted class B"
    if task == "regression":
        # Do not let evaluation outcomes set the discovery disagreement normalization.
        discovery_targets = paired.groupby("row")["y"].first().reindex(rows).to_numpy()[discovery]
        scale = float(np.std(discovery_targets)) if len(discovery_targets) else 0.0
        scale = scale or 1.0
        threshold = float(DEFAULTS["regression_gap_std"] * scale)
        disputed = abs(paired.pred_a - paired.pred_b) > threshold
        rates = disputed.groupby(paired["row"]).mean().reindex(rows)
        disagreement_definition = {
            "kind": "abs(pred_a-pred_b) > frozen_threshold",
            "threshold": threshold, "std_multiplier": DEFAULTS["regression_gap_std"],
            "discovery_target_std": scale, "constant_target_fallback": 1.0,
            "normalization": "std of one target per distinct discovery row only",
        }
    group_count = lambda mask: int(len(np.unique(codes[mask]))) if groups is not None else None
    supported = lambda mask: int(mask.sum()) >= spec.min_rows and (
        groups is None or group_count(mask) >= spec.min_groups)
    result = {
        "schema_version": 1, "spec": asdict(spec), "pair": list(spec.pair), "recipes": recipes,
        "warning": WARNING, "status": "insufficient_support", "rules": [],
        "selection": "squared-error shallow regression tree on mean per-source-row disagreement; "
                     "leaves ranked by discovery disagreement descending, ties by tree traversal",
        "disagreement_definition": disagreement_definition,
        "aggregation": "mean over repeats within each original row, then equal-weight mean over rows",
        "tree": {"max_depth": 2, "max_leaf_nodes": max(2, spec.max_rules),
                 "min_samples_leaf": spec.min_rows, "random_state": spec.seed},
        "split": {"unit": "declared_group" if groups is not None else "original_row",
                  "method": "sorted source rows; groups factorized in first-row order; "
                            "RandomState permutation of sorted unit codes; evaluation first ceil(fraction*n), "
                            "bounded to 1..n-1 units",
                  "seed": spec.seed, "evaluation_fraction": spec.evaluation_fraction,
                  "discovery_rows": int(discovery.sum()), "evaluation_rows": int(evaluation.sum()),
                  "discovery_groups": group_count(discovery), "evaluation_groups": group_count(evaluation)},
        "quality_metric": "accuracy_a_minus_b" if task == "classification" else "mae_b_minus_a",
        "numeric_semantics": "finite discovery values clipped to +/-1e30 before median fitting; "
                             "nonfinite imputed; transformed values clipped to +/-1e30 and float32; "
                             "float64 threshold comparisons",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "privacy": "No raw feature records, category vocabulary, row IDs, or group IDs. "
                   "Thresholds/medians are learned aggregates; hashing is not an anonymity guarantee.",
        "feature_schema": [],
    }
    if not supported(discovery) or not supported(evaluation):
        return result
    safe = X.loc[:, list(spec.features)]
    schema = fit_schema(safe.iloc[rows[discovery]])
    result["feature_schema"] = schema
    if not rates.to_numpy()[discovery].any():
        result["status"] = "no_disagreement_in_discovery"
        return result
    encoded_discovery = transform(safe.iloc[rows[discovery]], schema)
    tree = DecisionTreeRegressor(max_depth=2, max_leaf_nodes=max(2, spec.max_rules),
                                 min_samples_leaf=spec.min_rows, random_state=spec.seed)
    tree.fit(encoded_discovery, rates.to_numpy()[discovery])
    frozen = []

    def walk(node, predicates):
        column = int(tree.tree_.feature[node])
        if column < 0:
            if predicates:
                frozen.append(predicates)
            return
        field = schema[column]
        for child, operator in ((tree.tree_.children_left[node], "<="),
                                (tree.tree_.children_right[node], ">")):
            walk(child, predicates + [{
                "column": column, "feature": field["feature"], "encoding": field["kind"],
                "operator": operator, "threshold": float(tree.tree_.threshold[node]),
            }])

    walk(0, [])
    frozen.sort(key=lambda predicates: -float(
        rates.to_numpy()[discovery][rule_mask(encoded_discovery, predicates)].mean()))
    # Rules are now frozen. Evaluation features and quality are accessed only below.
    encoded = transform(safe.iloc[rows], schema)
    if task == "classification":
        quality = paired["a_correct"].astype(float) - paired["b_correct"].astype(float)
    else:
        quality = abs(paired.pred_b - paired.y) - abs(paired.pred_a - paired.y)
    quality_by_row = quality.groupby(paired["row"]).mean().reindex(rows).to_numpy()
    for predicates in frozen:
        mask = rule_mask(encoded, predicates)
        dmask, emask = mask & discovery, mask & evaluation
        if not supported(dmask):
            continue
        enough = supported(emask)
        text = " AND ".join(f"{p['feature']}[{p['encoding']}] {p['operator']} {p['threshold']:.8g}"
                            for p in predicates)
        rule = {
            "rule_id": f"rule_{len(result['rules']) + 1}", "predicates": predicates, "text": text,
            "backend_a": spec.pair[0], "backend_b": spec.pair[1],
            "recipe_family_a": recipe_families(recipes[spec.pair[0]]),
            "recipe_family_b": recipe_families(recipes[spec.pair[1]]),
            "seed": spec.seed, "min_rows": spec.min_rows, "min_groups": spec.min_groups,
            "split_unit": result["split"]["unit"],
            "label": "multi_feature" if len({p["feature"] for p in predicates}) > 1 else "single_feature",
            "discovery_rows": int(dmask.sum()), "discovery_groups": group_count(dmask),
            "evaluation_rows": int(emask.sum()), "evaluation_groups": group_count(emask),
            "evaluation_entries": int(paired.row.isin(rows[emask]).sum()),
            "evaluation_status": "supported" if enough else "insufficient_support",
            "disagreement_rate": float(rates.to_numpy()[emask].mean()) if enough else None,
            "quality_difference": float(quality_by_row[emask].mean()) if enough else None,
            "quality_metric": result["quality_metric"],
        }
        result["rules"].append(rule)
        if len(result["rules"]) == spec.max_rules:
            break
    result["status"] = "ok" if result["rules"] else "no_supported_rules"
    return result


def recipe_families(records: list[dict]) -> str:
    """Keep unknown legacy identities explicit in both report and summary exports."""
    return ", ".join(sorted({str((r.get("recipe") or {}).get("family_sha256") or "unknown (legacy)")
                             for r in records})) or "unknown (no records)"


def write_exports(result: dict, directory) -> None:
    """Aggregate-only exports, including explicit statuses for empty discoveries."""
    (directory / JSON_FILE).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(result["rules"]).reindex(columns=SUMMARY_COLUMNS).to_csv(directory / CSV_FILE, index=False)
