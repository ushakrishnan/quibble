"""Backend terms and versioned recipe identity; no downloads or license approval."""
import hashlib
import json
from copy import deepcopy

from .env_info import _ver

REVIEWED = "2026-10-30"
LIBRARIES = {
    "sklearn": ("scikit-learn", "BSD-3-Clause",
                "https://github.com/scikit-learn/scikit-learn/blob/main/COPYING"),
    "lightgbm": ("lightgbm", "MIT", "https://github.com/lightgbm-org/LightGBM/blob/master/LICENSE"),
    "catboost": ("catboost", "Apache-2.0", "https://github.com/catboost/catboost/blob/master/LICENSE"),
    "autogluon": ("autogluon.tabular", "Apache-2.0",
                  "https://github.com/autogluon/autogluon/blob/master/LICENSE"),
    "tabpfn": ("tabpfn", "Apache-2.0", "https://github.com/PriorLabs/TabPFN/blob/main/LICENSE"),
    "tabicl_v2": ("tabicl", "BSD-3-Clause (adapter components); Apache-2.0 (unused forecast)",
                  "https://github.com/soda-inria/tabicl/blob/main/LICENSE"),
    "mitra_v2": ("autogluon.tabular", "Apache-2.0",
                 "https://github.com/autogluon/autogluon/blob/1.6.3/LICENSE"),
    "tabfm": ("tabfm", "Apache-2.0", "https://github.com/google-research/tabfm/blob/main/LICENSE"),
    "kumo_tabular": ("structured-data-models", "Apache-2.0",
                     "https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE"),
}
BACKEND_LIBRARIES = {name: "sklearn" for name in ("histgb", "weak", "dummy", "linear")}
BACKEND_LIBRARIES.update({name: name for name in LIBRARIES if name != "sklearn"})
WEIGHTS = {
    "tabfm": ("google/tabfm-1.0.0-pytorch", "TabFM Non-Commercial v1.0",
              "Non-commercial/non-production terms; review the full license before use."),
    "tabpfn": ("Prior-Labs/tabpfn_3_5", "TABPFN-3.5 License v1.0",
               "Non-commercial/non-production terms and professional-use eligibility; review the full license."),
    "tabicl_v2": ("jingang/TabICL", "BSD-3-Clause",
                  "Retain required notices; the adapter does not use Apache-licensed forecast components."),
    "mitra_v2": ("autogluon/mitra-classifier-2", "Apache-2.0",
                 "Both pinned task model cards label Apache-2.0; retain applicable notices."),
    "kumo_tabular": ("nvidia/Kumo-Tabular", "OpenMDW-1.1",
                     "Review model terms and any additional applicable conditions."),
}


def backend_metadata(name):
    from .backends import BACKENDS
    library = LIBRARIES.get(BACKEND_LIBRARIES.get(name))
    package, license_name, source = library or (None, "unknown", None)
    weights = {"license": "not applicable", "terms_url": None, "checkpoint_id": None,
               "revision": None, "sha256": None, "identity_status": "not applicable",
               "restrictions": "No pretrained weights in this adapter."}
    if name in WEIGHTS:
        checkpoint, terms, restrictions = WEIGHTS[name]
        weights.update(license=terms, checkpoint_id=checkpoint,
                       terms_url=f"https://huggingface.co/{checkpoint}/blob/main/LICENSE",
                       identity_status="unknown: library-selected revision/checksum not captured",
                       restrictions=restrictions)
        weights["terms_scope"] = "Reference checkpoint only; actual loaded revision/checksum unknown."
        if name == "tabicl_v2":
            from .tabicl import ARTIFACTS, REVISION
            weights.update(revision=REVISION, artifacts=deepcopy(ARTIFACTS),
                           terms_url=f"https://huggingface.co/jingang/TabICL/blob/{REVISION}/README.md",
                           identity_status="pinned expected artifacts; actual verification recorded after loading",
                           terms_scope="Pinned model card and expected artifact hashes; not proof of a completed fit.")
        elif name == "mitra_v2":
            from .mitra import ARTIFACTS, SOURCE_REVISION, SOURCE_URL
            cards = {task: f"https://huggingface.co/{item['repository']}/blob/{item['revision']}/README.md"
                     for task, item in ARTIFACTS.items()}
            weights.update(checkpoint_id="Mitra-v2 task-specific pinned snapshots",
                           artifacts=deepcopy(ARTIFACTS), terms_url=cards["classification"],
                           task_terms_urls=cards, source_revision=SOURCE_REVISION, source_url=SOURCE_URL,
                           identity_status="pinned expected artifacts; actual verification recorded after loading",
                           terms_scope="Pinned task model cards; hashes are verified before local loading, "
                                       "not a legal-use guarantee or a completed-fit claim.")
    elif name == "autogluon" or library is None:
        weights.update(license="unknown", identity_status="unknown",
                       restrictions="Depends on selected model families/checkpoints; review separately.")
    return {"schema": 1, "backend": name,
            "library": {"package": package, "version": _ver(package) if package else None,
                        "license": license_name, "source_url": source},
            "weights": weights, "reviewed": REVIEWED if library else None,
            "access": "Library installation does not approve downloads or grant usage rights; "
                      "upstream access requirements may change.",
            "verification": BACKENDS[name].verification if library else "unverified custom backend",
            "verification_evidence": "docs/index.md#trust-levels-what-has-actually-been-run",
            "scope": "Reference terms, not legal advice; dataset and transitive-component terms are separate."}


def package_names(name):
    library = LIBRARIES.get(BACKEND_LIBRARIES.get(name))
    names = (library[0],) if library else ()
    if name == "mitra_v2":
        from .mitra import VERSIONS
        return tuple(VERSIONS) + ("loguru", "einx", "omegaconf", "einops")
    return names + (("torch",) if name in WEIGHTS else ())


def license_label(snapshot):
    if not snapshot:
        return "unknown (legacy: no runtime license snapshot)"
    return (f"code: {snapshot['library']['license']}; "
            f"weights: {snapshot['weights']['license']}")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recipe_hashes(requested):
    family = {k: v for k, v in requested.items() if k != "cell"}
    algorithm = {k: family[k] for k in ("schema", "backend", "adapter_revision", "tier",
                                      "parameters", "checkpoint", "search", "packages",
                                      "adaptation", "other_parameters")}
    return {"sha256": digest(requested), "family_sha256": digest(family),
            "algorithm_sha256": digest(algorithm)}


def recipe_snapshot(name, cfg, ms):
    from .backends import BACKENDS
    params = {}
    if name in ("histgb", "weak", "lightgbm", "tabpfn"):
        params["random_state"] = "bound to cell.model_seed"
    if name == "weak":
        params.update(max_iter=2, learning_rate=0.05)
    elif name == "lightgbm":
        params.update(verbose=-1, n_jobs=cfg.threads or -1)
    elif name == "catboost":
        params.update(random_seed="bound to cell.model_seed", verbose=0, allow_writing_files=False,
                      thread_count=cfg.threads or -1, task_type="GPU" if cfg.device == "cuda" else "CPU")
    elif name == "tabfm":
        params.update(n_estimators=cfg.tabfm_n_estimators, model="tabfm_v1_0_0_pytorch", use_cache=False)
    elif name == "tabpfn":
        params["max_train_rows"] = cfg.tabpfn_max_rows or (10_000 if cfg.device == "cuda" else 1_000)
    elif name == "tabicl_v2":
        from .tabicl import PARAMETERS, VERSION
        params.update(PARAMETERS, package_version=VERSION, random_state="bound to cell.model_seed",
                      n_jobs=cfg.threads or None, fine_tuning=False, regression_prediction="mean",
                      max_train_rows=10_000 if cfg.device == "cuda" else 1_000, max_features=500,
                      allow_auto_download=False)
    elif name == "mitra_v2":
        from .mitra import CLASSIFIER_PARAMETERS, REGRESSION_PARAMETERS, SOURCE_REVISION
        params.update(classification=dict(CLASSIFIER_PARAMETERS),
                      regression=dict(REGRESSION_PARAMETERS), source_revision=SOURCE_REVISION,
                      adaptation_time_limit_s=cfg.timeout_s, max_train_rows=512, max_features=32,
                      max_classes=10, device="explicit CUDA only", threads=cfg.threads,
                      seed="requested; full control unverified", allow_auto_download=False)
    elif name == "kumo_tabular":
        params.update(size=cfg.kumo_size, num_estimators=cfg.kumo_n_estimators, pretrained=True)
    elif name == "autogluon":
        params.update(presets=cfg.ag_preset, time_limit=cfg.ag_time_limit,
                      num_cpus=cfg.threads or "auto", num_gpus=1 if cfg.device == "cuda" else 0)
    elif name == "dummy":
        params.update(classification="prior", regression="mean")
    elif name == "linear":
        params.update(classification="logistic regression, max_iter=2000",
                      regression="ridge, alpha=1.0", preparation="median/scale/one-hot")
    weights = backend_metadata(name)["weights"]
    packages = ("numpy", "pandas", "scipy", "scikit-learn") + package_names(name)
    requested = {"schema": 1, "backend": name, "adapter_revision": 1, "tier": "defaults",
                 "parameters": params, "other_parameters": "defaults of the recorded library version",
                 "checkpoint": {key: weights[key] for key in ("checkpoint_id", "revision", "sha256",
                                                             "identity_status")},
                 "search": {"status": "not requested"}, "adaptation": "adapter/library defaults",
                 "packages": {p: _ver(p) for p in dict.fromkeys(packages)},
                 "preprocess": "native" if name == "autogluon" else cfg.preprocess,
                 "device": cfg.device, "cv": cfg.cv, "cv_seed": cfg.cv_seed,
                 "run_type": cfg.run_type, "seed_controlled": BACKENDS[name].seed_controlled,
                 "cell": {"split_seed": cfg.split_seed, "model_seed": ms}}
    if name in ("tabicl_v2", "mitra_v2"):
        requested["checkpoint"]["artifacts"] = deepcopy(weights["artifacts"])
    if name == "mitra_v2":
        from .mitra import REGRESSION_PARAMETERS
        requested["adaptation"] = {
            "classification": "zero-shot; no parameter fine-tuning",
            "regression": {**REGRESSION_PARAMETERS, "time_limit_s": cfg.timeout_s,
                           "stage": "prediction-time fitting; isolated pure-inference latency unavailable"},
        }
    if cfg.tier == "tuned":
        from .tuning import TREES, protocol
        requested["campaign_tier"] = "tuned"
        if name in TREES:
            requested.update(tier="tuned", search={"status": "requested", **protocol(cfg, name)},
                             adaptation="training-only inner selection, then outer-training refit")
        else:
            # Keep the actual foundation/default recipe label; opting into tree tuning
            # never implies that another model was tuned or fine-tuned.
            requested["search"] = {
                "status": "not supported", "reason": "Tree search only; adapter recipe unchanged",
                "campaign_protocol": protocol(cfg),
            }
    return {"requested": requested, **recipe_hashes(requested),
            "resolved": {"status": "not fitted"},
            "identity_complete": False}


def resolved_recipe(recipe, backend, name, context):
    result = deepcopy(recipe)
    parameters = None
    if name in ("histgb", "weak", "lightgbm", "catboost", "dummy"):
        parameters = (backend.model.get_all_params() if name == "catboost"
                      else backend.model.get_params(deep=False))
        # Reject unrecordable objects rather than writing unstable repr/object addresses.
        json.dumps(parameters, allow_nan=False)
    elif name == "linear":
        parameters = backend.model.named_steps["model"].get_params(deep=False)
    result["resolved"] = {"status": "fitted", "parameters": parameters,
                          "parameter_status": "recorded" if parameters is not None else
                                              "not exposed by adapter; requested recipe only",
                          "backend_info": dict(backend.info), "context": context}
    result["resolved_sha256"] = digest(result["resolved"])
    result["full_sha256"] = digest({"requested": result["requested"], "resolved": result["resolved"]})
    result["identity_complete"] = (parameters is not None and
                                  recipe["requested"]["checkpoint"]["identity_status"] == "not applicable")
    return result


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or not isinstance(recipe.get("requested"), dict):
        raise ValueError("Invalid recipe metadata; use a separate output directory.")
    if recipe["requested"].get("schema") != 1:
        raise ValueError("Unsupported recipe schema; use a separate output directory.")
    try:
        hashes = recipe_hashes(recipe["requested"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid recipe fields; use a separate output directory.") from exc
    if any(recipe.get(key) != value for key, value in hashes.items()):
        raise ValueError("Recipe digest mismatch; restore original evidence or use a separate output directory.")
    if recipe.get("resolved", {}).get("status") == "fitted" and \
            recipe.get("resolved_sha256") != digest(recipe["resolved"]):
        raise ValueError("Resolved recipe digest mismatch; restore original evidence.")
    if recipe.get("resolved", {}).get("status") == "fitted" and recipe.get("full_sha256") != \
            digest({"requested": recipe["requested"], "resolved": recipe["resolved"]}):
        raise ValueError("Full recipe digest mismatch; restore original evidence.")
    return recipe


def check_reuse(saved, requested, writing=False):
    if saved.get("recipe") is not None and (
            saved["recipe"].get("requested", {}).get("tier", "defaults") !=
            requested["requested"]["tier"]):
        raise ValueError("Recipe mismatch (tier): use a separate --out root.")
    if "error" in saved:
        return
    if saved.get("recipe") is not None:
        existing = validate_recipe(saved["recipe"])
        if existing["algorithm_sha256"] != requested["algorithm_sha256"]:
            raise ValueError("Recipe mismatch (tier/parameters/checkpoint/search/dependencies). "
                             "Use a separate --out; --force does not mix recipes.")
        if writing and existing["family_sha256"] != requested["family_sha256"]:
            raise ValueError("Recipe family mismatch (preprocessing/device/CV/protocol). "
                             "Use a separate --out instead of overwriting or mixing recipes.")
    elif writing:
        raise ValueError("Legacy recipe metadata is incomplete. Keep the old evidence and "
                         "use a separate --out for new runs.")


def validate_pool_recipes(runs):
    usable = [r for r in runs if "error" not in r and "refused" not in r]
    recipes = [r.get("recipe") for r in usable]
    if not any(r is not None for r in recipes):
        return "legacy: incomplete recipe metadata"
    if any(r is None for r in recipes):
        raise ValueError("Mixed legacy and versioned recipe evidence; use a separate output directory.")
    for run, recipe in zip(usable, recipes):
        validate_recipe(recipe)
        if recipe["requested"]["backend"] != run["backend"] or \
                recipe["requested"]["cell"] != {
                    "split_seed": run["split_seed"], "model_seed": run["model_seed"]}:
            raise ValueError("Recipe cell identity does not match the saved run.")
    families = {validate_recipe(r)["family_sha256"] for r in recipes}
    if len(families) > 1:
        raise ValueError("Mixed recipe families across splits; use a separate output directory.")
    return "recorded"
