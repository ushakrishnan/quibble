"""A backend = a class with fit(...) and predict(...). See README: adding one takes ~20 lines."""
import shutil
import tempfile
import time

import numpy as np
import pandas as pd


class Refused(Exception):
    """Backend cannot handle this task/size. Recorded as a result, not a crash."""


def is_categorical(col):
    """object, string and category columns; written without select_dtypes so it behaves the same
    on pandas 2 and on pandas 3, where string columns get their own dtype."""
    return (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype))


def cat_cols(X):
    return [c for c in X.columns if is_categorical(X[c])]


class Prep:
    """Learned once on train; applied cheaply per call (keeps latency honest)."""

    def __init__(self, mode):
        self.mode, self.cats = mode, {}

    def fit(self, X):
        self.cats = {c: pd.Categorical(X[c]).categories for c in cat_cols(X)}
        return self

    def transform(self, X):
        X = X.copy()
        for c, cats in self.cats.items():
            # unseen values become missing before the categorical is built, which keeps the
            # behaviour pandas is deprecating (constructing with values outside the categories)
            col = X[c]
            s = pd.Categorical(col.where(col.isin(cats)), categories=cats)
            if self.mode == "native":
                X[c] = s
            else:
                codes = s.codes.astype(float)
                X[c] = np.where(codes >= 0, codes, np.nan)
        return X


class Backend:
    name = ""
    tasks = ("classification", "regression")
    gpu = False  # can this backend use the GPU?
    cuda_exclusion = ""
    native_ok = False  # supports native categoricals (QUIBBLE_PREPROCESS=native)?
    info = {}
    can_save = False
    baseline = False  # a deliberately simple reference model
    load_s = 0.0  # seconds spent building the model (pretrained weights); fit_s excludes it
    seed_controlled = True  # does the model seed make training reproducible?
    verification = "tested"  # what has actually been run: shown by `quibble-ml doctor`

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        raise NotImplementedError

    def predict(self, X):  # proba (n, k) for classification, (n,) for regression
        raise NotImplementedError

    def close(self):
        pass

    def save(self, directory):
        raise NotImplementedError(f"Fitted-model retention is not supported for {self.name}")

    def diagnostics(self, path):
        return False

    def _check(self, task):
        if task not in self.tasks:
            raise Refused(f"{self.name} does not support {task}")


class _SK(Backend):
    def save(self, directory):
        import joblib
        directory.mkdir(parents=True, exist_ok=False)
        joblib.dump(self, directory / "model.joblib")

    def make(self, task, seed, cfg, device, cat_idx):
        raise NotImplementedError

    def transform(self, X):
        return self.prep.transform(X)

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        self._check(task)
        self.task = task
        mode = "native" if (cfg.preprocess == "native" and self.native_ok) else "ordinal"
        self.prep = Prep(mode).fit(Xtr)
        Xa = self.transform(Xtr)
        cat_idx = [Xa.columns.get_loc(c) for c in self.prep.cats] if mode == "native" else []
        t = time.perf_counter()
        self.model = self.make(task, seed, cfg, device, cat_idx)  # pretrained weights load here
        if getattr(self, "tuning_params", None) is not None:
            if self.name not in ("histgb", "lightgbm", "catboost"):
                raise Refused(f"Parameter search is not supported for {self.name}")
            self.model.set_params(**self.tuning_params)
        load_s = time.perf_counter() - t
        self.model.fit(Xa, ytr)
        self.load_s = load_s
        self.info = {"preprocess_effective": mode,
                     "device_effective": device if self.gpu else "cpu"}

    def predict(self, X):
        Xb = self.transform(X)
        return self.model.predict_proba(Xb) if self.task == "classification" \
            else self.model.predict(Xb)


class HistGB(_SK):
    name, native_ok = "histgb", True
    cuda_exclusion = "scikit-learn histogram gradient boosting has no CUDA implementation."
    can_save = True

    def make(self, task, seed, cfg, device, cat_idx):
        from sklearn.ensemble import (HistGradientBoostingClassifier as Cl,
                                      HistGradientBoostingRegressor as Rg)
        kw = {"random_state": seed}
        if cat_idx:
            kw["categorical_features"] = "from_dtype"
        return (Cl if task == "classification" else Rg)(**kw)


class Weak(HistGB):
    """Deliberately handicapped sanity candidate: a swap report must flag it. Hidden from 'all'."""
    name, internal = "weak", True

    def make(self, task, seed, cfg, device, cat_idx):
        m = super().make(task, seed, cfg, device, cat_idx)
        m.set_params(max_iter=2, learning_rate=0.05)
        return m


class LightGBM(_SK):
    name, native_ok = "lightgbm", True
    cuda_exclusion = ("This adapter uses LightGBM's CPU backend. GPU support requires a compatible "
                      "build and adapter; LightGBM's CUDA backend is not supported on Windows.")
    can_save = True

    def make(self, task, seed, cfg, device, cat_idx):
        from lightgbm import LGBMClassifier as Cl, LGBMRegressor as Rg
        return (Cl if task == "classification" else Rg)(
            random_state=seed, verbose=-1, n_jobs=cfg.threads or -1)


class CatBoost(_SK):
    name, native_ok, gpu = "catboost", True, True
    can_save = True

    def make(self, task, seed, cfg, device, cat_idx):
        from catboost import CatBoostClassifier as Cl, CatBoostRegressor as Rg
        kw = dict(random_seed=seed, verbose=0, allow_writing_files=False, thread_count=cfg.threads or -1,
                  task_type="GPU" if device == "cuda" else "CPU")
        if cat_idx:
            kw["cat_features"] = cat_idx
        return (Cl if task == "classification" else Rg)(**kw)

    def transform(self, X):
        X = super().transform(X)
        if self.prep.mode == "native":  # CatBoost wants strings, no NaN in cat columns
            for c in self.prep.cats:
                X[c] = X[c].astype(object).where(X[c].notna(), "__nan__").astype(str)
        return X


class TabPFN(_SK):
    name, gpu = "tabpfn", True
    verification = "experimental: offline Wine/diabetes CPU and CUDA holdouts verified with TabPFN9.2.0/v3.5; broader coverage unverified"

    def make(self, task, seed, cfg, device, cat_idx):
        from tabpfn import TabPFNClassifier as Cl, TabPFNRegressor as Rg
        return (Cl if task == "classification" else Rg)(device=device, random_state=seed)

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        limit = cfg.tabpfn_max_rows or (10_000 if device == "cuda" else 1_000)  # verify vs docs
        if len(Xtr) > limit:
            raise Refused(f"train rows {len(Xtr)} > {limit} (device={device})")
        if Xtr.shape[1] > 500:
            raise Refused(f"{Xtr.shape[1]} features > 500")
        if task == "classification" and ytr.nunique() > 10:
            raise Refused(f"{ytr.nunique()} classes > 10")
        super().fit(Xtr, ytr, task, seed, cfg, device)


class TabICLv2(_SK):
    name, gpu = "tabicl_v2", True
    verification = "experimental: CPU/CUDA Wine and diabetes holdouts verified; broader coverage unverified"

    def make(self, task, seed, cfg, device, cat_idx):
        from tabicl import TabICLClassifier, TabICLRegressor
        from .env_info import _ver
        from .tabicl import ARTIFACTS, PARAMETERS, VERSION, checkpoint
        if _ver("tabicl") != VERSION:
            raise Refused(f"TabICLv2 requires tabicl=={VERSION}; use the locked tabicl extra.")
        path = checkpoint(task, cfg.tabicl_allow_download)
        estimator = TabICLClassifier if task == "classification" else TabICLRegressor

        class TimedEstimator(estimator):
            def _load_model(self):
                started = time.perf_counter()
                try:
                    return super()._load_model()
                finally:
                    self.quibble_weight_load_s = time.perf_counter() - started

        return TimedEstimator(**PARAMETERS, model_path=path, allow_auto_download=False,
                              checkpoint_version=ARTIFACTS[task]["filename"],
                              device=device, random_state=seed, n_jobs=cfg.threads or None,
                              verbose=False)

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        self._check(task)
        if device not in ("cpu", "cuda"):
            raise Refused("TabICLv2 supports only Quibble's explicit cpu or cuda device policy.")
        limit = 10_000 if device == "cuda" else 1_000
        if len(Xtr) > limit or Xtr.shape[1] > 500:
            raise Refused(f"TabICLv2 policy limit: {limit} train rows and 500 features ({device}).")
        super().fit(Xtr, ytr, task, seed, cfg, device)
        from .tabicl import ARTIFACTS, PARAMETERS, REPOSITORY, REVISION
        self.load_s += self.model.quibble_weight_load_s
        self.classes = sorted(ytr.unique().tolist()) if task == "classification" else None
        self.info.update(tabicl_parameters=dict(PARAMETERS), tabicl_n_jobs=cfg.threads or None,
                         checkpoint={"repository": REPOSITORY, "revision": REVISION,
                                     **ARTIFACTS[task], "sha256_verified": True},
                         download_allowed=cfg.tabicl_allow_download, fine_tuning=False,
                         regression_prediction="mean", weight_load_timing="pinned _load_model hook")

    def predict(self, X):
        query = self.transform(X)
        if self.task == "regression":
            result = np.asarray(self.model.predict(query, output_type="mean"), dtype=float)
            if result.shape != (len(X),) or not np.isfinite(result).all():
                raise ValueError("TabICLv2 returned invalid regression predictions.")
            return result
        columns = list(self.model.classes_)
        if len(columns) != len(self.classes) or set(columns) != set(self.classes):
            raise ValueError("TabICLv2 probability classes do not match training classes.")
        result = np.asarray(self.model.predict_proba(query), dtype=float)
        if result.shape != (len(X), len(columns)) or not np.isfinite(result).all() \
                or (result < 0).any() or (result > 1).any() \
                or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
            raise ValueError("TabICLv2 returned invalid class probabilities.")
        return result[:, [columns.index(c) for c in self.classes]]

    def close(self):
        if hasattr(self, "model"):
            del self.model


class MitraV2(_SK):
    name, gpu, seed_controlled = "mitra_v2", True, False
    verification = "experimental: CUDA Wine/diabetes holdouts verified; broader coverage unverified"

    def make(self, task, seed, cfg, device, cat_idx):
        from .mitra import CLASSIFIER_PARAMETERS, checkpoint
        path = checkpoint(task, cfg.mitra_allow_download)
        if task == "classification":
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
            return MitraClassifier(**CLASSIFIER_PARAMETERS, hf_model=str(path),
                                   device=device, seed=seed)
        from mitra_finetune import MitraFinetune
        return MitraFinetune(checkpoint_dir=path, problem_type="regression", device=device,
                             random_state=seed, num_bag_folds=2, in_process=True,
                             eval_metric="root_mean_squared_error", time_limit=cfg.timeout_s)

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        from .mitra import (ARTIFACTS, CLASSIFIER_PARAMETERS, REGRESSION_PARAMETERS,
                            SOURCE_REVISION, cpu_budget, verify_runtime)
        self._check(task)
        if device != "cuda":
            raise Refused("Mitra-v2 currently supports only explicit CUDA; CPU/MPS are unverified.")
        if len(Xtr) > 512 or Xtr.shape[1] > 32:
            raise Refused("Mitra-v2 conservative policy limit: 512 training rows and 32 predictors.")
        self.classes = sorted(ytr.unique().tolist()) if task == "classification" else None
        if self.classes is not None and not 2 <= len(self.classes) <= 10:
            raise Refused("Mitra-v2 classification requires two to ten training classes.")
        verify_runtime()
        self.threads, self.prediction_adapts = cfg.threads, task == "regression"
        target = ytr.map({label: i for i, label in enumerate(self.classes)}) \
            if self.classes is not None else ytr
        started = time.perf_counter()
        with cpu_budget(cfg.threads):
            super().fit(Xtr, target, task, seed, cfg, device)
        self.load_s = time.perf_counter() - started
        artifact = ARTIFACTS[task]
        self.info.update(
            checkpoint={"repository": artifact["repository"], "revision": artifact["revision"],
                        "filename": "model.safetensors", "sha256": artifact["files"]["model.safetensors"],
                        "config_sha256": artifact["files"]["config.json"], "sha256_verified": True},
            mitra_source_revision=SOURCE_REVISION, fine_tuning=self.prediction_adapts,
            prediction_adapts=self.prediction_adapts, seed_controlled=False,
            mitra_parameters=dict(REGRESSION_PARAMETERS if self.prediction_adapts
                                  else CLASSIFIER_PARAMETERS),
            adaptation_time_limit_s=cfg.timeout_s if self.prediction_adapts else None,
            adaptation_steps_completed=None if self.prediction_adapts else 0,
            load_s_scope="checked snapshot and context registration" if self.prediction_adapts
                         else "zero-shot checkpoint loading and context preparation",
            weight_load_s=None, adaptation_s=None if self.prediction_adapts else 0.0,
            prediction_only_s=None, single_row_latency_available=not self.prediction_adapts,
            weight_loading_stage="fused prediction-time loading/adaptation/inference"
                                 if self.prediction_adapts else "fit",
            download_allowed=cfg.mitra_allow_download,
            regression_prediction="distributional-head mean" if self.prediction_adapts else None)

    def predict(self, X):
        from contextlib import nullcontext
        from .mitra import adaptation_workspace, cpu_budget
        query = self.transform(X)
        with cpu_budget(self.threads), (
                adaptation_workspace() if self.prediction_adapts else nullcontext()):
            result = np.asarray(self.model.predict(query) if self.prediction_adapts
                                else self.model.predict_proba(query), dtype=float)
        expected = (len(X),) if self.prediction_adapts else (len(X), len(self.classes))
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("Mitra-v2 returned invalid or nonfinite predictions.")
        if not self.prediction_adapts and ((result < 0).any() or (result > 1).any()
                                          or not np.allclose(result.sum(1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Mitra-v2 returned invalid class probabilities.")
        return result

    def close(self):
        if hasattr(self, "model"):
            del self.model


class TabFM(_SK):
    name, native_ok, gpu = "tabfm", True, True
    verification = "smoke-tested on CPU; limited full-data evidence"

    def make(self, task, seed, cfg, device, cat_idx):
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
        model = tabfm_v1_0_0_pytorch.load(model_type=task, device=device, use_cache=False)
        estimator = TabFMClassifier if task == "classification" else TabFMRegressor
        return estimator(model=model, random_state=seed, n_estimators=cfg.tabfm_n_estimators)

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        super().fit(Xtr, ytr, task, seed, cfg, device)
        self.info["tabfm_n_estimators"] = cfg.tabfm_n_estimators


class KumoTabular(Backend):
    name, native_ok, gpu = "kumo_tabular", True, True
    verification = "smoke-tested on CPU; limited full-data evidence"

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        import sdm
        import torch
        self._check(task)
        self.task, self.device = task, device
        self.prep = Prep(cfg.preprocess).fit(Xtr)
        Xa = self.prep.transform(Xtr)
        self.stypes = sdm.infer_stypes(Xa)
        x = sdm.TableTensor.from_pandas(df=Xa, stypes=self.stypes, device=device)
        target = ytr.rename("__y__").to_frame()
        y_stypes = sdm.infer_stypes(target, overrides={
            "__y__": "categorical" if task == "classification" else "numerical"})
        y = sdm.TableTensor.from_pandas(df=target, stypes=y_stypes, device=device)
        self.classes = [str(c) for c in np.unique(ytr)] if task == "classification" else None
        t = time.perf_counter()
        self.model = sdm.models.KumoTabular(
            task=task, size=cfg.kumo_size, pretrained=True, device=device)
        load_s = time.perf_counter() - t
        generator = torch.Generator(device=device).manual_seed(seed)
        self.model.fit(x=x, y=y, num_estimators=cfg.kumo_n_estimators, generator=generator)
        self.load_s = load_s
        self.info = {"preprocess_effective": cfg.preprocess, "device_effective": device,
                     "kumo_size": cfg.kumo_size, "kumo_n_estimators": cfg.kumo_n_estimators}

    def predict(self, X):
        import sdm
        x = sdm.TableTensor.from_pandas(
            df=self.prep.transform(X), stypes=self.stypes, device=self.device)
        pred = self.model.predict(x).to_pandas()
        if len(pred) != len(X):
            raise ValueError("KumoTabular prediction row count does not match input.")
        if self.task == "classification":
            pred = pred.rename(columns=str)
            if len(pred.columns) != len(self.classes) or set(pred.columns) != set(self.classes):
                raise ValueError("Expected KumoTabular probability columns for each training class.")
            P = pred[self.classes].to_numpy(dtype=float)
            if (not np.isfinite(P).all() or (P < 0).any() or (P > 1).any()
                    or not np.allclose(P.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("KumoTabular returned invalid class probabilities.")
            return P
        columns = [f"q{i:03d}" for i in range(1, 1000)]
        if len(pred.columns) != len(columns) or set(pred.columns) != set(columns):
            raise ValueError("Expected KumoTabular regression quantile columns q001 through q999.")
        quantiles = pred[columns].to_numpy(dtype=float)
        if not np.isfinite(quantiles).all():
            raise ValueError("KumoTabular returned nonfinite regression quantiles.")
        return quantiles.mean(axis=1)

    def close(self):
        if hasattr(self, "model"):
            self.model.clear()


class AutoGluon(Backend):
    name, gpu = "autogluon", True
    seed_controlled = False  # the model seed is accepted but AutoGluon runs are not seeded here
    verification = "tested on small CPU datasets; GPU runs excluded"
    cuda_exclusion = ("AutoGluon's GPU budget does not guarantee GPU-only model execution; "
                      "its ensemble can include CPU components.")
    can_save = True

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        from autogluon.tabular import TabularPredictor
        self._check(task)
        self.task, self.path = task, tempfile.mkdtemp(prefix="quibble_ag_")
        df = Xtr.copy()
        df["__y__"] = ytr.values
        self.p = TabularPredictor(label="__y__", verbosity=0, path=self.path,
                                  problem_type="regression" if task == "regression" else None)
        self.p.fit(df, presets=cfg.ag_preset, time_limit=cfg.ag_time_limit,
                   num_gpus=1 if device == "cuda" else 0,
                   num_cpus=cfg.threads if cfg.threads else "auto")
        self.info = {"preprocess_effective": "native", "device_effective": device,
                     "seed_controlled": False,
                     "ag_preset": cfg.ag_preset, "ag_time_limit": cfg.ag_time_limit}

    def predict(self, X):
        if self.task != "classification":
            return self.p.predict(X).to_numpy()
        proba = self.p.predict_proba(X)
        return proba[sorted(proba.columns)].to_numpy()  # columns in label order, as every adapter returns

    def close(self):
        shutil.rmtree(getattr(self, "path", ""), ignore_errors=True)

    def save(self, directory):
        shutil.copytree(self.path, directory)

    def diagnostics(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p.leaderboard(silent=True).to_csv(path, index=False)
        return True


class Dummy(Backend):
    """Reference: class priors (classification) or the training mean (regression)."""
    name, baseline = "dummy", True

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        from sklearn.dummy import DummyClassifier, DummyRegressor
        self._check(task)
        self.task = task
        self.model = (DummyClassifier(strategy="prior") if task == "classification"
                      else DummyRegressor(strategy="mean")).fit(Xtr, ytr)
        self.info = {"preprocess_effective": "none", "device_effective": "cpu"}

    def predict(self, X):
        return self.model.predict_proba(X) if self.task == "classification" else self.model.predict(X)


class _Linear(Backend):
    """Reference linear model: median-imputed, scaled numerics plus one-hot categoricals."""
    baseline = True

    def make(self, task, seed):
        raise NotImplementedError

    def fit(self, Xtr, ytr, task, seed, cfg, device):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        self._check(task)
        self.task = task
        cats = cat_cols(Xtr)
        nums = [c for c in Xtr.columns if c not in cats]
        parts = []
        if nums:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums))
        if cats:
            parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                               OneHotEncoder(handle_unknown="ignore")), cats))
        self.model = Pipeline([("prep", ColumnTransformer(parts)), ("model", self.make(task, seed))])
        self.model.fit(Xtr, ytr)
        self.info = {"preprocess_effective": "onehot", "device_effective": "cpu"}

    def predict(self, X):
        return self.model.predict_proba(X) if self.task == "classification" else self.model.predict(X)


class Linear(_Linear):
    """Logistic regression (classification) or ridge regression (regression)."""
    name = "linear"

    def make(self, task, seed):
        from sklearn.linear_model import LogisticRegression, Ridge
        return LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)


BACKENDS = {b.name: b for b in
            (Dummy, Linear, HistGB, Weak, LightGBM, CatBoost, TabPFN, TabICLv2, MitraV2,
             TabFM, KumoTabular, AutoGluon)}


def device_exclusion(name, device):
    if device != "cuda":
        return ""
    backend = BACKENDS[name]
    return backend.cuda_exclusion or ("" if backend.gpu else "This backend has no CUDA support.")


def select_backends(names, device):
    excluded = {name: reason for name in names if (reason := device_exclusion(name, device))}
    return [name for name in names if name not in excluded], excluded
