"""A backend = a class with fit(...) and predict(...). See README: adding one takes ~20 lines."""
import shutil
import tempfile

import numpy as np
import pandas as pd


class Refused(Exception):
    """Backend cannot handle this task/size. Recorded as a result, not a crash."""


def cat_cols(X):
    return list(X.select_dtypes(include=["object", "category"]).columns)


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
            s = pd.Categorical(X[c], categories=cats)
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
        self.model = self.make(task, seed, cfg, device, cat_idx)
        self.model.fit(Xa, ytr)
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


class TabFM(_SK):
    name, native_ok, gpu = "tabfm", True, True

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
        self.model = sdm.models.KumoTabular(
            task=task, size=cfg.kumo_size, pretrained=True, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        self.model.fit(x=x, y=y, num_estimators=cfg.kumo_n_estimators, generator=generator)
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
                     "ag_preset": cfg.ag_preset, "ag_time_limit": cfg.ag_time_limit}

    def predict(self, X):
        return (self.p.predict_proba(X) if self.task == "classification"
                else self.p.predict(X)).to_numpy()

    def close(self):
        shutil.rmtree(getattr(self, "path", ""), ignore_errors=True)

    def save(self, directory):
        shutil.copytree(self.path, directory)

    def diagnostics(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p.leaderboard(silent=True).to_csv(path, index=False)
        return True


BACKENDS = {b.name: b for b in
            (HistGB, Weak, LightGBM, CatBoost, TabPFN, TabFM, KumoTabular, AutoGluon)}


def device_exclusion(name, device):
    if device != "cuda":
        return ""
    backend = BACKENDS[name]
    return backend.cuda_exclusion or ("" if backend.gpu else "This backend has no CUDA support.")


def select_backends(names, device):
    excluded = {name: reason for name in names if (reason := device_exclusion(name, device))}
    return [name for name in names if name not in excluded], excluded
