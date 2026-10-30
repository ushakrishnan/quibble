"""Dataset registry. OpenML ids are from memory: run `python -m quibble.datasets --verify`."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

C, R = "classification", "regression"


class UnsupportedSplit(ValueError):
    """The evaluation split cannot support the shared classifier probability contract."""


# name: (openml id, task, expected OpenML name)
OPENML = {
    "credit-g": (31, C, "credit-g"), "diabetes": (37, C, "diabetes"),
    "adult": (1590, C, "adult"), "bank-marketing": (1461, C, "bank-marketing"),
    "phoneme": (1489, C, "phoneme"), "blood-transfusion": (1464, C, "blood-transfusion-service-center"),
    "kc1": (1067, C, "kc1"), "vehicle": (54, C, "vehicle"),
    "abalone": (183, R, "abalone"), "cpu_act": (197, R, "cpu_act"),
    "wine_quality": (287, R, "wine_quality"), "house_sales": (42092, R, "house_sales"),
    "diamonds": (42225, R, "diamonds"), "bike_sharing": (42712, R, "Bike_Sharing_Demand"),
}
BUILTIN = {"breast_cancer": C, "wine": C, "diabetes_reg": R}  # offline smoke tests
RESERVED = {"report", "runs", "smoke", "artifacts"}


def data_root():
    from .cache import project_path
    from .config import load_dotenv
    load_dotenv()
    return project_path(os.environ.get("QUIBBLE_DATA_DIR", "").strip() or "data")


def valid_name(name):
    return isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)) \
        and name.lower() not in RESERVED


def local_names():
    root = data_root()
    bundled = Path(__file__).with_name("sample_data")
    return sorted({p.name for directory in (bundled, root) if directory.exists()
                  for p in directory.iterdir() if p.is_dir() and valid_name(p.name)
                  and p.name not in BUILTIN and p.name not in OPENML
                  and (p / "dataset.json").is_file()})


def known(name):
    return name in BUILTIN or name in OPENML or name in local_names()


SPLIT_KINDS = ("random", "group", "temporal")


def split_kind(spec):
    """Effective split scheme for a local manifest: time ordering wins, then groups, else random."""
    if spec.get("split"):
        return spec["split"]
    return "temporal" if "time_column" in spec else "group" if "group_column" in spec else "random"


def local_spec(name):
    if not valid_name(name):
        raise ValueError("Local dataset names must contain letters, digits, hyphens or underscores "
                         "and cannot be report, runs, smoke, or artifacts")
    if name.lower() in {n.lower() for n in (*BUILTIN, *OPENML)}:
        raise ValueError(f"{name!r} is a registered dataset name; rename the local dataset folder")
    directory = data_root() / name
    if not directory.exists():
        directory = Path(__file__).with_name("sample_data") / name
    manifest = directory / "dataset.json"
    if not manifest.is_file():
        raise ValueError(f"Unknown dataset {name!r}; expected local manifest {manifest}")
    spec = json.loads(manifest.read_text(encoding="utf-8-sig"))
    allowed = {"file", "target", "task", "drop_columns", "categorical_columns",
               "time_column", "group_column", "split"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"{manifest}: expected an object with only {sorted(allowed)}")
    for key in ("file", "target", "task"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{manifest}: {key} must be a nonempty string")
    if spec["task"] not in (C, R):
        raise ValueError(f"{manifest}: task must be classification or regression")
    for key in ("drop_columns", "categorical_columns"):
        value = spec.setdefault(key, [])
        if not isinstance(value, list) or any(not isinstance(c, str) or not c for c in value) \
                or len(value) != len(set(value)):
            raise ValueError(f"{manifest}: {key} must be a list of unique column names")
    if spec["target"] in spec["drop_columns"] + spec["categorical_columns"] \
            or set(spec["drop_columns"]) & set(spec["categorical_columns"]):
        raise ValueError(f"{manifest}: target, dropped, and categorical columns must not overlap")
    for key in ("time_column", "group_column"):
        if key in spec and (not isinstance(spec[key], str) or not spec[key].strip()):
            raise ValueError(f"{manifest}: {key} must be a nonempty column name")
    if "split" in spec and spec["split"] not in SPLIT_KINDS:
        raise ValueError(f"{manifest}: split must be one of {', '.join(SPLIT_KINDS)}")
    if spec.get("split") == "temporal" and "time_column" not in spec:
        raise ValueError(f"{manifest}: split 'temporal' needs time_column")
    if spec.get("split") == "group" and "group_column" not in spec:
        raise ValueError(f"{manifest}: split 'group' needs group_column")
    structural = [spec[k] for k in ("time_column", "group_column") if k in spec]
    if len(structural) != len(set(structural)) or {spec["target"], *spec["drop_columns"],
                                                    *spec["categorical_columns"]} & set(structural):
        raise ValueError(f"{manifest}: target, dropped, categorical, time and group columns must not overlap")
    if "\\" in spec["file"]:
        raise ValueError(f"{manifest}: file must be a relative path inside the dataset folder "
                         "(use forward slashes)")
    relative = Path(spec["file"])
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f"{manifest}: file must be a relative path inside the dataset folder")
    if path.suffix.lower() not in (".csv", ".parquet"):
        raise ValueError(f"{manifest}: file must end in .csv or .parquet")
    return spec, path


def is_categorical(col):
    return (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype))


def _source(spec, path, raw):
    file_hash = hashlib.sha256(raw).hexdigest()
    signature = json.dumps(spec, sort_keys=True).encode("utf-8") + file_hash.encode("ascii")
    synthetic = None
    generation = path.parent / "generation.json"
    if generation.is_file():
        metadata = generation.read_bytes()
        provenance = json.loads(metadata)
        generation_spec = provenance.get("specification") if isinstance(provenance, dict) else None
        if not isinstance(provenance, dict) or provenance.get("synthetic") is not True \
                or provenance.get("generator") != "quibble.generate" \
                or provenance.get("sha256") != file_hash or provenance.get("version") != 1 \
                or not isinstance(generation_spec, dict) \
                or type(generation_spec.get("seed")) is not int:
            raise ValueError(f"{generation}: generated-source metadata does not match the input file; "
                             "restore the original generated bundle or create a new dataset")
        synthetic = {"generator": provenance["generator"], "version": provenance["version"],
                     "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                     "seed": provenance["specification"]["seed"]}
        signature += synthetic["metadata_sha256"].encode("ascii")
    return {"kind": "local", "path": str(path), "sha256": file_hash,
            "fingerprint": hashlib.sha256(signature).hexdigest(),
            **({"synthetic": synthetic} if synthetic else {}), **spec}


def source_info(name):
    if name in BUILTIN or name in OPENML:
        return None
    spec, path = local_spec(name)
    return _source(spec, path, path.read_bytes())


def validate_sources(out, name, source):
    """Prevent feature/schema changes from mixing with saved predictions for the same name."""
    for path in (Path(out) / name).glob("*_ss*_ms*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "error" in record or "refused" in record:
            continue
        saved = record.get("dataset_source")
        if (source or saved) and (not source or not saved
                                  or saved["fingerprint"] != source["fingerprint"]):
            raise ValueError(f"Local dataset source differs from saved results for {name}. "
                             "Restore the original file/manifest or use a new --out; "
                             "--force does not permit mixing changed source data.")
        if source and source.get("task") == C and split_kind(source) != "random":
            expected = list(range(len(source["class_labels"]))) if source.get("class_labels") else None
            classes = record.get("prediction_classes")
            if classes is None or (expected is not None and classes != expected):
                raise ValueError(f"{path.name}: saved group/temporal classification lacks matching "
                                 "class-order validation; regenerate with a new --out.")


def _local_fetch(name):
    spec, path = local_spec(name)
    raw = path.read_bytes()
    if path.suffix.lower() == ".csv":
        rows = csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        header = next(rows, [])
        if not header or any(not c.strip() or c != c.strip() for c in header) \
                or len(header) != len(set(header)):
            raise ValueError(f"{path}: CSV needs unique, nonempty headers without surrounding spaces")
        for number, row in enumerate(rows, 2):
            if row and len(row) != len(header):
                raise ValueError(f"{path}: CSV row {number} has {len(row)} fields, expected {len(header)}")
        dtypes = {c: "string" for c in spec["categorical_columns"]}
        if spec["task"] == C:
            dtypes[spec["target"]] = "string"
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=dtypes)
    else:
        try:
            frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
        except ImportError as e:
            raise ImportError("Parquet input requires pyarrow; install requirements.txt "
                              "or the package's [data] extra") from e
    if not frame.columns.is_unique or any(not isinstance(c, str) or not c.strip()
                                           or c != c.strip() for c in frame.columns):
        raise ValueError(f"{path}: columns must have unique, nonempty string names without surrounding spaces")
    structural = [spec[k] for k in ("time_column", "group_column") if k in spec]
    required = {spec["target"], *spec["drop_columns"], *spec["categorical_columns"], *structural}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path}: missing declared columns: {sorted(missing)}")
    if len(frame) < 4:
        raise ValueError(f"{path}: at least 4 labeled rows are required")
    y = frame[spec["target"]]
    if y.isna().any() or y.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: target contains missing or empty values; provide labeled rows only")
    if spec["task"] == R:
        y = pd.to_numeric(y, errors="raise").astype(float)
        if not np.isfinite(y).all():
            raise ValueError(f"{path}: regression target must contain finite numeric values")
    else:
        if y.nunique() < 2 or y.value_counts().min() < 2:
            raise ValueError(f"{path}: classification needs at least 2 classes and 2 rows per class")
        if pd.api.types.is_numeric_dtype(y) and not np.isfinite(y).all():
            raise ValueError(f"{path}: classification labels must be finite")
    times = groups = None
    if "time_column" in spec:
        raw_t = frame[spec["time_column"]]
        try:
            times = (raw_t.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(raw_t)
                     else pd.to_datetime(raw_t).astype("int64").to_numpy())
            if raw_t.isna().any() or (times.dtype.kind == "f" and not np.isfinite(times).all()):
                raise ValueError("missing values")
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(f"{path}: time_column {spec['time_column']!r} must be numeric or parseable "
                             f"dates with no missing values ({e})") from e
    if "group_column" in spec:
        raw_g = frame[spec["group_column"]]
        if raw_g.isna().any():
            raise ValueError(f"{path}: group_column {spec['group_column']!r} has missing values")
        groups = raw_g.astype(str).to_numpy()
    X = frame.drop(columns=[spec["target"], *spec["drop_columns"], *structural]).copy()
    if X.shape[1] == 0:
        raise ValueError(f"{path}: at least one feature must remain after dropping columns")
    for column in X:
        values = X[column]
        if column in spec["categorical_columns"] or pd.api.types.is_string_dtype(values) \
                or pd.api.types.is_object_dtype(values) or pd.api.types.is_bool_dtype(values) \
                or isinstance(values.dtype, pd.CategoricalDtype):
            if any(not pd.api.types.is_scalar(v) for v in values.dropna()):
                raise ValueError(f"{path}: feature {column!r} contains nested values; use scalar columns")
            X[column] = values.astype("string").astype(object)
        elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_complex_dtype(values):
            X[column] = values.astype(float)
            if np.isinf(X[column]).any():
                raise ValueError(f"{path}: feature {column!r} contains infinity")
        else:
            raise ValueError(f"{path}: unsupported feature type {values.dtype} for {column!r}; "
                             "drop it or explicitly declare it categorical")
    X.attrs["quibble_source"] = _source(spec, path, raw)
    kind = split_kind(spec)
    X.attrs["quibble_split"] = {"kind": kind, **{k: spec[k] for k in ("time_column", "group_column")
                                                   if k in spec}}
    if times is not None:
        X.attrs["quibble_times"] = times
    if groups is not None:
        X.attrs["quibble_groups"] = groups
    return X, y, name


def task_of(name):
    return BUILTIN[name] if name in BUILTIN else OPENML[name][1] if name in OPENML else local_spec(name)[0]["task"]


def _fetch(name, data_home=None):
    from .cache import prepare_caches
    paths = prepare_caches()
    data_home = data_home or str(paths["openml"])
    if (name in BUILTIN or name in OPENML) and (data_root() / name / "dataset.json").exists():
        raise ValueError(f"{name!r} is a registered dataset name; rename the local dataset folder")
    if name in BUILTIN:
        fn = {"breast_cancer": skd.load_breast_cancer, "wine": skd.load_wine,
              "diabetes_reg": skd.load_diabetes}[name]
        X, y = fn(as_frame=True, return_X_y=True)
        return X, y, name
    if name not in OPENML:
        try:
            return _local_fetch(name)
        except csv.Error as e:
            raise ValueError(f"{name}: invalid CSV: {e}") from e
    b = skd.fetch_openml(data_id=OPENML[name][0], as_frame=True, data_home=data_home)
    return b.data, b.target, b.details.get("name")


def load(name, data_home=None):
    X, y, _ = _fetch(name, data_home)
    task = task_of(name)
    if task == C:
        encoder = LabelEncoder().fit(y)
        y = pd.Series(encoder.transform(y))
        if "quibble_source" in X.attrs:
            X.attrs["quibble_source"]["class_labels"] = [str(c) for c in encoder.classes_]
    else:
        y = pd.Series(np.asarray(y, dtype=float))
    return X.reset_index(drop=True), y, task


def _balanced_group_folds(groups, k, seed):
    """Fold id per row; whole groups per fold, groups shuffled then placed in the smallest fold."""
    uniq, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) < k:
        raise ValueError(f"group split needs at least {k} distinct groups, found {len(uniq)}")
    order = np.random.RandomState(seed).permutation(len(uniq))
    load, fold_of = np.zeros(k), np.zeros(len(uniq), dtype=int)
    for g in order:
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += counts[g]
    return fold_of[inverse]


def _split_positions(X, y, split_id, task, cv, cv_seed, cfg):
    """Train and test row positions for the 'group' and 'temporal' schemes."""
    n = len(X)
    if cfg["kind"] == "group":
        groups = X.attrs["quibble_groups"]
        if cv == "holdout":
            n_groups = len(np.unique(groups))
            if n_groups < 2:
                raise ValueError("group split needs at least 2 distinct groups")
            k = min(10, n_groups)  # about 30% of the groups, as whole balanced folds, are the test set
            te = np.flatnonzero(_balanced_group_folds(groups, k, split_id) < max(1, round(0.3 * k)))
        else:
            k, r = (int(v) for v in cv.split("x"))
            repeat, f = divmod(split_id, k)
            te = np.flatnonzero(_balanced_group_folds(groups, k, cv_seed + repeat) == f)
        tr = np.setdiff1d(np.arange(n), te)
    else:
        order = np.argsort(X.attrs["quibble_times"], kind="stable")
        if cv == "holdout":
            cut = int(round(n * 0.7))
            tr, te = order[:cut], order[cut:]
        else:
            k, r = (int(v) for v in cv.split("x"))
            if r != 1:
                raise ValueError("temporal splits use forward chaining, which has no repeats: "
                                 f"use --cv {k}x1")
            edges = np.linspace(0, n, k + 2).round().astype(int)
            tr, te = order[:edges[split_id + 1]], order[edges[split_id + 1]:edges[split_id + 2]]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("split produced an empty train or test set; use fewer folds or more data")
    if task == C:
        missing = sorted(set(y.unique().tolist()) - set(y.iloc[tr].unique().tolist()))
        if missing:
            raise UnsupportedSplit(f"{cfg['kind']} split training data is missing dataset classes "
                                   f"{missing}; all classes must occur in training to preserve "
                                   "probability-column order. Use another valid evaluation window "
                                   "or provide more representative training data.")
        if y.iloc[tr].nunique() < 2:
            raise UnsupportedSplit("the training part of this split contains fewer than 2 classes")
    return np.sort(tr), np.sort(te)


def split(X, y, split_id, task, cv="holdout", cv_seed=0):
    """Returns Xtr, Xte, ytr, yte, test_row_positions.
    random (default): holdout is a stratified 70/30 split seeded by split_id; 'KxR' is the split_id-th
    fold of R repeats of K-fold CV. group: whole groups (group_column) stay on one side.
    temporal: the future (time_column) is always the test set; 'Kx1' is forward chaining.
    The same split_id gives identical test rows for every backend."""
    cfg = X.attrs.get("quibble_split") or {"kind": "random"}
    idx = np.arange(len(X))
    if cfg["kind"] != "random":
        tr, te = _split_positions(X, y, split_id, task, cv, cv_seed, cfg)
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], te
    if task == C and "quibble_source" in X.attrs:
        minimum = 2 if cv == "holdout" else int(cv.split("x")[0])
        if y.value_counts().min() < minimum:
            raise ValueError(f"Local classification needs at least {minimum} rows per class "
                             f"for {cv}; use fewer folds or provide more labeled rows")
    if cv == "holdout":
        strat = y if task == C and y.value_counts().min() >= 2 else None
        tr, te = train_test_split(idx, test_size=0.3, stratify=strat, random_state=split_id)
    else:
        k, r = (int(v) for v in cv.split("x"))
        if task == C and y.value_counts().min() >= k:
            it = RepeatedStratifiedKFold(n_splits=k, n_repeats=r, random_state=cv_seed).split(idx, y)
        else:
            it = RepeatedKFold(n_splits=k, n_repeats=r, random_state=cv_seed).split(idx)
        tr, te = list(it)[split_id]
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], te


def meta(X, y, task):
    cat = sum(1 for c in X.columns if is_categorical(X[c]))
    n = len(X)
    m = {"rows": n, "features": X.shape[1], "task": task,
         "cat_share": round(cat / X.shape[1], 3),
         "missing_share": round(float(X.isna().to_numpy().mean()), 4),
         "tier": "small" if n < 5_000 else "medium" if n < 50_000 else "large"}
    if task == C:
        m["classes"] = int(y.nunique())
        m["majority_share"] = round(float(y.value_counts(normalize=True).iloc[0]), 3)
    return m


def verify(names, data_home):
    out = {}
    for n in names:
        try:
            X, y, oname = _fetch(n, data_home)
            task = task_of(n)
            y = pd.Series(LabelEncoder().fit_transform(y)) if task == C else \
                pd.Series(np.asarray(y, dtype=float))
            m = meta(X.reset_index(drop=True), y, task)
            exp = OPENML[n][2] if n in OPENML else n
            m["openml_name"] = oname
            m["name_ok"] = str(oname).lower() == exp.lower()
            if "quibble_source" in X.attrs:
                m["dataset_source"] = X.attrs["quibble_source"]
            out[n] = m
            flag = "" if m["name_ok"] else f"  <-- NAME MISMATCH (expected {exp})"
            print(f"{n}: {m}{flag}")
        except Exception as e:
            out[n] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{n}: ERROR {e}")
    return out


def main():
    from .config import _s
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true", help="list built-in, OpenML, and local dataset names")
    ap.add_argument("--names", help="comma list; default = all registered and discovered local datasets")
    a = ap.parse_args()
    if a.list:
        for origin, names in (("builtin", BUILTIN), ("openml", OPENML), ("local", local_names())):
            for name in names:
                print(f"{name}: {origin}")
        return 0
    names = a.names.split(",") if a.names else list(OPENML) + list(BUILTIN) + local_names()
    res = verify(names, _s("QUIBBLE_DATA_HOME", ".cache/openml"))
    with open("datasets_meta.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote datasets_meta.json")
    return 1 if any("error" in record or not record.get("name_ok", False)
                    for record in res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
