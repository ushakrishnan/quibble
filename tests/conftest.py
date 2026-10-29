"""Pin every QUIBBLE_* setting before quibble is imported, so a developer's .env cannot leak in."""
import os

for k, v in {"QUIBBLE_DEVICE": "cpu", "QUIBBLE_THREADS": "1", "QUIBBLE_JOBS": "1",
             "QUIBBLE_CV": "holdout", "QUIBBLE_SPLIT_SEED": "0", "QUIBBLE_MODEL_SEEDS": "0",
             "QUIBBLE_PREPROCESS": "ordinal", "QUIBBLE_BOOT": "100", "QUIBBLE_SINGLE_N": "3",
             "QUIBBLE_TIMEOUT_S": "300", "QUIBBLE_TABPFN_MAX_ROWS": "",
             "QUIBBLE_TABFM_N_ESTIMATORS": "4", "QUIBBLE_KUMO_SIZE": "small",
             "QUIBBLE_KUMO_N_ESTIMATORS": "4", "QUIBBLE_KEEP_MODELS": "0",
             "QUIBBLE_CACHE_DIR": ".cache", "QUIBBLE_DATA_HOME": "",
             "QUIBBLE_DATA_DIR": "data",
             "QUIBBLE_SKLEARN_CACHE_DIR": "", "QUIBBLE_HF_CACHE_DIR": "",
             "QUIBBLE_TRANSFORMERS_CACHE_DIR": "",
             "_QUIBBLE_RUN_TYPE": "benchmark", "_QUIBBLE_WORKFLOW": "run",
             "_QUIBBLE_RESOLVED_DEVICE": ""}.items():
    os.environ[k] = v

import pytest  # noqa: E402

from quibble.config import Config  # noqa: E402
from quibble.run import run_one  # noqa: E402

BACKENDS = ("histgb", "lightgbm", "weak")
DATASETS = ("breast_cancer", "diabetes_reg")


@pytest.fixture(scope="session")
def cv_results(tmp_path_factory):
    """3-fold CV results for 3 backends on a classification and a regression dataset."""
    out = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUIBBLE_OUT_DIR", str(out))
        mp.setenv("QUIBBLE_CV", "3x1")
        for ss in range(3):
            mp.setenv("QUIBBLE_SPLIT_SEED", str(ss))  # in CV mode this is the split id
            cfg = Config.from_env()
            for ds in DATASETS:
                for b in BACKENDS:
                    run_one(b, ds, 0, cfg)
    return out
