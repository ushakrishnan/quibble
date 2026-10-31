---
title: Configuration
description: Environment settings, precedence, device selection, and parallelism.
---

## Configuration

These settings are the controls for an experiment: which machine to use, how
long to run, how to divide the examples, and what to save. Keep different
recipes in different output folders. Changing a setting does not retroactively
change the experiment already recorded in an old report.

## Precedence
1. **Command-line flags** (highest). They are written into the environment before settings are read.
2. **Real environment variables.**
3. **`.env` file** in the current directory. Real environment variables are never overridden by it.
4. **Built-in defaults.**

`.env` format: `KEY=value`, `#` comments (also trailing ` # comment`), optional quotes around values.
Create settings only if absent, then edit them. `.env` is in `.gitignore`.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

## All settings

| Variable | Default | Meaning |
|---|---|---|
| `QUIBBLE_DEVICE` | `auto` | `cpu`, `cuda`, or `auto`. `cuda` raises an error if CUDA is not available (no silent CPU fallback). `auto` uses CUDA when torch reports it. |
| `QUIBBLE_PREPROCESS` | `ordinal` | `ordinal` or `native` categorical handling. |
| `QUIBBLE_CV` | `holdout` | `holdout`, or `KxR` (e.g. `5x2`). |
| `QUIBBLE_SPLIT_SEED` | `0` | Holdout seed, or (in a single CV run) the split id. Sweeps set it per cell. |
| `QUIBBLE_CV_SEED` | `0` | Seed for assigning rows to CV folds. |
| `QUIBBLE_MODEL_SEEDS` | `0,1,2` | Model seeds for sweeps. |
| `QUIBBLE_BOOT` | `500` | Bootstrap resamples for intervals. Lower it for very large datasets (see below). |
| `QUIBBLE_THREADS` | `0` | Threads per run. 0 = all cores, or cores divided by jobs when jobs > 1. |
| `QUIBBLE_JOBS` | `1` | Parallel CPU runs in a sweep. |
| `QUIBBLE_TIMEOUT_S` | `900` | Per-run timeout in sweeps. 0 disables. |
| `QUIBBLE_SINGLE_N` | `30` | Number of single-row predictions timed for latency percentiles. |
| `QUIBBLE_AG_PRESET` | `medium_quality` | AutoGluon preset. |
| `QUIBBLE_AG_TIME_LIMIT` | `120` | AutoGluon training time limit in seconds (fixed across datasets for fairness). |
| `QUIBBLE_TABPFN_MAX_ROWS` | *(blank)* | Quibble's training-row refusal limit. Blank means 10,000 on `cuda` and 1,000 on `cpu`; these are conservative harness policies, not verified library capacity limits. |
| `QUIBBLE_TABICL_ALLOW_DOWNLOAD` | `0` | `1` explicitly permits pinned TabICLv2 checkpoint downloads after reviewing terms. Otherwise cached-only; missing weights produce a recorded refusal. This is not a licence acceptance switch. |
| `QUIBBLE_MITRA_ALLOW_DOWNLOAD` | `0` | Strict `0` or `1`; `1` explicitly permits the optional Mitra adapter's pinned checkpoint downloads. This is separate from installation, tuning approval and usage rights. |
| `QUIBBLE_TABFM_N_ESTIMATORS` | `4` | Positive TabFM ensemble count. Full training splits outside explicit smoke checks. |
| `QUIBBLE_KUMO_SIZE` | `small` | KumoTabular pretrained model size: `small`, `medium`, or `large`. |
| `QUIBBLE_KUMO_N_ESTIMATORS` | `4` | Positive KumoTabular ensemble count. |
| `QUIBBLE_KEEP_MODELS` | `0` | `1` retains fitted tree/AutoGluon models; `0` removes temporary fits. Foundation fitted estimators are not serialized. |
| `QUIBBLE_TIER` | `defaults` | `defaults` keeps existing estimator behavior; `tuned` explicitly requests training-only tree search. Separate output roots are required. |
| `QUIBBLE_TUNE_TRIALS` | `20` | Candidate settings per tree, 1..1000 and no more than its finite domain size. |
| `QUIBBLE_TUNE_FOLDS` | `3` | Inner folds per candidate, 2..20. Impossible or class-unsafe splits fail without random fallback. |
| `QUIBBLE_TUNE_SEED` | `0` | Candidate ordering and inner-fold seed; estimator seed remains the outer cell's model seed. |
| `QUIBBLE_TUNE_METRIC` | `auto` | `auto` selects log loss for classification and RMSE for regression; explicit `log_loss` or `rmse` must match the task. Lower is better. |
| `QUIBBLE_TUNE_TIMEOUT_S` | `0` | Optional cooperative deadline for search plus refit, checked around fits. 0 disables it. It cannot interrupt a running fit. |
| `QUIBBLE_OUT_DIR` | `results` | Where results are written. |
| `QUIBBLE_CACHE_DIR` | `.cache` | Central runtime download cache; relative paths resolve from the project root. |
| `QUIBBLE_SKLEARN_CACHE_DIR` | *(blank)* | Optional scikit-learn cache override; blank follows `QUIBBLE_CACHE_DIR/sklearn`. |
| `QUIBBLE_HF_CACHE_DIR` | *(blank)* | Optional Hugging Face cache override; blank follows `QUIBBLE_CACHE_DIR/huggingface`. |
| `QUIBBLE_TRANSFORMERS_CACHE_DIR` | *(blank)* | Optional legacy Transformers cache override; blank shares `QUIBBLE_HF_CACHE_DIR/hub`. |
| `QUIBBLE_DATA_HOME` | *(blank)* | OpenML location; blank follows `QUIBBLE_CACHE_DIR/openml`. Explicit relative paths also resolve from the project root. |
| `QUIBBLE_DATA_DIR` | `data` | Collection of named local dataset folders with `dataset.json` plus CSV/Parquet. Relative paths resolve from the project root; not a download cache. |

Invalid values (`QUIBBLE_DEVICE=tpu`, `QUIBBLE_CV=five`, ...) stop the run with a `config error`.

## Tree tuning controls

`run`, `benchmark`, and `demo` accept `--tier`, `--tune-trials`, `--tune-folds`,
`--tune-seed`, `--tune-metric`, and `--tune-timeout-s`. They override the matching
environment variables. Tuning is opt-in; its settings have no effect on a
defaults-tier fit or default recipe hash.

The versioned `tree-equal-trials-v1` protocol searches only HistGB, LightGBM and
CatBoost. It uses a seeded, without-replacement ordering of a fixed finite grid.
The best mean inner-fold score wins; exact ties select the first trial in that
ordering. It is an **equal-trial**, not equal-time or equal-compute protocol.
Twenty candidates and three folds mean 60 inner fits plus one final refit per
tree, outer split and model seed when all candidates succeed.

The outer-training partition alone determines inner folds, learned categorical
preprocessing and selection. Temporal folds are forward-only, tied times cannot
cross the boundary, and declared groups stay disjoint. A local temporal manifest
may declare `purge_gap` as a nonnegative **row count** removed before each outer
and inner validation boundary. It is not a time-duration or label-horizon purge.
Both train and validation must contain every outer-training class. Unsafe or
impossible folds fail explicitly; there is no shuffled fallback. Iteration counts
are searched explicitly; library-internal early stopping is disabled for tuned
trees so it cannot make an uncontrolled random validation split.

Use a new `--out` root for each tier and search protocol. `.quibble-tier.json`
guards the root even when existing cells failed; changing the tier, trial/fold
budget, seed, metric, deadline or domains requires a separate root, not `--force`.
Foundation adapters can accompany tuned trees but keep their actual
`recipe.requested.tier` and adaptation settings. Their search status explicitly
says `not supported`; tree tuning does not fine-tune foundation models or change
their terms. Other unsupported model families are refused rather than silently
searched.

`--timeout` / `QUIBBLE_TIMEOUT_S` remains the authoritative subprocess timeout in
sweeps and benchmarks, and in supervised single-cell tuned CLI runs. A killed
worker is failed/incomplete, never a partial winner. Direct Python `run_one` and
`fit_tuned` calls are in-process APIs and need an external supervisor for a hard
timeout. Cooperative deadlines may overrun while a fit is running and are
recorded as such.

The browser's Methods and settings step defaults to **Defaults** independently
of inherited tuning environment variables. Selecting **Tuned trees** exposes the
budget, seed, metric and optional deadline. On Start, approve that exact tuning
request separately from full-run consent. Changing data, methods or settings
clears both approvals. A smoke tuning check still requires tuning approval.
The server validates the request and inner folds before creating a job, and
saves approved settings and the versioned protocol in `job.json`. Approval does
not enable checkpoint downloads. Relational tuning is refused until horizon-safe
inner folds have their own supported contract.

## Swap acceptance policy

Swap limits are separate from environment settings. The packaged
[swap_thresholds.json](../quibble_ml/swap_thresholds.json) supplies defaults shared by the CLI and UI.
Export with `quibble-ml swap --export-thresholds limits.json`, review and edit the copy, then pass
`--thresholds limits.json` to `swap`, `report`, or a full `benchmark`. Missing keys inherit
defaults; files are never auto-discovered. `--require-thresholds` requires a non-empty explicit
policy, including partial overrides. Result provenance lists every inherited default; this flag
does not certify those defaults were reviewed. In the workbench, full-run policy controls support editing, import, download, and
per-experiment persistence. See [Swap checks](swap.md#changing-the-limits) for meanings and validation.

## Scenario-planning providers

These settings affect specification drafting only, not the prediction methods being compared.
The [workbench guide](experiment-workbench.md#configure-your-planning-provider) covers setup and consent.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1/` | OpenAI-compatible inference URL; HTTPS except loopback |
| `LLM_MODEL` | blank | Provider model identifier or Azure deployment name |
| `LLM_AUTH_MODE` | `api_key` | `api_key`, `aad`, or loopback-only `none`; old `key` spelling accepted |
| `LLM_API_KEY` | blank | Key for the selected provider; never sent to the browser |
| `LLM_PROVIDER_NAME` | endpoint hostname | Optional display/provenance label; legacy Azure defaults to `Azure` |
| `LLM_AAD_CREDENTIAL` | `cli` | `cli` or noninteractive `default` Azure credential |
| `LLM_TOKEN_SCOPE` | `https://ai.azure.com/.default` | Entra inference token scope |
| `AZURE_TENANT_ID` | blank | Optional Azure CLI tenant; also used by Azure Identity environment credentials |
| `AZURE_SUBSCRIPTION_ID` | blank | Optional explicit Azure CLI subscription selection |

Standard `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` aliases work when their
generic equivalents are absent. A custom non-OpenAI `LLM_BASE_URL` requires its own
`LLM_API_KEY`, preventing accidental reuse of an unrelated provider's key.
Legacy `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT_NAME`,
`FOUNDRY_AUTH_MODE`, and `FOUNDRY_API_KEY` remain supported when a generic/OpenAI
endpoint is absent; the legacy mode defaults to `aad`. `FOUNDRY_AAD_CREDENTIAL`
and `AZURE_OPENAI_SCOPE` also remain credential/scope fallbacks.

`quibble-ml auth` or `quibble-ml web --login` performs explicit Azure CLI login.
`quibble-ml auth --check` never triggers login. Planning does not silently log in or fall
back to the local example or another provider. Key configuration/token checks do not prove inference RBAC
or endpoint compatibility.
Provider output must satisfy the generation schema. A schema-invalid response can
trigger one correction request with the validation error; there are no automatic
transport retries. Both requests can incur charges. A second invalid response is an
explicit error, not a local fallback.

## Centralized download caches

Quibble prepares caches before importing model libraries. It sets Hugging Face home, hub,
dataset and asset caches; TabPFN checkpoints; Torch Hub and compiler/extension caches;
Triton; scikit-learn/OpenML; and XDG defaults below the cache root. Inherited library-specific
cache variables are overridden so old shell settings cannot redirect downloads elsewhere.
Set the Quibble cache settings above to override locations. Pip/uv installation caches
are not changed. Existing files are not moved automatically.
Leave individual overrides blank to follow the central root. Explicit individual paths
take precedence and may intentionally point outside the root. Transformers shares the
Hugging Face hub by default to avoid storing the same weights in a second directory.

The CPU demo retains its fixed device. `benchmark` honors the device setting, but defaults
to CV `3x2` and model seed `0`; its flags override those defaults. Smoke mode fixes holdout
and seed `0`, samples at most 128 train/32 test rows, and uses an isolated output directory.

## Choosing CPU or GPU
```ini
# .env
QUIBBLE_DEVICE=cuda
```
- `cpu` pins all selected models to CPU.
- `cuda` runs CatBoost, TabPFN, TabFM, and KumoTabular on CUDA. Incompatible selections
  are visibly excluded, not run on CPU or replaced by other algorithms. If none remain,
  the command rejects the selection.
  Experimental TabICLv2 has passed small CPU/CUDA classification/regression
  holdouts; experimental Mitra-v2 has passed small CUDA holdouts with different
  classification/regression recipes. Neither implies broad hardware coverage.
- HistGB and Weak use scikit-learn, which has no CUDA implementation for these estimators.
  This LightGBM adapter uses CPU; GPU support needs a compatible build and adapter,
  and LightGBM's CUDA backend is not supported on Windows.
  AutoGluon is excluded because its GPU budget cannot guarantee GPU-only model execution.
  All remain available in CPU comparisons.
- Each result stores both the requested device and the effective device per backend, so a mixed
  table is never mislabelled.
- Auto chooses CUDA if available, otherwise CPU, once per sweep and pins it in all workers.
  The same exclusions apply when Auto resolves to CUDA. It does not benchmark speed.
  Resource failures never cause a
  silent CPU fallback. CPU requests stay on CPU even when CUDA hardware is available.
- Model resource failures, process exit codes, and timeouts are recorded while other cells
  continue. No configuration can guarantee sufficient memory for every model/dataset.
- Data preparation, metrics, and reporting still use CPU; the policy governs model execution.
  Exclusions and their reasons are saved in `invocation.json` and report participation.
  Historical mixed-device results retain their labels and warnings.
  Use separate roots for deliberate CPU-versus-CUDA experiments
  or changes to split assignments, rather than reusing incompatible cached cells.

## Threads and parallelism
- `--jobs N` runs N CPU runs at once; each gets `cores / N` threads unless you set `QUIBBLE_THREADS`.
- Parallel runs share the machine, so **latency and memory numbers become noisy**. Use `--jobs 1`
  for timings you intend to publish. The report warns when any result came from a parallel sweep.

## Memory note for the bootstrap
The bootstrap keeps a `QUIBBLE_BOOT x n_rows` integer matrix (8 bytes per cell). At 500 resamples a
48,000-row dataset needs roughly 190 MB; a 500,000-row dataset would need about 2 GB. Lower
`QUIBBLE_BOOT` (for example to 200) for very large datasets.

## Example `.env` files

Laptop, CPU only:
```ini
QUIBBLE_DEVICE=cpu
QUIBBLE_CV=5x2
QUIBBLE_MODEL_SEEDS=0
QUIBBLE_JOBS=2
```
Final benchmark, GPU box:
```ini
QUIBBLE_DEVICE=cuda
QUIBBLE_CV=5x2
QUIBBLE_MODEL_SEEDS=0,1,2,3,4
QUIBBLE_JOBS=1
QUIBBLE_OUT_DIR=results_cuda
```
