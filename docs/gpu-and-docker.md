---
title: GPU and Docker
description: CUDA device routing, optional model setup, and container execution.
---

## GPU and Docker

A GPU is a faster calculator for methods that support it, not evidence that a
model predicts better. Docker packages a runtime; it does not verify the model,
the data licence or the fairness of a comparison. Check both device labels and
the tested scope before comparing costs.

> [!NOTE]
> Native Windows CUDA detection, a CatBoost classification smoke, and local
> TabFM/KumoTabular Wine classification fits have
> succeeded. This is limited validation, not coverage of every GPU backend or task.
> WSL2 and Docker remain unvalidated. See [trust levels](index.md#trust-levels-what-has-actually-been-run).

## What uses the GPU
With `QUIBBLE_DEVICE=cuda`:

| Backend | Uses GPU |
|---|---|
| `catboost` | yes (`task_type="GPU"`) |
| `tabpfn` | yes (`device="cuda"`), and its row limit rises to 10,000 by default |
| `tabicl_v2` | yes; small pinned classification/regression CUDA holdouts verified |
| `mitra_v2` | CUDA-only experimental adapter; classification is zero-shot, regression performs prediction-time fine-tuning |
| `tabfm` | yes (PyTorch weights loaded with `device="cuda"`) |
| `kumo_tabular` | yes (model, tables, and seeded generator use `device="cuda"`) |
| `autogluon` | excluded: GPU budgets do not prevent CPU ensemble components |
| `histgb`, `weak` | excluded: these scikit-learn estimators have no CUDA implementation |
| `lightgbm` | excluded: this adapter uses CPU; CUDA backend is not supported on Windows |

`cuda` is **strict**: if torch cannot see CUDA, the command stops with `config error`. It never falls
back to CPU silently, so a benchmark can never claim GPU numbers it did not produce. `auto` picks CUDA
when available, otherwise CPU. Auto resolves once in the parent and is pinned in workers.
Memory failures are recorded without a CPU retry or hidden training-row reduction.
Exclusions are printed and saved in invocation metadata and report participation.
If all selected methods are excluded, choose compatible methods or use CPU.
Data preparation, metrics, and reporting still use CPU.
Historical mixed CPU/GPU measurements keep their original labels.

On `cuda`, GPU-capable backends run one at a time even with `--jobs > 1` (they would compete for
VRAM). On `cpu`, all selected methods run on CPU and can use multiple workers.

## Native Windows

Add `--extra windows-cuda` to the [locked setup](getting-started.md#install)
command, retaining all wanted model extras. This explicitly selects CUDA 12.8
builds of PyTorch 2.11.0 and torchvision 0.26.0 from the official PyTorch index.
The base setup does not install Torch. No separate CUDA Toolkit is needed, but a
compatible NVIDIA driver is required. Run `uv pip check` and
`python -m quibble_ml doctor` afterward. The extra is Windows-only; it does not
configure Linux or macOS GPU packages.

## Setting up WSL2 (Windows)
1. Install a current NVIDIA driver **on Windows** with WSL support. Do not install a Linux GPU driver
   inside WSL.
2. In the WSL shell, `nvidia-smi` should list your GPU.
3. Keep the repository in the WSL filesystem (for example `~/quibble`), not under `/mnt/c`, for speed.
4. Create a separate Linux environment with selected model extras:
   ```bash
   uv sync --locked --python 3.13 --extra web --extra data --extra tabfm --extra kumo
   source .venv/bin/activate
   uv pip check
   ```
5. Check: `quibble-ml doctor` should print `CUDA OK: <gpu name>, X of Y GB VRAM free.`
6. Set `QUIBBLE_DEVICE=cuda` in `.env`.
7. Only selected model libraries are installed. If doctor cannot see CUDA, inspect
   the locked Linux Torch build and driver compatibility. Do not copy Windows
   wheels or use `windows-cuda` to configure Linux. Any alternative wheel source
   needs an explicit manifest/lock change and validation; do not force an untracked
   Torch version over the model dependencies.

The new lock resolves the selected dependency graph; this does not establish a
new GPU fit. Earlier native Windows CUDA evidence is limited; WSL2 dependency
installation and GPU runtime remain unverified.
Windows and WSL environments are platform-specific; do not activate a Windows `.venv` inside WSL.

## Sharing the GPU with other software
If another process holds GPU memory (for example a model-serving process), `quibble-ml doctor` warns
when less than 40% of VRAM is free. For benchmark runs:
- stop the other process, or lower its memory reservation, before starting,
- remember that anything else using the GPU also perturbs timings.

`peak_torch_vram_mb` records only PyTorch allocations made by the run's own process.

## Docker
Files: `Dockerfile`, `docker-compose.yml`. They build on `python:3.13-slim` with uv
0.12.11 and run as a **non-root user (`quibble`, UID/GID 1000)**; adjust the `UID`/`GID` build arguments and the
`user:` line in the compose file if yours differ.

```bash
docker compose build
docker compose run --rm cpu quibble_ml.doctor
docker compose run --rm cpu quibble_ml.run --sweep --backends histgb,lightgbm --datasets breast_cancer
docker compose build --build-arg DEPENDENCY_EXTRAS=web,data,tabfm,kumo gpu
docker compose run --rm gpu quibble_ml.run --sweep --backends catboost --datasets breast_cancer
```
- The entrypoint is `python -m`, so the argument is a module name (`quibble_ml.run`, `quibble_ml.report`,
  `quibble_ml.analyze`, ...).
- `./results` and `./.cache` are mounted into the container so results and runtime download caches persist.
  The mounted directories must be writable by UID 1000 (`sudo chown -R 1000:1000 results .cache`).
- The `gpu` service requests one NVIDIA GPU and needs the NVIDIA Container Toolkit (on Windows,
  Docker Desktop with the WSL2 backend and GPU support).
- The default image syncs the locked `web,data` extras without development packages.
  Model libraries are opt-in through the comma-separated `DEPENDENCY_EXTRAS` build
  argument. GPU execution also needs a compatible Linux Torch build and host driver;
  the Windows CUDA extra is not a Linux configuration.
- The compose file sets `QUIBBLE_DEVICE` per service. `.env` is excluded from the
  image; pass other settings explicitly with `docker compose run -e QUIBBLE_JOBS=1 ...`.
- Container build/runtime and Linux CUDA compatibility remain unverified. A
  successful lock does not establish a successful Docker or GPU execution.
