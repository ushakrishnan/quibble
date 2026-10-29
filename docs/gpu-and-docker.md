---
title: GPU and Docker
description: CUDA device routing, optional model setup, and container execution.
---

## GPU and Docker

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

The [local setup](getting-started.md#install) installs CUDA 12.8
builds of PyTorch 2.11.0 and torchvision 0.26.0 through `requirements.txt`. No separate
Torch installation or CUDA Toolkit is needed. A compatible NVIDIA driver is required.
Run `python -m pip check` and `python -m quibble doctor` after installation.
These pins apply to Windows only; other platforms resolve their own Torch builds.

## Setting up WSL2 (Windows)
1. Install a current NVIDIA driver **on Windows** with WSL support. Do not install a Linux GPU driver
   inside WSL.
2. In the WSL shell, `nvidia-smi` should list your GPU.
3. Keep the repository in the WSL filesystem (for example `~/quibble`), not under `/mnt/c`, for speed.
4. Create the repository's local environment and install the same unified requirements:
   ```bash
   uv venv --python 3.13 --seed .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m pip install --no-deps -e .
   python -m pip check
   ```
5. Check: `quibble doctor` should print `CUDA OK: <gpu name>, X of Y GB VRAM free.`
6. Set `QUIBBLE_DEVICE=cuda` in `.env`.
7. All model libraries are already included. If doctor cannot see CUDA, the resolved Torch build
   may be CPU-only. Choose CUDA wheels compatible with the complete requirements and your driver;
   do not force an old Torch version over the model dependencies. Run `pip check` and doctor again.

Windows/Python 3.13 dependency resolution and limited native CUDA execution are checked;
WSL2 dependency resolution and GPU runtime remain unverified.
Windows and WSL environments are platform-specific; do not activate a Windows `.venv` inside WSL.

## Sharing the GPU with other software
If another process holds GPU memory (for example a model-serving process), `quibble doctor` warns
when less than 40% of VRAM is free. For benchmark runs:
- stop the other process, or lower its memory reservation, before starting,
- remember that anything else using the GPU also perturbs timings.

`peak_torch_vram_mb` records only PyTorch allocations made by the run's own process.

## Docker
Files: `Dockerfile`, `docker-compose.yml`. They build on `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`
and run as a **non-root user (`ushak`, UID/GID 1000)**; adjust the `UID`/`GID` build arguments and the
`user:` line in the compose file if yours differ.

```bash
docker compose build
docker compose run --rm cpu quibble.doctor
docker compose run --rm cpu quibble.run --sweep --backends histgb,lightgbm --datasets breast_cancer
docker compose run --rm gpu quibble.run --sweep --backends catboost --datasets breast_cancer
```
- The entrypoint is `python -m`, so the argument is a module name (`quibble.run`, `quibble.report`,
  `quibble.analyze`, ...).
- `./results` and `./.cache` are mounted into the container so results and runtime download caches persist.
  The mounted directories must be writable by UID 1000 (`sudo chown -R 1000:1000 results .cache`).
- The `gpu` service requests one NVIDIA GPU and needs the NVIDIA Container Toolkit (on Windows,
  Docker Desktop with the WSL2 backend and GPU support).
- The Dockerfile installs the unified requirements, including model libraries. Their dependencies
  may replace the base image's Torch build; check CUDA compatibility inside the container.
- The compose file sets `QUIBBLE_DEVICE` per service. Other settings can come from the
  `.env` copied into the image at build time; rebuilding is needed after changing that file,
  or pass settings explicitly with `docker compose run -e QUIBBLE_JOBS=1 ...`.
