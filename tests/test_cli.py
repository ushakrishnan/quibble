import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from quibble_ml import __version__


def quibble(*args, **kw):
    return subprocess.run([sys.executable, "-m", "quibble_ml", *args], capture_output=True, text=True, **kw)


def test_version_and_help():
    assert quibble("--version").stdout.strip() == f"quibble-ml {__version__}"
    h = quibble("--help").stdout
    for cmd in ("run", "analyze", "swap", "report", "demo", "doctor"):
        assert cmd in h


def test_legacy_namespace_cannot_shadow_cli(tmp_path):
    (tmp_path / "quibble.py").write_text(
        'raise RuntimeError("The unrelated quibble package must not be imported")\n',
        encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = quibble("--version", cwd=tmp_path, env=env, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"quibble-ml {__version__}"


def test_command_dispatch_uses_only_the_new_namespace():
    from quibble_ml.cli import COMMANDS
    assert all(module.startswith("quibble_ml.") for module in COMMANDS.values())


def test_unknown_command_fails():
    assert quibble("nope").returncode == 2


def test_cuda_flag_fails_with_clear_message_without_cuda(tmp_path):
    r = quibble("run", "--backend", "histgb", "--dataset", "wine", "--device", "cuda",
                "--out", str(tmp_path))
    if r.returncode == 0:  # a CUDA machine: nothing to assert here
        return
    assert "config error" in r.stderr or "config error" in r.stdout


def test_end_to_end_sweep_via_cli(tmp_path):
    out = str(tmp_path)
    r = quibble("run", "--sweep", "--backends", "histgb,weak", "--datasets", "wine",
                "--cv", "3x1", "--model-seeds", "0", "--out", out)
    assert r.returncode == 0 and "6 to run" in r.stdout
    records = [json.loads(p.read_text()) for p in tmp_path.glob("*/*_ss*_ms*.json")]
    assert len({record["run_id"] for record in records}) == 1
    assert all(record["timestamp"] for record in records)
    archived = list((tmp_path / "runs").glob("*/*/snapshot.json"))
    assert len(archived) == 1  # one parent snapshot, not one snapshot per child
    r2 = quibble("run", "--sweep", "--backends", "histgb,weak", "--datasets", "wine",
                 "--cv", "3x1", "--model-seeds", "0", "--out", out)
    assert "0 to run, 6 already done" in r2.stdout  # resume
    assert list((tmp_path / "runs").glob("*/*/snapshot.json")) == archived
    assert quibble("analyze", "--dataset", "wine", "--out", out).returncode == 0
    assert quibble("report", "--out", out).returncode == 0


def test_demo_produces_successful_runs_and_report(tmp_path):
    r = quibble("demo", "--out", str(tmp_path), "--cv", "2x1", timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    records = list(tmp_path.glob("*/*_ss*_ms*.json"))
    assert len(records) == 18
    for path in records:
        record = json.loads(path.read_text())
        assert "error" not in record and "refused" not in record, record
        assert record["peak_rss_mb"] > 0
    report = (tmp_path / "report" / "report.md").read_text()
    assert "histgb -> weak: **FAIL**" in report


@pytest.fixture
def demo_calls(monkeypatch, tmp_path):
    from quibble_ml import report, run
    for key, value in {"QUIBBLE_OUT_DIR": str(tmp_path), "QUIBBLE_CV": "holdout",
                       "QUIBBLE_JOBS": "1", "QUIBBLE_DEVICE": "cpu",
                       "QUIBBLE_MODEL_SEEDS": "0"}.items():
        monkeypatch.setenv(key, value)
    sweep = Mock()
    build = Mock(return_value=tmp_path / "report" / "report.md")
    monkeypatch.setattr(run, "sweep", sweep)
    monkeypatch.setattr(report, "build", build)
    return sweep, build


@pytest.mark.parametrize(("selection", "expected", "swaps"), [
    ("histgb,tabfm,kumo_tabular", ["histgb", "tabfm", "kumo_tabular"],
     [("histgb", "tabfm"), ("histgb", "kumo_tabular")]),
    ("tabfm,kumo_tabular", ["tabfm", "kumo_tabular"], [("tabfm", "kumo_tabular")]),
    ("kumo_tabular", ["kumo_tabular"], []),
    ("histgb, tabfm,histgb", ["histgb", "tabfm"], [("histgb", "tabfm")]),
])
def test_demo_selected_backends_and_swap_checks(demo_calls, tmp_path, capsys, selection, expected, swaps):
    from quibble_ml.cli import main
    sweep, build = demo_calls
    assert main(["demo", "--backends", selection, "--out", str(tmp_path), "--force"]) == 0
    backends, datasets, cfg, force = sweep.call_args.args
    assert backends == expected and datasets == ["breast_cancer", "wine", "diabetes_reg"]
    assert cfg.device == "cpu" and cfg.model_seeds == (0,) and force is True
    build.assert_called_once_with(str(tmp_path), datasets, swaps=swaps)
    output = capsys.readouterr().out
    assert f"Demo backends: {', '.join(expected)} (CPU)" in output
    assert "deliberately bad" not in output


def test_demo_defaults_and_sanity_message_unchanged(demo_calls, tmp_path, capsys):
    from quibble_ml.cli import main
    sweep, build = demo_calls
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert sweep.call_args.args[0] == ["histgb", "lightgbm", "weak"]
    assert sweep.call_args.args[3] is False
    assert build.call_args.kwargs["swaps"] == [("histgb", "lightgbm"), ("histgb", "weak")]
    assert "deliberately bad" in capsys.readouterr().out


def test_demo_uses_shared_results_setting_without_out_override(demo_calls, tmp_path):
    from quibble_ml.cli import main
    sweep, build = demo_calls
    assert main(["demo"]) == 0
    assert sweep.call_args.args[2].out_dir == str(tmp_path)
    assert build.call_args.args[0] == str(tmp_path)


@pytest.mark.parametrize("selection", ["", "histgb,,tabfm", "histgb,unknown"])
def test_demo_rejects_invalid_backends(demo_calls, tmp_path, capsys, selection):
    from quibble_ml.cli import main
    sweep, build = demo_calls
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--backends", selection, "--out", str(tmp_path)])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    sweep.assert_not_called()
    build.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_demo_with_custom_backends_via_cli(tmp_path):
    r = quibble("demo", "--backends", "histgb,weak", "--cv", "holdout", "--jobs", "1",
                "--out", str(tmp_path), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    records = list(tmp_path.glob("*/*_ss*_ms*.json"))
    assert len(records) == 6
    for path in records:
        record = json.loads(path.read_text())
        assert record["backend"] in ("histgb", "weak")
        assert "error" not in record and "refused" not in record
    report = (tmp_path / "report" / "report.md").read_text()
    assert "histgb -> weak: **FAIL**" in report
    assert "Swap check: histgb -> lightgbm" not in report
