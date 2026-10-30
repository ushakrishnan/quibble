"""Guards against the documentation drifting away from the code."""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from quibble.cli import COMMANDS

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
TEXT = {f: f.read_text(encoding="utf-8") for f in DOCS}


def without_fences(text: str) -> str:
    return re.sub(r"(?ms)^[ \t]*```.*?^[ \t]*```[^\n]*", "", text)


def heading_ids(text: str) -> set[str]:
    text = without_fences(text)
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.M):
        slug = re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
        anchor, suffix = slug, 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{slug}-{suffix}"
        anchors.add(anchor)
    return anchors


def test_docs_exist():
    names = {f.name for f in DOCS}
    for n in ("index.md", "getting-started.md", "concepts.md", "cli-reference.md", "configuration.md",
              "backends.md", "datasets.md", "running-benchmarks.md", "analysis.md", "swap.md",
              "reports.md", "results-format.md", "gpu-and-docker.md", "troubleshooting.md",
              "development.md", "limitations.md", "project-plan.md", "testing-your-data.md",
              "README.md", "about.md", "experiment-workbench.md"):
        assert n in names, n


def test_threshold_links_and_policy_workflow_match_the_packaged_resource():
    assert (ROOT / "quibble" / "swap_thresholds.json").is_file()
    assert not (ROOT / "swap_thresholds.example.json").exists()
    assert '"swap_thresholds.json"' in (ROOT / "pyproject.toml").read_text()
    for path, text in TEXT.items():
        assert "swap_thresholds.example.json" not in text, path
    guide = TEXT[ROOT / "docs" / "swap.md"]
    for detail in ("--export-thresholds", "--require-thresholds", "Custom policy",
                   "../quibble/swap_thresholds.json", "Download effective policy"):
        assert detail in guide


def test_release_metadata_is_consistent():
    from quibble import __version__

    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{__version__}"' in metadata
    readme = TEXT[ROOT / "README.md"]
    assert f"Status: v{__version__} pre-release" in readme
    assert f"/releases/tag/v{__version__}" in readme
    for path in (ROOT / "README.md", ROOT / "docs" / "README.md", ROOT / "docs" / "about.md"):
        assert "unreleased" not in TEXT[path].lower()


def test_standalone_docs_and_current_browser_actions():
    home = TEXT[ROOT / "docs" / "README.md"]
    assert "[Getting started](getting-started.md)" in home
    assert "[Documentation index](index.md)" in home
    setup = TEXT[ROOT / "docs" / "getting-started.md"]
    assert "uv venv --python 3.13 --seed .venv" in setup
    assert "pip install -r requirements.txt" in setup
    assert "../README.md" not in setup
    for path in (ROOT / "README.md", ROOT / "docs" / "about.md",
                 ROOT / "docs" / "experiment-workbench.md"):
        text = TEXT[path]
        for action in ("Save and continue", "Discard changes and continue", "Past experiments"):
            assert action in text, path.name
        assert "About" in text and "GitHub" in text and "Docs" in text
    assert "Use as-is" not in TEXT[ROOT / "docs" / "experiment-workbench.md"]


def test_bundled_dataset_inventory_is_documented():
    from quibble.web import ASSETS
    guide = TEXT[ROOT / "docs" / "datasets.md"]
    for directory in (ROOT / "quibble" / "sample_data").iterdir():
        if directory.is_dir():
            assert f"| `{directory.name}` |" in guide
    assert (ASSETS / "about.html").is_file()
    assert '"web_assets/*.html"' in (ROOT / "pyproject.toml").read_text()


def test_public_blog_uses_current_titles_and_public_assets():
    blog = ROOT / "docs" / "blog"
    articles = {
        "01-same-accuracy-different-mistakes.md": "Same Accuracy, Different Mistakes",
        "02-making-prediction-comparisons-worth-trusting.md":
            "Making Prediction Comparisons Worth Trusting",
    }
    assert {path.name for path in blog.glob("*.md")} == {"README.md", *articles}
    for filename, title in articles.items():
        text = TEXT[blog / filename]
        assert f"title: {title}" in text
        assert "../assets/blog/" in text
        assert "../docs/" not in text
        assert "superdocs" not in text and "FundamentalInfo" not in text
    for entry in ("README.md", "about.md", "index.md"):
        assert "(blog/README.md)" in TEXT[ROOT / "docs" / entry]
    for filename in ("evidence-pipeline.svg", "evidence-pipeline.png",
                     "evidence-stages.svg", "evidence-stages.png",
                     "wine-smoke-evidence.json", "wine-smoke-summary.csv"):
        assert (ROOT / "docs" / "assets" / "blog" / filename).is_file()


def test_dataset_guide_covers_complete_registry_and_locations():
    from quibble.datasets import BUILTIN, OPENML
    text = TEXT[ROOT / "docs" / "datasets.md"]
    for name in BUILTIN:
        assert f"| `{name}` |" in text
    for name, (identifier, _, expected) in OPENML.items():
        assert f"| `{name}` | {identifier} |" in text
        assert expected in text
        assert f"https://www.openml.org/d/{identifier})" in text
    for filename in ("breast_cancer.csv", "wine_data.csv", "diabetes_data_raw.csv.gz",
                     "diabetes_target.csv.gz"):
        assert filename in text
    for section in ("What is stored where, and when", "How and when a command uses data",
                    "Registered, downloaded, verified, and actually evaluated",
                    "Reading the saved Wine predictions"):
        assert section in text
    for source in ("https://huggingface.co/google/tabfm-1.0.0-pytorch",
                   "https://huggingface.co/nvidia/Kumo-Tabular",
                   "https://huggingface.co/Prior-Labs/tabpfn_3_5"):
        assert source in text


def test_readme_uses_local_requirements_setup():
    text = TEXT[ROOT / "README.md"]
    assert r".\.venv\Scripts\Activate.ps1" in text
    assert "pip install -r requirements.txt" in text
    assert "pip install --no-deps -e ." in text
    assert 'pip install -e ".[' not in text
    assert not re.search(r"LTMStarter[\\/]+\.(?:venv|cache)", text)
    assert "torch==2.7.0" not in text
    assert "[Testing your own data](docs/testing-your-data.md)" in text


def test_local_data_guide_covers_the_implemented_manifest_and_workflow():
    text = TEXT[ROOT / "docs" / "testing-your-data.md"]
    for key in ("file", "target", "task", "drop_columns", "categorical_columns"):
        assert f'"{key}"' in text
    for detail in ("--list", "--verify --names payments", "--smoke", "--cv 3x2",
                   "dataset.json", "data.parquet", "QUIBBLE_DATA_DIR",
                   "fingerprint", "Privacy and publication", "first 40 feature columns"):
        assert detail in text


def test_requirements_cover_backends_reporting_and_tests():
    from packaging.requirements import Requirement
    entries = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    requirements = {req.name.lower(): req for line in entries
                    if line and not line.startswith(("#", "--")) for req in [Requirement(line)]}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "tabpfn",
                 "tabfm", "structured-data-models", "autogluon.tabular", "matplotlib", "psutil",
                 "pytest", "pyarrow"):
        assert name in requirements, name
    assert "pytorch" in requirements["tabfm"].extras
    assert requirements["psutil"].marker.evaluate({"sys_platform": "win32"})
    assert not requirements["psutil"].marker.evaluate({"sys_platform": "linux"})
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in entries
    for name, version in (("torch", "2.11.0+cu128"), ("torchvision", "0.26.0+cu128")):
        requirement = requirements[name]
        assert str(requirement.specifier) == f"=={version}"
        assert requirement.marker.evaluate({"sys_platform": "win32"})
        assert not requirement.marker.evaluate({"sys_platform": "linux"})
        assert not requirement.marker.evaluate({"sys_platform": "darwin"})


def test_requirements_use_one_file():
    assert not (ROOT / "requirements-extra.txt").exists()


def test_relative_links_resolve():
    for f, text in TEXT.items():
        for m in re.finditer(r"\]\((<[^>]+>|[^)\s]+)\)", text):
            target = urlsplit(m.group(1).strip("<>"))
            if target.scheme or target.netloc:
                continue
            path = (f.parent / unquote(target.path)).resolve() if target.path else f
            assert path.exists(), f"{f.name}: broken link {m.group(1)}"
            if target.fragment and path.suffix == ".md":
                anchors = heading_ids(path.read_text(encoding="utf-8"))
                assert unquote(target.fragment) in anchors, f"{f.name}: broken anchor {m.group(1)}"


def test_docs_have_metadata_and_labeled_balanced_fences():
    for f, text in TEXT.items():
        frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.S)
        if f == ROOT / "README.md":
            assert frontmatter is None, "GitHub README should not render a metadata table"
            assert text.startswith("# Quibble\n")
        else:
            assert frontmatter, f"{f.name}: missing frontmatter"
            for key in ("title", "description"):
                assert re.search(rf"^{key}:\s+\S", frontmatter[1], flags=re.M), f"{f.name}: missing {key}"
            assert not re.search(r"^# ", without_fences(text), flags=re.M), f"{f.name}: title is already in metadata"
        inside = False
        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                if not inside:
                    assert line.strip() != "```", f"{f.name}: code fence needs a language"
                inside = not inside
        assert not inside, f"{f.name}: unclosed code fence"


def test_readme_illustrates_the_real_workflow():
    text = TEXT[ROOT / "README.md"]
    screenshots = re.findall(r"!\[[^\]]+\]\((docs/assets/screenshots/[^)]+)\)", text)
    assert screenshots == ["docs/assets/screenshots/workbench-results.png"]
    for heading in ("## In plain English", "## Technical overview"):
        assert heading in text
        assert text.index(heading) < text.index("## See the workflow")
    assert "docs/experiment-workbench.md#visual-walkthrough" in text
    guide = TEXT[ROOT / "docs" / "experiment-workbench.md"]
    for name in ("workbench-choose-data.png", "workbench-configure-run.png", "workbench-results.png"):
        assert f"(assets/screenshots/{name})" in guide
    assert "real local full comparison" in text
    assert "assets/screenshots/*.png" in (ROOT / "pyproject.toml").read_text()


def test_env_template_has_no_duplicate_settings():
    entries = re.findall(r"^(QUIBBLE_[A-Z_]+)=", (ROOT / ".env.example").read_text(), flags=re.M)
    assert len(entries) == len(set(entries)), "Duplicate keys silently preserve the first value"


def test_env_template_cache_defaults_follow_root(tmp_path, monkeypatch):
    import os

    from quibble.cache import prepare_caches
    from quibble.config import load_dotenv

    entries = re.findall(r"^(QUIBBLE_[A-Z_]+)=", (ROOT / ".env.example").read_text(), flags=re.M)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in entries:
        monkeypatch.delenv(key, raising=False)
    load_dotenv(str(ROOT / ".env.example"))
    monkeypatch.setenv("QUIBBLE_CACHE_DIR", str(tmp_path))
    paths = prepare_caches()
    assert paths["root"] == tmp_path
    assert paths["sklearn"] == tmp_path / "sklearn"
    assert paths["huggingface"] == tmp_path / "huggingface"
    assert paths["openml"] == tmp_path / "openml"


def test_every_env_var_in_docs_exists_in_code():
    code = "".join(p.read_text() for p in (ROOT / "quibble").glob("*.py")) + (ROOT / ".env.example").read_text()
    known = set(re.findall(r"QUIBBLE_[A-Z_]+", code))
    for f, text in TEXT.items():
        for v in set(re.findall(r"QUIBBLE_[A-Z][A-Z_]+[A-Z]", text)):
            assert v in known, f"{f.name} mentions unknown setting {v}"


def test_every_setting_is_documented():
    documented = (ROOT / "docs" / "configuration.md").read_text()
    for v in set(re.findall(r"QUIBBLE_[A-Z_]+", (ROOT / ".env.example").read_text())):
        assert v in documented, f"{v} missing from docs/configuration.md"


def test_commands_shown_in_docs_are_real():
    valid = set(COMMANDS) | {"demo"}
    for f, text in TEXT.items():
        found = set(re.findall(r"`quibble ([a-z]+)[^`]*`", text))
        for block in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S):  # fenced code only
            found |= set(re.findall(r"^\s*(?:\$ )?quibble ([a-z]+)", block, flags=re.M))
        for cmd in found:
            assert cmd in valid, f"{f.name}: unknown command 'quibble {cmd}'"


def test_every_command_is_documented():
    ref = (ROOT / "docs" / "cli-reference.md").read_text()
    for cmd in set(COMMANDS) | {"demo"}:
        assert f"quibble {cmd}" in ref, cmd


@pytest.mark.parametrize("command", sorted((set(COMMANDS) | {"demo"}) - {"doctor"}))
def test_documented_flags_exist(command):
    """Flags documented in each command section must appear in its generated help."""
    import subprocess
    import sys
    help_text = subprocess.run([sys.executable, "-m", "quibble", command, "--help"],
                               capture_output=True, text=True, check=True).stdout
    ref = (ROOT / "docs" / "cli-reference.md").read_text()
    section = ref.split(f"## `quibble {command}`")[1].split("\n## ", 1)[0]
    for flag in set(re.findall(r"`(--[a-z-]+)", section)):
        assert flag in help_text, f"documented flag {flag} not accepted by `quibble {command}`"
