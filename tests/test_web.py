"""Loopback HTTP boundary and real generated-data experiment coverage."""
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from quibble import generate
from quibble.web import Server, Workbench, render_report
from quibble.web import ASSETS


def test_workflow_has_one_data_choice_and_shared_run_screen():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    for identifier in ("choose-existing", "choose-new", "nav-data", "nav-run", "nav-results",
                       "nav-history", "save", "change-dataset"):
        assert html.count(f'id="{identifier}"') == 1
    data = html.split('<section id="data-choice"', 1)[1].split("</section>", 1)[0]
    experiment = html.split('<section id="run-path"', 1)[1].split("</section>", 1)[0]
    assert 'id="dataset"' in data and 'id="existing-review"' in data
    assert 'id="dataset"' not in experiment and 'id="spec"' not in experiment
    assert 'id="chosen-dataset"' in experiment
    assert '<section id="new-path" aria-labelledby="scenario-heading" hidden>' in html
    assert '<section id="run-path" aria-labelledby="experiment-heading" hidden>' in html
    workspace = html.split('<nav class="workspace-nav"', 1)[1].split("</nav>", 1)[0]
    steps = html.split('<nav id="workflow-steps"', 1)[1].split("</nav>", 1)[0]
    assert 'id="nav-workflow"' in workspace and 'id="nav-history"' in workspace
    assert 'id="nav-history"' not in steps
    assert all(f'id="nav-{name}"' in steps for name in ("data", "run", "results"))
    assert 'id="scenario-sample"' not in html
    assert 'id="continue-new"' not in html
    assert '>Save and continue</button>' in html
    actions = html.split('<button id="save"', 1)[1].split("</div>", 1)[0]
    assert 'id="discard-variant"' in actions
    assert '>Discard changes and continue</button>' in actions


def test_unified_catalog_labels_tasks_and_sources(app):
    catalog = {item["name"]: item for item in app.state()["dataset_catalog"]}
    assert catalog["wine"]["kind"] == "multiclass classification"
    assert catalog["breast_cancer"]["kind"] == "binary classification"
    assert catalog["diabetes_reg"]["task"] == "regression"
    assert catalog["wine"]["group"] == "Built-in benchmarks"
    assert catalog["retail_clothing_regression"]["group"] == "Bundled synthetic examples"
    spec = generate.recommendation(rows=100, fields=4)
    spec["name"] = "custom_saved"
    generate.save(spec)
    catalog = {item["name"]: item for item in app.state()["dataset_catalog"]}
    assert catalog["custom_saved"]["group"] == "Saved datasets"


def test_product_header_and_about_are_separate_from_workflow():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    from quibble.web import render_header
    assert "__PRODUCT_HEADER__" in html
    header = render_header("workbench")
    assert '<h1 class="brand">Quibble</h1>' in header
    assert 'href="/about" target="_blank" rel="noopener noreferrer"' in header
    assert 'href="https://github.com/ushakrishnan/quibble"' in header
    assert "Not published" not in header
    about = (ASSETS / "about.html").read_text(encoding="utf-8")
    for heading in ("Why Quibble?", "What Quibble does", "Methods and compute", "What the evidence means",
                    "Local operation and provider calls", "Documentation and project status"):
        assert heading in about
    assert '<script src="/landing.js" defer></script>' in about
    assert 'class="button-link" href="/workbench">Open workbench</a>' in about


def test_documentation_navigation_covers_guides_and_anchors():
    from quibble.web import DOCS, DOC_NAV, documentation_files, render_document
    filenames = [name for _, names in DOC_NAV for name in names]
    assert len(filenames) == len(set(filenames))
    assert set(filenames) == {path.relative_to(DOCS).as_posix()
                              for path in DOCS.rglob("*.md")}
    files = documentation_files()
    route = "/docs/getting-started.md"
    rendered = render_document(route, files[route], files).decode()
    assert "<title>Getting started | Quibble</title>" in rendered
    assert 'aria-label="Documentation"' in rendered
    assert 'href="/docs/getting-started.md" aria-current="page"' in rendered
    assert 'aria-label="On this page"' in rendered
    assert 'href="#install"' in rendered and 'id="install"' in rendered
    assert 'rel="prev" href="/docs/about.md"' in rendered
    assert 'rel="next" href="/docs/experiment-workbench.md"' in rendered
    assert 'href="#document-content">Skip to content' in rendered
    assert 'aria-label="Breadcrumb"' in rendered
    for route in ("/docs/README.md", "/docs/blog/02-making-prediction-comparisons-worth-trusting.md"):
        rendered = render_document(route, files[route], files).decode()
        assert rendered.count('rel="prev"') + rendered.count('rel="next"') == 1
    blog_route = "/docs/blog/01-same-accuracy-different-mistakes.md"
    blog = render_document(blog_route, files[blog_route], files).decode()
    sidebar = blog.split('<aside class="docs-sidebar">', 1)[1].split("</aside>", 1)[0]
    assert 'aria-label="Blog series"' in sidebar
    assert "Documentation navigation" not in sidebar
    assert "/docs/getting-started.md" not in sidebar
    assert 'rel="prev" href="/docs/blog/README.md"' in blog
    assert 'rel="next" href="/docs/blog/02-making-prediction-comparisons-worth-trusting.md"' in blog
    assert 'aria-label="Breadcrumb"><a href="/blog">Blog</a>' in blog
    guide = render_document("/docs/README.md", files["/docs/README.md"], files).decode()
    sidebar = guide.split('<aside class="docs-sidebar">', 1)[1].split("</aside>", 1)[0]
    assert "/docs/blog/" not in sidebar


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIBBLE_OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("QUIBBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "")
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "")
    for name in ("LLM_BASE_URL", "LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "")
    instance = Workbench()
    yield instance
    instance.close()


def example(app):
    return app.action("/api/recommend", {"rows": 100, "fields": 4, "seed": 42, "task": "classification"})


def test_review_and_preview_required(app):
    result = example(app)
    body = {"spec": result["spec"], "plan_id": result["plan_id"], "reviewed": True}
    with pytest.raises(ValueError, match="Preview"):
        app.action("/api/save", body)
    app.action("/api/preview", body)
    body["spec"]["seed"] = 43
    with pytest.raises(ValueError, match="Preview"):
        app.action("/api/save", body)
    app.action("/api/preview", body)
    saved = app.action("/api/save", body)
    assert saved["rows"] == 100
    assert saved["name"] in app.state()["datasets"]
    body["plan_id"] = "browser-created"
    with pytest.raises(ValueError, match="session"):
        app.action("/api/save", body)


def test_existing_generated_review_reuse_and_save_new(app):
    spec = generate.recommendation(rows=100, fields=4)
    saved = generate.save(spec)
    directory = app.root.parent.parent / "data" / spec["name"]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    review = app.action("/api/inspect", {"dataset": spec["name"]})
    assert review["rows"] == 100 and review["predictors"] == 4
    assert len(review["sample"]) == 20
    assert review["generation"]["specification"] == spec
    with pytest.raises(ValueError, match="Review"):
        app.action("/api/use-existing", {"dataset": spec["name"]})
    body = {"dataset": spec["name"], "reviewed": True, "fingerprint": review["fingerprint"]}
    assert app.action("/api/use-existing", body)["name"] == saved["name"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    edit = app.action("/api/edit-existing", {"dataset": spec["name"]})
    save_body = {"spec": edit["spec"], "plan_id": edit["plan_id"], "reviewed": True}
    app.action("/api/preview", save_body)
    with pytest.raises(ValueError, match="already exists"):
        app.action("/api/save", save_body)
    save_body["spec"]["name"] = "grocery_variant"
    with pytest.raises(ValueError, match="Preview"):
        app.action("/api/save", save_body)
    app.action("/api/preview", save_body)
    assert app.action("/api/save", save_body)["name"] == "grocery_variant"
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_existing_review_rejects_changed_source(app):
    from quibble import datasets
    spec = generate.recommendation(rows=100)
    generate.save(spec)
    review = app.action("/api/inspect", {"dataset": spec["name"]})
    _, path = datasets.local_spec(spec["name"])
    metadata_path = path.parent / "generation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["specification"]["assumptions"].append("Updated provenance.")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since review"):
        app.action("/api/use-existing", {"dataset": spec["name"], "reviewed": True,
                                       "fingerprint": review["fingerprint"]})


def test_builtin_review_and_unknown_dataset(app):
    review = app.action("/api/inspect", {"dataset": "wine"})
    assert review["rows"] == 178 and review["generation"] is None
    assert app.action("/api/use-existing", {"dataset": "wine", "reviewed": True,
                                          "fingerprint": review["fingerprint"]})["name"] == "wine"
    with pytest.raises(ValueError, match="Only generated"):
        app.action("/api/edit-existing", {"dataset": "wine"})
    for name in ("../escape", [], "not_found"):
        with pytest.raises(ValueError, match="existing dataset"):
            app.action("/api/inspect", {"dataset": name})


@pytest.mark.parametrize(("name", "task"), [
    ("grocery_sample", "classification"), ("grocery_regression", "regression"),
    ("retail_clothing_binary", "classification"), ("retail_clothing_regression", "regression"),
])
def test_ready_made_samples_are_saved_reviewable_and_reusable(app, monkeypatch, name, task):
    monkeypatch.setattr(generate, "generate", lambda *a: pytest.fail("Must not regenerate saved samples"))
    assert name in app.state()["datasets"]
    result = app.action("/api/inspect", {"dataset": name})
    assert result["rows"] == 1000 and result["predictors"] == 8 and result["task"] == task
    assert len(result["sample"]) == 20
    assert result["generation"]["specification"]["name"] == name
    assert result["source"]["synthetic"]
    assert app.action("/api/use-existing", {"dataset": name, "reviewed": True,
                                          "fingerprint": result["fingerprint"]})["name"] == name


def test_imported_dataset_review_preserves_label_names(app):
    from quibble import datasets
    directory = datasets.data_root() / "imported"
    directory.mkdir(parents=True)
    (directory / "dataset.json").write_text(json.dumps({
        "file": "data.csv", "target": "decision", "task": "classification"}))
    (directory / "data.csv").write_text("amount,decision\n1,approve\n2,reject\n3,approve\n4,reject\n")
    review = app.action("/api/inspect", {"dataset": "imported"})
    assert review["generation"] is None
    assert review["sample"][0]["decision"] == "approve"
    assert review["target_summary"] == {"approve": 2, "reject": 2}
    assert review["source"]["class_labels"] == ["approve", "reject"]


def test_provider_consent_and_no_silent_fallback(app, monkeypatch):
    from quibble import planner
    def fail(*args):
        raise planner.PlannerError("provider unavailable")
    monkeypatch.setattr(planner, "plan", fail)
    with pytest.raises(ValueError, match="Confirm"):
        app.action("/api/plan", {})
    with pytest.raises(planner.PlannerError, match="provider unavailable"):
        app.action("/api/plan", {"consent": True, "scenario": "test", "rows": 100,
                                "fields": 4, "seed": 42, "task": "classification"})


def test_report_state_does_not_expose_secrets(app, monkeypatch):
    monkeypatch.setenv("FOUNDRY_API_KEY", "do-not-expose-this")
    assert "do-not-expose-this" not in json.dumps(app.state())


def test_formatted_report_preserves_structure_and_rewrites_only_known_figures():
    html = render_report(
        "# Results\n\n## Interpretation\n\n**Strong** and *emphasis* with `code`.\n\n"
        "| Method | Score |\n|---|---|\n| histgb | 0.9 |\n\n"
        "- One\n- Two\n\n> Synthetic only\n\n```text\nexample\n```\n\n"
        "![quality](fig_quality.png)\n\n[external](https://example.com)\n\n"
        "[local](../analysis/rows.csv)\n",
        "123-abc", ["fig_quality.png"])
    for tag in ("h1", "h2", "strong", "em", "code", "table", "th", "td", "ul", "blockquote", "pre"):
        assert f"<{tag}>" in html
    assert 'src="/api/jobs/123-abc/figures/fig_quality.png"' in html
    assert 'href="https://example.com"' in html
    assert "../analysis" not in html and ">local</a>" in html


def test_formatted_report_cannot_execute_html_or_fetch_unapproved_images():
    html = render_report(
        '<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n'
        '<iframe src="https://evil.example"></iframe>\n\n'
        '[bad](javascript:alert(1))\n\n![remote](https://evil.example/tracker.png)\n\n'
        '![data](data:image/png;base64,abc)\n\n![path](../private.png)\n\n'
        '![quoted](fig_quality.png "onerror=alert(1)")',
        "123-abc", ["fig_quality.png"])
    assert "<script" not in html and "<iframe" not in html
    assert "<img src=x" not in html
    assert 'href="javascript:' not in html
    assert html.count("<img ") == 1
    assert 'src="/api/jobs/123-abc/figures/fig_quality.png"' in html
    assert "tracker.png" not in html and "../private.png" not in html


def test_single_workbench_lease(app):
    with pytest.raises(ValueError, match="Another workbench"):
        Workbench()


def test_output_root_uses_dotenv_before_initializing_storage(tmp_path, monkeypatch):
    root = tmp_path / "configured-results"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUIBBLE_OUT_DIR", raising=False)
    (tmp_path / ".env").write_text(f"QUIBBLE_OUT_DIR={root}\n", encoding="utf-8")
    instance = Workbench()
    try:
        assert instance.root == root / "web"
    finally:
        instance.close()


def test_http_origin_token_and_path_boundaries(app):
    server = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def request(path, data=None, headers=None):
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        return urllib.request.urlopen(req, timeout=10)
    try:
        landing = request("/").read().decode()
        assert "Why Quibble?" in landing
        assert 'href="/workbench">Open workbench</a>' in landing
        assert server.token not in landing
        page = request("/workbench").read().decode()
        assert server.token in page
        assert "Content-Security-Policy" in request("/").headers
        from quibble import __version__
        about = request("/about")
        assert about.headers["Content-Type"] == "text/html; charset=utf-8"
        about_html = about.read().decode()
        assert f"Version {__version__}" in about_html
        assert "Pre-release" in about_html
        assert "unreleased" not in about_html.lower()
        assert "__QUIBBLE_VERSION__" not in about_html
        assert server.token not in about_html
        assert "Version" not in about_html.split("<header>", 1)[1].split("</header>", 1)[0]
        assert 'data-page="about" aria-current="page"' in about_html
        assert 'href="/blog"' in about_html
        blog = request("/blog").read().decode()
        from quibble.web import render_header
        for path, current in (("/", "about"), ("/workbench", "workbench"), ("/about", "about"),
                              ("/blog", "blog"), ("/docs/README.md", "docs")):
            rendered = request(path).read().decode()
            assert render_header(current) in rendered
            assert "__PRODUCT_HEADER__" not in rendered
        assert 'href="/docs/blog/01-same-accuracy-different-mistakes.md"' in blog
        article = request("/docs/blog/01-same-accuracy-different-mistakes.md").read().decode()
        assert 'src="/docs/assets/blog/evidence-stages.svg"' in article
        assert 'href="/docs/analysis.md"' in article
        assert 'id="why-not-upload-the-csv-to-chatgpt"' in article
        assert "title: Same Accuracy" not in article
        assert request("/docs/assets/blog/evidence-stages.png").headers["Content-Type"] == "image/png"
        evidence = json.loads(request("/docs/assets/blog/wine-smoke-evidence.json").read())
        assert evidence["dataset"]["name"] == "wine"
        for forbidden in ("/docs/../.env", "/docs/%2e%2e/.env",
                          "/docs/assets/blog/missing.svg", "/docs/FundamentalInfo/private.md"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                request(forbidden)
            assert caught.value.code == 404
        state = json.loads(request("/api/state").read())
        assert not state["provider"]["configured"]
        with pytest.raises(urllib.error.HTTPError) as caught:
            request("/api/recommend", b"{}", {"Content-Type": "application/json"})
        assert caught.value.code == 403
        headers = {"Content-Type": "application/json", "X-Quibble-Token": server.token}
        with pytest.raises(urllib.error.HTTPError) as caught:
            request("/api/recommend", b"{}", {**headers, "Origin": "https://evil.example"})
        assert caught.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as caught:
            request("/api/state", headers={"Host": "evil.example"})
        assert caught.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            request("/api/jobs/..%5c..%5c.env")
        with pytest.raises(urllib.error.HTTPError) as caught:
            request("/.env")
        assert caught.value.code == 404
        body = json.dumps({"rows": 100, "fields": 3, "seed": 42, "task": "regression"}).encode()
        result = json.loads(request("/api/recommend", body, headers).read())
        assert result["spec"]["rows"] == 100
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("task,mode", [("classification", "smoke"), ("regression", "full")])
def test_real_generated_dataset_benchmark_and_report(app, task, mode):
    spec = generate.recommendation(rows=100, fields=4, task=task)
    generate.save(spec)
    record = app.start({"dataset": spec["name"], "backends": ["histgb", "lightgbm"],
                        "mode": mode, "device": "cpu", "cv": "holdout", "full_confirmed": True})
    app.thread.join(timeout=120)
    assert not app.thread.is_alive()
    finished = app.job(record["id"])
    assert finished["status"] == "completed", finished["log"]
    assert finished["exit_code"] == 0
    assert finished["directory"] == str(app.root / record["id"])
    assert finished["report_available"] and finished["report"]
    assert "Interpretation" in finished["report"]
    assert spec["name"] in finished["report"]
    assert "### Synthetic experiment" in finished["report"]
    assert "<table>" in finished["report_html"]
    assert "<h3>Interpretation" in finished["report_html"]
    assert f'/api/jobs/{record["id"]}/figures/' in finished["report_html"]
    assert finished["figures"]
    assert app.figure(record["id"], finished["figures"][0]).startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        app.figure(record["id"], "..\\..\\.env")
    assert ("SMOKE" in finished["report"].upper()) if mode == "smoke" else ("Swap" in finished["report"])
    assert app.state()["jobs"][0]["id"] == record["id"]


def test_run_rejects_arbitrary_commands_and_unconfirmed_full(app):
    with pytest.raises(ValueError, match="methods"):
        app.start({"dataset": "wine", "backends": ["histgb; echo secret"]})
    with pytest.raises(ValueError, match="Confirm"):
        app.start({"dataset": "wine", "backends": ["histgb"], "mode": "full",
                   "device": "cpu", "cv": "holdout"})


@pytest.mark.parametrize("overrides", [None, {"max_quality_drop": 0.02}])
def test_full_run_persists_effective_policy_and_passes_custom_file(app, monkeypatch, overrides):
    from pathlib import Path
    from quibble.thresholds import DEFAULTS
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("quibble.web.subprocess.run", worker)
    assert app.state()["threshold_defaults"] == DEFAULTS
    record = app.start({"dataset": "wine", "backends": ["histgb"], "mode": "full",
                        "device": "cpu", "cv": "holdout", "full_confirmed": True,
                        "thresholds": overrides})
    app.thread.join(10)
    saved = app.job(record["id"])
    effective = {**DEFAULTS, **(overrides or {})}
    assert saved["thresholds"] == effective
    assert saved["thresholds_default"] is (overrides is None)
    policy_path = app.root / record["id"] / "thresholds.json"
    assert json.loads(policy_path.read_text()) == effective
    command = commands[0]
    if overrides:
        assert Path(command[command.index("--thresholds") + 1]) == policy_path
        assert "--require-thresholds" in command
    else:
        assert "--thresholds" not in command and "--require-thresholds" not in command
    app.close()
    restarted = Workbench()
    try:
        assert restarted.job(record["id"])["thresholds"] == effective
    finally:
        restarted.close()


@pytest.mark.parametrize("overrides", [{}, [], {"typo": 1}, {"max_quality_drop": True},
                                       {"confident_prob": 2}, {"primary": "rmse"}])
def test_web_rejects_invalid_policy_before_creating_job(app, overrides):
    with pytest.raises(ValueError):
        app.start({"dataset": "wine", "backends": ["histgb"], "mode": "full",
                   "device": "cpu", "cv": "holdout", "full_confirmed": True,
                   "thresholds": overrides})
    assert app.thread is None
    assert not list(app.root.glob("*/job.json"))


def test_smoke_does_not_apply_or_save_a_swap_policy(app, monkeypatch):
    monkeypatch.setattr("quibble.web.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    body = {"dataset": "wine", "backends": ["histgb"], "mode": "smoke",
            "device": "cpu", "cv": "holdout"}
    with pytest.raises(ValueError, match="Smoke"):
        app.start({**body, "thresholds": {"max_quality_drop": 0.02}})
    record = app.start(body)
    app.thread.join(10)
    assert app.job(record["id"])["thresholds"] is None
    assert not (app.root / record["id"] / "thresholds.json").exists()


def test_previous_incomplete_status_is_explicit(app):
    path = app.root / "0123"
    path.mkdir()
    app._write(path / "job.json", {"id": "0123", "status": "running"})
    app.close()
    restarted = Workbench()
    try:
        assert json.loads((path / "job.json").read_text())["status"] == "interrupted"
    finally:
        restarted.close()


def test_completed_report_can_be_reopened_after_server_restart(app):
    directory = app.root / "0123-abcd"
    report = directory / "output" / "report" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Saved report\n\n## Interpretation\n\nA completed experiment.\n", encoding="utf-8")
    app._write(directory / "job.json", {"id": directory.name, "status": "completed"})
    app.close()
    restarted = Workbench()
    try:
        job = restarted.job(directory.name)
        assert job["status"] == "completed"
        assert job["directory"] == str(directory)
        assert "A completed experiment." in job["report"]
        assert "<h2>Interpretation</h2>" in job["report_html"]
        assert any(j["id"] == directory.name for j in restarted.state()["jobs"])
    finally:
        restarted.close()


def test_failed_job_is_not_success_and_active_job_blocks_second(app, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def run(command, **kwargs):
        kwargs["stdout"].write("Deliberately failed benchmark.\n")
        started.set()
        assert release.wait(10)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr("quibble.web.subprocess.run", run)
    body = {"dataset": "wine", "backends": ["histgb"], "mode": "smoke", "device": "cpu", "cv": "holdout"}
    record = app.start(body)
    assert started.wait(10)
    try:
        with pytest.raises(ValueError, match="already"):
            app.start(body)
    finally:
        release.set()
        app.thread.join(10)
    finished = app.job(record["id"])
    assert finished["status"] == "failed" and finished["exit_code"] == 1
    assert not finished["report_available"] and finished["report"] is None
    assert "Deliberately failed" in finished["log"]


@pytest.mark.parametrize(("device", "available", "resolved"), [
    ("cpu", True, "cpu"), ("auto", False, "cpu"), ("auto", True, "cuda"), ("cuda", True, "cuda"),
])
def test_web_pins_device_and_persists_exclusions(app, monkeypatch, device, available, resolved):
    from quibble import config
    monkeypatch.setattr(config, "cuda_available", lambda: available)
    captured = []

    def worker(command, **kwargs):
        captured.append(kwargs["env"]["_QUIBBLE_RESOLVED_DEVICE"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("quibble.web.subprocess.run", worker)
    record = app.start({"dataset": "wine", "backends": ["histgb", "catboost", "autogluon"],
                        "mode": "smoke", "device": device, "cv": "holdout"})
    app.thread.join(10)
    assert captured == [resolved]
    saved = app.job(record["id"])
    assert saved["device_resolved"] == resolved
    assert saved["included_backends"] == (
        ["catboost"] if resolved == "cuda" else ["histgb", "catboost", "autogluon"])
    assert set(saved["excluded_backends"]) == (
        {"histgb", "autogluon"} if resolved == "cuda" else set())
    assert app.state()["cuda_available"] == available
    assert "lightgbm" in app.state()["cuda_exclusions"]


def test_web_rejects_empty_gpu_selection_and_unavailable_cuda(app, monkeypatch):
    from quibble import config
    body = {"dataset": "wine", "backends": ["histgb"], "mode": "smoke",
            "device": "cuda", "cv": "holdout"}
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    with pytest.raises(ValueError, match="No selected methods"):
        app.start(body)
    monkeypatch.setattr(config, "cuda_available", lambda: False)
    with pytest.raises(ValueError, match="not available"):
        app.start({**body, "backends": ["catboost"]})
    assert app.thread is None
