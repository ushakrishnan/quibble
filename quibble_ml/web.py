"""Loopback-only experiment workbench; not a multi-user deployment server."""
import argparse
import html
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

from . import __version__, datasets, planner, generate
from .cache import project_path
from .history import new_run_id
from .thresholds import DEFAULTS, policy_provenance, resolve_thresholds, validate_primary
from .uploads import Uploads, LIMITS as UPLOAD_LIMITS
from .relational_uploads import RelationalUploads

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("web_assets")
MAX_BODY = 128_000
DOC_NAV = (
    ("Start here", ("README.md", "what-quibble-does.md", "about.md", "getting-started.md", "experiment-workbench.md", "concepts.md")),
    ("Run experiments", ("testing-your-data.md", "datasets.md", "backends.md",
                         "configuration.md", "running-benchmarks.md", "gpu-and-docker.md")),
    ("Understand results", ("reports.md", "analysis.md", "swap.md", "results-format.md")),
    ("Reference & support", ("cli-reference.md", "troubleshooting.md", "limitations.md",
                             "index.md", "development.md", "project-plan.md")),
    ("Ideas & rationale", ("rationale.md", "recommend-abstain.md")),
    ("Blog", ("blog/README.md", "blog/01-same-accuracy-different-mistakes.md",
              "blog/02-making-prediction-comparisons-worth-trusting.md")),
)
DOCS = Path(__file__).with_name("docs")
if not DOCS.is_dir():
    DOCS = Path(__file__).resolve().parents[1] / "docs"


def render_report(markdown, identifier, figures, *, local_links=None, local_images=None):
    try:
        import bleach
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise ValueError("Install the [web] extra to render reports (uv sync --extra web).") from exc
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = renderer.parse(markdown)
    anchors = set()
    for index, block in enumerate(tokens):
        if block.type == "heading_open":
            slug = re.sub(r"[^\w -]", "", tokens[index + 1].content.lower()).replace(" ", "-") or "section"
            anchor, suffix = slug, 0
            while anchor in anchors:
                suffix += 1
                anchor = f"{slug}-{suffix}"
            anchors.add(anchor)
            block.attrSet("id", anchor)
    for block in tokens:
        for token in block.children or []:
            if token.type == "image":
                source = token.attrGet("src")
                if local_images and source in local_images:
                    token.attrSet("src", local_images[source])
                elif source in figures:
                    token.attrSet("src", f"/api/jobs/{quote(identifier, safe='')}/figures/{quote(source, safe='')}")
                else:
                    token.type, token.tag, token.attrs, token.children = "text", "", {}, None
            elif token.type == "link_open":
                href = token.attrGet("href") or ""
                if local_links and href in local_links:
                    token.attrSet("href", local_links[href])
                    continue
                if not href.startswith("#") and urlsplit(href).scheme not in ("https", "http"):
                    token.attrs.pop("href", None)
                else:
                    token.attrSet("rel", "noreferrer noopener")
    html = renderer.renderer.render(tokens, renderer.options, {})
    return bleach.clean(
        html, tags={"p", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em",
                    "ul", "ol", "li", "blockquote", "pre", "code", "hr", "br",
                    "table", "thead", "tbody", "tr", "th", "td", "a", "img"},
        attributes={"a": ["href", "title", "rel"], "img": ["src", "alt", "title"],
                    **{f"h{level}": ["id"] for level in range(1, 7)},
                    "ol": ["start"]}, protocols={"http", "https"}, strip=True)


def documentation_files():
    paths = [*DOCS.glob("*.md"), *DOCS.glob("blog/*.md"),
             *DOCS.glob("assets/blog/*.png"), *DOCS.glob("assets/blog/*.svg"),
             *DOCS.glob("assets/blog/*.csv"), *DOCS.glob("assets/blog/*.json"),
             *DOCS.glob("assets/screenshots/*.png")]
    return {"/docs/" + path.relative_to(DOCS).as_posix(): path for path in paths}


def render_header(page):
    header = (ASSETS / "header.html").read_text(encoding="utf-8")
    header = header.replace(f'data-page="{page}"', f'data-page="{page}" aria-current="page"')
    if page == "workbench":
        for href in ("/blog", "/about", "/docs/README.md"):
            header = header.replace(f'href="{href}"',
                                    f'href="{href}" target="_blank" rel="noopener noreferrer"')
    return header


def document_title(path):
    match = re.search(r"^title:\s*(.+)$", path.read_text(encoding="utf-8"), flags=re.M)
    return match[1].strip().strip("\"'") if match else path.stem.replace("-", " ")


def document_navigation(route, files, content):
    groups, pages = [], []
    is_blog = route.startswith("/docs/blog/")
    navigation_label = "Blog series" if is_blog else "Documentation"
    current_group = navigation_label
    for group, filenames in DOC_NAV:
        if (group == "Blog") != is_blog:
            continue
        items = []
        for filename in filenames:
            target = "/docs/" + filename
            if target not in files:
                continue
            title = document_title(files[target])
            pages.append((target, title))
            current = ' aria-current="page"' if target == route else ""
            if current:
                current_group = group
            items.append(f'<li><a href="{quote(target, safe="/")}"{current}>{html.escape(title)}</a></li>')
        groups.append(f'<div class="docs-nav-group"><h2>{html.escape(group)}</h2>'
                      f'<ul>{"".join(items)}</ul></div>')
    sidebar = (f'<aside class="docs-sidebar"><details open><summary>{navigation_label} navigation</summary>'
               f'<nav aria-label="{navigation_label}">{"".join(groups)}</nav></details></aside>')
    headings = re.findall(r'<h([23]) id="([^"]+)">(.*?)</h\1>', content, flags=re.S)
    toc = []
    for level, anchor, label in headings:
        label = html.escape(html.unescape(re.sub(r"<[^>]+>", "", label)))
        toc.append(f'<li class="toc-level-{level}"><a href="#{anchor}">{label}</a></li>')
    on_page = ('<aside class="docs-toc"><nav aria-label="On this page"><h2>On this page</h2>'
               f'<ul>{"".join(toc)}</ul></nav></aside>') if toc else ""
    adjacent = []
    for index, (target, _) in enumerate(pages):
        if target != route:
            continue
        for offset, label, relation in ((-1, "Previous", "prev"), (1, "Next", "next")):
            if 0 <= index + offset < len(pages):
                destination, title = pages[index + offset]
                adjacent.append(f'<a rel="{relation}" href="{quote(destination, safe="/")}">'
                                f'<span>{label}</span>{html.escape(title)}</a>')
        break
    pagination = f'<nav class="docs-pagination" aria-label="Adjacent pages">{"".join(adjacent)}</nav>'
    return sidebar, on_page, pagination, current_group


def render_document(route, path, files, *, source_links=None):
    from markdown_it import MarkdownIt
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        text = text.split("\n---\n", 1)[1]
    links, images = {}, {}
    for block in MarkdownIt("commonmark").parse(text):
        for token in block.children or []:
            attribute = "src" if token.type == "image" else "href"
            source = token.attrGet(attribute)
            if not source:
                continue
            destination = urljoin(route, source)
            target = urlsplit(destination)
            if unquote(target.path) in files:
                (images if token.type == "image" else links)[source] = destination
            elif token.type != "image" and source_links and unquote(target.path) in source_links:
                links[source] = source_links[unquote(target.path)] + (
                    "#" + target.fragment if target.fragment else "")
    content = render_report(text, "", [], local_links=links, local_images=images)
    title = html.escape(document_title(path))
    sidebar, on_page, pagination, group = document_navigation(route, files, content)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title} | Quibble</title><script src="/theme.js"></script>'
            '<link rel="stylesheet" href="/style.css"></head>'
            '<body><a class="skip-link" href="#document-content">Skip to content</a>'
            f'{render_header("blog" if route.startswith("/docs/blog/") else "docs")}'
            f'<main class="docs-layout">{sidebar}<div class="docs-reading">'
            f'<nav class="docs-breadcrumb" aria-label="Breadcrumb">'
            f'<a href="{"/blog" if route.startswith("/docs/blog/") else "/docs/README.md"}">'
            f'{"Blog" if route.startswith("/docs/blog/") else "Docs"}</a>'
            f'<span aria-hidden="true"> / </span>{html.escape(group)}</nav>'
            f'<article id="document-content" class="report docs-article" tabindex="-1">{content}</article>'
            f'{pagination}</div>{on_page}</main></body></html>').encode("utf-8")


class Workbench:
    def __init__(self):
        from .config import load_dotenv
        load_dotenv()
        self.root = project_path(os.environ.get("QUIBBLE_OUT_DIR", "").strip() or "results") / "web"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease = (self.root / ".workbench.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if not os.fstat(self.lease.fileno()).st_size:
                    self.lease.write(b"0")
                    self.lease.flush()
                self.lease.seek(0)
                msvcrt.locking(self.lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lease.close()
            raise ValueError("Another workbench is using this results directory.") from exc
        self.lock = threading.Lock()
        self.thread = None
        self.active = None
        self.plans = {}
        self.previews = set()
        self.reviewed_sources = {}
        try:
            self.uploads = Uploads(self.root)
            self.relational_uploads = RelationalUploads(self.root)
            saved_registry = datasets.data_root() / "relational-registry.json"
            if datasets.relational_registry() is None and saved_registry.is_file():
                os.environ["QUIBBLE_RELATIONAL_REGISTRY"] = str(saved_registry.resolve())
            for path in self.root.glob("*/job.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] in ("queued", "running"):
                    record.update(status="interrupted", error="The web server stopped before a final status was recorded. Inspect the log before rerunning.")
                    self._write(path, record)
        except (OSError, ValueError, KeyError, TypeError):
            self.lease.close()
            raise

    @staticmethod
    def _write(path, value):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def state(self):
        from .backends import BACKENDS, select_backends
        from .provenance import backend_metadata
        from .config import cuda_available
        try:
            settings = planner.Settings.from_env()
            provider = {"configured": True, "model": settings.model, "name": settings.provider,
                        "endpoint": settings.endpoint,
                        "authentication": settings.mode, "credential": settings.credential}
        except planner.PlannerError as exc:
            provider = {"configured": False, "error": str(exc)}
        catalog, dataset_errors = [], []
        for name in sorted(set(datasets.BUILTIN) | set(datasets.local_names()) | set(datasets.relational_names())):
            try:
                X, y, task = datasets.load(name)
            except (ValueError, OSError, ImportError) as exc:
                dataset_errors.append({"name": name, "error": str(exc)[:700]})
                logger.warning("Dataset unavailable during catalog validation (%s).", type(exc).__name__)
                continue
            source = X.attrs.get("quibble_source")
            group = "Built-in benchmarks" if name in datasets.BUILTIN else (
                "Bundled synthetic examples" if source and
                source.get("path") and Path(source["path"]).is_relative_to(Path(__file__).with_name("sample_data")) else
                "Saved datasets")
            kind = "regression" if task == "regression" else (
                "binary classification" if y.nunique() == 2 else "multiclass classification")
            catalog.append({"name": name, "task": task, "kind": kind, "group": group})
        return {"datasets": [item["name"] for item in catalog], "dataset_catalog": catalog,
                "dataset_errors": dataset_errors,
                "backends": [n for n, b in BACKENDS.items() if not getattr(b, "internal", False)],
                "upload_limits": UPLOAD_LIMITS,
                "backend_metadata": {n: backend_metadata(n) for n, b in BACKENDS.items()
                                     if not getattr(b, "internal", False)},
                "cuda_available": cuda_available(),
                "cuda_exclusions": select_backends(list(BACKENDS), "cuda")[1],
                "provider": provider, "limits": generate.LIMITS,
                "threshold_defaults": dict(DEFAULTS),
                "jobs": [json.loads(p.read_text(encoding="utf-8"))
                         for p in sorted(self.root.glob("*/job.json"), reverse=True)][:30]}

    def _remember(self, result):
        token = secrets.token_hex(16)
        # Plans are server-owned; the browser can edit the spec but not attach fake planner provenance.
        self.plans[token] = result["planner"]
        while len(self.plans) > 100:
            del self.plans[next(iter(self.plans))]
        return {**result, "plan_id": token}

    @staticmethod
    def _signature(spec, plan_id):
        return json.dumps([spec, plan_id], sort_keys=True)

    def action(self, route, body):
        if route == "/api/environment":
            return self.environment(body)
        if route in ("/api/verify/swap", "/api/verify/gate"):
            return self.verify(route.rsplit("/", 1)[1], body)
        if route == "/api/compare-jobs":
            return self.compare_jobs(body)
        if route in ("/api/relational/start", "/api/relational/review",
                     "/api/relational/confirm", "/api/relational/cancel"):
            method = route.rsplit("/", 1)[1]
            result = getattr(self.relational_uploads, method)(body)
            if method == "confirm":
                self.reviewed_sources[result["name"]] = result["fingerprint"]
            return result
        if route in ("/api/uploads/review", "/api/uploads/confirm", "/api/uploads/cancel"):
            method = route.rsplit("/", 1)[1]
            result = getattr(self.uploads, method)(body)
            if method == "confirm":
                self.reviewed_sources[result["name"]] = result["fingerprint"]
            return result
        if route == "/api/inspect":
            name = body["dataset"]
            if not isinstance(name, str) or name not in set(datasets.BUILTIN) | set(datasets.local_names()) | set(datasets.relational_names()):
                raise ValueError("Choose an existing dataset from the list.")
            X, y, task = datasets.load(name)
            source = X.attrs.get("quibble_source")
            target = source.get("target", "outcome") if source else "outcome"
            frame = X.copy()
            labels = source.get("class_labels") if source else None
            displayed_y = y.map(dict(enumerate(labels))) if labels else y
            frame[target] = displayed_y
            metadata = None
            if source and source.get("synthetic"):
                metadata = json.loads((Path(source["path"]).parent / "generation.json").read_text(encoding="utf-8"))
            return {"name": name, "task": task, "target": target,
                    "rows": len(X), "predictors": X.shape[1], "columns": list(frame.columns),
                    "sample": json.loads(frame.head(20).to_json(orient="records")),
                    "missing": {column: int(frame[column].isna().sum()) for column in frame},
                    "source": source, "generation": metadata,
                    "relational": X.attrs.get("quibble_relational", {}).get("audit"),
                    "fingerprint": source["fingerprint"] if source else f"builtin:{name}",
                    "target_summary": {str(k): int(v) for k, v in displayed_y.value_counts().items()}
                    if task == "classification" else
                    {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())}}
        if route == "/api/use-existing":
            if body.get("reviewed") is not True:
                raise ValueError("Review the existing sample and confirm before using it.")
            result = self.action("/api/inspect", body)
            if result["fingerprint"] != body.get("fingerprint"):
                raise ValueError("Dataset changed since review. Inspect it again before continuing.")
            self.reviewed_sources[result["name"]] = result["fingerprint"]
            return {"name": result["name"], "fingerprint": result["fingerprint"]}
        if route == "/api/edit-existing":
            result = self.action("/api/inspect", body)
            if not result["generation"]:
                raise ValueError("Only generated datasets have a generation specification to edit.")
            return self._remember({"spec": result["generation"]["specification"],
                                   "planner": {"derived_from": result["name"],
                                               "source_fingerprint": result["fingerprint"],
                                               "original_planner": result["generation"].get("planner")}})
        if route == "/api/recommend":
            spec = generate.recommendation(body["rows"], body["fields"], body["seed"], body["task"])
            return self._remember({"spec": spec, "planner": None})
        if route == "/api/plan":
            if body.get("consent") is not True:
                raise ValueError("Confirm that your scenario text may be sent to the configured planning provider.")
            return self._remember(planner.plan(body["scenario"], body["rows"], body["fields"],
                                              body["seed"], body["task"]))
        if route in ("/api/preview", "/api/save"):
            plan_id = body["plan_id"]
            if not isinstance(plan_id, str) or plan_id not in self.plans:
                raise ValueError("Create a plan or recommendation in this server session first.")
            spec = body["spec"]
            signature = self._signature(spec, plan_id)
            if route == "/api/preview":
                result = generate.preview(spec)
                self.previews.add(signature)
                if len(self.previews) > 100:
                    self.previews = {signature}
                return result
            if body.get("reviewed") is not True or signature not in self.previews:
                raise ValueError("Preview the current specification and confirm review before saving.")
            provenance = self.plans[plan_id]
            return generate.save(spec, {"planning": provenance, "reviewed_specification": True})
        if route == "/api/run":
            return self.start(body)
        raise ValueError("Unknown action")

    def start(self, body):
        from .backends import BACKENDS, select_backends
        from .config import resolve_device
        from .provenance import backend_metadata
        from .tuning import DOMAINS, TREES, protocol
        from types import SimpleNamespace
        name, methods = body["dataset"], body["backends"]
        if name not in set(datasets.BUILTIN) | set(datasets.local_names()) | set(datasets.relational_names()):
            raise ValueError("Choose a built-in or local dataset from the list.")
        if not isinstance(methods, list) or not 1 <= len(methods) <= 4 \
                or any(not isinstance(n, str) or n not in BACKENDS for n in methods) \
                or len(set(methods)) != len(methods):
            raise ValueError("Choose 1-4 distinct available prediction methods.")
        if body.get("device") not in ("cpu", "cuda", "auto") \
                or body.get("mode") not in ("smoke", "full") \
                or body.get("cv") not in ("holdout", "3x1", "3x2"):
            raise ValueError("Invalid device, mode, or split scheme.")
        if body.get("mode") == "full" and body.get("full_confirmed") is not True:
            raise ValueError("Confirm the full comparison; selected methods may download weights or take substantial time.")
        tier = body.get("tier", "defaults")
        if tier not in ("defaults", "tuned"):
            raise ValueError("Invalid comparison tier.")
        tuning = {"tune_trials": body.get("tune_trials", 20),
                  "tune_folds": body.get("tune_folds", 3),
                  "tune_seed": body.get("tune_seed", 0),
                  "tune_metric": body.get("tune_metric", "auto"),
                  "tune_timeout_s": body.get("tune_timeout_s", 0)}
        if tier == "defaults":
            tuning = {"tune_trials": 20, "tune_folds": 3, "tune_seed": 0,
                      "tune_metric": "auto", "tune_timeout_s": 0}
        for key, low, high in (("tune_trials", 1, 1000), ("tune_folds", 2, 20),
                                ("tune_seed", 0, 2**32 - 1)):
            if type(tuning[key]) is not int or not low <= tuning[key] <= high:
                raise ValueError(f"{key} must be an integer in {low}..{high}.")
        import math
        deadline = tuning["tune_timeout_s"]
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) \
                or not 0 <= deadline <= sys.float_info.max:
            raise ValueError("tune_timeout_s must be finite and nonnegative.")
        if tuning["tune_metric"] not in ("auto", "log_loss", "rmse"):
            raise ValueError("Invalid tuning selection metric.")
        if tier == "tuned" and body.get("tuning_confirmed") is not True:
            raise ValueError("Explicit tuning approval is required for this exact configuration.")
        overrides = body.get("thresholds")
        if body["mode"] == "smoke" and overrides is not None:
            raise ValueError("Smoke runs do not use swap thresholds.")
        effective = resolve_thresholds(overrides, require=overrides is not None)
        if self.thread is not None and self.thread.is_alive():
            raise ValueError("An experiment is already running. Wait for its completion.")
        try:
            resolved = resolve_device(body["device"])
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        included, excluded = select_backends(methods, resolved)
        if not included:
            raise ValueError("No selected methods support GPU-only execution. "
                             "Select CatBoost, TabPFN, TabFM or KumoTabular, or choose CPU.")
        if tier == "tuned":
            from .provenance import WEIGHTS
            if not set(included) & set(TREES):
                raise ValueError("Tuned tier requires at least one enabled supported tree.")
            if set(included) - set(TREES) - set(WEIGHTS):
                raise ValueError("Tuned tier supports tree search plus unchanged foundation recipes only.")
            for method in set(included) & set(TREES):
                size = math.prod(len(values) for values in DOMAINS[method].values())
                if tuning["tune_trials"] > size:
                    raise ValueError(f"{method} has only {size} finite candidates; reduce tune_trials.")
        # Validate source and split feasibility before starting expensive subprocess work.
        X, y, task = datasets.load(name)
        source = X.attrs.get("quibble_source")
        if source and (source.get("uploaded") or source.get("kind") == "relational") and (
                body.get("fingerprint") != source["fingerprint"] or
                self.reviewed_sources.get(name) != source["fingerprint"]):
            raise ValueError("Uploaded dataset changed or was not reviewed in this session. Review it again.")
        if source and source.get("kind") == "relational":
            if tier == "tuned":
                raise ValueError("Relational tuning requires horizon-safe inner folds; choose defaults.")
            if body["cv"] != "holdout" or resolved != "cpu" or any(
                    method not in {"histgb", "lightgbm", "catboost"} for method in methods):
                raise ValueError("Relational browser baseline supports CPU local trees and official holdout only; no native route.")
        validate_primary(effective, task)
        if tier == "tuned" and tuning["tune_metric"] not in (
                "auto", "rmse" if task == "regression" else "log_loss"):
            raise ValueError("Tuning selection metric is incompatible with this dataset task.")
        from .config import parse_cv
        cv = "holdout" if body["mode"] == "smoke" else body["cv"]
        if cv == "holdout":
            split_ids = (int(os.getenv("QUIBBLE_SPLIT_SEED", "0")),)
        else:
            k, repeats = parse_cv(cv)
            split_ids = range(k * repeats)
        for split_id in split_ids:
            Xtr, _, ytr, _, _ = datasets.split(
                X, y, split_id, task, cv, int(os.getenv("QUIBBLE_CV_SEED", "0")))
            if tier == "tuned":
                from .tuning import inner_splits
                if body["mode"] == "smoke":
                    from .run import smoke_positions
                    positions = smoke_positions(ytr, 128, int(os.getenv("QUIBBLE_CV_SEED", "0"))
                                                + split_id, task)
                    Xtr, ytr = datasets.subset_rows(Xtr, positions), ytr.iloc[positions]
                inner_splits(Xtr, ytr, task, tuning["tune_folds"], tuning["tune_seed"])
        identifier = new_run_id()
        directory = self.root / identifier
        directory.mkdir()
        command = [sys.executable, "-m", "quibble_ml", "benchmark", "--datasets", name,
                   "--backends", ",".join(methods), "--device", body["device"], "--jobs", "1",
                   "--model-seeds", "0", "--out", str(directory / "output")]
        # Always pin the tier and search settings, ignoring inherited tuning environment.
        command += ["--tier", tier]
        for key, value in tuning.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        if body["mode"] == "smoke":
            command.append("--smoke")
        else:
            command += ["--cv", body["cv"]]
            self._write(directory / "thresholds.json", effective)
            if overrides is not None:
                self._write(directory / "threshold_overrides.json", overrides)
                command += ["--thresholds", str(directory / "threshold_overrides.json"), "--require-thresholds"]
        record = {"id": identifier, "dataset": name, "backends": methods, "device": body["device"],
                  "backend_metadata": {n: backend_metadata(n) for n in methods},
                  "dataset_source": source,
                  "device_resolved": resolved, "included_backends": included, "excluded_backends": excluded,
                  "mode": body["mode"], "status": "queued", "exit_code": None,
                  "full_confirmed": body["mode"] == "full" and body.get("full_confirmed") is True,
                  "tier": tier, "campaign_tier": tier, **tuning,
                  "tuning_confirmed": tier == "tuned" and body.get("tuning_confirmed") is True,
                  "tuning_protocol": protocol(SimpleNamespace(**tuning)) if tier == "tuned"
                                     else {"status": "not requested"},
                  "thresholds": effective if body["mode"] == "full" else None,
                  "thresholds_default": not overrides if body["mode"] == "full" else None,
                  "threshold_provenance": policy_provenance(overrides) if body["mode"] == "full" else None,
                  "report_available": False, "error": None}
        self._write(directory / "job.json", record)
        self.active = identifier
        self.thread = threading.Thread(target=self._run, args=(directory, command, record),
                                       name="quibble-experiment", daemon=False)
        self.thread.start()
        return record.copy()

    def _run(self, directory, command, record):
        try:
            record["status"] = "running"
            self._write(directory / "job.json", record)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["_QUIBBLE_RESOLVED_DEVICE"] = record["device_resolved"]
            if (record.get("dataset_source") or {}).get("uploaded") or (
                    record.get("dataset_source") or {}).get("kind") == "relational":
                env["_QUIBBLE_REVIEWED_SOURCE"] = json.dumps({
                    "dataset": record["dataset"], "fingerprint": record["dataset_source"]["fingerprint"]})
            with (directory / "run.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        cwd=Path(__file__).resolve().parents[1], check=False)
            reports = list((directory / "output").glob("report/report.md")) \
                + list((directory / "output" / "smoke").glob("*/report/report.md"))
            record.update(exit_code=result.returncode, report_available=bool(reports),
                          status="completed" if result.returncode == 0 else "failed",
                          error=None if result.returncode == 0 else
                          "Some requested work did not complete successfully. Read the log and any partial report.")
        except (OSError, ValueError):
            logger.exception("Experiment execution failed")
            record.update(status="failed", error="Experiment process or result storage failed. Check the server terminal and job log.")
        finally:
            try:
                self._write(directory / "job.json", record)
            except OSError:
                logger.exception("Unable to persist final experiment status")

    def job(self, identifier):
        if not isinstance(identifier, str) or not identifier \
                or any(c not in "0123456789-abcdef" for c in identifier):
            raise ValueError("Invalid experiment identifier")
        directory = self._contained(self.root / identifier, self.root)
        record_path = self._contained(directory / "job.json", directory)
        if not record_path.is_file():
            raise ValueError("Unknown experiment")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["directory"] = str(directory)
        log_path = self._contained(directory / "run.log", directory)
        if log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 24000))
                record["log"] = stream.read().decode("utf-8", errors="replace")
        else:
            record["log"] = "Experiment is queued."
        reports = list((directory / "output").glob("report/report.md")) \
            + list((directory / "output" / "smoke").glob("*/report/report.md"))
        for path in reports:
            self._contained(path, directory)
        record["report"] = reports[0].read_text(encoding="utf-8") if reports else None
        record["figures"] = [p.name for p in reports[0].parent.glob("fig_*.png")] if reports else []
        record["report_html"] = render_report(record["report"], identifier, record["figures"]) if reports else None
        record["report_sections"] = []
        if record["report_html"]:
            for level, anchor, title in re.findall(
                    r'<h([1-6]) id="([^"]+)">(.*?)</h\1>', record["report_html"], re.S):
                record["report_sections"].append({
                    "id": anchor, "level": int(level),
                    "title": html.unescape(re.sub(r"<[^>]+>", "", title))})
        verification_dir = directory / "verifications"
        record["verifications"] = []
        for path in sorted(verification_dir.glob("*.json"), reverse=True)[:30]:
            self._contained(path, directory)
            record["verifications"].append(json.loads(path.read_text(encoding="utf-8")))
        return record

    @staticmethod
    def _contained(path, base):
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("Evidence path leaves its saved job directory.")
        return path

    def environment(self, body):
        """Explicit, deliberately smaller than CLI doctor: no paths, env, host or GPU probes."""
        if body != {"requested": True}:
            raise ValueError("Explicitly request local diagnostics.")
        from . import env_info
        from .backends import BACKENDS
        from .provenance import backend_metadata
        return {"collected_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Package inventory only; no GPU initialization, credentials, paths or connectivity checks.",
                "python": sys.version.split()[0], "os": sys.platform,
                "packages": {name: env_info._ver(name) for name in env_info.PKGS},
                "backends": {name: backend_metadata(name) for name, backend in BACKENDS.items()
                             if not getattr(backend, "internal", False)}}

    def _evidence(self, identifier):
        """Read-only preflight. Unknown or partial provenance can never become approval."""
        import numpy as np
        from .backends import BACKENDS
        from .pool import load_pool, coverage_status
        from .provenance import validate_recipe, digest
        job = self.job(identifier)
        if job.get("id") != identifier:
            raise ValueError("Saved job identity does not match its directory.")
        directory = self._contained(self.root / identifier, self.root)
        out = self._contained(directory / "output", directory)
        for path in out.rglob("*"):
            self._contained(path, directory)
        name, methods = job.get("dataset"), job.get("backends")
        if name not in set(datasets.BUILTIN) | set(datasets.local_names()) | set(datasets.relational_names()):
            raise ValueError("Saved source is not an available offline dataset.")
        if not isinstance(methods, list) or not methods or any(n not in BACKENDS for n in methods):
            raise ValueError("Saved job has unsupported method identities.")
        issues, pools, records, coverage = [], {}, {}, {}
        if job.get("mode") != "full":
            issues.append("Smoke evidence is a setup check, not full evaluation.")
        if job.get("status") != "completed":
            issues.append("Execution is not completed.")
        invocation_path = out / "invocation.json"
        invocation = json.loads(invocation_path.read_text(encoding="utf-8")) if invocation_path.is_file() else {}
        cells = invocation.get("cells", [])
        if not cells or invocation.get("excluded_backends") or job.get("excluded_backends"):
            issues.append("Required invocation coverage is absent or includes excluded methods.")
        if set(invocation.get("requested_backends", invocation.get("backends", []))) != set(methods):
            issues.append("Saved job methods differ from the complete requested invocation.")
        if any(c.get("dataset") != name or c.get("backend") not in methods or
               c.get("model_seed") != 0 for c in cells):
            issues.append("Invocation includes unsupported source, method or model-seed cells.")
        if job.get("mode") == "full":
            try:
                policy = resolve_thresholds(job.get("thresholds"), require=True)
                policy_path = self._contained(directory / "thresholds.json", directory)
                if not policy_path.is_file() or json.loads(policy_path.read_text(encoding="utf-8")) != policy:
                    issues.append("Saved job policy and policy artifact differ or are missing.")
            except ValueError:
                policy = None
                issues.append("Saved acceptance policy is missing or invalid.")
        else:
            policy = None
        X, current_y, task = datasets.load(name)
        source = X.attrs.get("quibble_source")
        try:
            datasets.validate_sources(out, name, source)
        except ValueError:
            issues.append("Current source fingerprint differs from saved evidence.")
        if (job.get("dataset_source") or {}).get("fingerprint") != (source or {}).get("fingerprint"):
            issues.append("Job source fingerprint differs from the available source.")
        for method in methods:
            try:
                pool, runs = load_pool(out, name, method, 0)
                required = {c["split_seed"] for c in cells if c.get("backend") == method
                            and c.get("dataset") == name and c.get("model_seed") == 0}
                cover = coverage_status(required, pool, runs)
                coverage[method] = {key: cover[key] for key in ("required", "completed", "missing")}
                if not required or cover["missing"] or set(pool) != required:
                    issues.append(f"{method}: incomplete required prediction coverage.")
                for run in runs:
                    if "error" in run or "refused" in run:
                        issues.append(f"{method}: failed or refused cell.")
                        continue
                    recipe = validate_recipe(run.get("recipe"))
                    if recipe.get("resolved", {}).get("status") != "fitted":
                        raise ValueError("Recipe is not fitted.")
                    request = recipe["requested"]
                    if request["backend"] != method or request["cell"] != {
                            "split_seed": run["split_seed"], "model_seed": 0}:
                        raise ValueError("Recipe cell differs.")
                    if run.get("dataset") != name or run.get("model_seed") != 0:
                        raise ValueError("Recorded source or model seed differs.")
                    if request.get("run_type") == "smoke" or run.get("task") != task:
                        raise ValueError("Task or smoke recipe differs.")
                    if task == "classification" and run.get("prediction_classes") != list(range(current_y.nunique())):
                        raise ValueError("Prediction class meaning or order is incomplete.")
                    context = recipe["resolved"]["context"]
                    if context.get("dataset") != name or context.get("task") != task \
                            or context.get("dataset_fingerprint") != (source or {}).get("fingerprint") \
                            or context.get("split") != run.get("split"):
                        raise ValueError("Resolved recipe source or split context differs.")
                    saved_split = pool.get(run["split_seed"])
                    if saved_split is not None and recipe["resolved"]["context"].get(
                            "test_indices_sha256") != digest(saved_split["idx"].tolist()):
                        raise ValueError("Recipe test row identities differ from saved arrays.")
                for data in pool.values():
                    idx = data["idx"]
                    if idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= len(current_y)) \
                            or not np.array_equal(data["y"], current_y.iloc[idx].to_numpy()) \
                            or len(data["P"]) != len(idx) or not np.isfinite(data["P"]).all():
                        raise ValueError("Saved labels or row identities differ from the source.")
                pools[method], records[method] = pool, runs
            except (ValueError, KeyError, TypeError, OSError):
                issues.append(f"{method}: invalid recipe, labels, rows or prediction artifacts.")
        # A digest binds all local inputs, without exposing raw source paths or private rows.
        import hashlib
        evidence_hash = hashlib.sha256()
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".npy"):
                evidence_hash.update(path.relative_to(out).as_posix().encode())
                evidence_hash.update(path.read_bytes())
        summary = {key: job.get(key) for key in (
            "id", "dataset", "mode", "status", "backends", "device", "device_resolved",
            "tier", "campaign_tier")}
        summary.update(task=task, source_fingerprint=(job.get("dataset_source") or {}).get("fingerprint"),
                       policy=policy, policy_sha256=digest(policy), coverage=coverage,
                       issues=list(dict.fromkeys(issues)), evidence_sha256=evidence_hash.hexdigest())
        provenance = job.get("threshold_provenance") or {}
        summary["policy_provenance"] = {key: provenance[key] for key in ("kind", "inherited", "provided")
                                        if key in provenance}
        summary["cells"] = [
            {"backend": method, "split_seed": run.get("split_seed"), "model_seed": run.get("model_seed"),
             "device": run.get("device_effective", run.get("device")),
             "recipe_sha256": (run.get("recipe") or {}).get("full_sha256"),
             "evaluation": {key: (run.get("recipe") or {}).get("requested", {}).get(key)
                            for key in ("cv", "cv_seed", "preprocess", "run_type", "tier")},
             "class_order_sha256": digest(run.get("prediction_classes")),
             "metrics": {key: run[key] for key in ("accuracy", "log_loss", "rmse", "mae", "auc")
                         if isinstance(run.get(key), (int, float))}}
            for method, runs in records.items() for run in runs]
        return summary, pools, records, out

    def verify(self, kind, body):
        from .provenance import digest
        allowed = {"job_id", "incumbent", "candidate"} | ({"min_gain"} if kind == "gate" else set())
        if set(body) - allowed:
            raise ValueError("Verification uses the immutable saved policy; unsupported request fields.")
        if self.thread is not None and self.thread.is_alive():
            raise ValueError("Wait for the active experiment before checking saved evidence.")
        summary, pools, records, out = self._evidence(body.get("job_id"))
        incumbent, candidate = body.get("incumbent"), body.get("candidate")
        if incumbent == candidate or any(name not in summary["backends"] for name in (incumbent, candidate)):
            raise ValueError("Choose two distinct methods recorded in this job.")
        if summary["issues"]:
            raise ValueError("Saved evidence cannot support verification: " + " ".join(summary["issues"]))
        from .pool import align
        align([pools[incumbent], pools[candidate]])
        gain = body.get("min_gain", 0.0)
        import math
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain < 0:
            raise ValueError("Baseline minimum gain must be finite and nonnegative.")
        # Separate, time-bounded analysis process: no fitting, provider calls or model loading.
        request = {"kind": kind, "dataset": summary["dataset"], "incumbent": incumbent,
                   "candidate": candidate, "out": str(out), "thresholds": summary["policy"],
                   "min_gain": gain}
        code = (
            "import json,sys; from quibble_ml import swap,gate; "
            "a=json.loads(sys.argv[1]); "
            "r=swap.evaluate(a['dataset'],a['incumbent'],a['candidate'],out=a['out'],"
            "thresholds=a['thresholds'],require_thresholds=True) if a['kind']=='swap' else "
            "gate.evaluate(a['dataset'],a['candidate'],baseline=a['incumbent'],out=a['out'],"
            "min_gain=a['min_gain'],metric=a['thresholds']['primary']); "
            "print(json.dumps(r,allow_nan=False))")
        artifact = {"id": new_run_id(), "kind": kind, "job_id": summary["id"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "inputs": {key: request[key] for key in ("incumbent", "candidate", "min_gain")},
                    "policy": summary["policy"], "policy_sha256": digest(summary["policy"]),
                    "policy_provenance": summary["policy_provenance"],
                    "policy_replay": "Saved effective policy supplied explicitly. Evaluator policy-origin flags describe this replay; policy_provenance describes the original selection.",
                    "coverage": summary["coverage"], "evidence_sha256": summary["evidence_sha256"],
                    "caveats": ["Saved-evidence analysis only; no models fitted.",
                                "PASS is not production-safety approval."]}
        try:
            completed = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(request)],
                                       capture_output=True, text=True, timeout=120, check=False,
                                       cwd=Path(__file__).resolve().parents[1])
            if completed.returncode:
                raise ValueError("Scientific evaluator rejected these artifacts.")
            result = json.loads(completed.stdout)
            if result.get("overall") not in ("PASS", "FAIL", "INCONCLUSIVE"):
                raise ValueError("Unknown scientific verdict.")
            if kind == "gate":
                result["settings"] = "Recorded model recipes; inspect each method's saved tier and search provenance."
            artifact.update(status="completed", result=result)
        except (ValueError, OSError, subprocess.TimeoutExpired):
            artifact.update(status="error", error="Saved-evidence analysis failed or exceeded 120 seconds. No scientific verdict.")
        directory = self._contained(self.root / summary["id"] / "verifications", self.root / summary["id"])
        directory.mkdir(exist_ok=True)
        self._write(directory / (artifact["id"] + ".json"), artifact)
        return artifact

    def compare_jobs(self, body):
        import numpy as np
        from .provenance import digest
        identifiers = body.get("job_ids")
        if set(body) != {"job_ids"} or not isinstance(identifiers, list) or len(identifiers) != 2 \
                or identifiers[0] == identifiers[1]:
            raise ValueError("Select exactly two distinct saved job IDs.")
        evidence = [self._evidence(identifier) for identifier in identifiers]
        left, right = [item[0] for item in evidence]
        differences = []
        root_compatible = all(left.get(key) == right.get(key) for key in ("dataset", "task", "source_fingerprint"))
        if not root_compatible:
            differences.append("Different sources or targets: configuration/report comparison only; no common quality ranking.")
        if not left["source_fingerprint"] or not right["source_fingerprint"]:
            differences.append("Source fingerprint is unrecorded; compatibility is unknown.")
        for key in ("mode", "backends", "device_resolved", "campaign_tier", "policy_sha256"):
            if left.get(key) != right.get(key):
                differences.append(f"Different {key}.")
        if left["issues"] or right["issues"]:
            differences.append("Incomplete or invalid evidence; no approval or paired claim.")
        pair_validated = []
        if root_compatible and not left["issues"] and not right["issues"]:
            shared = set(evidence[0][1]) & set(evidence[1][1])
            for method in sorted(shared):
                a, b = evidence[0][1][method], evidence[1][1][method]
                if set(a) != set(b) or any(
                        not np.array_equal(a[s][key], b[s][key]) for s in set(a) & set(b)
                        for key in ("idx", "y")):
                    differences.append(f"{method}: incompatible test splits, rows or labels.")
                    continue
                ra, rb = evidence[0][2][method], evidence[1][2][method]
                if digest([r["recipe"]["requested"] for r in ra]) != digest([r["recipe"]["requested"] for r in rb]):
                    differences.append(f"{method}: different recipes or execution settings.")
                if {left.get("campaign_tier"), right.get("campaign_tier")} == {"defaults", "tuned"}:
                    from .tuning import TREES, load_tier_pair
                    if method in TREES:
                        defaults, tuned = (evidence if left["campaign_tier"] == "defaults" else evidence[::-1])
                        try:
                            for split in a:
                                load_tier_pair(defaults[3], tuned[3], left["dataset"], method, split, 0)
                            pair_validated.append(method)
                        except (ValueError, OSError, KeyError):
                            differences.append(f"{method}: cross-tier pairing refused.")
        if not root_compatible:
            for summary in (left, right):
                for cell in summary["cells"]:
                    cell.pop("metrics", None)
        return {"jobs": [left, right], "compatibility": (
                    "incompatible" if not root_compatible else "unknown" if
                    not left["source_fingerprint"] or left["issues"] or right["issues"] else
                    "differences" if differences else "matched"),
                "differences": differences, "validated_cross_tier_methods": pair_validated,
                "presentation": "descriptive only" if root_compatible else "configuration only",
                "paired_verdict": None,
                "caveat": "No winner, equivalence or paired statistical verdict is inferred from side-by-side scores."}

    def figure(self, identifier, name):
        record = self.job(identifier)
        if name not in record["figures"] or Path(name).name != name:
            raise ValueError("Unknown report figure")
        directory = self.root / identifier / "output"
        reports = list(directory.glob("report/report.md")) \
            + list((directory / "smoke").glob("*/report/report.md"))
        return self._contained(reports[0].parent / name, self.root / identifier).read_bytes()

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            print("Waiting for the active experiment to finish before closing. Per-fit timeouts still apply.")
            self.thread.join()
        if not self.lease.closed:
            self.lease.close()


class Server(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = os.name != "nt"

    def __init__(self, address, app):
        super().__init__(address, Handler)
        self.app = app
        self.token = secrets.token_urlsafe(32)
        self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class Handler(BaseHTTPRequestHandler):
    server: Server

    def setup(self):
        self.request.settimeout(15)
        super().setup()

    def log_message(self, format, *args):
        logger.info("%s %s", self.client_address[0], format % args)

    def respond(self, value, status=200, content_type="application/json; charset=utf-8"):
        payload = json.dumps(value, allow_nan=False).encode("utf-8") if content_type.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def host_allowed(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            self.respond({"error": "Only this loopback origin is allowed."}, 403)
            return False
        return True

    def do_GET(self):
        if not self.host_allowed():
            return
        route = urlsplit(self.path).path
        try:
            if route == "/api/state":
                return self.respond(self.server.app.state())
            if route.startswith("/api/jobs/"):
                parts = unquote(route.removeprefix("/api/jobs/")).split("/")
                if len(parts) == 3 and parts[1] == "figures":
                    return self.respond(self.server.app.figure(parts[0], parts[2]), content_type="image/png")
                if len(parts) != 1:
                    raise ValueError("Invalid experiment route")
                return self.respond(self.server.app.job(parts[0]))
            if route.startswith("/api/datasets/"):
                name = unquote(route.removeprefix("/api/datasets/"))
                if name not in datasets.local_names():
                    raise ValueError("Unknown local dataset")
                _, path = datasets.local_spec(name)
                mime = "text/csv; charset=utf-8" if path.suffix.lower() == ".csv" else "application/octet-stream"
                return self.respond(path.read_bytes(), content_type=mime)
            if route == "/blog" or route.startswith("/docs/"):
                route = "/docs/blog/README.md" if route == "/blog" else unquote(route)
                files = documentation_files()
                if route not in files:
                    return self.respond({"error": "Documentation page not found."}, 404)
                path = files[route]
                if path.suffix == ".md":
                    return self.respond(render_document(route, path, files),
                                        content_type="text/html; charset=utf-8")
                if path.suffix == ".json":
                    return self.respond(json.loads(path.read_text(encoding="utf-8")))
                mime = {".png": "image/png", ".svg": "image/svg+xml",
                        ".csv": "text/csv; charset=utf-8", ".json": "application/json; charset=utf-8"}
                return self.respond(path.read_bytes(), content_type=mime[path.suffix])
            files = {"/": ("about.html", "text/html; charset=utf-8"),
                     "/workbench": ("index.html", "text/html; charset=utf-8"),
                     "/about": ("about.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
                     "/landing.js": ("landing.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if route not in files:
                return self.respond({"error": "Not found"}, 404)
            name, content_type = files[route]
            payload = (ASSETS / name).read_bytes()
            if route in ("/", "/about", "/workbench"):
                payload = payload.replace(b"__PRODUCT_HEADER__",
                                          render_header("workbench" if route == "/workbench" else "about").encode("utf-8"))
            if route == "/workbench":
                payload = payload.replace(b"__CSRF_TOKEN__", self.server.token.encode("ascii"))
            elif route in ("/", "/about"):
                payload = payload.replace(b"__QUIBBLE_VERSION__", __version__.encode("ascii"))
            self.respond(payload, content_type=content_type)
        except (ValueError, OSError) as exc:
            logger.warning("Read failed: %s", exc)
            self.respond({"error": str(exc)}, 400)

    def do_POST(self):
        if not self.host_allowed():
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in self.server.allowed_hosts} \
                or not secrets.compare_digest(self.headers.get("X-Quibble-Token", ""), self.server.token):
            return self.respond({"error": "Invalid origin or session token. Reload the local page."}, 403)
        if urlsplit(self.path).path in ("/api/uploads", "/api/relational/table"):
            relational_table = urlsplit(self.path).path == "/api/relational/table"
            if self.headers.get("Content-Type") != "application/octet-stream" or self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                return self.respond({"error": "Use a bounded binary file upload."}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                maximum = 16_000_000 if relational_table else UPLOAD_LIMITS["file_bytes"]
                if not 0 < size <= maximum:
                    self.close_connection = True
                    return self.respond({"error": "Relational table must be 1-16,000,000 bytes."
                                         if relational_table else "Upload must be 1-50 MiB."}, 413)
                if not self.server.app.lock.acquire(blocking=False):
                    self.close_connection = True
                    return self.respond({"error": "Another operation is in progress. Please wait."}, 409)
                try:
                    if relational_table:
                        result = self.server.app.relational_uploads.receive(
                            self.headers.get("X-Relational-Stage", ""),
                            self.headers.get("X-Relational-Table", ""),
                            unquote(self.headers.get("X-Upload-Name", "")), self.rfile, size)
                    else:
                        result = self.server.app.uploads.receive(
                            self.rfile, size, unquote(self.headers.get("X-Upload-Name", "")))
                finally:
                    self.server.app.lock.release()
                return self.respond(result)
            except (ValueError, OSError) as exc:
                logger.warning("Local upload rejected (%s).", type(exc).__name__)
                self.close_connection = True
                return self.respond({"error": str(exc)}, 400)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json" \
                or self.headers.get("Transfer-Encoding"):
            return self.respond({"error": "Use a bounded JSON request."}, 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                raise ValueError("Request body must be 1-128000 bytes")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if not self.server.app.lock.acquire(blocking=False):
                return self.respond({"error": "Another operation is in progress. Please wait."}, 409)
            try:
                result = self.server.app.action(urlsplit(self.path).path, body)
            finally:
                self.server.app.lock.release()
            self.respond(result)
        except (ValueError, KeyError, TypeError, OSError, planner.PlannerError) as exc:
            if urlsplit(self.path).path.startswith("/api/uploads/"):
                logger.warning("Local upload review rejected (%s).", type(exc).__name__)
            else:
                logger.warning("Operation rejected: %s", exc)
            self.respond({"error": str(exc)}, 400)


def create_parser():
    parser = argparse.ArgumentParser(prog="quibble-ml web", description="Local-only dataset and experiment workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the workbench in your browser")
    parser.add_argument("--login", action="store_true", help="explicit Azure CLI sign-in before serving")
    parser.add_argument("--device-code", action="store_true", help="use device-code sign-in with --login")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.device_code and not args.login:
        parser.error("Choose a port in 1-65535; --device-code requires --login")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.login:
        previous = sys.argv
        try:
            sys.argv = ["quibble-ml auth"] + (["--device-code"] if args.device_code else [])
            if planner.main():
                return 1
        finally:
            sys.argv = previous
    app = None
    try:
        app = Workbench()
        with Server(("127.0.0.1", args.port), app) as server:
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"Quibble workbench: {url}\nLocal-only. Scenario planning sends your entered description to your configured provider only after consent.", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local workbench.")
    except (OSError, ValueError) as exc:
        logger.error("Workbench could not start: %s", exc)
        return 1
    finally:
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
