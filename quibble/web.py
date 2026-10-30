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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

from . import __version__, datasets, planner, generate
from .cache import project_path
from .history import new_run_id
from .thresholds import DEFAULTS, policy_provenance, resolve_thresholds, validate_primary

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("web_assets")
MAX_BODY = 128_000
DOC_NAV = (
    ("Start here", ("README.md", "about.md", "getting-started.md", "experiment-workbench.md", "concepts.md")),
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
        raise ValueError("Install requirements.txt or quibble-ml[web] to render reports.") from exc
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = renderer.parse(markdown)
    if local_links is not None:
        anchors = set()
        for index, block in enumerate(tokens):
            if block.type == "heading_open":
                slug = re.sub(r"[^\w -]", "", tokens[index + 1].content.lower()).replace(" ", "-")
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
        try:
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
        from .config import cuda_available
        try:
            settings = planner.Settings.from_env()
            provider = {"configured": True, "model": settings.model, "name": settings.provider,
                        "endpoint": settings.endpoint,
                        "authentication": settings.mode, "credential": settings.credential}
        except planner.PlannerError as exc:
            provider = {"configured": False, "error": str(exc)}
        catalog = []
        for name in sorted(set(datasets.BUILTIN) | set(datasets.local_names())):
            X, y, task = datasets.load(name)
            source = X.attrs.get("quibble_source")
            group = "Built-in benchmarks" if name in datasets.BUILTIN else (
                "Bundled synthetic examples" if source and
                Path(source["path"]).is_relative_to(Path(__file__).with_name("sample_data")) else
                "Saved datasets")
            kind = "regression" if task == "regression" else (
                "binary classification" if y.nunique() == 2 else "multiclass classification")
            catalog.append({"name": name, "task": task, "kind": kind, "group": group})
        return {"datasets": [item["name"] for item in catalog], "dataset_catalog": catalog,
                "backends": [n for n, b in BACKENDS.items() if not getattr(b, "internal", False)],
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
        if route == "/api/inspect":
            name = body["dataset"]
            if not isinstance(name, str) or name not in set(datasets.BUILTIN) | set(datasets.local_names()):
                raise ValueError("Choose an existing dataset from the list.")
            X, y, task = datasets.load(name)
            source = X.attrs.get("quibble_source")
            target = source["target"] if source else "outcome"
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
        name, methods = body["dataset"], body["backends"]
        if name not in set(datasets.BUILTIN) | set(datasets.local_names()):
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
        # Validate source and split feasibility before starting expensive subprocess work.
        X, y, task = datasets.load(name)
        validate_primary(effective, task)
        from .config import parse_cv
        cv = "holdout" if body["mode"] == "smoke" else body["cv"]
        if cv == "holdout":
            split_ids = (int(os.getenv("QUIBBLE_SPLIT_SEED", "0")),)
        else:
            k, repeats = parse_cv(cv)
            split_ids = range(k * repeats)
        for split_id in split_ids:
            datasets.split(X, y, split_id, task, cv, int(os.getenv("QUIBBLE_CV_SEED", "0")))
        identifier = new_run_id()
        directory = self.root / identifier
        directory.mkdir()
        command = [sys.executable, "-m", "quibble", "benchmark", "--datasets", name,
                   "--backends", ",".join(methods), "--device", body["device"], "--jobs", "1",
                   "--model-seeds", "0", "--out", str(directory / "output")]
        if body["mode"] == "smoke":
            command.append("--smoke")
        else:
            command += ["--cv", body["cv"]]
            self._write(directory / "thresholds.json", effective)
            if overrides is not None:
                self._write(directory / "threshold_overrides.json", overrides)
                command += ["--thresholds", str(directory / "threshold_overrides.json"), "--require-thresholds"]
        record = {"id": identifier, "dataset": name, "backends": methods, "device": body["device"],
                  "device_resolved": resolved, "included_backends": included, "excluded_backends": excluded,
                  "mode": body["mode"], "status": "queued", "exit_code": None,
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
        directory = self.root / identifier
        record_path = directory / "job.json"
        if not record_path.is_file():
            raise ValueError("Unknown experiment")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["directory"] = str(directory)
        log_path = directory / "run.log"
        if log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 24000))
                record["log"] = stream.read().decode("utf-8", errors="replace")
        else:
            record["log"] = "Experiment is queued."
        reports = list((directory / "output").glob("report/report.md")) \
            + list((directory / "output" / "smoke").glob("*/report/report.md"))
        record["report"] = reports[0].read_text(encoding="utf-8") if reports else None
        record["figures"] = [p.name for p in reports[0].parent.glob("fig_*.png")] if reports else []
        record["report_html"] = render_report(record["report"], identifier, record["figures"]) if reports else None
        return record

    def figure(self, identifier, name):
        record = self.job(identifier)
        if name not in record["figures"] or Path(name).name != name:
            raise ValueError("Unknown report figure")
        directory = self.root / identifier / "output"
        reports = list(directory.glob("report/report.md")) \
            + list((directory / "smoke").glob("*/report/report.md"))
        return (reports[0].parent / name).read_bytes()

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
            logger.warning("Operation rejected: %s", exc)
            self.respond({"error": str(exc)}, 400)


def create_parser():
    parser = argparse.ArgumentParser(prog="quibble web", description="Local-only dataset and experiment workbench")
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
            sys.argv = ["quibble auth"] + (["--device-code"] if args.device_code else [])
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
