"""Export the public documentation reader for static GitHub Pages hosting."""
import argparse
import html
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from . import __version__
from .web import ASSETS, DOCS, documentation_files, render_document, render_header

REPOSITORY = "https://github.com/ushakrishnan/quibble"


def static_path(route):
    if route == "/":
        return "index.html"
    if route in ("/about", "/blog", "/workbench"):
        return route.strip("/") + "/index.html"
    return route.lstrip("/").removesuffix(".md") + (".html" if route.endswith(".md") else "")


def rewrite_urls(page, base_path):
    def replace(match):
        attribute, value = match.groups()
        target = urlsplit(html.unescape(value))
        if target.scheme or target.netloc or not target.path.startswith("/"):
            return match[0]
        destination = base_path + "/" + quote(static_path(unquote(target.path)), safe="/")
        if target.query:
            destination += "?" + target.query
        if target.fragment:
            destination += "#" + target.fragment
        return f'{attribute}="{html.escape(destination, quote=True)}"'
    return re.sub(r'\b(href|src)="([^"]*)"', replace, page)


def build_site(output, base_path="/quibble"):
    base_path = base_path.rstrip("/")
    if base_path and (not re.fullmatch(r"/[A-Za-z0-9._/-]+", base_path)
                      or any(part in ("", ".", "..") for part in base_path[1:].split("/"))):
        raise ValueError("Base path must be empty or a URL path such as /quibble.")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Site output must be a new or empty directory; existing files are not removed.")
    output.mkdir(parents=True, exist_ok=True)
    files = documentation_files()
    source_root = DOCS.parent
    sources = [*source_root.glob("quibble_ml/*.py"), *source_root.glob("tests/*.py")]
    sources.extend(source_root / name for name in (
        "README.md", "LICENSE", ".env.example", "pyproject.toml", "uv.lock",
        "quibble_ml/swap_thresholds.json", "Dockerfile", "docker-compose.yml"))
    source_links = {
        "/" + path.relative_to(source_root).as_posix():
        REPOSITORY + "/blob/main/" + quote(path.relative_to(source_root).as_posix(), safe="/")
        for path in sources if path.is_file()
    }

    def write_page(route, page):
        destination = output / static_path(route)
        destination.parent.mkdir(parents=True, exist_ok=True)
        page = page.replace("LOCAL WORKBENCH", "DOCS &amp; BLOG")
        page = page.replace('<script src="/landing.js" defer></script>', "")
        page = page.replace(' target="_blank"', "")
        destination.write_text(rewrite_urls(page, base_path), encoding="utf-8")

    for route, path in files.items():
        if path.suffix == ".md":
            write_page(route, render_document(route, path, files, source_links=source_links).decode("utf-8"))
        else:
            destination = output / static_path(route)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    about = (ASSETS / "about.html").read_text(encoding="utf-8")
    about = about.replace("__PRODUCT_HEADER__", render_header("about"))
    about = about.replace("__QUIBBLE_VERSION__", __version__)
    about = about.replace(
        "If you opened this page from an experiment, switch back to its original tab to keep your unsaved work.",
        "This is the public documentation site. Experiments run in your own local Quibble environment.")
    write_page("/", about)
    write_page("/about", about)
    write_page("/blog", render_document(
        "/docs/blog/README.md", files["/docs/blog/README.md"], files, source_links=source_links).decode("utf-8"))
    workbench = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Run the workbench locally | Quibble</title>'
        '<script src="/theme.js"></script>'
        '<link rel="stylesheet" href="/style.css"></head><body>'
        '<a class="skip-link" href="#local-workbench">Skip to content</a>'
        + render_header("workbench")
        + '<main id="local-workbench" tabindex="-1"><section><h2>Run the workbench on your machine</h2>'
        '<p>GitHub Pages hosts these guides and articles, not the Python server or model execution. '
        'It does not upload data or run experiments.</p>'
        '<p><a href="/docs/getting-started.md">Follow the installation guide</a>, then run:</p>'
        '<pre><code>python -m quibble_ml web --open</code></pre>'
        '<p>Open the address printed by your local server, choose Open workbench, '
        'then review a ready-made dataset and start a smoke comparison.</p>'
        '</section></main></body></html>')
    write_page("/workbench", workbench)
    shutil.copyfile(ASSETS / "style.css", output / "style.css")
    shutil.copyfile(ASSETS / "theme.js", output / "theme.js")
    (output / ".nojekyll").touch()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build") / "pages")
    parser.add_argument("--base-path", default="/quibble")
    args = parser.parse_args(argv)
    try:
        output = build_site(args.out, args.base_path)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Site build failed: {exc}\n")
    print(f"Static documentation built at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
