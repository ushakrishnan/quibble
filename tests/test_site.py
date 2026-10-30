"""Static Pages exports use public assets and work below a repository URL prefix."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from quibble.site import build_site


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.urls.append(attrs[attribute])


@pytest.mark.parametrize("base_path", ["/quibble", "", "/nested/project"])
def test_static_site_has_no_broken_internal_links(tmp_path, base_path):
    output = build_site(tmp_path / "site", base_path)
    pages = {}
    for path in output.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert "__PRODUCT_HEADER__" not in text and "__QUIBBLE_VERSION__" not in text
        assert 'href="/api/' not in text and 'src="/api/' not in text
        assert 'src="/app.js"' not in text and "landing.js" not in text
        assert f'src="{base_path}/theme.js"' in text
        assert 'aria-label="Color theme"' in text
        parser = Links()
        parser.feed(text)
        pages[path] = parser
    for path, parser in pages.items():
        for value in parser.urls:
            target = urlsplit(value)
            if target.scheme or target.netloc:
                continue
            if not target.path:
                destination = path
            else:
                assert target.path.startswith(base_path + "/"), (path, value)
                relative = unquote(target.path[len(base_path) + 1:])
                destination = output / relative
            assert destination.is_file(), (path, value)
            if target.fragment and destination in pages:
                assert unquote(target.fragment) in pages[destination].anchors, (path, value)
    assert (output / ".nojekyll").is_file()
    assert (output / "theme.js").is_file()
    for name in ("workbench-choose-data.png", "workbench-configure-run.png", "workbench-results.png"):
        assert (output / "docs" / "assets" / "screenshots" / name).is_file()
        guide = (output / "docs" / "experiment-workbench.html").read_text(encoding="utf-8")
        assert f'src="{base_path}/docs/assets/screenshots/{name}"' in guide
    assert (output / "docs" / "assets" / "blog" / "evidence-stages.svg").is_file()
    article = output / "docs" / "blog" / "02-making-prediction-comparisons-worth-trusting.html"
    text = article.read_text(encoding="utf-8")
    assert "https://github.com/ushakrishnan/quibble/blob/main/quibble/datasets.py" in text
    assert "https://github.com/ushakrishnan/quibble/blob/main/quibble/swap_thresholds.json" in text
    assert 'aria-label="Blog series"' in text
    assert "GitHub Pages hosts these guides" in (output / "workbench" / "index.html").read_text()
    names = {path.name for path in output.rglob("*")}
    assert not {".env", "data.csv", "job.json", "FundamentalInfo", ".git"} & names


def test_static_build_preserves_existing_files(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        build_site(tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("base_path", ["https://example.com", "../project", "/../project",
                                     "/project//docs", "/project?query", "/project/<script>"])
def test_static_build_rejects_invalid_prefix(tmp_path, base_path):
    with pytest.raises(ValueError, match="Base path"):
        build_site(tmp_path / "site", base_path)
    assert not (tmp_path / "site").exists()
