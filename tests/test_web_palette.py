"""Keep the neutral palette and readable semantic colors consistent between themes."""
import re

import pytest

from quibble.web import ASSETS


def palette(theme):
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    selector = ":root" if theme == "light" else ':root[data-theme="dark"]'
    block = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert block is not None
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block[1]))


def rgb(color):
    color = color.removeprefix("#")
    if len(color) == 3:
        color = "".join(component * 2 for component in color)
    return tuple(int(color[index:index + 2], 16) / 255 for index in (0, 2, 4))


def luminance(color):
    components = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                  for value in rgb(color)]
    return sum(value * weight for value, weight in zip(components, (0.2126, 0.7152, 0.0722)))


def contrast(foreground, background):
    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_base_surfaces_and_text_are_neutral(theme):
    colors = palette(theme)
    for name in ("bg", "surface", "surface-2", "surface-3", "border", "border-strong",
                 "text", "text-2", "muted"):
        red, green, blue = rgb(colors["--" + name])
        assert red == green == blue, name
    assert "--accent" not in colors
    assert "--accent-soft" not in colors


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_and_controls_have_readable_contrast(theme):
    colors = palette(theme)
    pairs = [("text", "bg"), ("text", "surface"), ("text-2", "surface-2"),
             ("muted", "bg"), ("muted", "surface"), ("muted", "surface-2"),
             ("on-primary", "primary"), ("on-primary", "primary-hover"),
             ("primary-ink", "surface"), ("primary-ink", "primary-soft"),
             ("ok", "ok-soft"), ("bad", "bad-soft"), ("warn", "warn-soft")]
    for foreground, background in pairs:
        ratio = contrast(colors["--" + foreground], colors["--" + background])
        assert ratio >= 4.5, (theme, foreground, background, ratio)
    for background in ("surface", "surface-2"):
        assert contrast(colors["--border-strong"], colors["--" + background]) >= 3


def test_header_uses_theme_surfaces_instead_of_gradients():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    assert "gradient(" not in css
    header = re.search(r"\nheader\s*\{([^}]+)\}", css)
    assert header is not None
    assert "background: var(--surface)" in header[1]
    assert "color: var(--text)" in header[1]
    assert "color-scheme: dark; border-color:" not in css
