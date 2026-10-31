"""Keep the neutral palette and readable semantic colors consistent between themes."""
import re

import pytest

from quibble_ml.web import ASSETS


ACCENTS = ["blue", "teal", "indigo", "violet", "fuchsia", "slate"]


def palette(theme, accent=None, preference="standard"):
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    selector = ":root" if theme == "light" else ':root[data-theme="dark"]'
    block = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert block is not None
    colors = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block[1]))
    if accent:
        selectors = [f':root[data-accent="{accent}"]']
        if theme == "dark":
            selectors.append(f':root[data-theme="dark"][data-accent="{accent}"]')
        if preference == "high":
            selectors.append(':root[data-contrast="high"]')
            if theme == "dark":
                selectors.append(':root[data-theme="dark"][data-contrast="high"]')
        for selector in selectors:
            override = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
            assert override is not None, selector
            colors.update(re.findall(r"(--[\w-]+):\s*([^;]+);", override[1]))
    return colors


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
@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("preference", ["standard", "high"])
def test_text_and_controls_have_readable_contrast(theme, accent, preference):
    colors = palette(theme, accent, preference)
    pairs = [("text", "bg"), ("text", "surface"), ("text-2", "surface-2"),
             ("muted", "bg"), ("muted", "surface"), ("muted", "surface-2"),
             ("on-primary", "primary"), ("on-primary", "primary-hover"),
             ("primary-ink", "surface"), ("primary-ink", "primary-soft"),
             ("ok", "ok-soft"), ("bad", "bad-soft"), ("warn", "warn-soft")]
    surfaces = ("bg", "surface", "surface-2", "surface-3", "primary-soft")
    pairs += [(text, surface) for text in ("text", "text-2", "muted", "primary-ink")
              for surface in surfaces]
    for foreground, background in pairs:
        ratio = contrast(colors["--" + foreground], colors["--" + background])
        assert ratio >= 4.5, (theme, foreground, background, ratio)
    for foreground in ("border-strong", "primary", "primary-hover", "primary-line", "focus"):
        for background in surfaces:
            ratio = contrast(colors["--" + foreground], colors["--" + background])
            assert ratio >= 3, (theme, accent, preference, foreground, background, ratio)
    # Focus is two-tone: a surface-colored 3px separator contrasts with the
    # primary control, then a 3px accent outline contrasts with its surroundings.
    for control in ("primary", "primary-hover"):
        assert contrast(colors["--surface"], colors["--" + control]) >= 3


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("preference", ["standard", "high"])
def test_accents_cannot_redefine_semantic_verdicts(theme, preference):
    baseline = palette(theme)
    for accent in ACCENTS:
        colors = palette(theme, accent, preference)
        for name in ("ok", "ok-soft", "bad", "bad-soft", "warn", "warn-soft"):
            assert colors["--" + name] == baseline["--" + name]
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    for block in re.findall(r':root(?:\[data-theme="dark"\])?\[data-accent="[^"]+"\]\s*\{([^}]+)\}', css):
        assert set(re.findall(r"(--[\w-]+):", block)) == {
            "--primary", "--primary-hover", "--primary-ink", "--on-primary",
            "--primary-soft", "--primary-line", "--focus"}


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("accent", ACCENTS)
def test_high_contrast_increases_neutral_text_and_control_contrast(theme, accent):
    standard = palette(theme, accent)
    high = palette(theme, accent, "high")
    for foreground in ("text", "text-2", "muted", "border-strong"):
        for background in ("bg", "surface", "surface-2", "surface-3"):
            assert contrast(high["--" + foreground], high["--" + background]) > contrast(
                standard["--" + foreground], standard["--" + background])


def test_forced_colors_and_two_tone_focus_are_explicit():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    forced = css.split("@media (forced-colors: active)", 1)[1].split("@media print", 1)[0]
    for token in ("Canvas", "CanvasText", "Highlight", "HighlightText", "LinkText", "ButtonText"):
        assert token in forced
    assert ":root[data-theme][data-accent][data-contrast]" in forced
    assert "forced-color-adjust: none" not in css
    assert "outline: 3px solid var(--focus); outline-offset: 3px" in css
    assert "box-shadow: 0 0 0 3px var(--surface)" in css
    assert ":is(:hover, :focus) .nav-label { visibility: visible; }" in css


def test_header_uses_theme_surfaces_instead_of_gradients():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    assert "gradient(" not in css
    header = re.search(r"\nheader\s*\{([^}]+)\}", css)
    assert header is not None
    assert "background: var(--surface)" in header[1]
    assert "color: var(--text)" in header[1]
    assert "color-scheme: dark; border-color:" not in css
