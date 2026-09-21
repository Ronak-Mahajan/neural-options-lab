"""Static checks on the dashboard markup and stylesheet.

These parse frontend/index.html and frontend/styles.css as text. They cover
the properties a browser session would otherwise have to confirm: label
contrast, script load order, the ids app.js looks up, tap-target sizes and
the responsive rail.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
CSS = (FRONTEND / "styles.css").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")

AA_NORMAL_TEXT = 4.5


def _linear(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def _contrast(fg: tuple[float, float, float], bg: tuple[float, float, float]) -> float:
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _token(name: str) -> str:
    m = re.search(rf"--{re.escape(name)}:\s*([^;]+);", CSS)
    assert m, f"--{name} is not defined in styles.css"
    return m.group(1).strip()


def _hex(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return tuple(float(int(v[i:i + 2], 16)) for i in (0, 2, 4))


def _over(value: str, bg: tuple[float, float, float]) -> tuple[float, float, float]:
    """Resolve a token to an opaque colour over `bg`."""
    if value.startswith("#"):
        return _hex(value)
    r, g, b, a = (float(x) for x in re.findall(r"[\d.]+", value))
    return tuple(a * f + (1 - a) * k for f, k in zip((r, g, b), bg))


GROUNDS = {
    "page": (0.0, 0.0, 0.0),
    "well": _hex(_token("well")),
    "surface": _hex(_token("surface")),
    "raised": _hex(_token("raised")),
}
LABEL_TOKENS = ("ink-hi", "ink", "ink-mid", "ink-low")


@pytest.mark.parametrize("ground", GROUNDS)
@pytest.mark.parametrize("token", LABEL_TOKENS)
def test_label_token_meets_aa(token: str, ground: str) -> None:
    """Every label step is used for text under 18px, on every ground."""
    bg = GROUNDS[ground]
    ratio = _contrast(_over(_token(token), bg), bg)
    assert ratio >= AA_NORMAL_TEXT, f"--{token} on {ground}: {ratio:.2f}:1"


@pytest.mark.parametrize("ground", GROUNDS)
def test_label_tiers_stay_distinct(ground: str) -> None:
    """Adjacent steps differ by at least a quarter in contrast ratio."""
    bg = GROUNDS[ground]
    ratios = [_contrast(_over(_token(t), bg), bg) for t in LABEL_TOKENS]
    for brighter, dimmer in zip(ratios, ratios[1:]):
        assert brighter / dimmer >= 1.25, ratios


def test_no_opacity_on_label_text() -> None:
    """Opacity multiplies the token alpha and takes the text below AA."""
    m = re.search(r"\.kicker-note\s*\{([^}]*)\}", CSS)
    assert m and "opacity" not in m.group(1)


def test_scripts_are_deferred_in_order() -> None:
    """Plotly then app.js, both deferred, so the page paints first and
    Plotly is defined when app.js starts."""
    scripts = re.findall(r"<script\b([^>]*)>", HTML)
    srcs = [re.search(r'src="([^"]+)"', s).group(1) for s in scripts]
    assert [s.rsplit("/", 1)[-1] for s in srcs] == ["plotly-2.35.2.min.js", "app.js"]
    assert all(re.search(r"\bdefer\b", s) for s in scripts)
    assert not any(re.search(r"\basync\b", s) for s in scripts)


def test_no_inline_script_or_handlers() -> None:
    """Nothing inline can run before the deferred scripts or outside a CSP."""
    assert not re.search(r"<script\b(?![^>]*\bsrc=)[^>]*>", HTML)
    assert not re.search(r"\son[a-z]+\s*=", HTML)


def test_ids_app_js_looks_up_exist() -> None:
    ids = set(re.findall(r'\bid="([^"]+)"', HTML))
    wanted = set(re.findall(r'(?<![\w.])\$\("([^"]+)"\)', APP_JS))
    wanted |= set(re.findall(r'getElementById\("([^"]+)"\)', APP_JS))
    assert wanted, "no id lookups found in app.js"
    assert not sorted(wanted - ids)


def test_tab_buttons_name_existing_panes() -> None:
    ids = set(re.findall(r'\bid="([^"]+)"', HTML))
    tabs = re.findall(r'data-tab="([^"]+)"', HTML)
    assert tabs == ["tab-pricing", "tab-hedging", "tab-stream", "tab-ai"]
    assert set(tabs) <= ids


def test_help_ring_hit_area() -> None:
    """The ring stays 15px; its ::after box makes the target at least 40px.
    The inset is measured from the padding box inside the 1px border."""
    ring = re.search(r"\.help\s*\{([^}]*)\}", CSS).group(1)
    assert "width: 15px" in ring and "height: 15px" in ring
    assert "position: relative" in ring
    inset = re.search(r"\.help::after\s*\{[^}]*inset:\s*-(\d+)px", CSS)
    assert inset and (15 - 2) + 2 * int(inset.group(1)) >= 40


@pytest.mark.parametrize("selector", [r"\.panel-details > summary", r"\.lede-more"])
def test_one_line_controls_have_hit_slop(selector: str) -> None:
    """A 16px line box plus the vertical inset on both sides reaches 40px."""
    inset = re.search(selector + r"::after\s*\{[^}]*inset:\s*-(\d+)px", CSS)
    assert inset and 16 + 2 * int(inset.group(1)) >= 40


def _media_block(query: str) -> str:
    """Bodies of every `@media (query)` block in the stylesheet, joined."""
    bodies = []
    for m in re.finditer(re.escape(f"@media ({query})"), CSS):
        depth, i = 0, CSS.index("{", m.end())
        for j in range(i, len(CSS)):
            depth += {"{": 1, "}": -1}.get(CSS[j], 0)
            if depth == 0:
                bodies.append(CSS[i + 1:j])
                break
    assert bodies, f"no @media ({query}) block"
    return "\n".join(bodies)


def test_rail_scrolls_inside_its_sticky_box() -> None:
    base = re.search(r"\n\.rail\s*\{([^}]*)\}", CSS).group(1)
    assert "position: sticky" in base
    assert "max-height: calc(100vh" in base and "overflow-y: auto" in base
    one_column = _media_block("max-width: 1100px")
    rail = re.search(r"\.rail\s*\{([^}]*)\}", one_column).group(1)
    assert "position: static" in rail and "max-height: none" in rail
    assert "overflow: visible" in rail


def test_rail_collapses_wherever_the_shell_is_one_column() -> None:
    """The summary row and the collapsed state live in the same breakpoint
    that stacks the rail above the stage."""
    one_column = _media_block("max-width: 1100px")
    toggle = re.search(r"\.rail-toggle\s*\{([^}]*)\}", one_column).group(1)
    assert "display: flex" in toggle and "grid-column: 1 / -1" in toggle
    assert ".rail.collapsed > .rail-section" in one_column
    assert ".rail-toggle" not in _media_block("max-width: 800px")
    # app.js decides the initial collapse with matchMedia; the width it tests
    # has to be the breakpoint that carries the collapsed rules.
    widths = set(re.findall(r'matchMedia\("\(max-width: (\d+)px\)"\)', APP_JS))
    assert "1100" in widths, widths


def test_per_endpoint_updaters_are_single_flight() -> None:
    """Each pricing endpoint's updater runs through latestOnly, so a dragged
    slider cannot apply a stale response over a newer one."""
    for updater in ("updatePrice", "updateConvergence", "updateSurface",
                    "updateXAI", "updateIVSurface"):
        assert re.search(r"latestOnly\(" + updater + r"\)", APP_JS), updater
        assert re.search(r"async function " + updater +
                         r"\(isCurrent = \(\) => true\)", APP_JS), updater


def test_benchmark_failure_is_reported_in_panel() -> None:
    m = re.search(r"async function updateBenchmark.*?^}", APP_JS,
                  re.S | re.M)
    assert m and 'panelMessage("plot-latency"' in m.group(0)


def test_refused_explain_resets_stale_attributions() -> None:
    m = re.search(r"async function updateXAI.*?^}", APP_JS, re.S | re.M)
    body = m.group(0)
    assert "lastAttributions = null" in body
    assert "lastBaselinePrice = null" in body
    assert 'panelMessage("plot-xai"' in body


def test_hedge_verdict_and_chip_share_one_test() -> None:
    """The winner named by the verdict is the only source of the green chip,
    and it exists only when the paired test separates the top two."""
    assert re.search(r"winner: called \? first\.key : null", APP_JS)
    assert re.search(r'verdict\.winner === key \? "good" : ""', APP_JS)
    # No chip is highlighted from a raw point-estimate minimum.
    assert "=== best ?" not in APP_JS


def test_static_copy_has_no_dashes_or_stale_labels() -> None:
    for name, text in (("index.html", HTML), ("styles.css", CSS)):
        assert chr(0x2014) not in text and "&mdash;" not in text, name
    stale = r"independent simulation|fair value|per calendar day|per contract ·"
    assert not re.search(stale, HTML, re.I)
