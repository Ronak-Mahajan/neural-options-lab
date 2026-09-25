"""Static checks on the dashboard's accessibility and chart wiring.

These read frontend/index.html, frontend/styles.css and frontend/app.js as
text, like test_port_frontend.py. They pin the properties a keyboard or
screen-reader visitor depends on (focus rings, names, states, live regions)
and the chart and request behaviour that a browser session would otherwise
have to confirm.
"""
from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
CSS = (FRONTEND / "styles.css").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")

IDS = set(re.findall(r'\bid="([^"]+)"', HTML))


def _tags(pattern: str) -> list[str]:
    return re.findall(pattern, HTML)


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf'\b{name}="([^"]*)"', tag)
    return m.group(1) if m else None


def _function(name: str) -> str:
    """The body of a top-level `function name(...) {...}` in app.js."""
    start = APP_JS.index(f"function {name}(")
    depth, i = 0, APP_JS.index("{", start)
    for j in range(i, len(APP_JS)):
        depth += {"{": 1, "}": -1}.get(APP_JS[j], 0)
        if depth == 0:
            return APP_JS[i:j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


# Keyboard focus -------------------------------------------------------------

def test_every_control_has_a_visible_focus_ring() -> None:
    assert re.search(r"(^|\n):focus-visible\s*\{[^}]*outline:\s*2px solid", CSS)


def test_range_sliders_show_keyboard_focus() -> None:
    """The track has outline:none, so focus needs its own mark on the track
    and on the thumb in both engines."""
    assert re.search(r'input\[type="range"\]:focus-visible\s*\{[^}]*outline:\s*2px', CSS)
    for engine in ("-webkit-slider-thumb", "-moz-range-thumb"):
        assert re.search(
            rf'input\[type="range"\]:focus-visible::{engine}\s*\{{[^}}]*box-shadow', CSS), engine


def test_outline_none_always_has_a_replacement() -> None:
    """Every rule that removes the outline on focus draws something else."""
    for sel, body in re.findall(r"([^{}]+)\{([^}]*outline:\s*none[^}]*)\}", CSS):
        sel = sel.strip()
        if ":focus" not in sel:
            continue
        if sel.startswith("main"):
            continue  # programmatic focus target of the skip link, not a control
        assert "box-shadow" in body or "border-color" in body, sel


def test_skip_link_is_first_and_targets_the_results() -> None:
    body = HTML[HTML.index("<body>"):]
    first = re.search(r"<(a|button|input|select|textarea)\b[^>]*>", body).group(0)
    assert 'class="skip-link"' in first and 'href="#results"' in first
    main = re.search(r"<main\b[^>]*>", HTML).group(0)
    assert 'id="results"' in main and 'tabindex="-1"' in main
    assert re.search(r"\.skip-link:focus[^{]*\{[^}]*top:\s*\d+px", CSS)


def test_header_scrolls_away_on_short_narrow_viewports() -> None:
    """At 320x256 (400% zoom) a sticky header would cover over half the view."""
    m = re.search(r"@media \(max-width: 1100px\) and \(max-height: 500px\)\s*\{(.*?)\n\}",
                  CSS, re.S)
    assert m and re.search(r"\.cmd-bar\s*\{\s*position:\s*static", m.group(1))


# Names, roles and states ----------------------------------------------------

def test_tabs_are_a_roving_tablist() -> None:
    assert re.search(r'<nav class="tab-nav" role="tablist"', HTML)
    tabs = _tags(r'<button class="tab-btn[^"]*"[^>]*>')
    assert len(tabs) == 4
    assert all(_attr(t, "role") == "tab" for t in tabs)
    assert [_attr(t, "aria-selected") for t in tabs].count("true") == 1
    assert [_attr(t, "tabindex") for t in tabs] == ["0", "-1", "-1", "-1"]
    for t in tabs:
        pane = _attr(t, "aria-controls")
        assert pane in IDS
        assert re.search(rf'id="{pane}"[^>]*role="tabpanel"[^>]*aria-labelledby="{_attr(t, "id")}"',
                         HTML)
    show = _function("showTab")
    assert 'setAttribute("aria-selected"' in show and "tabIndex" in show
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in APP_JS


def test_segmented_buttons_report_their_pressed_state() -> None:
    groups = re.findall(r'<div class="segmented[^"]*" id="([^"]+)"([^>]*)>(.*?)</div>', HTML, re.S)
    buttons = 0
    for gid, attrs, inner in groups:
        assert 'role="group"' in attrs and "aria-label=" in attrs, gid
        btns = re.findall(r'<button[^>]*class="seg-btn[^"]*"[^>]*>', inner)
        buttons += len(btns)
        pressed = [_attr(b, "aria-pressed") for b in btns]
        assert pressed.count("true") == 1 and pressed.count("false") == len(btns) - 1, gid
        assert all(_attr(b, "type") == "button" for b in btns), gid
    assert buttons == 13
    paint = _function("paintSegmented")
    assert 'setAttribute("aria-pressed"' in paint
    # Both writers of the selection go through it.
    assert "paintSegmented(box, btn.dataset.value)" in _function("bindSegmented")
    assert "paintSegmented(" in _function("setSegmented")


def test_selection_survives_forced_colours() -> None:
    m = re.search(r"@media \(forced-colors: active\)\s*\{(.*?)\n\}", CSS, re.S)
    assert m and ".seg-btn.active" in m.group(1) and ".tab-btn.active" in m.group(1)
    assert 'input[type="range"]' in m.group(1)


def test_every_help_button_has_a_name_and_an_expanded_state() -> None:
    helps = _tags(r'<button[^>]*class="help"[^>]*>')
    assert len(helps) >= 15
    for b in helps:
        label = _attr(b, "aria-label")
        assert label and label.startswith("About "), b[:80]
    # The glossary term is a real button, so the keyboard reaches it.
    assert re.search(r'<button type="button" class="glossary" data-help=', HTML)
    assert "<span class=\"glossary\"" not in HTML
    # Chips built in app.js carry the same attributes.
    chip = _function("hedgeStatChip")
    for attr in ("aria-label", "aria-expanded", "aria-controls"):
        assert attr in chip
    for attr in ('"aria-expanded", "true"', '"aria-describedby", "help-bubble"'):
        assert attr in APP_JS
    assert re.search(r'e\.key === "Escape"\) close\(\)', APP_JS)


def test_polite_live_regions_exist_and_are_debounced() -> None:
    for rid in ("quote-status", "hedge-status", "note-status", "rail-status", "help-live"):
        assert re.search(rf'id="{rid}" role="status"', HTML), rid
    assert re.search(r'id="stream-sub" role="status"', HTML)
    announce = _function("announce")
    assert "setTimeout" in announce and "clearTimeout" in announce
    # The per-tick readouts are never live: at 15 a second they would flood.
    for tile in ("ws-spot", "ws-price", "ws-delta", "ws-ticks"):
        tag = re.search(rf'<[^>]*id="{tile}"[^>]*>', HTML).group(0)
        assert "aria-live" not in tag and "role=" not in tag


def test_refused_values_say_why() -> None:
    mark = _function("markInvalid")
    assert 'setAttribute("aria-invalid", "true")' in mark
    assert 'classList.add("invalid")' in mark
    assert 'removeAttribute("aria-invalid")' in _function("clearInvalid")
    for field in ("spot", "strike", "maturity", "sigma", "rate"):
        box = re.search(rf'<input[^>]*id="val-{field}"[^>]*>', HTML).group(0)
        slot = _attr(box, "aria-describedby")
        assert slot == f"err-{field}" and slot in IDS
    # The contract-size fields refuse with the same reason slot and a quiet blur.
    size = APP_JS[APP_JS.index("function bindSizeField("):APP_JS.index('bindSizeField("in-qty"')]
    assert "markInvalid(el, reason)" in size and "clearInvalid(el)" in size
    assert "commit(true)" in size
    assert 'classList.add("invalid")' not in size
    for field, slot in (("in-qty", "err-qty"), ("in-mult", "err-mult")):
        box = re.search(rf'<input[^>]*id="{field}"[^>]*>', HTML).group(0)
        assert _attr(box, "aria-describedby") == slot and slot in IDS
    # The premium field flags a refusal to assistive technology too.
    solve = _function("solveImpliedVol")
    assert 'box.setAttribute("aria-invalid", "true")' in solve
    assert 'box.removeAttribute("aria-invalid")' in solve


def test_every_chart_has_a_name_and_a_summary() -> None:
    plots = _tags(r'<div class="plot[^"]*" id="(plot-[^"]+)"')
    assert len(plots) >= 10
    for pid in plots:
        tag = re.search(rf'<div class="plot[^"]*" id="{pid}"[^>]*>', HTML).group(0)
        assert _attr(tag, "role") == "figure", pid
        assert (_attr(tag, "aria-label") or "").startswith("Chart of"), pid
        assert _attr(tag, "aria-describedby") == pid + "-summary" and pid + "-summary" in IDS
        assert f'describeChart("{pid}"' in APP_JS, pid


# Motion ---------------------------------------------------------------------

def test_number_tween_honours_reduced_motion() -> None:
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in APP_JS
    assert "reduceMotion.matches" in _function("animateNumber")


# Charts ---------------------------------------------------------------------

def test_convergence_band_draws_no_markers() -> None:
    band = re.search(r'fill: "toself"', APP_JS)
    assert band
    window = APP_JS[band.start() - 400:band.start() + 200]
    assert 'mode: "lines"' in window


def test_holdings_are_stacked_panels_not_overlaid_axes() -> None:
    assert "overlaying" not in APP_JS
    start = APP_JS.index('Plotly.react("plot-holdings"')
    layout = APP_JS[start:APP_JS.index("PLOT_CONFIG", start)]
    assert re.search(r"yaxis:.*?domain: \[0, 0\.62\]", layout, re.S)
    assert re.search(r"yaxis2:.*?domain: \[0\.72, 1\]", layout, re.S)


def test_short_dated_surface_has_smile_slices() -> None:
    assert 'id="plot-ivsmile"' in HTML
    assert 'Plotly.react("plot-ivsmile"' in APP_JS


def test_two_dimensional_charts_do_not_drag_zoom() -> None:
    base = APP_JS[APP_JS.index("const PLOT_BASE"):APP_JS.index("const PLOT_CONFIG")]
    assert re.search(r"dragmode:\s*false", base)
    assert "scrollZoom: false" in APP_JS


def test_time_to_expiry_is_the_one_term() -> None:
    assert "maturity (y)" not in APP_JS
    assert not re.search(r'aria-label="[^"]*maturity', HTML, re.I)


def test_loading_copy_uses_the_ellipsis_character() -> None:
    for name, text in (("index.html", HTML), ("app.js", APP_JS)):
        strings = re.findall(r'"([^"\n]*)"', text)
        assert not [s for s in strings if re.search(r"[A-Za-z ]\.\.\.$|^\.\.\.$", s)], name


def test_live_spot_tile_is_not_in_the_model_colour() -> None:
    assert '$("ws-spot").className = "v mono";' in APP_JS
    assert '$("ws-spot").className = "v mono live"' not in APP_JS


# Requests and sockets -------------------------------------------------------

def test_requests_have_a_deadline() -> None:
    api = _function("api")
    assert "signal: timeoutSignal(" in api
    assert "AbortSignal.timeout" in _function("timeoutSignal")
    # The default deadline exceeds the server's 30 s queue wait plus a job.
    ms = int(re.search(r"const API_TIMEOUT_MS = (\d+);", APP_JS).group(1))
    assert ms >= 60000


def test_websocket_handlers_bind_their_own_socket() -> None:
    ws = _function("wsConnect")
    assert "const sock = new WebSocket(" in ws
    assert "ws.send(" not in ws and "sock.send(" in ws
    for handler in ("onopen", "onmessage", "onclose", "onerror"):
        m = re.search(rf"sock\.{handler} = [^{{]*\{{\s*(?://[^\n]*\n\s*)*([^\n]*)", ws)
        assert m and "if (ws !== sock) return;" in m.group(1), handler
    assert not re.search(r"\bws\.on(open|message|close|error)\b", ws)


def test_hidden_surfaces_draw_when_their_group_opens() -> None:
    boot = APP_JS[APP_JS.rindex("await runPrice();"):]
    boot = boot[:boot.index("})();")]
    assert "await runSurface()" not in boot and "await runIVSurface()" not in boot
    assert "refreshSurfaces()" in boot


def test_plotly_script_is_pinned_with_sri() -> None:
    tag = re.search(r"<script\b[^>]*plotly-2\.35\.2\.min\.js[^>]*>", HTML).group(0)
    assert 'integrity="sha384-cCVCZkAjYNxaYKbM8lsArLznDF/SvMFr1jcZrvOpSTCa0W40ZAdLzHCEulnUa5i7"' in tag
    assert 'crossorigin="anonymous"' in tag


# Contrast -------------------------------------------------------------------

def test_accent_text_on_its_tint_uses_the_text_step() -> None:
    for sel in (r"\.tag-nn", r"\.contract-pill\.live"):
        body = re.search(sel + r"\s*\{([^}]*)\}", CSS).group(1)
        assert "var(--neural-text)" in body, sel


def test_idle_feed_dims_figures_not_labels() -> None:
    assert not re.search(r"\.stream-stats\.idle\s*\{[^}]*opacity", CSS)
    assert re.search(r"\.stream-stats\.idle \.v\s*\{[^}]*opacity", CSS)
    # The idle price drops the live colour, so the dimmed figure is ink-hi.
    assert '$("ws-price").className = "v mono";' in _function("wsIdle")
