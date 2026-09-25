"""Acceptance checks for the dashboard, run in headless Chromium.

Usage: verify_ux.py <base_url> [out_dir]
Every browser context is created with bypass_csp=True. The checks wait on
string predicates (page.wait_for_function), which Playwright evaluates with
eval, and the server's Content-Security-Policy correctly allows no eval. The
page's own scripts need no such exemption; a run without bypass_csp that
counts CSP violations is the check for that.
Covers: first-visit provenance copy, the status pill and model card, the
agreement hero, grouped sections, per-panel disclosures, the hedging verdict
and chips, the report copy, the short-dated regime, tap-to-reveal help,
responsive behaviour and URL state.
"""
import json
import os
import sys

# A pw-browsers directory next to this file, when there is one, holds the
# browsers; otherwise Playwright uses PLAYWRIGHT_BROWSERS_PATH or its default.
_local_browsers = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pw-browsers")
if os.path.isdir(_local_browsers):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", _local_browsers)
from playwright.sync_api import sync_playwright

base = sys.argv[1].rstrip("/")
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    os.makedirs(out, exist_ok=True)
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def overflow(page):
    return page.evaluate("""Math.round(Math.max(0, ...[...document.querySelectorAll('body *')]
        .filter(e => e.offsetParent !== null && !(function(el){for(let a=el.parentElement;a;a=a.parentElement){const o=getComputedStyle(a).overflowX;if(o==='auto'||o==='scroll')return true;}return false;})(e))
        .map(e => e.getBoundingClientRect().right)) - innerWidth)""")


with sync_playwright() as pw:
    browser = pw.chromium.launch()

    # ───────────────────────────────────────────────── desktop first visit
    ctx = browser.new_context(bypass_csp=True, viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(base + "/", wait_until="networkidle", timeout=180_000)
    page.wait_for_function("document.getElementById('nn-price').textContent.startsWith('$')", timeout=180_000)
    page.wait_for_function("!!document.getElementById('plot-convergence').data", timeout=180_000)
    page.wait_for_timeout(1200)

    check("no console errors",
          not [e for e in errors if "Failed to load resource" not in e],
          "; ".join(errors)[:300])
    check("no horizontal overflow", overflow(page) <= 0, f"overflow={overflow(page)}px")

    # The owner's question: is a ticker required, and where do numbers come from?
    contract = page.evaluate("document.getElementById('contract-text').textContent")
    check("contract line names the instrument being priced",
          "Asian call" in contract and "$100 stock" in contract, contract[:160])
    lede_text = page.evaluate("document.querySelector('.lede-body').innerText")
    check("the page says a ticker is optional without being asked",
          "ticker" in lede_text.lower()
          and page.evaluate("document.getElementById('contract-pill').textContent") == "Example",
          lede_text[:150].replace(chr(10), " "))
    check("contract pill reads Example before any ticker",
          page.evaluate("document.getElementById('contract-pill').textContent") == "Example")
    check("ticker section is marked optional and is last in the rail",
          page.evaluate("""(() => {
            const secs=[...document.querySelectorAll('.rail-section h2')].map(h=>h.textContent);
            return secs[secs.length-1].toLowerCase().includes('optional');
          })()"""), page.evaluate("[...document.querySelectorAll('.rail-section h2')].map(h=>h.textContent).join(' | ')"))

    # The "133 parameters on the task bar" complaint.
    badge = page.evaluate("document.getElementById('model-badge-text').textContent")
    check("status pill shows status, not a spec string", badge == "Model ready", badge)
    check("badge carries no raw numbers", not any(c.isdigit() for c in badge), badge)
    page.click("#model-badge")
    page.wait_for_timeout(300)
    card = page.evaluate("document.getElementById('model-card').hidden ? '' : document.getElementById('model-card').innerText")
    low = card.lower()
    check("model card opens with labelled facts",
          "133,889" in card and "accuracy" in low and "trained range" in low,
          card[:180].replace(chr(10), " · "))

    # The hero reports agreement, not an unstable wall-clock ratio.
    tags = page.evaluate("[...document.querySelectorAll('.hero .panel-tag')].map(e => e.textContent.replace('?',''))")
    check("hero leads with the price, the position and the cross-check",
          len(tags) == 3 and "Model price" in tags[0] and "Position" in tags[1]
          and "Monte Carlo" in tags[2], " | ".join(tags))
    check("the cross-check carries the verdict",
          "Cross-check" in page.evaluate("document.querySelector('#card-agreement .dial-label').textContent"))
    check("the position card carries the export",
          page.evaluate("!!document.querySelector('#card-position #btn-copy-quote')"))
    agr = page.evaluate("document.getElementById('agreement').textContent")
    check("the cross-check states the gap against a stated tolerance",
          "from the simulation" in agr
          and ("error bar" in agr or "measured error" in agr or "wider than both" in agr), agr)
    check("speedup ratio is gone from the hero",
          "×" not in page.evaluate("document.getElementById('speedup').textContent"),
          page.evaluate("document.getElementById('speedup').textContent"))
    timing = page.evaluate("document.getElementById('timing-line').textContent")
    check("timings reported once, under the hero", "On this server" in timing, timing[:110])

    # Four-decimal Greeks need units; the model's credentials need a home a
    # phone can reach, since the status pill is hidden on small screens.
    units = page.evaluate("[...document.querySelectorAll('.greek-unit')].map(e => e.textContent)")
    check("every Greek carries its unit", len(units) == 5 and all(units), " | ".join(units))
    page.evaluate("document.getElementById('group-accuracy').open = true")
    page.wait_for_timeout(400)
    tiles = page.evaluate("[...document.querySelectorAll('#accuracy-stats .hedge-stat')].map(c => c.querySelector('.k').textContent)")
    check("accuracy panel states the model's credentials", len(tiles) >= 4, " | ".join(tiles))
    estat = page.evaluate("document.getElementById('error-stat').textContent")
    check("accuracy stat names the centre as well as the width",
          ("mean " in estat or "systematic bias" in estat)
          and ("95% of errors" in estat or "95th percentile" in estat), estat[:150])
    page.evaluate("document.getElementById('group-accuracy').open = false")

    # Diagnostics are grouped and collapsed rather than strewn across headers.
    check("pricing tab has grouped sections",
          page.evaluate("document.querySelectorAll('#tab-pricing .group').length") >= 3)
    check("accuracy and domain sections start collapsed",
          page.evaluate("!document.getElementById('group-accuracy').open && !document.getElementById('group-domain').open"))
    check("per-panel details exist for method and throughput notes",
          page.evaluate("document.querySelectorAll('#tab-pricing .panel-details').length") >= 3)
    check("no raw completeness-error string on the page",
          "completeness err" not in page.evaluate("document.body.innerText"))
    check("no 'params ·' spec fragment on the page",
          "params ·" not in page.evaluate("document.body.innerText"))

    # Help is tappable, not hover-only.
    page.evaluate("document.querySelector('.greek-name .help').click()")
    page.wait_for_timeout(250)
    check("tap-to-reveal help works",
          not page.evaluate("document.getElementById('help-bubble').hidden")
          and len(page.evaluate("document.getElementById('help-bubble').textContent")) > 20)
    page.keyboard.press("Escape")

    # A pricer has to accept an exact strike, not the nearest slider notch.
    for field, typed, want in (("val-strike", "137.42", 137.42),
                               ("val-sigma", "18.3%", 0.183),
                               ("val-rate", "4.37", 0.0437)):
        page.fill("#" + field, typed)
        page.press("#" + field, "Enter")
        page.wait_for_timeout(1800)
    st3 = page.evaluate("({k: state.strike, s: state.sigma, r: state.rate})")
    check("parameters accept exact typed values",
          abs(st3["k"] - 137.42) < 1e-9 and abs(st3["s"] - 0.183) < 1e-9
          and abs(st3["r"] - 0.0437) < 1e-9, json.dumps(st3))
    page.fill("#val-maturity", "30d")
    page.press("#val-maturity", "Enter")
    page.wait_for_timeout(1800)
    check("expiry accepts a unit suffix",
          abs(page.evaluate("state.maturity") - 30 / 252) < 1e-9,
          str(page.evaluate("state.maturity")))
    page.fill("#val-maturity", "1d")
    page.press("#val-maturity", "Enter")
    page.wait_for_timeout(1500)
    check("a one-day expiry can be typed",
          not page.evaluate("document.getElementById('val-maturity').classList.contains('invalid')")
          and abs(page.evaluate("state.maturity") - 0.004) < 1e-9,
          str(page.evaluate("state.maturity")))
    page.fill("#val-sigma", "500")
    page.press("#val-sigma", "Enter")
    page.wait_for_timeout(500)
    check("out-of-range entry is refused, not clamped",
          page.evaluate("document.getElementById('val-sigma').classList.contains('invalid')")
          and abs(page.evaluate("state.sigma") - 0.183) < 1e-9)
    page.evaluate("document.getElementById('val-sigma').blur()")
    page.wait_for_timeout(400)

    # The numbers have to describe a position, and be able to leave the page.
    page.fill("#in-qty", "-5")
    page.press("#in-qty", "Enter")
    page.wait_for_timeout(900)
    pos = page.evaluate("document.getElementById('pos-sub').textContent")
    val = page.evaluate("document.getElementById('pos-value').textContent")
    check("a position has a size and a direction",
          "short" in pos.lower() and "5" in pos and "−" in val,
          val + "  /  " + pos)
    d_unit = float(page.evaluate("document.getElementById('g-delta').textContent")
                   .replace("−", "-").replace(",", ""))
    page.evaluate("document.querySelector('#greek-basis .seg-btn[data-value=position]').click()")
    page.wait_for_timeout(400)
    d_pos = float(page.evaluate("document.getElementById('g-delta').textContent")
                  .replace("−", "-").replace(",", ""))
    want = d_unit * -5 * 100
    check("Greeks scale to the whole position",
          abs(d_pos - want) <= max(0.02, abs(want) * 0.002),
          f"per contract {d_unit} x -500 = {want:.2f}, shown {d_pos}")
    page.evaluate("document.querySelector('#greek-basis .seg-btn[data-value=unit]').click()")
    page.wait_for_timeout(300)
    check("size travels in the shareable link", "qty=-5" in page.url, page.url[-90:])
    rows = page.evaluate("""(() => {
        const fn = window.quoteRows; return fn ? fn().map(r => r[0]) : []; })()""")
    check("the quote carries contract, inputs, price, Greeks and the check",
          any("Instrument" in r for r in rows) and any("Produced" in r for r in rows)
          and any("Delta per share" in r for r in rows)
          and any("whole position" in r for r in rows)
          and any("Cross-check" in r for r in rows), str(len(rows)) + " rows")
    page.fill("#in-qty", "1")
    page.press("#in-qty", "Enter")
    page.wait_for_timeout(700)

    # The pricer has to run backwards too: a premium in, a volatility out.
    page.goto(base + "/", wait_until="networkidle", timeout=180_000)
    page.wait_for_function("document.getElementById('nn-price').textContent.startsWith('$')", timeout=180_000)
    page.wait_for_timeout(1200)
    page.fill("#in-target-price", "9.50")
    page.click("#btn-solve-iv")
    page.wait_for_function("document.getElementById('btn-solve-iv').textContent === 'Solve'", timeout=120_000)
    page.wait_for_timeout(2200)
    solved = page.evaluate("document.getElementById('nn-price').textContent")
    check("a premium solves back to a volatility that reproduces it",
          solved.startswith("$9.50"),
          solved + " after solving for $9.50, sigma " + str(page.evaluate("state.sigma")))
    before_sigma = page.evaluate("state.sigma")
    page.fill("#in-target-price", "95")
    page.click("#btn-solve-iv")
    page.wait_for_function("document.getElementById('btn-solve-iv').textContent === 'Solve'", timeout=120_000)
    page.wait_for_timeout(500)
    note = page.evaluate("document.getElementById('iv-solve-note').textContent")
    check("an unreachable premium reports the range instead of clamping",
          "No volatility between" in note and "left unchanged" in note
          and abs(page.evaluate("state.sigma") - before_sigma) < 1e-12, note[:120])

    # A panel that cannot compute says so instead of shimmering forever.
    page.goto(base + "/?spot=100&strike=137&T=0.02", wait_until="networkidle", timeout=180_000)
    page.wait_for_timeout(6000)
    states = page.evaluate("""['plot-convergence','plot-surface','plot-xai','plot-ivsurface'].map(id => {
        const el = document.getElementById(id);
        return el.querySelector('.shimmer') ? 'shimmer' : 'settled'; })""")
    check("no panel is left shimmering on a refused contract",
          all(s == "settled" for s in states), " | ".join(states))
    script_errors = [e for e in errors if "Failed to load resource" not in e]
    check("no script errors on a refused contract", not script_errors,
          "; ".join(script_errors)[:200])

    page.goto(base + "/", wait_until="networkidle", timeout=180_000)
    page.wait_for_function("document.getElementById('nn-price').textContent.startsWith('$')", timeout=180_000)
    page.wait_for_timeout(1500)
    if out:
        page.screenshot(path=os.path.join(out, "pricing.png"), full_page=True)

    # Expiry quick-picks reach the short-dated regime, which is a sliver of track.
    page.evaluate("document.querySelector('#maturity-quickpick .pick[data-t=\"0.02\"]').click()")
    page.wait_for_function("document.getElementById('val-maturity').value === '5d'", timeout=30_000)
    page.wait_for_timeout(2500)
    c0 = page.evaluate("document.getElementById('contract-text').textContent")
    check("short-dated regime is named as a European option",
          "European call" in c0 and "5 trading-day" in c0, c0[:120])
    check("rail note switches with the regime",
          "European" in page.evaluate("document.getElementById('rail-note').textContent"))

    # URL state still round-trips.
    page.goto(base + "/?tab=pricing&spot=160&strike=100&T=1&sigma=0.25&rate=0.04&type=put",
              wait_until="networkidle", timeout=180_000)
    page.wait_for_function("document.getElementById('nn-price').textContent.startsWith('$')", timeout=180_000)
    st = page.evaluate("({spot: state.spot, type: state.optionType, T: state.maturity})")
    check("URL state restored", st["spot"] == 160 and st["type"] == "put", json.dumps(st))
    check("contract line follows the URL state",
          "out-of-the-money" in page.evaluate("document.getElementById('contract-text').textContent"),
          page.evaluate("document.getElementById('contract-text').textContent")[:110])
    ctx.close()

    # ───────────────────────────────────────────────────────────── hedging
    ctx = browser.new_context(bypass_csp=True, viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    herrs = []
    page.on("pageerror", lambda e: herrs.append(str(e)))
    page.goto(base + "/?tab=hedging", wait_until="networkidle", timeout=180_000)
    page.wait_for_timeout(800)
    check("hedging tab has a real empty state, not a shimmer",
          page.evaluate("!!document.getElementById('hedge-empty')")
          and page.evaluate("document.querySelectorAll('#tab-hedging .shimmer').length") == 0)
    check("contract line persists onto the hedging tab",
          page.evaluate("!!document.getElementById('contract-line') && getComputedStyle(document.getElementById('contract-line')).display") != "none")
    check("contract line says the hedging run uses its own instrument",
          "its own 30-day at-the-money call" in
          page.evaluate("document.getElementById('contract-scope').textContent"),
          page.evaluate("document.getElementById('contract-scope').textContent"))
    check("inputs the hedging run ignores are dimmed, with a reason",
          page.evaluate("document.getElementById('in-sigma').closest('.param').classList.contains('rail-offtab')")
          and page.evaluate("document.getElementById('in-spot').closest('.rail-section').hasAttribute('inert')")
          and not page.evaluate("document.getElementById('rail-inactive-note').hidden"))
    page.click("#btn-hedge")
    page.wait_for_timeout(1000)
    page.wait_for_function("!document.getElementById('btn-hedge').disabled && document.querySelectorAll('#hedge-stats .hedge-stat').length > 0", timeout=300_000)
    page.wait_for_timeout(800)

    verdict = page.evaluate("document.getElementById('hedge-verdict').textContent")
    check("hedging states a verdict in words",
          ("smallest worst-5% loss" in verdict or "are level on worst-5% loss" in verdict)
          and len(verdict) > 120, verdict[:150])
    chips = page.evaluate("[...document.querySelectorAll('#hedge-stats .hedge-stat')].map(c => c.querySelector('.k').textContent + '=' + c.querySelector('.v').textContent + (c.querySelector('.v').classList.contains('good') ? '[best]' : ''))")
    check("chips are named by strategy, not by CVaR jargon",
          any("learned policy" in c for c in chips)
          and any("delta hedge" in c for c in chips)
          and all(c.startswith(("Worst-5% loss", "Trading cost", "Average P&L")) for c in chips)
          and not any("CVaR" in c for c in chips), " | ".join(chips))
    # A winner is highlighted only when the paired bootstrap separates the
    # top two; a level top two shows no best chip, matching the verdict text.
    n_best = sum("[best]" in c for c in chips)
    if "are level on worst-5% loss" in verdict:
        check("no chip is crowned when the paired test calls the top two level",
              n_best == 0, f"best={n_best} | " + " | ".join(chips))
    else:
        check("exactly one chip is highlighted as best",
              n_best == 1, f"best={n_best} | " + " | ".join(chips))
    check("tail-loss chips carry their error bars",
          sum("±" in c for c in chips) >= 3, " | ".join(chips))
    sub = page.evaluate("document.getElementById('hedge-sub').textContent")
    check("subtitle says where the volatility and rate come from",
          "not from the contract panel" in sub or "from the contract panel" in sub, sub[:140])
    check("loss convention is defined once, in dollars",
          "worst 5% of" in page.evaluate("document.getElementById('hedge-convention').textContent")
          and "bootstrap standard error" in page.evaluate("document.getElementById('hedge-convention').textContent"))
    check("method footnote says the baselines are vol-matched",
          "vol-matched" in page.evaluate("document.getElementById('hedge-method').textContent"),
          page.evaluate("document.getElementById('hedge-method').textContent")[:120])
    check("no page errors on hedging", not herrs, "; ".join(herrs)[:200])
    if out:
        page.screenshot(path=os.path.join(out, "hedging.png"), full_page=True)

    # ────────────────────────────────────────────────────────────── report
    page.goto(base + "/?tab=ai", wait_until="networkidle", timeout=180_000)
    page.wait_for_timeout(1500)
    check("report names its writer",
          "rule-based narrator" in page.evaluate("document.getElementById('report-lede').textContent")
          or "language model" in page.evaluate("document.getElementById('report-lede').textContent"),
          page.evaluate("document.getElementById('report-lede').textContent")[:120])
    page.click("#btn-risk")
    page.wait_for_timeout(600)
    page.wait_for_function("!document.getElementById('btn-risk').disabled", timeout=300_000)
    page.wait_for_timeout(600)
    rpt = page.evaluate("document.getElementById('ai-report').textContent")
    check("report drops the 'generic underlying' placeholder",
          "generic underlying" not in rpt)
    check("report does not call the costed delta hedge frictionless",
          "frictionless" not in rpt)
    check("report names the contract it priced", "Asian call" in rpt, rpt[:110])
    check("report separates the priced contract from the hedged one",
          "30-day at-the-money call" in rpt)
    check("report formats money with the sign before the currency",
          "$-" not in rpt, rpt[:200])
    if out:
        page.screenshot(path=os.path.join(out, "report.png"), full_page=True)
    ctx.close()

    # ──────────────────────────────────────────────────── tablet and phone
    for name, vw, vh in (("tablet", 820, 1180), ("phone", 390, 844)):
        ctx = browser.new_context(bypass_csp=True, viewport={"width": vw, "height": vh},
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(base + "/", wait_until="networkidle", timeout=180_000)
        page.wait_for_function("document.getElementById('nn-price').textContent.startsWith('$')", timeout=180_000)
        page.wait_for_timeout(1500)
        ov = overflow(page)
        check(f"{name}: no console errors", not errs, "; ".join(errs)[:200])
        check(f"{name}: no horizontal overflow", ov <= 0, f"overflow={ov}px")
        if name == "phone":
            y = page.evaluate("Math.round(document.getElementById('nn-price').getBoundingClientRect().top + scrollY)")
            check("phone: the price is above the fold", y < vh, f"price at y={y}")
            check("phone: rail summary reads as a sentence",
                  "at-the-money" in page.evaluate("document.getElementById('rail-summary').textContent"),
                  page.evaluate("document.getElementById('rail-summary').textContent"))
            check("phone: the rail has a visible Edit control",
                  page.evaluate("getComputedStyle(document.querySelector('.rail-edit')).display") != "none")
            check("phone: all four tabs fit without clipping",
                  page.evaluate("""(() => {
                    const n = document.querySelector('.tab-nav');
                    const last = n.lastElementChild.getBoundingClientRect();
                    return getComputedStyle(n).display === 'grid' && last.right <= innerWidth + 1;
                  })()"""))
            check("phone: the byline is present",
                  "Ronak Mahajan" in page.evaluate("document.body.innerText"))
        if out:
            page.screenshot(path=os.path.join(out, f"{name}.png"), full_page=False)
        ctx.close()

    # ─────────────────────────────────────────────────────────  methodology
    ctx = browser.new_context(bypass_csp=True, viewport={"width": 390, "height": 844})
    page = ctx.new_page()
    r = page.goto(base + "/methodology", wait_until="load", timeout=60_000)
    check("methodology page served", r is not None and r.status == 200, str(r.status if r else None))
    check("methodology: no overflow on phone", overflow(page) <= 0, f"overflow={overflow(page)}px")
    imgs = page.evaluate("Promise.all([...document.querySelectorAll('figure.fig img')].map(async i => (await fetch(i.getAttribute('src'))).ok))")
    check("methodology figures load", len(imgs) >= 5 and all(imgs), str(imgs))
    ctx.close()
    browser.close()

print()
print("FAILURES:", fails) if fails else print("ALL UX CHECKS PASSED")
sys.exit(1 if fails else 0)
