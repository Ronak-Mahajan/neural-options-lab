"""README, data notice and the three quant documents against the code and artifacts.

Each test reads a source of truth (the served index.html, the FastAPI route
decorators, HedgingEngine.compare's defaults, artifacts/eval.json,
docs/heston_reference.json, git's list of tracked data files) and requires the
prose to agree with it, so the README describes the product that is served and
the documents keep their stated scope.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
DATA_README = ROOT / "data" / "README.md"
NOTICE = ROOT / "NOTICE"

TELLS = re.compile(r"—|&mdash;|previously|no longer|used to|has been fixed|now fixed|honest|actually",
                   re.IGNORECASE)


def _flat(path: Path) -> str:
    """The file with runs of whitespace collapsed, so a reflow cannot break a phrase match."""
    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def readme() -> str:
    return _flat(README)


# ---- the README describes the served product ----

def test_readme_names_the_served_tabs(readme):
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    tabs = [t.strip() for t in
            re.findall(r'<button class="tab-btn[^"]*"[^>]*\bdata-tab="[^"]+"[^>]*>([^<]+)</button>', html)]
    assert tabs == ["Quote", "Hedge", "Live", "Desk note"], tabs
    assert "The dashboard has four tabs." in readme
    for tab in tabs:
        assert f"**{tab}.**" in readme, tab
    for stale in ("Pricing Lab", "AI Risk Analyst", "three sections", "two commands", "live-news"):
        assert stale not in readme, stale


def test_readme_api_table_lists_every_route(readme):
    src = (ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8")
    routes = []
    for kind, path, methods in re.findall(
            r'@app\.(get|post|websocket|api_route)\(\s*"([^"]+)"(?:\s*,\s*methods=\[([^\]]*)\])?', src):
        if kind == "api_route":
            kind = re.findall(r'"(\w+)"', methods)[0].lower()
        routes.append(({"get": "GET", "post": "POST", "websocket": "WS"}[kind], path))
    assert len(routes) >= 15, routes
    missing = [f"{v} {p}" for v, p in routes if f"`{v} {p}`" not in readme]
    assert not missing, missing


def test_readme_hedge_path_count_matches_compare_defaults(readme):
    src = (ROOT / "backend" / "quant" / "hedging.py").read_text(encoding="utf-8")
    sig = re.search(r"def compare\(self.*?n_paths: int = (\d+).*?seeds: tuple\[int, \.\.\.\] = \(([\d, ]+)\)",
                    src, re.S)
    assert sig, "HedgingEngine.compare's signature has drifted from this test"
    per_seed, seeds = int(sig.group(1)), [s for s in sig.group(2).split(",") if s.strip()]
    total = per_seed * len(seeds)
    assert f"same {total:,} simulated paths" in readme
    assert f"five seeds of {per_seed:,}" in readme


def test_readme_layout_lists_every_quant_module(readme):
    layout = README.read_text(encoding="utf-8").split("## Repository layout")[1].split("```")[1]
    modules = sorted(p.name for p in (ROOT / "backend" / "quant").glob("*.py") if p.name != "__init__.py")
    missing = [m for m in modules if m not in layout]
    assert not missing, missing
    for script in ("asian_arbitrage_audit.py", "joint_skew_refit.py"):
        assert script in layout and (ROOT / "scripts" / script).exists()


def test_readme_reference_path_count_matches_eval(readme):
    ref = json.loads((ROOT / "artifacts" / "eval.json").read_text(encoding="utf-8"))["ref_paths"]
    assert f"vs {ref:,}-path references (the `eval.json` protocol)" in readme
    assert f"against {ref:,}-path references that `artifacts/eval.json` provides" in readme
    assert "500,000-path references that `artifacts/eval.json`" not in readme
    assert "vs 500,000-path references (the `eval.json`" not in readme


def test_readme_scopes_the_variance_reduction_factor(readme):
    for place in (readme.split("## Results at a glance")[1].split("## Try it")[0],
                  README.read_text(encoding="utf-8").split("| Monte Carlo engine |")[1].split("\n")[0]):
        assert "at-the-money one-year call at 20% vol" in place
        assert "5x to 10x" in place
    assert "24.0x" not in readme and "24.5x" not in readme


def test_readme_scopes_the_hedging_error_bars(readme):
    assert "404 ± 7 vs 493 ± 11 bp" in readme
    assert "bootstrap standard errors over one set of 15,000 evaluation paths for one trained network" in readme
    assert "between 48 and 99 bp" in readme


def test_readme_real_paths_claims_match_the_bootstrap(readme):
    bullet = readme.split("**Deep hedging on real paths.**")[1].split("- **")[0]
    assert "resolved on BTC" in bullet and "point estimate on SPY" in bullet
    assert "0.44% to 0.92%" in bullet
    assert "On BTC the 50 bp mean difference is unresolved" in bullet


def _minus(s: str) -> str:
    return s.replace("-", "−")


def test_readme_headline_numbers_match_their_artifacts(readme):
    """The headline figures are formatted from the artifact that produces them."""
    ev = json.loads((ROOT / "artifacts" / "eval.json").read_text(encoding="utf-8"))
    rmse = ev["ensemble"]["price"]["rmse_bps"]
    assert f"Price RMSE is {rmse:.2f} basis points of strike on {ev['n_points']} held-out points" in readme

    fit = json.loads((DOCS / "atm_skew_term_structure.json").read_text(encoding="utf-8"))["spy"]["fits"]
    market = fit["market_pooled"]
    assert _minus(f"exponent {market['b']:.3f} ± {market['se_b']:.3f}") in readme

    raw = (DOCS / "hedging_real_paths.txt").read_text(encoding="utf-8")
    cells, cost, asset = {}, None, None
    for line in raw.splitlines():
        if m := re.match(r"== cost ([\d.]+) ==", line):
            cost = round(float(m.group(1)) * 1e4)
        elif m := re.match(r"=== (\S+):", line):
            asset = m.group(1)
        elif m := re.match(r"\s+(deep|delta)\s+(\S+)\s+\S+\s+(\S+)", line):
            cells[(asset, cost, m.group(1))] = (float(m.group(2)), float(m.group(3)))
        elif m := re.match(r"\s+paired deep-delta: mean \S+, deep better on (\d+)%", line):
            cells[(asset, cost, "win")] = int(m.group(1))
    bullet = readme.split("**Deep hedging on real paths.**")[1].split("- **")[0]
    spy, btc = ("SPY", 10), ("BTC-USD", 10)
    assert (f"SPY {cells[(*spy, 'delta')][1]:.3f} vs {cells[(*spy, 'deep')][1]:.3f}; "
            f"BTC {cells[(*btc, 'delta')][1]:.3f} vs {cells[(*btc, 'deep')][1]:.3f}") in bullet
    spy50 = ("SPY", 50)
    assert _minus(f"{cells[(*spy50, 'deep')][0] * 100:.2f}% vs {cells[(*spy50, 'delta')][0] * 100:.2f}%, "
                  f"on {cells[(*spy50, 'win')]}% of windows") in bullet


@pytest.mark.parametrize("readme_phrase, doc, doc_phrase", [
    ("27.9 (95% interval 22.9 to 33.5)", "approximation_benchmark.md",
     "a ratio of 27.9 (95% interval 22.9 to 33.5"),
    ("404 ± 7 vs 493 ± 11 bp", "deep_hedging_regimes.md", "404 ± 7 against 493 ± 11 bp"),
    ("78.5 of those 88.5 bp", "deep_hedging_regimes.md", "88.5 bp, of which 78.5 bp is mean"),
    ("paired 95% interval 0.44% to 0.92%", "hedging_real_paths.md", "| +0.0070 | +0.0044 to +0.0092 |"),
    ("10% at 126 days at the calibrated vol-of-vol", "atm_skew_term_structure.md",
     "| 126 | 3.94 | -0.334 | -0.364 | -0.382 | -0.391 | -0.396 | -0.403 | 9.6% |"),
])
def test_readme_numbers_live_in_a_pinned_doc(readme, readme_phrase, doc, doc_phrase):
    assert readme_phrase in readme
    assert doc_phrase in _flat(DOCS / doc)


# ---- the three quant documents ----

def test_real_paths_doc_carries_block_bootstrap_intervals():
    text = _flat(DOCS / "hedging_real_paths.md")
    for gone in ("well outside", "2.5x too small", "about 64 independent", "about 94 independent"):
        assert gone not in text, gone
    for row in ("| SPY, 10 bp | +0.0135 | −0.0033 to +0.0375 |",
                "| SPY, 50 bp | +0.0065 | −0.0122 to +0.0334 | +0.0070 | +0.0044 to +0.0092 |",
                "| BTC-USD, 10 bp | +0.0292 | +0.0169 to +0.0412 |",
                "| BTC-USD, 50 bp | +0.0246 | +0.0129 to +0.0360 | +0.0017 | −0.0014 to +0.0052 |"):
        assert row in text, row
    assert "1.2x to 1.9x the i.i.d. ones" in text


def test_real_paths_bootstrap_rows_use_committed_point_estimates():
    """The point columns of the bootstrap table are differences of the committed tables."""
    raw = (DOCS / "hedging_real_paths.txt").read_text(encoding="utf-8")
    cells, cost, asset = {}, None, None
    for line in raw.splitlines():
        if m := re.match(r"== cost ([\d.]+) ==", line):
            cost = f"{round(float(m.group(1)) * 1e4)} bp"
        elif m := re.match(r"=== (\S+):", line):
            asset = m.group(1)
        elif m := re.match(r"\s+(deep|delta)\s+(\S+)\s+\S+\s+(\S+)", line):
            cells[(asset, cost, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    text = _flat(DOCS / "hedging_real_paths.md")
    for asset in ("SPY", "BTC-USD"):
        for cost in ("10 bp", "50 bp"):
            deep, delta = cells[(asset, cost, "deep")], cells[(asset, cost, "delta")]
            cvar = f"{deep[1] - delta[1]:+.4f}".replace("-", "−")
            assert f"| {asset}, {cost} | {cvar} |" in text, (asset, cost, cvar)


def test_heston_doc_scopes_the_time_step_claim():
    text = _flat(DOCS / "heston_reference.md")
    assert "O(dt)" not in text
    run = next(r for r in json.loads((DOCS / "heston_reference.json").read_text(encoding="utf-8"))
               ["verification"]["monte_carlo"] if r["T"] == 1.0)
    assert run["n_steps"] == 400 and "The test runs at 400 steps" in text
    assert f"`{run['se'][2]:.4f}` at `T = 1`" in text
    assert "no asymptotic rate is established" in text


def test_atm_skew_doc_states_the_time_step_bias():
    text = _flat(DOCS / "atm_skew_term_structure.md")
    assert "rules out a numerical artefact" not in text
    assert "was not studied" not in text
    assert "| 126 | 3.94 | -0.334 | -0.364 | -0.382 | -0.391 | -0.396 | -0.403 | 9.6% |" in text
    assert "-0.3187 at 50 steps and -0.3088 extrapolated" in text
    spy = json.loads((DOCS / "atm_skew_term_structure.json").read_text(encoding="utf-8"))["spy"]
    fit = spy["fits"]["model_short"]
    assert f"{fit['b']:.3f} +- {fit['se_b']:.3f}" in text


# ---- the measurements behind the quoted numbers are committed ----

def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script, output", [
    ("variance_reduction.py", "variance_reduction.json"),
    ("hedger_seed_spread.py", "deep_hedging_seed_spread.json"),
    ("heston_step_convergence.py", "heston_step_convergence.json"),
    ("rb_step_convergence.py", "rb_step_convergence/summary.json"),
    ("hedge_real_paths_block.py", "hedging_real_paths_block.json"),
])
def test_each_study_ships_its_script_and_output(script, output):
    assert (ROOT / "scripts" / script).exists(), script
    assert (DOCS / output).exists(), output


def test_variance_reduction_figures_come_from_the_committed_run(readme):
    rows = _json(DOCS / "variance_reduction.json")["rows"]

    def span(**c):
        v = [r["sd_ratio"] for r in rows if all(r[k] == x for k, x in c.items())]
        assert v, c
        return f"{min(v):.0f}x to {max(v):.0f}x"
    atm = dict(S=100, K=100, T=1.0, sigma=0.2)
    assert f"{span(**atm, n_paths=5000)} at 5,000 paths and {span(**atm, n_paths=20000)} at 20,000" in readme
    others = [r["sd_ratio"] for r in rows
              if (r["S"], r["T"], r["sigma"]) in ((100, 1.0, 0.6), (80, 0.25, 0.3), (130, 2.0, 0.7))]
    assert f"it is {min(others):.0f}x to {max(others):.0f}x" in readme


def test_seed_spread_table_and_readme_come_from_the_committed_run(readme):
    rows = _json(DOCS / "deep_hedging_seed_spread.json")["rows"]
    at50 = [r for r in rows if r["cost"] == 0.005]
    gaps = [-r["deep_minus_delta_bp"] for r in at50]
    assert len(at50) == 8 and all(g > 0 for g in gaps)
    assert f"between {min(gaps):.0f} and {max(gaps):.0f} bp, with the policy ahead in all eight cases" in readme
    doc = _flat(DOCS / "deep_hedging_regimes.md")
    for r in at50:
        cell = (f"| {r['train_seed']} | {r['eval_seeds'][0]}-{r['eval_seeds'][-1]} | "
                f"{r['deep']['cvar_bp']:.1f} ± {r['deep']['se_bp']:.1f} |")
        assert cell in doc, cell


def test_heston_step_shift_comes_from_the_committed_run():
    run = _json(DOCS / "heston_step_convergence.json")
    k110 = run["K"].index(110.0)
    d = run["diffs"]["800-200"]
    phrase = f"`{d['diff'][k110]:.4f} ± {d['se'][k110]:.4f}`"
    assert phrase in _flat(DOCS / "heston_reference.md"), phrase
    for pair in ("100-50", "200-100", "400-200", "800-400"):
        x = run["diffs"][pair]
        assert f"`{x['diff'][k110]:.4f} ± {x['se'][k110]:.4f}`" in _flat(DOCS / "heston_reference.md"), pair


def test_rough_bergomi_step_bias_comes_from_the_committed_runs():
    summary = _json(DOCS / "rb_step_convergence" / "summary.json")
    text = _flat(DOCS / "atm_skew_term_structure.md")
    for r in summary["rows"]:
        assert f"| {r['psi_bias50_rel']:.1%} |" in text, r["file"]
    s145 = next(x for x in summary["two_point_slopes"] if x["window_days"] == [1, 45])
    assert f"{s145['slope_n50']:.4f} at 50 steps and {s145['slope_extrapolated']:.4f} extrapolated" in text
    assert min(r["min_consecutive_psi_diff_z"] for r in summary["rows"]) >= 3.5


def test_real_path_intervals_come_from_the_committed_run():
    block = _json(DOCS / "hedging_real_paths_block.json")["block"]
    text = _flat(DOCS / "hedging_real_paths.md")

    def fmt(x):
        return f"{x:+.4f}".replace("-", "−")
    for asset in ("SPY", "BTC-USD"):
        for cost, label in ((0.001, "10 bp"), (0.005, "50 bp")):
            m = block[asset][f"cost_{cost}"]["mbb_b6"]
            lo, hi = m["cvar_diff_ci95"]
            assert f"{fmt(lo)} to {fmt(hi)}" in text, (asset, label)
            lo, hi = m["mean_diff_ci95"]
            assert f"{fmt(lo)} to {fmt(hi)}" in text, (asset, label)


# ---- data licensing ----

def _tracked(prefix: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.splitlines() if line]


def test_data_notice_covers_every_tracked_dataset():
    text = _flat(DATA_README)
    for path in ("`data/surfaces/equity/`", "`data/surfaces/deribit/`", "`data/btc_series/`",
                 "`data/pricing_map*/`", "`artifacts/deribit_snapshot_btc_20260805T214815Z.json`"):
        assert path in text, path
    assert "not relicensed" in text and "not covered by the MIT License" in text
    counts = {"data/surfaces/equity": "208 gzipped JSON", "data/surfaces/deribit": "418 gzipped JSON",
              "data/btc_series": "17 (", "data/pricing_map*": "482 `.npz` shards"}
    for prefix, phrase in counts.items():
        files = _tracked(prefix)
        if files is None:
            pytest.skip("git is not available to list tracked data")
        if not files:
            pytest.skip("not a git checkout")
        n = len(files)
        assert phrase.startswith(str(n)), (prefix, n, phrase)
        assert phrase in text, phrase


def test_notice_and_readme_scope_the_licence(readme):
    notice = _flat(NOTICE)
    assert "MIT License does not extend to the third-party market data" in notice
    for path in ("data/surfaces/equity/", "data/surfaces/deribit/",
                 "artifacts/deribit_snapshot_btc_20260805T214815Z.json"):
        assert path in notice, path
    licence = readme.split("## License")[1]
    assert "[data/README.md](data/README.md)" in licence and "[NOTICE](NOTICE)" in licence
    assert "not relicensed" in licence
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


@pytest.mark.parametrize("path", [README, DATA_README, NOTICE,
                                  DOCS / "hedging_real_paths.md",
                                  DOCS / "atm_skew_term_structure.md",
                                  DOCS / "heston_reference.md"], ids=lambda p: p.name)
def test_visitor_text_has_no_em_dashes(path):
    text = path.read_text(encoding="utf-8")
    assert "—" not in text and "&mdash;" not in text


@pytest.mark.parametrize("path", [DATA_README, NOTICE], ids=lambda p: p.name)
def test_data_notices_carry_no_narration_tells(path):
    hits = [(i, line) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if TELLS.search(re.sub(r"`[^`]*`", "", line))]
    assert not hits, hits
