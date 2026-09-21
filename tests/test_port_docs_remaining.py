"""The volatility, market, Asian-audit and hedging documents against their sources.

The documents under docs/ are written by hand around numbers that live in
committed artifacts (docs/*.json, artifacts/*) or come out of deterministic
calls (`HedgingEngine.compare()`). Each test here formats a number from its
source the way the document quotes it and requires the document to carry it, so
a figure cannot drift from what produced it. Where one quantity has two values
(a recorded and a refitted map RMSE, a validation draw and an audit grid, an
offline and a served run), the test requires both values and the label that
tells them apart.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
VOL_DOCS = tuple(DOCS / name for name in (
    "heston_reference.md", "no_arbitrage_surface.md", "joint_skew_refit.md",
    "atm_skew_term_structure.md", "real_market_data.md"))
REAL_PATHS = DOCS / "hedging_real_paths.md"
REGIMES = DOCS / "deep_hedging_regimes.md"
ASIAN_MD = DOCS / "asian_arbitrage_audit.md"
SNAPSHOT = ARTIFACTS / "deribit_snapshot_btc_20260805T214815Z.json"

MINUS = "−"

# The final grep of the tells checklist: history narration, candour markers,
# cleft tails, counting openers and em dashes.
TELLS = re.compile(
    r"previously|used to|no longer|retract|has been fixed|now fixed|now reports|honest|"
    r"genuinely|actually|deliberately|silently|quietly|hypothesis tested|"
    r"measured, not asserted|the easy, wrong|entitled to say|as it should|whole point|"
    r"which is why|which is what|which is how|worth (stating|noting|taking)|"
    r"deserves emphasis|most (interesting|dangerous)|Two things|Three things|"
    r"Two conclusions|Two further|—|&mdash;|READ THIS|IMPORTANT:",
    re.IGNORECASE)


def _text(name: str) -> str:
    """The document with runs of whitespace collapsed, so a reflow cannot break a phrase match."""
    return " ".join((DOCS / name).read_text(encoding="utf-8").split())


def _json(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _u(x: float, digits: int) -> str:
    """A number as the Unicode-minus documents print it."""
    return f"{x:.{digits}f}".replace("-", MINUS)


def _num(cell: str) -> float:
    return float(cell.replace(MINUS, "-").replace("+", "").replace("*", "").strip())


# ---- the five volatility and market documents ----

@pytest.mark.parametrize("doc", VOL_DOCS + (REAL_PATHS, REGIMES, ASIAN_MD),
                         ids=lambda p: p.name)
def test_cited_files_exist(doc):
    """Every `artifacts/...`, `docs/...`, `tests/...` and `scripts/...` path a document cites exists."""
    cited = set(re.findall(r"`((?:artifacts|docs|tests|scripts|backend)/[\w./]+\.\w+)`",
                           doc.read_text(encoding="utf-8")))
    assert cited, f"{doc.name} cites no files; the citation pattern has drifted"
    missing = sorted(p for p in cited if not (ROOT / p).exists())
    assert not missing, f"{doc.name} cites files absent from the tree: {missing}"


@pytest.mark.parametrize("doc", VOL_DOCS + (REAL_PATHS, REGIMES, ASIAN_MD),
                         ids=lambda p: p.name)
def test_prose_carries_no_checklist_tells(doc):
    hits = [(i, line) for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if TELLS.search(re.sub(r"`[^`]*`", "", line))]
    assert not hits, hits


@pytest.mark.parametrize("doc", VOL_DOCS + (REAL_PATHS, REGIMES, ASIAN_MD),
                         ids=lambda p: p.name)
def test_headings_are_noun_phrases(doc):
    bad = [line for line in doc.read_text(encoding="utf-8").splitlines()
           if re.match(r"#{1,6}\s*(\d[\d.]*\s*)?(What|Why|Where|How|Is)\b", line)
           or re.match(r"#{1,6}\s.*(\?|, not .*)$", line)]
    assert not bad, bad


def test_heston_map_rmse_recorded_and_refit():
    """1.07 is the recorded map RMSE and 1.10 the refit; the document labels each."""
    text, s = _text("heston_reference.md"), _json("heston_reference.json")["calibration"]["summary"]
    assert f"{s['map_refit_mean']:.3f} ± {s['map_refit_sd']:.3f} refit" in text
    assert f"{s['map_recorded_mean']:.3f} recorded" in text
    assert f"{s['map_recomputed_mean']:.3f}" in text
    summary = text.split("## 1.")[0]
    assert re.search(rf"refitted on the same quotes reaches {s['map_refit_mean']:.2f}", summary)
    assert re.search(rf"recorded for those captures in\s+\S+ is {s['map_recorded_mean']:.2f}\b", summary)
    assert f"{-s['heston_LS_minus_map_refit_mean']:.2f} vol points better" in text
    assert f"{-s['heston_LS_minus_map_recorded_mean']:.2f} on the recorded figure" in text
    assert f"{s['heston_LS_mean']:.3f} ± {s['heston_LS_sd']:.3f}" in text


def test_heston_calibration_table_matches_artifact():
    text, d = _text("heston_reference.md"), _json("heston_reference.json")
    for cap in d["calibration"]["captures"]:
        fits, m = cap["fits"], cap["map"]
        p = fits["LS"]["params"]
        row = (f"| `{cap['info']['capture']}` | {fits['LS']['rmse_volpts']:.3f} | "
               f"{fits['LS-kappa10']['rmse_volpts']:.3f} | {fits['LS-kappa3']['rmse_volpts']:.3f} | "
               f"{m['recorded']['rmse_volpts']:.3f} | {m['refit']['rmse_volpts']:.3f} | "
               f"{p['kappa']:.0f} | {p['sigma_v']:.2f} | {p['rho']:.3f} | {math.sqrt(p['theta']):.3f} |")
        assert row in text, row


def test_heston_sigma_v_and_feller_ratio():
    """Vol-of-variance is 4.33 to 4.72 (mean 4.54); the Feller ratio is 0.22 to 0.28."""
    text, d = _text("heston_reference.md"), _json("heston_reference.json")
    fits = [c["fits"]["LS"] for c in d["calibration"]["captures"]]
    sv = [f["params"]["sigma_v"] for f in fits]
    fr = [f["feller_ratio"] for f in fits]
    assert f"between {min(sv):.2f} and {max(sv):.2f} (mean {statistics.mean(sv):.2f})" in text
    assert f"4.3 to 4.7 (mean {statistics.mean(sv):.1f})" in text
    assert f"is {min(fr):.2f} to {max(fr):.2f}" in text
    assert all(f["pinned"] for f in fits), "an LS optimum left the kappa bound; the text says every one is pinned"


def test_heston_protocol_matches_artifact():
    text, d = _text("heston_reference.md"), _json("heston_reference.json")
    b = d["protocol"]["bounds"]
    assert f"`kappa ∈ [{b['kappa'][0]:g}, {b['kappa'][1]:g}]`" in text
    assert f"`sigma_v ∈ [{b['sigma_v'][0]:g}, {b['sigma_v'][1]:g}]`" in text
    assert f"`rho ∈ [{b['rho'][0]:g}, {b['rho'][1]:g}]`" in text
    n = [c["info"]["n_quotes_capture"] for c in d["calibration"]["captures"]]
    assert f"{min(n)} to {max(n)} quotes each" in text
    s = d["skew"]["summary"]
    assert f"{-s['heston_LS_minus_market_sigmas']:.1f} market standard errors" in text
    rough = (s["rough_bergomi_b"] - s["market_b"]) / s["market_se_b"]
    assert f"{-rough:.1f} market standard errors away" in text


def test_surface_rmse_stated_on_both_grids():
    """0.22 vol points belongs to the validation draw and 0.38 to the audit grid."""
    text, d = _text("no_arbitrage_surface.md"), _json("no_arbitrage_surface.json")
    val, cmp_ = d["fit_metrics"]["validation"], d["comparison"]
    assert f"{val['iv_rmse_volpts_resolved']:.2f}" == "0.22" and f"{cmp_['iv_rmse_volpts_resolved']:.2f}" == "0.38"
    summary = text.split("## 1.")[0]
    for where in (summary, text.split("## 7.")[1]):
        assert "0.22" in where and "0.38" in where
        assert f"{val['n']:,}" in where and f"{val['n_resolved']:,}" in where
        assert f"{cmp_['n_resolved']:,}" in where and f"{cmp_['n_points']:,}" in where
    assert f"{cmp_['iv_rmse_volpts_resolved']:.3f} vol points" in text
    assert f"{val['iv_rmse_volpts_resolved']:.3f} vol points" in text
    assert f"{val['price_rmse_bps']:.3f} bps" in text and f"{cmp_['price_rmse_bps_all']:.2f} bps" in text


def test_surface_violation_rates_match_artifact():
    text, d = _text("no_arbitrage_surface.md"), _json("no_arbitrage_surface.json")
    iv, cov = d["teacher"]["iv_space"], d["teacher"]["coverage"]
    cal, but = iv["calendar_resolved"], iv["butterfly_resolved"]
    assert cal["n_violations"] == 99 and cal["n_points"] == 56638
    assert f"{100 * cal['violation_fraction']:.2f}% calendar" in text
    assert f"{100 * but['violation_fraction']:.2f}% butterfly" in text
    assert "0.18%" not in text, "99/56,638 is 0.17%, and the table carries 0.175%"
    assert f"{100 * cal['violation_fraction']:.3f}%" in text
    assert f"{100 * iv['butterfly_all_defined']['violation_fraction']:.2f}%" in text
    assert f"{100 * iv['calendar_all_defined']['violation_fraction']:.2f}%" in text
    assert f"{cov['below_intrinsic_worst_bps']:.1f} bps of strike" in text
    assert f"{100 * cov['resolved_fraction']:.1f}% of the box" in text
    for student in d["student"]["iv_space"].values():
        assert student["n_violations"] == 0
    rmse = [m["teacher_price_rmse_bps"] for m in d["mc_check"]]
    assert f"{min(rmse):.2f} to {max(rmse):.2f} bps" in text


def test_joint_refit_exponents_and_cost():
    text, d = _text("joint_skew_refit.md"), _json("joint_skew_refit.json")
    mc, market = d["mc"], d["market"]["pooled_8_captures"]
    for key in ("served", "joint", "pure", "interior"):
        fit = mc[f"{key}_ladder_fit"]
        assert f"{_u(fit['b'], 3)} ± {fit['se_b']:.3f}" in text, key
    dist = {k: abs(mc[f"{k}_ladder_fit"]["b"] - market["b"]) / market["se_b"] for k in ("served", "joint", "pure")}
    assert f"{dist['served']:.1f} market standard errors" in text
    assert (f"joint parameters are {dist['joint']:.1f} SE" in text
            and f"calibration file is {dist['served']:.1f} SE away" in text
            and f"pure-smile optimum {dist['pure']:.1f} SE" in text)
    cb = d["cost_benefit"]
    assert f"{cb['rmse_cost_vs_pure_smile_mean_3_captures']:.2f} vol points of" in text
    assert f"{cb['rmse_cost_vs_pure_smile_volpts']:.2f} on the" in text
    gaps = [mc[f"{k}_smile"]["rmse_mc_volpts"] - mc[f"{k}_smile"]["rmse_map_volpts"]
            for k in ("served", "joint", "pure", "interior")]
    assert text.count(f"{min(gaps):.2f}-{max(gaps):.2f} vol points") >= 2


def test_joint_refit_names_served_checkpoint():
    """The served 0DTE surrogate carries the accepted 20 August fit, which differs from the calibration file."""
    text = _text("joint_skew_refit.md")
    dyn = _json("no_arbitrage_surface.json")["dynamics"]
    served = _json("joint_skew_refit.json")["served"]["params"]
    assert served["accepted"] is False and served["H"] != dyn["H"]
    assert f"η {dyn['eta']:.3f}, ρ {_u(dyn['rho'], 3)}, H {dyn['H']:.3f}" in text
    assert "was trained on" not in text.split("## 4.")[1].split("## 5.")[0]


def test_atm_skew_headline_numbers_match_artifact():
    text, spy = _text("atm_skew_term_structure.md"), _json("atm_skew_term_structure.json")["spy"]
    fits = spy["fits"]
    assert f"{fits['model_short']['b']:.3f} +- {fits['model_short']['se_b']:.3f}" in text
    assert f"{fits['market_pooled']['b']:.3f} +- {fits['market_pooled']['se_b']:.3f}" in text
    rows = spy["market_per_capture"]
    w = [1.0 / r["se_b"] ** 2 for r in rows]
    mean = sum(wi * r["b"] for wi, r in zip(w, rows)) / sum(w)
    chi2 = sum(((r["b"] - mean) / r["se_b"]) ** 2 for r in rows)
    assert f"chi^2 of {chi2:.1f} on {len(rows) - 1} degrees of freedom" in text
    below = sum(r["se_b"] < spy["market_spread"]["sd_b"] for r in rows)
    assert below == 5 and "five of the eight per-capture SEs" in text
    thu = statistics.mean(r["b"] for r in rows if r["pricing_time"].startswith("2026-08-20"))
    fri = statistics.mean(r["b"] for r in rows if r["pricing_time"].startswith("2026-08-21"))
    assert f"(mean {thu:.3f})" in text and f"(mean {fri:.3f})" in text


@pytest.fixture(scope="module")
def deribit_surface():
    from backend.quant.deribit import load_snapshot
    from backend.quant.surface import build_surface
    return build_surface(load_snapshot(SNAPSHOT))


def test_market_doc_quote_flags_and_example(deribit_surface):
    text = _text("real_market_data.md")
    quotes = list(deribit_surface.quotes)
    flagged = [q for q in quotes if q.flags]
    assert f"Of {len(quotes)} quotes, {len(flagged)} are flagged" in text
    counts: dict[str, int] = {}
    for q in flagged:
        for flag in q.flags:
            counts[flag] = counts.get(flag, 0) + 1
    for flag, n in counts.items():
        assert f"| `{flag}` | {n} |" in text
    assert f"sum to {sum(counts.values())}" in text
    put = next(q for q in quotes if q.right == "put" and q.strike == 70000.0
               and q.tenor * 365 * 24 < 12)
    assert "low_vega" in put.flags
    assert f"{put.iv_mark * 100:.1f} vol points" in text and f"{put.exchange_mark_iv:.1f}" in text
    assert f"{put.strike - put.forward:,.2f} of its {put.mark_btc * put.forward:,.2f} dollar mark" in text


def test_market_doc_forward_and_spread_statements(deribit_surface):
    text = _text("real_market_data.md")
    report = json.loads((ARTIFACTS / "deribit_arbitrage_report.json").read_text(encoding="utf-8"))
    rows = report["forward_consistency"]
    gaps = [abs(r["synthetic_vs_future_median_bps"]) for r in rows]
    assert len(rows) == 12 and f"{statistics.median(gaps):.1f} bps" in text and f"{max(gaps):.2f} bps" in text
    assert all(g < r["bracket_width_median_bps"] / 2 for g, r in zip(gaps, rows))
    by_expiry: dict[int, list[float]] = {}
    for q in deribit_surface.clean():
        if q.right == "put":
            by_expiry.setdefault(round(q.tenor * 365), []).append((q.ask_usd - q.bid_usd) / q.forward * 1e4)
    widest = max(by_expiry, key=lambda e: statistics.median(by_expiry[e]))
    assert widest == 232
    assert f"median bid-ask of {statistics.median(by_expiry[widest]):.0f} bps" in text
    assert f"maximum of {max(by_expiry[widest]):.0f}" in text


# ---- the Asian audit document beyond the rates test_asian_audit.py pins ----

@pytest.fixture(scope="module")
def asian_report() -> dict:
    path = DOCS / "asian_arbitrage_audit.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def asian_markdown() -> str:
    if not ASIAN_MD.exists():
        pytest.skip(f"{ASIAN_MD} not present")
    return ASIAN_MD.read_text(encoding="utf-8")


def test_softplus_head_imposes_positivity():
    """C >= 0 holds by construction: each member ends in Softplus and the shared
    output scale is positive, so the document reports that row as imposed."""
    from backend.quant.engine import PricingEngine
    from scripts import asian_arbitrage_audit as aud

    checkpoint = ARTIFACTS / "model.pt"
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint} not present")
    engine = PricingEngine(checkpoint)
    assert engine._output_scale > 0.0
    for member in engine.members:
        assert isinstance(member.out, nn.Softplus)
    # Far out of the money at the shortest audited maturity and lowest vol.
    m = np.array([0.5, 0.5])
    T = np.array([0.05, 2.0])
    sigma = np.array([0.05, 0.05])
    rate = np.array([0.04, 0.04])
    assert np.all(aud.unit_strike_price(engine, m, T, sigma, rate) > 0.0)


def test_asian_doc_quotes_positivity_floor_from_json(asian_report, asian_markdown):
    worst = asian_report["conditions"]["positivity_bps"]["all"]["worst_value"]
    assert f"{worst:.3f} bps" in asian_markdown
    zero_rows = [row for row in asian_report["arbitration"]
                 if row["m"] == 0.5 and row["mc_price_bps"] == 0.0
                 and row["curran_bps"] < 5e-4]
    assert len(zero_rows) == 3
    net = [row["network_bps"] for row in zero_rows]
    assert f"{min(net):.2f} to" in asian_markdown
    assert f"{max(net):.2f} bps" in asian_markdown.replace("\n", " ")


def test_asian_doc_quotes_violation_counts_from_json(asian_report, asian_markdown):
    cond = asian_report["conditions"]
    assert f"{cond['convexity_d2C_dK2_ge_0']['all']['n_violations']:,}" in asian_markdown
    assert f"{cond['delta_ge_0']['all']['n_violations']} points" in asian_markdown.replace("\n", " ")
    fly = {row["label"]: row for row in asian_report["butterfly_check"]}
    resolved = fly["worst butterfly, resolved region"]
    assert f"{resolved['network_butterfly_bps']:.3f} bps" in asian_markdown


def test_asian_doc_scopes_vega_as_black_scholes(asian_markdown):
    flat = asian_markdown.replace("\n", " ")
    assert "Carr, Ewald & Xiao 2008" in flat
    assert "forward strip" in flat
    assert "Every row except vega is a static-arbitrage condition" in flat


def test_asian_doc_typography(asian_markdown):
    assert "**" not in asian_markdown
    assert chr(0x2014) not in asian_markdown          # em dash


# ---- the hedging documents ----

def test_real_paths_tables_match_raw_output():
    """Every hedger row of hedging_real_paths.txt appears in the document's tables."""
    label = {"deep": "deep", "delta": "delta", "whalley_wilmott": "Whalley-Wilmott"}
    table = [[c.replace("*", "").strip() for c in line.strip().strip("|").split("|")]
             for line in REAL_PATHS.read_text(encoding="utf-8").splitlines()
             if line.startswith("| ")]
    cost = None
    checked = 0
    for line in (DOCS / "hedging_real_paths.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"== cost ([\d.]+) ==", line)
        if m:
            cost = f"{round(float(m.group(1)) * 1e4)} bp"
            continue
        m = re.match(r"\s+(deep|delta|whalley_wilmott)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if not m:
            continue
        raw = [float(v) for v in m.groups()[1:]]
        matches = [row for row in table
                   if len(row) == 7 and row[0] == cost and row[1] == label[m.group(1)]
                   and [_num(c) for c in row[2:]] == pytest.approx(raw, abs=1e-9)]
        assert matches, f"no table row for {m.group(1)} at {cost}: {raw}"
        checked += 1
    assert checked == 12


def test_real_paths_checkpoint_identity():
    """hedger.pt and the v1 unconstrained-measure checkpoint are the same bytes, as stated."""
    import hashlib

    md5 = {name: hashlib.md5((ARTIFACTS / name).read_bytes()).hexdigest()
           for name in ("hedger.pt", "hedger_v1_unconstrained_measure.pt")}
    assert md5["hedger.pt"] == md5["hedger_v1_unconstrained_measure.pt"]
    text = _text("hedging_real_paths.md")
    assert f"both MD5 `{md5['hedger.pt']}`" in text


def _two_run_table() -> dict[str, tuple[str, str]]:
    rows = {}
    for line in REGIMES.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 3 and cells[0] and not set(cells[0]) <= {"-"}:
            rows[cells[0]] = (cells[1], cells[2])
    return rows


def _level(cell: str) -> tuple[float, float]:
    value, se = re.match(r"([\d.]+) ± ([\d.]+) bp", cell).groups()
    return float(value), float(se)


def test_regimes_offline_column_matches_json():
    """The offline-matrix column of the 50 bp table is the JSON cell it names."""
    data = _json("deep_hedging_regimes.json")
    cell = data["cells"]["rbergomi_jumps|0.005"]
    sources = {"deep CVaR_95": cell["deep"]["rbergomi_jumps"],
               "delta CVaR_95": cell["baselines"]["delta"],
               "Whalley-Wilmott CVaR_95": cell["baselines"]["whalley_wilmott"],
               "linear CVaR_95": cell["baselines"]["linear"]}
    table = _two_run_table()
    for name, src in sources.items():
        value, se = _level(table[name][0])
        assert value == pytest.approx(src["cvar95"] * 1e4, abs=0.05)
        assert se == pytest.approx(src["cvar95_se"] * 1e4, abs=0.05)
    assert data["protocol"]["seeds"] == [17, 18, 19, 20, 21]
    assert data["protocol"]["n_boot"] == 500


def test_regimes_served_column_regenerates():
    """The served-run column and the paired table come out of `compare()`.

    Same call as POST /api/hedge with rough dynamics at 50 bp: sigma and rate
    from the checkpoint's measure_params, the rough-jumps measure only.
    """
    from backend.quant.hedging import HedgingEngine

    eng = HedgingEngine(ARTIFACTS / "hedger_rbergomi_jumps.pt")
    params = eng.meta["measure_params"]
    out = eng.compare(params["xi"] ** 0.5, params["rate"], 0.005,
                      primary="rbergomi_jumps", measures=("rbergomi_jumps",))

    table = _two_run_table()
    for name, key in (("deep CVaR_95", "deep"), ("delta CVaR_95", "delta"),
                      ("Whalley-Wilmott CVaR_95", "whalley_wilmott"),
                      ("linear CVaR_95", "linear")):
        value, se = _level(table[name][1])
        assert value == pytest.approx(out[key]["cvar95"] * 1e4, abs=0.06)
        assert se == pytest.approx(out[key]["cvar95_se"] * 1e4, abs=0.06)

    pairs = out["paired_bootstrap"]["pairs"]
    keys = {"deep − delta": "deep|delta",
            "deep − Whalley-Wilmott": "deep|whalley_wilmott",
            "deep − linear": "deep|linear",
            "delta − Whalley-Wilmott": "delta|whalley_wilmott",
            "Whalley-Wilmott − linear": "whalley_wilmott|linear"}
    checked = 0
    for line in REGIMES.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or cells[0] not in keys:
            continue
        pair = pairs[keys[cells[0]]]
        lo, hi = (_num(v) for v in cells[5].strip("[]").split(","))
        dollars = 100.0                      # per option at a $100 strike
        assert _num(cells[1]) == pytest.approx(pair["diff"] * dollars, abs=1.5e-3)
        assert _num(cells[2]) == pytest.approx(pair["se"] * dollars, abs=1.5e-3)
        assert _num(cells[3]) == pytest.approx(pair["hypot_se"] * dollars, abs=1.5e-3)
        assert _num(cells[4]) == pytest.approx(pair["corr"], abs=6e-3)
        assert lo == pytest.approx(pair["ci_low"] * dollars, abs=1.5e-3)
        assert hi == pytest.approx(pair["ci_high"] * dollars, abs=1.5e-3)
        checked += 1
    assert checked == 5
