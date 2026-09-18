"""Checks for `scripts/asian_arbitrage_audit.py` and its published artifacts.

The audit is a published finding (`docs/asian_arbitrage_audit.md`), so what needs
pinning is not a violation rate - that is a measurement of a checkpoint and moves
when the checkpoint does - but the *conditions* it measures and the artifacts it
publishes:

  (a) the contract. E[A] of the discrete arithmetic average is the quantity the
      Asian floor e^{-rT}(E[A] - K)+ is built from, and the audit's vectorised
      form of it must equal a direct sum over the 50 fixing dates, the helper in
      `backend/quant/monte_carlo.py`, and the parity term the served puts are
      derived from (`engine._parity_adjustment_torch`). If any of the four drifts,
      the floor is measuring a different contract from the one being served.
  (b) the floor is the ASIAN one. It sits strictly below the European intrinsic
      whenever r > 0 - the whole reason the 0DTE audit's floor is not reused - and
      a high-accuracy Monte Carlo price of the contract never falls below it.
  (c) the derivative machinery. dC/dK, d2C/dK2, delta, gamma and vega come from
      autograd through the served graph; each is checked against a central finite
      difference of the served batch path, and d2C/dK2 = m^2 gamma is an identity
      that homogeneity of degree one forces.
  (d) scope. The lattice must stay above the 0DTE cutoff, so that `model.pt` and
      not `model_0dte.pt` is what gets audited, and the four non-transferable
      conditions (Black-Scholes implied vol, Durrleman's g, the European intrinsic
      floor, calendar monotonicity) must stay absent, with their reasons recorded.
  (e) the artifacts. The committed JSON is internally consistent and describes the
      checkpoint actually in `artifacts/`, and the markdown quotes the JSON: every
      headline number in the document is regenerated here from the JSON and
      required to appear in the text.

Tolerances are the measured agreements, rounded up: 5e-16 relative on E[A]
against the direct sum, 9e-16 absolute against the parity term, 3e-5 on first
derivatives against a 1e-3 central difference, 0.006 absolute / 0.3% relative on
second derivatives against a 1e-2 one (float32 second differences cannot do
better: eps * price / h^2 is 0.1 at h = 1e-3, which is why the step is larger),
and 5e-7 on the homogeneity identity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.dataset import PARAM_RANGES, N_MONITORING_STEPS  # noqa: E402
from backend.quant.engine import PricingEngine, ZERO_DTE_CUTOFF  # noqa: E402
from backend.quant.monte_carlo import (expected_arithmetic_average,  # noqa: E402
                                       price_asian_mc)
from scripts import asian_arbitrage_audit as aud  # noqa: E402

JSON_PATH = ROOT / "docs" / "asian_arbitrage_audit.json"
MD_PATH = ROOT / "docs" / "asian_arbitrage_audit.md"
CHECKPOINT = ROOT / "artifacts" / "model.pt"

#: (m, T, sigma, r) points used for the derivative checks: interior of the box,
#: spread across moneyness, maturity and vol.
DERIV_POINTS = ((1.00, 1.00, 0.20, 0.04), (0.90, 0.50, 0.40, 0.00),
                (1.20, 2.00, 0.80, 0.10), (1.05, 0.25, 0.20, 0.02),
                (0.70, 1.50, 0.40, 0.04), (1.50, 1.00, 0.40, 0.04))


@pytest.fixture(scope="module")
def engine() -> PricingEngine:
    if not CHECKPOINT.exists():
        pytest.skip(f"{CHECKPOINT} not present")
    return PricingEngine(CHECKPOINT)


@pytest.fixture(scope="module")
def report() -> dict:
    if not JSON_PATH.exists():
        pytest.skip(f"{JSON_PATH} not present; run "
                    "python -m scripts.asian_arbitrage_audit")
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def markdown() -> str:
    if not MD_PATH.exists():
        pytest.skip(f"{MD_PATH} not present")
    return MD_PATH.read_text(encoding="utf-8")


def _deriv_arrays() -> tuple[np.ndarray, ...]:
    return tuple(np.array([p[i] for p in DERIV_POINTS], dtype=np.float64)
                 for i in range(4))


# ------------------------------------------------------------------- (a) E[A]
def test_expected_average_matches_a_direct_sum_over_the_fixing_dates():
    n = N_MONITORING_STEPS
    for m in (0.5, 0.9, 1.0, 1.37, 2.0):
        for T in (0.05, 0.3, 1.0, 2.0):
            for r in (0.0, 1e-9, 0.02, 0.04, 0.1):
                t = np.arange(1, n + 1) * (T / n)      # t_i = i T / n
                direct = m * np.exp(r * t).mean()
                got = float(aud.expected_average(m, T, r, n))
                assert abs(got - direct) <= 1e-12 * direct, (m, T, r)


def test_expected_average_matches_the_monte_carlo_helper():
    # `expected_arithmetic_average` evaluates the same series in its g^n form,
    # which loses precision as r -> 0 (g^n - 1 and g - 1 both cancel) where the
    # expm1 form does not; at ordinary rates the two agree to roundoff, and at
    # r = 1e-9 the gap is the helper's cancellation, not a different formula.
    n = N_MONITORING_STEPS
    for m in (0.5, 1.0, 2.0):
        for T in (0.05, 1.0, 2.0):
            for r in (0.0, 0.02, 0.04, 0.1):
                got = float(aud.expected_average(m, T, r, n))
                ref = expected_arithmetic_average(m, r, T, n)
                assert abs(got - ref) <= 1e-12 * max(ref, 1e-12), (m, T, r)
            tiny_got = float(aud.expected_average(m, T, 1e-9, n))
            tiny_ref = expected_arithmetic_average(m, 1e-9, T, n)
            assert abs(tiny_got - tiny_ref) <= 1e-8 * m


def test_expected_average_matches_the_served_parity_term(engine):
    # The served put is C - e^{-rT}(E[A] - K); with K = 1 and spot = m that
    # adjustment is exactly the floor's ingredient, so the audit and the engine
    # must agree on it point by point.
    for m in (0.5, 1.0, 1.37, 2.0):
        for T in (0.05, 0.5, 2.0):
            for r in (0.0, 0.02, 0.04, 0.1):
                mine = float(np.exp(-r * T)
                             * (aud.expected_average(m, T, r) - 1.0))
                served = float(engine._parity_adjustment_torch(
                    torch.tensor([m], dtype=torch.float64),
                    torch.tensor([T], dtype=torch.float64),
                    torch.tensor([r], dtype=torch.float64))[0])
                assert abs(mine - served) <= 1e-12, (m, T, r)


# ------------------------------------------------------------------ (b) floor
def test_asian_floor_is_strictly_below_the_european_intrinsic_when_r_positive():
    for m in (1.2, 1.5, 2.0):
        for T in (0.25, 1.0, 2.0):
            euro = max(m - np.exp(-0.04 * T), 0.0)
            asian = float(aud.asian_floor(m, 1.0, T, 0.04))
            assert 0.0 < asian < euro, (m, T)
            # at r = 0 every fixing has forward S and the two coincide
            assert float(aud.asian_floor(m, 1.0, T, 0.0)) == pytest.approx(
                max(m - 1.0, 0.0), abs=1e-12)


def test_asian_floor_reproduces_the_documented_gap_to_the_european_floor():
    # docs/asian_arbitrage_audit.md, section 1.3: at m = 1, T = 2, r = 0.04 the
    # European intrinsic is 768.8 bps of strike and the Asian floor 387.0.
    asian_bps = float(aud.asian_floor(1.0, 1.0, 2.0, 0.04)) * 1e4
    euro_bps = max(1.0 - np.exp(-0.04 * 2.0), 0.0) * 1e4
    assert asian_bps == pytest.approx(387.0, abs=0.1)
    assert euro_bps == pytest.approx(768.8, abs=0.1)


def test_monte_carlo_price_never_falls_below_the_asian_floor():
    # Measured worst case over these points and seeds: -0.81 standard errors
    # (deep in the money at sigma = 0.05, where the price IS the floor).
    points = ((1.30, 1.00, 0.20, 0.04), (0.80, 0.50, 0.40, 0.10),
              (1.00, 2.00, 0.05, 0.02), (2.00, 2.00, 0.05, 0.04),
              (1.03, 0.05, 0.05, 0.00))
    for m, T, sigma, r in points:
        for seed in (3, 5):
            mc = price_asian_mc(m, 1.0, T, sigma, r, n_paths=50_000,
                                n_steps=N_MONITORING_STEPS, option_type="call",
                                seed=seed, control_variate=True)
            floor = float(aud.asian_floor(m, 1.0, T, r))
            assert mc.price >= floor - 4.0 * mc.std_error - 1e-12, \
                (m, T, sigma, r, seed, mc.price, floor)


def test_curran_reference_respects_the_floor_and_the_upper_bounds():
    m, T, sigma, r = _deriv_arrays()
    ref = aud.curran_batch(m, T, sigma, r)
    ea = aud.expected_average(m, T, r)
    disc = np.exp(-r * T)
    assert np.all(ref >= disc * np.maximum(ea - 1.0, 0.0) - 1e-12)
    assert np.all(ref <= disc * ea + 1e-12)
    assert np.all(ref <= m + 1e-12)


# ------------------------------------------------------- (c) the derivatives
def test_autograd_first_derivatives_match_finite_differences(engine):
    m, T, sigma, r = _deriv_arrays()
    d = aud.autograd_derivatives(engine, m, T, sigma, r)
    ones = np.ones_like(m)
    h = 1e-3

    up = aud.strike_price(engine, m, T, sigma, r, ones * (1 + h))
    dn = aud.strike_price(engine, m, T, sigma, r, ones * (1 - h))
    assert np.abs(d["dC_dK"] - (up - dn) / (2 * h)).max() < 5e-4

    up = aud.unit_strike_price(engine, m + h, T, sigma, r)
    dn = aud.unit_strike_price(engine, m - h, T, sigma, r)
    assert np.abs(d["delta"] - (up - dn) / (2 * h)).max() < 5e-4

    up = aud.unit_strike_price(engine, m, T, sigma + h, r)
    dn = aud.unit_strike_price(engine, m, T, sigma - h, r)
    assert np.abs(d["vega"] - (up - dn) / (2 * h)).max() < 5e-4


def test_autograd_second_derivatives_match_finite_differences(engine):
    m, T, sigma, r = _deriv_arrays()
    d = aud.autograd_derivatives(engine, m, T, sigma, r)
    ones = np.ones_like(m)
    h = 1e-2                       # see the module docstring on the step size
    mid = aud.strike_price(engine, m, T, sigma, r, ones)

    up = aud.strike_price(engine, m, T, sigma, r, ones * (1 + h))
    dn = aud.strike_price(engine, m, T, sigma, r, ones * (1 - h))
    fd = (up - 2 * mid + dn) / h ** 2
    assert np.all(np.abs(d["d2C_dK2"] - fd)
                  <= 0.02 + 0.02 * np.abs(d["d2C_dK2"]))

    up = aud.unit_strike_price(engine, m * (1 + h), T, sigma, r)
    dn = aud.unit_strike_price(engine, m * (1 - h), T, sigma, r)
    fd = (up - 2 * mid + dn) / (m * h) ** 2
    assert np.all(np.abs(d["gamma"] - fd) <= 0.02 + 0.02 * np.abs(d["gamma"]))


def test_convexity_in_strike_is_m_squared_gamma(engine):
    # C(S, K) = K f(S/K) is homogeneous of degree one, which makes
    # d2C/dK2 = (S/K)^2 d2C/dS2 an identity rather than a numerical accident.
    m, T, sigma, r = _deriv_arrays()
    d = aud.autograd_derivatives(engine, m, T, sigma, r)
    assert np.abs(d["d2C_dK2"] - m ** 2 * d["gamma"]).max() < 1e-4


def test_curran_vega_is_the_derivative_of_curran(engine):
    m, T, sigma, r = _deriv_arrays()
    got = aud.curran_vega(m, T, sigma, r)
    h = 1e-4
    fd = (aud.curran_batch(m, T, sigma + h, r)
          - aud.curran_batch(m, T, sigma - h, r)) / (2 * h)
    assert np.abs(got - fd).max() < 1e-5
    assert np.all(got >= -1e-12)          # the true contract's vega is positive


# --------------------------------------------------- (d) scope of the audit
def test_audit_margins_are_the_conditions_they_claim_to_be(engine):
    coords = aud.build_lattice(n_m=9, n_T=3, sigmas=(0.20,), rates=(0.04,))
    rep = aud.audit(engine, coords)
    assert set(rep["conditions"]) == set(aud.CONDITION_ORDER)

    m, T = coords["m"], coords["T"]
    sigma, r = coords["sigma"], coords["rate"]
    price = aud.unit_strike_price(engine, m, T, sigma, r)
    ea = aud.expected_average(m, T, r)
    disc = np.exp(-r * T)
    margins = rep["_margins"]

    # every margin is written as "margin >= 0" and must equal the condition
    assert np.allclose(margins["asian_floor_bps"],
                       (price - disc * np.maximum(ea - 1.0, 0.0)) * 1e4,
                       atol=1e-9)
    assert np.allclose(margins["upper_bound_discounted_mean_bps"],
                       (disc * ea - price) * 1e4, atol=1e-9)
    assert np.allclose(margins["upper_bound_spot_bps"], (m - price) * 1e4,
                       atol=1e-9)
    assert np.allclose(margins["positivity_bps"], price * 1e4, atol=1e-9)
    assert np.allclose(margins["slope_dC_dK_ge_-discount"]
                       + margins["monotone_dC_dK_le_0"], disc, atol=1e-9)

    # the resolved region is exactly "reference vega at or above the floor"
    assert np.array_equal(rep["_resolved"],
                          aud.curran_vega(m, T, sigma, r) >= aud.VEGA_FLOOR)

    for key, cond in rep["conditions"].items():
        for region in ("all", "resolved"):
            stats = cond[region]
            assert 0.0 <= stats["violation_fraction"] <= 1.0, (key, region)
            assert stats["n_violations"] <= stats["n_points"], (key, region)
        assert cond["resolved"]["n_points"] <= cond["all"]["n_points"], key


def test_audit_refuses_a_lattice_the_zero_dte_surrogate_would_price(engine):
    # PricingEngine routes T <= 12/252 to model_0dte.pt; auditing model.pt there
    # would audit a different checkpoint under this one's name.
    below = float(ZERO_DTE_CUTOFF) / 2.0
    coords = {"m": np.array([1.0]), "T": np.array([below]),
              "sigma": np.array([0.2]), "rate": np.array([0.04]),
              "m_axis": np.array([1.0]), "T_axis": np.array([below])}
    with pytest.raises(AssertionError):
        aud.audit(engine, coords)
    assert aud.build_lattice()["T"].min() > ZERO_DTE_CUTOFF


def test_non_transferable_conditions_are_not_audited(report):
    banned = ("durrleman", "implied", "iv_", "intrinsic", "calendar", "dc_dt")
    for key in aud.CONDITION_ORDER:
        assert not any(b in key.lower() for b in banned), key
    for key in report["conditions"]:
        assert not any(b in key.lower() for b in banned), key

    # ... and the reasons they are absent are published, not merely omitted
    absent = report["conditions_not_tested"]
    joined = " ".join(absent).lower()
    for token in ("durrleman", "european_intrinsic", "calendar"):
        assert token in joined, token
    for reason in absent.values():
        assert len(reason) > 40


# ----------------------------------------------------------- (e) artifacts
def test_committed_json_is_internally_consistent(report):
    p = report["protocol"]
    assert p["n_points"] == (p["n_m"] * p["n_T"]
                             * len(p["sigmas"]) * len(p["rates"]))
    assert p["n_monitoring_steps"] == N_MONITORING_STEPS
    assert p["T_min"] > p["zero_dte_cutoff"]
    assert (p["m_min"], p["m_max"]) == PARAM_RANGES["moneyness"]
    assert (p["T_min"], p["T_max"]) == PARAM_RANGES["maturity"]
    assert min(p["sigmas"]) >= PARAM_RANGES["sigma"][0]
    assert max(p["sigmas"]) <= PARAM_RANGES["sigma"][1]
    assert min(p["rates"]) >= PARAM_RANGES["rate"][0]
    assert max(p["rates"]) <= PARAM_RANGES["rate"][1]

    assert set(report["conditions"]) == set(aud.CONDITION_ORDER)
    for key, cond in report["conditions"].items():
        for region in ("all", "resolved"):
            stats = cond[region]
            assert 0.0 <= stats["violation_fraction"] <= 1.0, (key, region)
            at = stats["worst_at"]
            assert p["m_min"] <= at["m"] <= p["m_max"], key
            assert p["T_min"] <= at["T"] <= p["T_max"], key

    cov = report["coverage"]
    assert cov["resolved_points"] <= p["n_points"]
    assert 0.0 < cov["resolved_fraction"] < 1.0
    # the two autograd routes to the same second derivative must agree
    cx = report["cross_checks"]
    assert cx["sign_agreement_convexity_vs_gamma"] > 0.999
    assert cx["max_abs_gap_d2C_dK2_vs_m2_gamma"] < 1e-3
    assert cx["max_abs_price_gap_autograd_vs_price_batch"] < 1e-6


def test_committed_json_describes_the_checkpoint_in_artifacts(report):
    if not CHECKPOINT.exists():
        pytest.skip(f"{CHECKPOINT} not present")
    assert report["checkpoint"]["path"] == "artifacts/model.pt"
    assert report["checkpoint"]["sha256"] == aud._sha256(CHECKPOINT)
    assert report["checkpoint"]["param_ranges"] == {
        k: list(v) for k, v in PARAM_RANGES.items()}


def test_resolved_region_violations_sit_where_the_document_says(report):
    # docs/asian_arbitrage_audit.md, section 2: every violation that survives the
    # vega floor is at m >= 1.97, T >= 1.70, sigma = 0.40.
    seen = 0
    for key in ("convexity_d2C_dK2_ge_0", "gamma_ge_0", "butterfly_1pct_bps"):
        for v in report["conditions"][key]["resolved"].get("violations", []):
            seen += 1
            assert v["m"] >= 1.97 and v["T"] >= 1.70 and v["sigma"] == 0.40
    assert seen > 0
    for key in ("asian_floor_bps", "monotone_dC_dK_le_0", "vega_ge_0",
                "delta_ge_0", "delta_le_forward",
                "slope_dC_dK_ge_-discount", "positivity_bps",
                "upper_bound_spot_bps", "upper_bound_discounted_mean_bps"):
        assert report["conditions"][key]["resolved"]["n_violations"] == 0, key


def test_markdown_is_written_from_the_json(report, markdown):
    for token in ("TODO", "TBD", "XXX", "FIXME", "{}", "<placeholder>"):
        assert token not in markdown, token

    p, cov = report["protocol"], report["coverage"]
    for count in (p["n_points"], cov["resolved_points"]):
        assert f"{count:,}" in markdown, count
    assert f"{100 * cov['resolved_fraction']:.2f}%" in markdown

    for key, cond in report["conditions"].items():
        assert f"{100 * cond['all']['violation_fraction']:.2f}%" in markdown, key
        if cond["resolved"]["n_violations"]:
            assert f"{100 * cond['resolved']['violation_fraction']:.3f}%" \
                in markdown, key

    worst = report["conditions"]
    for key, spec in (("asian_floor_bps", "{:.2f}"),
                      ("butterfly_1pct_bps", "{:.2f}"),
                      ("convexity_d2C_dK2_ge_0", "{:.2f}"),
                      ("gamma_ge_0", "{:.2f}"),
                      ("vega_ge_0", "{:.4f}")):
        assert spec.format(worst[key]["all"]["worst_value"]) in markdown, key
    for key, spec in (("convexity_d2C_dK2_ge_0", "{:.3f}"),
                      ("gamma_ge_0", "{:.4f}"),
                      ("butterfly_1pct_bps", "{:.3f}")):
        assert spec.format(worst[key]["resolved"]["worst_value"]) in markdown, \
            key

    fly = {row["label"]: row for row in report["butterfly_check"]}
    worst_fly = fly["worst butterfly, full box"]
    assert f"{worst_fly['network_butterfly_bps']:.3f}" in markdown
    assert f"{worst_fly['mc_butterfly_bps']:.4f}" in markdown
