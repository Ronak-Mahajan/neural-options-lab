"""Fast checks for scripts/atm_skew_term_structure.py (whole file < 30 s).

The script itself takes minutes at its default path budget; these tests run
its building blocks at a small budget and check the properties that must hold
regardless of noise:

  (a) with rho < 0 the model ATM skew is negative, and |psi| decays with
      maturity (5 days steeper than 63 days) - the qualitative rough-vol
      signature, at a budget that resolves it;
  (b) the empirical extractor returns a finite slope with a finite, positive
      standard error on a committed SPY capture;
  (c) a flat-vol Black-Scholes surface inverts to psi = 0 through the same
      stencil code path (sanity of the inversion and the difference);
  (d) the power-law and local-quadratic fitters recover known slopes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.calibrate import bs_call  # noqa: E402
from scripts import atm_skew_term_structure as ats  # noqa: E402

CAPTURES = sorted((ROOT / "data" / "surfaces" / "equity").glob("spy_*.json.gz"))


def test_model_skew_negative_and_steeper_at_short_maturity():
    params = {"spot": 100.0, "rate": 0.03, "xi": 0.04, "eta": 2.0,
              "rho": -0.7, "H": 0.15}
    rows = ats.model_skew_curve(params, T_days=(5, 63), n_paths=50_000,
                                n_reps=2, seed=11)
    by_T = {r["T_days"]: r for r in rows}
    psi5, psi63 = by_T[5.0]["psi"], by_T[63.0]["psi"]
    assert math.isfinite(psi5) and math.isfinite(psi63)
    assert psi5 < 0.0 and psi63 < 0.0, (psi5, psi63)
    assert abs(psi5) > abs(psi63), (psi5, psi63)
    # the across-seed SE must be finite and small relative to the effect
    assert by_T[5.0]["se"] > 0.0 and by_T[5.0]["se"] < 0.5 * abs(psi5)
    # the 2h estimate must agree with the h estimate to well within |psi|
    assert abs(by_T[5.0]["psi_2h"] - psi5) < 0.3 * abs(psi5)


@pytest.mark.skipif(not CAPTURES, reason="no committed SPY capture")
def test_empirical_extractor_returns_finite_slopes():
    rows, info = ats.market_skew_from_capture(CAPTURES[0])
    assert info["n_quotes"] > 0
    assert len(rows) >= 3, "expected several expiries with enough ATM quotes"
    for r in rows:
        assert math.isfinite(r["psi"]), r
        assert math.isfinite(r["se"]) and r["se"] > 0.0, r
        assert r["n"] >= 6 and r["k_min"] < 0.0 < r["k_max"], r
        assert 0.0 < r["atm_iv"] < 1.0, r
    # SPY: shorter expiries carry the steeper skew, and it is negative
    rows = sorted(rows, key=lambda r: r["tau"])
    assert sum(r["psi"] < 0.0 for r in rows) >= len(rows) - 1
    assert abs(rows[0]["psi"]) > abs(rows[-1]["psi"])


def test_flat_vol_surface_has_zero_central_difference_skew():
    spot, rate, T, sigma = 100.0, 0.02, 10.0 / 252.0, 0.20
    h = ats.moneyness_step(sigma ** 2, T)
    F = spot * math.exp(rate * T)
    ks = np.asarray(ats.STENCIL) * h
    strikes = F * np.exp(ks)
    prices = np.array([bs_call(spot, float(K), T, sigma, rate) for K in strikes])
    s = ats.skew_from_stencil(ks, strikes, prices, spot, T, rate,
                              se_prices=np.full(5, 1e-3))
    assert abs(s["atm_iv"] - sigma) < 1e-6
    assert abs(s["psi"]) < 1e-6 and abs(s["psi_2h"]) < 1e-6
    assert s["se_naive"] > 0.0 and math.isfinite(s["se_naive"])


def test_flat_vol_surface_skew_is_zero_at_several_maturities():
    spot, rate, sigma = 250.0, 0.04, 0.12
    for d in (1, 5, 63):
        T = d / 252.0
        h = ats.moneyness_step(sigma ** 2, T)
        ks = np.asarray(ats.STENCIL) * h
        strikes = spot * math.exp(rate * T) * np.exp(ks)
        prices = np.array([bs_call(spot, float(K), T, sigma, rate) for K in strikes])
        s = ats.skew_from_stencil(ks, strikes, prices, spot, T, rate)
        assert abs(s["psi"]) < 1e-6, (d, s)


def test_power_law_fit_recovers_exponent():
    rng = np.random.default_rng(3)
    T = np.geomspace(1 / 252, 0.5, 12)
    b_true, C = -0.30, 0.4
    se = 0.01 * C * T ** b_true
    psi = -(C * T ** b_true + rng.normal(0.0, se))
    f = ats.fit_power_law(T, psi, se)
    assert abs(f["b"] - b_true) < 4.0 * f["se_b"], f
    assert f["n"] == 12 and f["dof"] == 10
    assert abs(f["C"] - C) / C < 0.1
    # masking restricts the fit range
    g = ats.fit_power_law(T, psi, se, mask=T <= 45 / 252)
    assert g["n"] == int((T <= 45 / 252).sum())


def test_local_quadratic_recovers_slope_at_zero():
    rng = np.random.default_rng(5)
    k = np.linspace(-0.03, 0.03, 25)
    a, b, c = 0.15, -0.9, 4.0
    iv = a + b * k + c * k ** 2 + rng.normal(0.0, 1e-4, size=k.size)
    hs = np.full(k.size, 5e-3)
    fit = ats.local_quadratic_skew(k, iv, hs, band=0.05)
    assert fit is not None
    assert abs(fit["psi"] - b) < 4.0 * fit["se"] + 1e-3
    assert abs(fit["atm_iv"] - a) < 1e-3
    # a one-sided strip (no quotes above ATM) is refused rather than extrapolated
    assert ats.local_quadratic_skew(k[k < 0], iv[k < 0], hs[k < 0], band=0.05) is None
