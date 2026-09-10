"""Fast checks for backend/quant/heston.py (whole file about 15 s).

  (a) the published COS reference value (Fang & Oosterlee 2008, Table 3):
      S = K = 100, T = 1, r = q = 0, v0 = 0.0175, kappa = 1.5768,
      theta = 0.0398, sigma_v = 0.5751, rho = -0.5711 -> call = 5.785155450;
  (b) put-call parity between the directly priced call and put to 1e-10;
  (c) the Black-Scholes limit sigma_v -> 0 with v0 = theta to 1e-8, and the
      smoothness of the approach (the departure is linear in sigma_v: the
      physical first-order skew, not a numerical artefact);
  (d) convergence in N at a pinned truncation range: the error falls at
      least geometrically between 2^5, 2^6, 2^7 and sits at the floating
      floor from 2^8 on; and, on a pinned WIDE range where the 2^8 -> 2^10
      -> 2^12 ladder is resolvable, it falls geometrically there too;
  (e) an independent full-truncation Euler Monte Carlo at 200k paths agrees
      within 3 standard errors;
  plus: the closed-form cumulants against the cf's own derivatives, the
  vectorised Black-76 inversion against calibrate.implied_vol, the
  short-maturity ATM skew limit rho sigma_v / (4 sqrt(v0)), the truncation
  self-check, and a small synthetic calibration round trip.
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

from backend.quant import heston as H  # noqa: E402
from backend.quant.calibrate import Quote, bs_call, implied_vol  # noqa: E402

FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, sigma_v=0.5751, rho=-0.5711)
FO_REF = 5.785155450        # Fang & Oosterlee (2008), Table 3


# ── (a) published reference ────────────────────────────────────────────────
def test_fo2008_reference_value():
    # FO's own range (L = 12, no widening): 8.0e-9 from the published value
    for N in (2 ** 8, 2 ** 10, 2 ** 12):
        c = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=N, L=12.0,
                          range_tol=float("inf"), **FO)[0]
        assert abs(c - FO_REF) < 1e-7, (N, c)
    # default (range self-check widens L to 27): 1.6e-8 from the published
    # value, which is the residual truncation of the reference itself
    for N in (2 ** 10, 2 ** 12):
        c = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=N, **FO)[0]
        assert abs(c - FO_REF) < 1e-7, (N, c)
    assert abs(H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** 10, **FO)[0]
               - 5.785155434377) < 1e-9


# ── (b) parity ─────────────────────────────────────────────────────────────
def test_put_call_parity_direct_prices():
    S, T, r, q = 100.0, 0.75, 0.03, 0.01
    K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    c = H.heston_call(S, K, T, r, q, **FO)
    p = H.heston_put(S, K, T, r, q, **FO)
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert np.max(np.abs(c - p - parity)) < 1e-10
    # and the parity-mapped call agrees with the direct one
    cp = H.heston_call(S, K, T, r, q, via_parity=True, **FO)
    assert np.max(np.abs(c - cp)) < 1e-10


# ── (c) Black-Scholes limit ────────────────────────────────────────────────
def test_black_scholes_limit():
    K = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    sig, r, T = 0.2, 0.05, 1.0
    bs = np.array([bs_call(100.0, k, T, sig, r) for k in K])
    for sv in (0.0, 1e-9):
        c = H.heston_call(100.0, K, T, r, 0.0, v0=sig ** 2, kappa=1.5,
                          theta=sig ** 2, sigma_v=sv, rho=-0.7)
        assert np.max(np.abs(c - bs)) < 1e-8, (sv, c - bs)
    # the departure from BS is LINEAR in sigma_v (first-order skew): the
    # rewritten cf is smooth through sigma_v = 0, not merely patched there
    errs = [np.max(np.abs(H.heston_call(100.0, K, T, r, 0.0, v0=sig ** 2, kappa=1.5,
                                        theta=sig ** 2, sigma_v=sv, rho=-0.7) - bs))
            for sv in (1e-6, 1e-4)]
    assert 90.0 < errs[1] / errs[0] < 110.0, errs
    # deterministic but non-flat variance curve (v0 != theta) at sigma_v = 0
    v0, th, kap, T2 = 0.03, 0.06, 2.0, 0.7
    vbar = th * T2 + (v0 - th) * (1.0 - math.exp(-kap * T2)) / kap
    c = H.heston_call(100.0, K, T2, 0.02, 0.0, v0=v0, kappa=kap, theta=th,
                      sigma_v=0.0, rho=-0.5)
    bs2 = np.array([bs_call(100.0, k, T2, math.sqrt(vbar / T2), 0.02) for k in K])
    assert np.max(np.abs(c - bs2)) < 1e-8


# ── (d) convergence in N ───────────────────────────────────────────────────
def test_convergence_in_N_at_pinned_range():
    kw = dict(L=12.0, range_tol=float("inf"), **FO)
    ref = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** 14, **kw)[0]
    err = {n: abs(H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** n, **kw)[0] - ref)
           for n in (5, 6, 7, 8, 10, 12)}
    # geometric (in fact ~100x per doubling) while resolvable ...
    assert err[6] < 0.5 * err[5] and err[7] < 0.5 * err[6], err
    assert err[5] < 0.1 and err[7] < 1e-6
    # ... and at the floating-point floor from 2^8 on: a "halving" between
    # 2^8, 2^10 and 2^12 cannot be resolved because there is nothing left
    assert err[8] < 1e-10 and err[10] < 1e-10 and err[12] < 1e-10, err
    assert err[10] <= max(0.5 * err[8], 1e-10) and err[12] <= max(0.5 * err[10], 1e-10)


def test_convergence_in_N_resolvable_on_a_wide_range():
    """The spec's 2^8 -> 2^10 -> 2^12 ladder, where it IS resolvable: a
    pinned wide range (L = 40) at T = 1 trading day with the parameters a
    short-dated SPY calibration selects leaves few terms per unit of width,
    so the error sits above the floating floor at 2^8 and falls at least
    geometrically to 2^10 (measured: ~4e-5 -> ~1e-11 -> floor)."""
    pp = dict(v0=0.0016, kappa=100.0, theta=0.024, sigma_v=4.7, rho=-0.57)
    K = np.array([97.0, 100.0, 103.0])
    T = 1.0 / 252.0
    kw = dict(L=40.0, range_tol=float("inf"), **pp)
    ref = H.heston_call(100.0, K, T, 0.0, 0.0, N=2 ** 15, **kw)
    err = {n: float(np.max(np.abs(H.heston_call(100.0, K, T, 0.0, 0.0, N=2 ** n, **kw) - ref)))
           for n in (8, 10, 12)}
    assert err[8] > 1e-6, err                      # resolvable, not at the floor
    assert err[10] < 0.5 * err[8] and err[10] < 1e-9, err
    assert err[12] <= max(0.5 * err[10], 1e-10), err
    # and the default (self-checking) range at the same inputs is converged
    # to the floor already at 2^8
    d8 = H.heston_call(100.0, K, T, 0.0, 0.0, N=2 ** 8, **pp)
    d12 = H.heston_call(100.0, K, T, 0.0, 0.0, N=2 ** 12, **pp)
    assert np.max(np.abs(d8 - d12)) < 1e-9


# ── (e) Monte Carlo cross-check ────────────────────────────────────────────
def test_monte_carlo_cross_check():
    K = np.array([90.0, 100.0, 110.0])
    T, r = 0.5, 0.02
    cos = H.heston_call(100.0, K, T, r, 0.0, **FO)
    mc, se = H.heston_mc_call(100.0, K, T, r, 0.0, n_paths=200_000,
                              n_steps=400, seed=1, chunk=100_000, **FO)
    z = (mc - cos) / se
    assert np.all(np.abs(z) < 3.0), (cos, mc, se, z)
    assert np.all(se > 0.0) and np.all(se < 0.05)


# ── supporting checks ──────────────────────────────────────────────────────
def test_cumulants_match_characteristic_function():
    def numeric(T, **pp):
        h = 1e-3

        def f(u):
            return np.log(H.heston_cf(np.array([u], dtype=complex), T, **pp))[0]

        d1 = (f(h) - f(-h)) / (2 * h)
        d2 = (f(h) - 2 * f(0.0) + f(-h)) / h ** 2
        d1b = (f(h / 2) - f(-h / 2)) / h
        d2b = (f(h / 2) - 2 * f(0.0) + f(-h / 2)) / (h / 2) ** 2
        return (4 * d1b - d1).imag / 3, -(4 * d2b - d2).real / 3

    for T in (0.01, 0.5, 2.0):
        for pp in (FO, dict(v0=0.04, kappa=0.5, theta=0.09, sigma_v=2.0, rho=-0.9),
                   dict(v0=0.006, kappa=50.0, theta=0.02, sigma_v=5.0, rho=-0.6)):
            c1, c2 = H.heston_cumulants(T, **pp)
            n1, n2 = numeric(T, **pp)
            assert c2 > 0.0
            assert abs(c1 - n1) < 1e-9 * max(1.0, abs(c1)) + 1e-12, (T, pp, c1, n1)
            assert abs(c2 - n2) / c2 < 1e-5, (T, pp, c2, n2)


def test_range_self_check_widens_where_the_tails_demand_it():
    # FO's Feller-violating parameters at T = 1: L = 12 leaves a martingale
    # defect of ~4e-7 (a 4e-5 put error); the check widens to L = 27
    rep12 = H.cos_range_report(1.0, N=1024, L=12.0, range_tol=float("inf"), **FO)
    rep = H.cos_range_report(1.0, N=1024, L=12.0, **FO)
    assert rep12["defect"] > 1e-7 and rep["defect"] < 1e-10
    assert rep["L"] > 12.0
    # a short-dated case passes at L = 12 without widening
    short = H.cos_range_report(2.0 / 252.0, N=1024, L=12.0, **FO)
    assert short["L"] == 12.0 and short["defect"] < 1e-10


def test_vectorised_inversion_matches_calibrate_implied_vol():
    K = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
    T, r, q = 0.5, 0.03, 0.01
    iv = H.heston_implied_vol(100.0, K, T, r, q, **FO)
    c = H.heston_call(100.0, K, T, r, q, **FO)
    fwd_pv = 100.0 * math.exp(-q * T)
    ref = np.array([implied_vol(float(ci), fwd_pv, float(k), T, r) for ci, k in zip(c, K)])
    assert np.max(np.abs(iv - ref)) < 1e-9
    # negative skew with rho < 0 on the put wing; the call wing turns back up
    # (a smile, sigma_v = 0.58) so only the put side is monotone
    assert iv[0] > iv[1] > iv[2], iv
    # round trip of the Black-76 inversion, array maturities
    sig = np.array([0.05, 0.2, 0.8, 2.0])
    Kr = np.array([80.0, 100.0, 120.0, 60.0])
    Tr = np.array([0.02, 0.5, 1.0, 2.0])
    p = H.black76_price(100.0, Kr, Tr, sig, Kr >= 100.0)
    back = H.black76_implied_vol(p, 100.0, Kr, Tr, Kr >= 100.0)
    assert np.max(np.abs(back - sig)) < 1e-12
    # unpriceable quotes are NaN, not a number: a put at or below intrinsic,
    # a call above its cap F, and a non-finite price
    bad = H.black76_implied_vol(np.array([0.0, 0.0, 101.0, np.nan]), 100.0,
                                np.array([80.0, 120.0, 90.0, 100.0]), 0.02,
                                np.array([False, False, True, True]))
    assert np.all(np.isnan(bad)), bad
    # while an absurdly small but positive OTM price still has one (4.3%)
    tiny = H.black76_implied_vol(np.array([1e-300]), 100.0, np.array([80.0]), 0.02,
                                 np.array([False]))[0]
    assert np.isfinite(tiny) and 0.03 < tiny < 0.06


def test_short_maturity_atm_skew_limit():
    pp = dict(v0=0.04, kappa=2.0, theta=0.04, sigma_v=1.0, rho=-0.7)
    lim = H.heston_short_skew_limit(pp["v0"], pp["sigma_v"], pp["rho"])
    assert abs(lim - (-0.875)) < 1e-12
    s = H.heston_atm_skew(1e-4, h=1e-4, **pp)
    assert abs(s["psi"] - lim) < 5e-4, (s["psi"], lim)
    # the skew is FLAT in log T at the short end: the classical fingerprint
    s1 = H.heston_atm_skew(1.0 / 252.0, h=1e-4, **pp)
    s2 = H.heston_atm_skew(4.0 / 252.0, h=1e-4, **pp)
    slope = (math.log(abs(s2["psi"])) - math.log(abs(s1["psi"]))) / math.log(4.0)
    assert abs(slope) < 0.03, slope
    # stencil convention: Richardson agrees with the fine derivative
    st = H.heston_atm_skew(20.0 / 252.0, **pp)
    fine = H.heston_atm_skew(20.0 / 252.0, h=1e-4, **pp)
    assert abs(st["psi_richardson"] - fine["psi"]) < 5 * st["truncation"] + 1e-3


def test_synthetic_calibration_round_trip():
    """Quotes generated by Heston itself must be recovered (RMSE ~ 0)."""
    truth = dict(v0=0.012, kappa=8.0, theta=0.03, sigma_v=1.5, rho=-0.65)
    rate, F0 = 0.03, 500.0
    quotes = []
    for i, tau in enumerate((0.02, 0.05, 0.12)):
        fwd_pv = F0 * math.exp(-rate * tau)
        K = F0 * np.exp(np.linspace(-0.12, 0.08, 21) * math.sqrt(tau / 0.05))
        iv = H.heston_implied_vol(F0, K, tau, 0.0, 0.0, **truth)
        for k, v in zip(K, iv):
            if not np.isfinite(v):
                continue
            mid = bs_call(fwd_pv, float(k), tau, float(v), rate)
            quotes.append(Quote(tau=tau, strike=float(k), mid_call=mid, iv=float(v),
                                vega=1.0, kind="C" if k >= fwd_pv else "P",
                                expiry=f"E{i}", fwd_pv=fwd_pv, half_spread_iv=0.1))
    fit = H.calibrate_heston(quotes, rate, N=256, n_random_starts=0)
    assert fit.rmse_volpts < 1e-3, fit
    assert fit.n_unpriceable == 0
    assert abs(fit.params["rho"] - truth["rho"]) < 0.02
    assert abs(fit.params["v0"] - truth["v0"]) / truth["v0"] < 0.05
    assert isinstance(fit.feller, bool) and fit.feller_ratio > 0
    d = fit.as_dict()
    assert "residuals" not in d and d["n_quotes"] == len(quotes)
    # holding kappa at the truth: the other four are recovered, kappa is
    # reported exactly as held with se = 0 and never as pinned
    fk = H.calibrate_heston(quotes, rate, N=256, n_random_starts=0,
                            fixed={"kappa": truth["kappa"]})
    assert fk.params["kappa"] == truth["kappa"] and fk.se["kappa"] == 0.0
    assert fk.fixed == {"kappa": truth["kappa"]}
    assert fk.rmse_volpts < 1e-3 and not any("kappa" in s for s in fk.pinned)
    assert abs(fk.params["rho"] - truth["rho"]) < 0.02
    # a wrong held kappa cannot be compensated exactly: the RMSE rises
    fw = H.calibrate_heston(quotes, rate, N=256, n_random_starts=0,
                            fixed={"kappa": 0.5})
    assert fw.params["kappa"] == 0.5 and fw.rmse_volpts > 10 * fk.rmse_volpts
    with pytest.raises(ValueError):
        H.calibrate_heston(quotes, rate, N=256, fixed={"nu": 1.0})
