"""Tests for backend/quant/asian_approx.py: the closed-form Asian approximations.

What is asserted, and why each tolerance is what it is:

* Parity. Both approximations produce puts from the exact Asian parity
  C - P = e^{-rT}(E[A] - K), so the residual is floating-point noise
  (measured ~1e-13 on prices of order 10). Tolerance 1e-10.
* n = 1. With a single monitoring date A = S_T, the exact moments are the
  Black-Scholes moments and both approximations must return the
  Black-Scholes price. Measured agreement ~1e-13; tolerance 1e-10.
* The moments here are the ones benchmarks.levy_asian_call already uses
  inline. Rebuilding its price from arithmetic_average_moments must give the
  identical number, so the two cannot drift apart.
* Agreement with Monte Carlo. Against price_asian_mc(seed=1, n_paths=200_000)
  the two approximations behave differently, and the tests say so rather
  than share one tolerance:
    - Curran (1994) lies within 4 standard errors at every near-the-money
      point (measured -1.2 to -2.6 SE, always below: it is a lower bound).
    - Turnbull-Wakeman sits ABOVE the reference by 20 to 60 standard errors
      (measured +1.7 to +7.4 bps of strike). That is the documented
      systematic error of lognormal moment matching, the very thing the
      neural surrogate is benchmarked against, so it is asserted as a
      property, not tolerated as noise. A 4-SE test on it would be a test
      that cannot pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from backend.quant.asian_approx import (arithmetic_average_moments,
                                        curran_call, curran_price,
                                        monitoring_times,
                                        turnbull_wakeman_price)
from backend.quant.benchmarks import levy_asian_call
from backend.quant.monte_carlo import (expected_arithmetic_average,
                                       geometric_asian_price, price_asian_mc)

N_STEPS = 50

# (spot, strike, maturity, sigma, rate): three near-the-money contracts
# spanning short/long maturity and 0.20-0.30 vol, all inside the trained box.
NEAR_THE_MONEY = [
    (100.0, 100.0, 1.0, 0.25, 0.04),
    (100.0, 95.0, 0.5, 0.20, 0.04),
    (100.0, 105.0, 2.0, 0.30, 0.04),
]

# Wider set for the identities that need no Monte Carlo.
PARAM_SETS = NEAR_THE_MONEY + [
    (100.0, 80.0, 2.0, 0.50, 0.10),
    (100.0, 130.0, 0.1, 0.15, 0.00),
    (100.0, 100.0, 0.05, 0.80, 0.07),
    (50.0, 100.0, 1.5, 0.35, 0.02),
]

APPROXIMATIONS = {
    "turnbull_wakeman": turnbull_wakeman_price,
    "curran": curran_price,
}


def bs_call(spot, strike, maturity, sigma, rate):
    sd = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / sd
    d2 = d1 - sd
    return spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2)


@pytest.fixture(scope="module")
def mc_reference():
    """200,000-path control-variate references, seed 1, one per contract."""
    return {p: price_asian_mc(*p, n_paths=200_000, n_steps=N_STEPS, seed=1)
            for p in NEAR_THE_MONEY}


# --------------------------------------------------------------------------- #
#  Exact identities
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,pricer", APPROXIMATIONS.items(),
                         ids=list(APPROXIMATIONS))
@pytest.mark.parametrize("params", PARAM_SETS)
def test_put_call_parity_holds_to_1e10(name, pricer, params):
    S, K, T, sig, r = params
    call = pricer(S, K, T, sig, r, N_STEPS, option_type="call")
    put = pricer(S, K, T, sig, r, N_STEPS, option_type="put")
    rhs = math.exp(-r * T) * (expected_arithmetic_average(S, r, T, N_STEPS) - K)
    assert abs((call - put) - rhs) < 1e-10, (
        f"{name}: parity residual {(call - put) - rhs:.3e}")


@pytest.mark.parametrize("params", PARAM_SETS)
def test_moments_reduce_to_black_scholes_when_n_is_1(params):
    """One monitoring date at T means A = S_T: the moments are the lognormal
    moments of S_T and both approximations collapse to Black-Scholes."""
    S, K, T, sig, r = params
    m1, m2 = arithmetic_average_moments(S, T, sig, r, n_steps=1)
    assert m1 == pytest.approx(S * math.exp(r * T), rel=1e-12)
    assert math.log(m2) - 2.0 * math.log(m1) == pytest.approx(sig ** 2 * T,
                                                              abs=1e-12)
    bs = bs_call(S, K, T, sig, r)
    assert abs(turnbull_wakeman_price(S, K, T, sig, r, 1) - bs) < 1e-10
    assert abs(curran_call(S, K, T, sig, r, 1) - bs) < 1e-10
    assert abs(curran_call(S, K, T, sig, r, 1, threshold="linear") - bs) < 1e-10


@pytest.mark.parametrize("params", PARAM_SETS)
def test_moments_are_the_ones_benchmarks_uses(params):
    """arithmetic_average_moments must reproduce benchmarks.levy_asian_call
    exactly when its formula is rebuilt from them, and M1 must equal the
    parity engine's E[A] (a geometric series vs a direct sum)."""
    S, K, T, sig, r = params
    m1, m2 = arithmetic_average_moments(S, T, sig, r, N_STEPS)
    assert m1 == pytest.approx(expected_arithmetic_average(S, r, T, N_STEPS),
                               rel=1e-12)
    v = math.log(m2) - 2.0 * math.log(m1)
    sd = math.sqrt(v)
    d1 = (math.log(m1 / K) + 0.5 * v) / sd
    rebuilt = math.exp(-r * T) * (m1 * norm.cdf(d1) - K * norm.cdf(d1 - sd))
    assert abs(rebuilt - levy_asian_call(S, K, T, sig, r, N_STEPS)) < 1e-12


def test_second_moment_uses_the_covariance_not_the_variance():
    """A wrong E[A^2] that used sigma^2 * t_i on the diagonal only (i.e.
    treated the monitoring dates as uncorrelated) would understate the
    variance of A. Check the double sum against a brute-force evaluation of
    E[S_i S_j] = S0^2 exp(r(t_i+t_j)) exp(sigma^2 min(t_i,t_j))."""
    S, T, sig, r, n = 100.0, 1.0, 0.25, 0.04, 7
    t = monitoring_times(T, n)
    brute = 0.0
    for i in range(n):
        for j in range(n):
            brute += S ** 2 * math.exp(r * (t[i] + t[j])) \
                * math.exp(sig ** 2 * min(t[i], t[j]))
    brute /= n ** 2
    _, m2 = arithmetic_average_moments(S, T, sig, r, n)
    assert m2 == pytest.approx(brute, rel=1e-13)
    diag_only = sum(S ** 2 * math.exp(2 * r * ti + sig ** 2 * ti)
                    for ti in t) / n ** 2
    assert m2 > diag_only        # the cross terms are what make A random


# --------------------------------------------------------------------------- #
#  Against Monte Carlo
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("threshold", ["exact", "linear"])
def test_curran_within_four_standard_errors_of_mc(mc_reference, threshold):
    for params, ref in mc_reference.items():
        S, K, T, sig, r = params
        approx = curran_call(S, K, T, sig, r, N_STEPS, threshold=threshold)
        z = (approx - ref.price) / ref.std_error
        assert abs(z) <= 4.0, (
            f"Curran({threshold}) at {params}: {approx:.6f} vs MC "
            f"{ref.price:.6f} +/- {ref.std_error:.6f} ({z:+.1f} SE)")


def test_curran_is_a_lower_bound_above_the_geometric_price(mc_reference):
    """Curran with the exact threshold equals the Jensen bound
    E[(E[A|G]-K)^+] <= E[(A-K)^+], and E[A|G] >= G gives the other side."""
    for params, ref in mc_reference.items():
        S, K, T, sig, r = params
        approx = curran_call(S, K, T, sig, r, N_STEPS)
        assert approx <= ref.price + 4.0 * ref.std_error
    for S, K, T, sig, r in PARAM_SETS:
        geo = geometric_asian_price(S, K, T, sig, r, N_STEPS, "call")
        assert curran_call(S, K, T, sig, r, N_STEPS) >= geo - 1e-12


def test_turnbull_wakeman_error_is_systematic_not_noise(mc_reference):
    """The moment-matched price is biased high near the money, by far more
    than the 200k-path reference can be wrong, and by less than 10 bps of
    strike at these vols. Both halves are the measured behaviour that the
    surrogate benchmark in the README relies on."""
    for params, ref in mc_reference.items():
        S, K, T, sig, r = params
        approx = turnbull_wakeman_price(S, K, T, sig, r, N_STEPS)
        z = (approx - ref.price) / ref.std_error
        err_bps = (approx - ref.price) / K * 1e4
        assert z > 4.0, f"TW at {params}: {z:+.1f} SE, expected a positive bias"
        assert err_bps < 10.0, f"TW at {params}: {err_bps:+.2f} bps of strike"


def test_curran_thresholds_agree_near_the_money():
    """Curran's first-order threshold and the exact solve differ by well under
    0.01 bps of strike at moderate vol (measured 2e-4 bps or less)."""
    for S, K, T, sig, r in NEAR_THE_MONEY:
        exact = curran_call(S, K, T, sig, r, N_STEPS, threshold="exact")
        linear = curran_call(S, K, T, sig, r, N_STEPS, threshold="linear")
        assert abs(exact - linear) / K * 1e4 < 0.01


def test_curran_deterministic_limit_is_discounted_forward_intrinsic():
    """sigma -> 0 makes A deterministic; the call is e^{-rT}(E[A]-K)^+."""
    S, K, T, r = 100.0, 90.0, 1.0, 0.04
    ea = expected_arithmetic_average(S, r, T, N_STEPS)
    exact = math.exp(-r * T) * (ea - K)
    assert curran_call(S, K, T, 0.0, r, N_STEPS) == pytest.approx(exact)
    assert curran_call(S, K, T, 1e-4, r, N_STEPS) == pytest.approx(exact,
                                                                   rel=1e-6)
