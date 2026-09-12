"""The volatility solver must return the volatility it was given."""

import numpy as np
import pytest

from backend.quant.engine import PricingEngine
from backend.quant.solve_vol import solve_implied_vol


@pytest.fixture(scope="module")
def engine():
    return PricingEngine()


CASES = [
    # spot, strike, maturity, rate, sigma, option_type
    (100.0, 100.0, 1.00, 0.04, 0.25, "call"),
    (100.0, 100.0, 1.00, 0.04, 0.25, "put"),
    (120.0, 100.0, 0.50, 0.02, 0.40, "call"),
    (90.0, 110.0, 1.50, 0.06, 0.15, "put"),
    (100.0, 100.0, 0.25, 0.00, 0.60, "call"),
]


@pytest.mark.parametrize("spot,strike,mat,rate,sigma,kind", CASES)
def test_round_trip_recovers_the_volatility(engine, spot, strike, mat, rate,
                                            sigma, kind):
    """Price at a known volatility, then solve the price back to it."""
    target = float(engine.price_batch(
        np.array([spot]), np.array([strike]), np.array([mat]),
        np.array([sigma]), np.array([rate]), option_type=kind)[0])

    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8)

    assert sol.bracketed
    # The price is what was inverted, so the price must match tightly; the
    # volatility follows to whatever precision the price curve supports.
    assert abs(sol.price_at_sigma - target) < 1e-5
    assert abs(sol.sigma - sigma) < 5e-3


def test_price_is_monotone_in_volatility(engine):
    """The bisection is only valid because the price rises with volatility."""
    sigmas = np.linspace(0.05, 0.80, 40)
    n = sigmas.shape[0]
    prices = engine.price_batch(
        np.full(n, 100.0), np.full(n, 100.0), np.full(n, 1.0),
        sigmas, np.full(n, 0.04), option_type="call")
    diffs = np.diff(prices.astype(np.float64))
    # Allow a hair of network noise, but the trend must be strictly upward.
    assert diffs.min() > -1e-4
    assert prices[-1] > prices[0] + 1.0


def test_target_below_the_floor_returns_the_lowest_volatility(engine):
    sol = solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, 0.0,
                            option_type="call")
    assert not sol.bracketed or sol.sigma == pytest.approx(sol.sigma_low)
    assert sol.sigma == pytest.approx(sol.sigma_low)


def test_target_above_the_ceiling_returns_the_highest_volatility(engine):
    sol = solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, 1e6,
                            option_type="call")
    assert not sol.bracketed
    assert sol.sigma == pytest.approx(sol.sigma_high)


def test_a_negative_target_is_rejected(engine):
    with pytest.raises(ValueError):
        solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, -1.0)


def test_an_inverted_bracket_is_rejected(engine):
    with pytest.raises(ValueError):
        solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, 5.0,
                          sigma_low=0.5, sigma_high=0.2)


def test_the_solver_stays_inside_the_requested_bracket(engine):
    sol = solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, 6.0,
                           sigma_low=0.10, sigma_high=0.45)
    assert 0.10 - 1e-9 <= sol.sigma <= 0.45 + 1e-9
