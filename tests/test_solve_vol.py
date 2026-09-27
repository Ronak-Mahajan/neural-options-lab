"""The volatility solver must return the volatility it was given."""

import numpy as np
import pytest

from backend.quant.engine import PricingEngine
from backend.quant.solve_vol import solve_implied_vol


@pytest.fixture(scope="module")
def engine():
    return PricingEngine()


def price_at(engine, spot, strike, mat, rate, sigma, kind):
    return float(engine.price_batch(
        np.array([spot]), np.array([strike]), np.array([mat]),
        np.array([sigma]), np.array([rate]), option_type=kind)[0])


def sweep(engine, spot, strike, mat, rate, kind, points=96):
    grid = np.linspace(0.05, 0.80, points)
    n = grid.shape[0]
    return engine.price_batch(
        np.full(n, spot), np.full(n, strike), np.full(n, mat), grid,
        np.full(n, rate), option_type=kind).astype(np.float64)


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
    target = price_at(engine, spot, strike, mat, rate, sigma, kind)

    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8)

    assert sol.bracketed
    # The price is what was inverted, so the price must match tightly; the
    # volatility follows to whatever precision the price curve supports.
    assert abs(sol.price_at_sigma - target) < 1e-5
    assert abs(sol.sigma - sigma) < 5e-3
    # Near the money the price rises with volatility: one root.
    assert len(sol.roots) == 1 and sol.roots[0] == sol.sigma


def test_price_rises_with_volatility_near_the_money(engine):
    """At the money the price curve is monotone, so the solve has one root
    there. Far from the money it is not (see the tests below)."""
    prices = sweep(engine, 100.0, 100.0, 1.0, 0.04, "call", points=40)
    diffs = np.diff(prices)
    # Allow a hair of network noise, but the trend must be strictly upward.
    assert diffs.min() > -1e-4
    assert prices[-1] > prices[0] + 1.0


# The dashboard's "far out-of-the-money put" preset. Its price falls from
# sigma 5% to a minimum near 22% before it rises, so the price the page shows
# at 25% is also reproduced near 18.5%, and both lie below the price at 5%.
PRESET = (160.0, 100.0, 1.0, 0.04, "put")

# These puts are the call less a parity term of order 1 per unit strike, so
# float32 resolves their price to a few 1e-6 at a strike of 100, and a solve
# matches the target to that resolution, not to the bisection tolerance.
PUT_PRICE_RESOLUTION = 1e-5


def test_the_far_otm_preset_solves_back_to_its_own_volatility(engine):
    spot, strike, mat, rate, kind = PRESET
    target = price_at(engine, spot, strike, mat, rate, 0.25, kind)
    assert target < price_at(engine, spot, strike, mat, rate, 0.05, kind)

    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8, sigma_hint=0.25)

    assert sol.bracketed
    assert abs(sol.sigma - 0.25) < 5e-3
    assert abs(sol.price_at_sigma - target) < PUT_PRICE_RESOLUTION
    assert len(sol.roots) == 2
    assert sol.roots[0] == pytest.approx(0.185, abs=5e-3)
    assert sol.roots[1] == sol.sigma
    # The hint picks the root, so a hint near the other one returns it.
    low = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8, sigma_hint=0.15)
    assert low.sigma == pytest.approx(sol.roots[0], abs=1e-6)


def test_every_crossing_is_found(engine):
    """This put's price crosses 0.005597 three times between 5% and 80%, and
    is floored at zero across a stretch in between."""
    spot, strike, mat, rate, kind, target = (200.0, 100.0, 0.25, 0.0, "put",
                                             0.005597)
    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8, sigma_hint=0.25)

    assert sol.bracketed
    assert len(sol.roots) == 3
    assert list(sol.roots) == sorted(sol.roots)
    for root, near in zip(sol.roots, (0.05, 0.52, 0.79)):
        assert abs(root - near) < 0.01
        assert abs(price_at(engine, spot, strike, mat, rate, root, kind)
                   - target) < PUT_PRICE_RESOLUTION
    assert sol.sigma == sol.roots[0]          # nearest the 25% hint
    high = solve_implied_vol(engine, spot, strike, mat, rate, target,
                             option_type=kind, sigma_hint=0.75)
    assert high.sigma == pytest.approx(sol.roots[2], abs=1e-4)


def test_a_zero_premium_on_the_floor_is_one_root(engine):
    """Where the price is floored at zero, a run of sweep points all hit a
    zero target; the run counts once and the hint picks the point in it."""
    spot, strike, mat, rate, kind = 200.0, 100.0, 0.25, 0.0, "put"
    prices = sweep(engine, spot, strike, mat, rate, kind)
    assert (prices == 0.0).sum() > 1

    sol = solve_implied_vol(engine, spot, strike, mat, rate, 0.0,
                            option_type=kind, sigma_hint=0.70)
    assert sol.bracketed
    assert len(sol.roots) == 1
    assert sol.price_at_sigma == 0.0
    assert abs(sol.sigma - 0.70) < 0.01


def test_a_price_below_the_first_sweep_point_is_still_reachable(engine):
    """The preset's sweep starts at its price at 5% and dips below it; a
    premium in the dip is reachable, and the reported range is the dip's
    minimum to the maximum, not the two ends of the sweep."""
    spot, strike, mat, rate, kind = PRESET
    prices = sweep(engine, spot, strike, mat, rate, kind)
    target = 0.5 * (prices.min() + prices[0])

    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8)
    assert sol.bracketed
    assert abs(sol.price_at_sigma - target) < PUT_PRICE_RESOLUTION
    assert (sol.low_price, sol.high_price) == (prices.min(), prices.max())


def test_a_price_below_the_sweep_minimum_is_out_of_reach(engine):
    """Below the lowest price the model makes anywhere in the range, nothing
    is bracketed, and the answer is the sweep point closest to the target,
    not an end of the range."""
    spot, strike, mat, rate, kind = PRESET
    prices = sweep(engine, spot, strike, mat, rate, kind)
    grid = np.linspace(0.05, 0.80, 96)
    target = 0.5 * prices.min()

    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind)
    assert not sol.bracketed
    assert sol.roots == ()
    assert (sol.low_price, sol.high_price) == (prices.min(), prices.max())
    assert sol.sigma == pytest.approx(grid[prices.argmin()])
    assert sol.price_at_sigma == pytest.approx(prices.min())
    assert sol.sigma_low < sol.sigma < sol.sigma_high


def test_without_a_hint_the_steepest_root_is_returned(engine):
    """No hint: report the root where the price moves most with volatility.
    On the preset that is the rising branch at 25%, not the falling one."""
    spot, strike, mat, rate, kind = PRESET
    target = price_at(engine, spot, strike, mat, rate, 0.25, kind)
    sol = solve_implied_vol(engine, spot, strike, mat, rate, target,
                            option_type=kind, tol=1e-8)
    grid = np.linspace(0.05, 0.80, 96)
    prices = sweep(engine, spot, strike, mat, rate, kind)
    slope = np.diff(prices) / np.diff(grid)
    cells = [int(np.searchsorted(grid, r)) - 1 for r in sol.roots]
    steepest = sol.roots[int(np.argmax([abs(slope[c]) for c in cells]))]
    assert sol.sigma == steepest


def test_target_below_the_floor_returns_the_lowest_volatility(engine):
    sol = solve_implied_vol(engine, 100.0, 100.0, 1.0, 0.04, 0.0,
                            option_type="call")
    assert not sol.bracketed
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
