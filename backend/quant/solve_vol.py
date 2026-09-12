"""Invert the surrogate: which volatility reproduces a given price?

Pricing a contract from a volatility is what the network does directly.
The question a person actually arrives with is the other way round - a price
is quoted, and they want the volatility implied by it. For the Asian payoff
there is no closed form to invert, so this solves the surrogate itself.

The surrogate's price is monotone increasing in volatility for a vanilla or
an average-rate call or put, so a bisection is both safe and exact to the
tolerance asked for. The first bracket is found with one batched evaluation
across the whole trained volatility range, which costs a single forward pass
rather than one per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolSolution:
    """The volatility that reproduces `target_price`, and how it was found."""

    sigma: float
    price_at_sigma: float
    target_price: float
    iterations: int
    # Set when the target lies outside what the model can produce anywhere in
    # its trained volatility range; sigma is then the closest achievable end.
    bracketed: bool
    low_price: float
    high_price: float
    sigma_low: float
    sigma_high: float


def solve_implied_vol(engine, spot: float, strike: float, maturity: float,
                      rate: float, target_price: float,
                      option_type: str = "call",
                      sigma_low: float | None = None,
                      sigma_high: float | None = None,
                      tol: float = 1e-6, max_iter: int = 60,
                      scan_points: int = 96) -> VolSolution:
    """Solve `engine(sigma) == target_price` for sigma.

    `tol` is on the price, in the same units as `target_price`, so the answer
    is as precise as the quote that produced it rather than to an arbitrary
    number of volatility decimals.
    """
    box = engine.meta.get("param_ranges", {})
    lo_default, hi_default = box.get("sigma", [0.05, 0.8])
    lo = float(lo_default if sigma_low is None else sigma_low)
    hi = float(hi_default if sigma_high is None else sigma_high)
    if not hi > lo:
        raise ValueError("sigma_high must exceed sigma_low")
    if not np.isfinite(target_price) or target_price < 0:
        raise ValueError("target_price must be a non-negative number")

    def price_many(sigmas: np.ndarray) -> np.ndarray:
        n = sigmas.shape[0]
        return engine.price_batch(
            np.full(n, float(spot)), np.full(n, float(strike)),
            np.full(n, float(maturity)), sigmas.astype(np.float64),
            np.full(n, float(rate)), option_type=option_type,
        ).astype(np.float64)

    # One batched sweep gives the whole price-versus-volatility curve, which
    # both brackets the root and reveals whether the target is reachable.
    grid = np.linspace(lo, hi, scan_points)
    prices = price_many(grid)
    p_lo, p_hi = float(prices[0]), float(prices[-1])

    if target_price <= p_lo:
        return VolSolution(lo, p_lo, target_price, 0, target_price >= p_lo,
                           p_lo, p_hi, lo, hi)
    if target_price >= p_hi:
        return VolSolution(hi, p_hi, target_price, 0, target_price <= p_hi,
                           p_lo, p_hi, lo, hi)

    # The sweep is monotone up to network noise; take the last crossing so a
    # tiny non-monotone wobble near the floor cannot pick a spurious bracket.
    above = np.nonzero(prices >= target_price)[0]
    j = int(above[0])
    a, b = float(grid[j - 1]), float(grid[j])
    fa = float(prices[j - 1]) - target_price

    it = 0
    while it < max_iter and (b - a) > 1e-9:
        mid = 0.5 * (a + b)
        fm = float(price_many(np.array([mid]))[0]) - target_price
        it += 1
        if abs(fm) <= tol:
            a = b = mid
            fa = fm
            break
        if (fa < 0) == (fm < 0):
            a, fa = mid, fm
        else:
            b = mid
    sigma = 0.5 * (a + b)
    return VolSolution(sigma, float(price_many(np.array([sigma]))[0]),
                       target_price, it, True, p_lo, p_hi, lo, hi)
