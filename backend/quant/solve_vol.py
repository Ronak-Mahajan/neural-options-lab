"""Invert the surrogate: which volatility reproduces a given price?

Pricing a contract from a volatility is what the network does directly.
The question a person actually arrives with is the other way round - a price
is quoted, and they want the volatility implied by it. For the Asian payoff
there is no closed form to invert, so this solves the surrogate itself.

The surrogate's price rises with volatility over most of its box, but not
everywhere: docs/asian_arbitrage_audit.md measures negative vega at 28.05%
of its lattice, all of it where the true vega is below 0.02 per unit of
strike and the price barely depends on volatility. A premium there can be
reproduced by more than one volatility. So the solver takes one batched
sweep across the trained volatility range, which costs a single forward
pass, and reads everything from it:

  * reachability and the reported price range come from the sweep's minimum
    and maximum, not from its two ends;
  * every grid cell where the sweep crosses the target is bisected, all
    cells together, one batched forward pass per step;
  * the answer is the root nearest `sigma_hint` (the volatility the caller
    started from), and every root is returned with it.
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
    # False when the target lies outside every price the model produces in
    # the searched volatility range; sigma is then the sweep point whose
    # price is closest to the target.
    bracketed: bool
    # The lowest and highest price across the sweep.
    low_price: float
    high_price: float
    sigma_low: float
    sigma_high: float
    # Every volatility in the range that reproduces the target, ascending.
    # Empty when the target is out of reach.
    roots: tuple[float, ...] = ()


def _pick(roots: list[float], slopes: list[float],
          sigma_hint: float | None) -> int:
    """Index of the root to report: the one nearest the hint, or without a
    hint the one where the price moves most with volatility, which is the
    volatility the premium pins down best."""
    if sigma_hint is not None:
        return int(np.argmin([abs(r - sigma_hint) for r in roots]))
    return int(np.argmax([abs(s) for s in slopes]))


def solve_implied_vol(engine, spot: float, strike: float, maturity: float,
                      rate: float, target_price: float,
                      option_type: str = "call",
                      sigma_low: float | None = None,
                      sigma_high: float | None = None,
                      tol: float = 1e-6, max_iter: int = 60,
                      scan_points: int = 96,
                      sigma_hint: float | None = None) -> VolSolution:
    """Solve `engine(sigma) == target_price` for sigma.

    `tol` is on the price, in the same units as `target_price`, so the answer
    is as precise as the quote that produced it rather than to an arbitrary
    number of volatility decimals.

    When several volatilities reproduce the price, `sigma` is the one nearest
    `sigma_hint`; with no hint it is the one in the steepest part of the
    price curve. `roots` lists all of them.
    """
    box = engine.meta.get("param_ranges", {})
    lo_default, hi_default = box.get("sigma", [0.05, 0.8])
    lo = float(lo_default if sigma_low is None else sigma_low)
    hi = float(hi_default if sigma_high is None else sigma_high)
    if not hi > lo:
        raise ValueError("sigma_high must exceed sigma_low")
    if not np.isfinite(target_price) or target_price < 0:
        raise ValueError("target_price must be a non-negative number")
    if sigma_hint is not None and not np.isfinite(sigma_hint):
        raise ValueError("sigma_hint must be a finite number")

    def price_many(sigmas: np.ndarray) -> np.ndarray:
        n = sigmas.shape[0]
        return engine.price_batch(
            np.full(n, float(spot)), np.full(n, float(strike)),
            np.full(n, float(maturity)), sigmas.astype(np.float64),
            np.full(n, float(rate)), option_type=option_type,
        ).astype(np.float64)

    # One batched sweep gives the whole price-versus-volatility curve.
    grid = np.linspace(lo, hi, scan_points)
    prices = price_many(grid)
    p_min, p_max = float(prices.min()), float(prices.max())

    if not p_min <= target_price <= p_max:
        k = int(np.argmin(np.abs(prices - target_price)))
        return VolSolution(float(grid[k]), float(prices[k]), target_price, 0,
                           False, p_min, p_max, lo, hi)

    d = prices - target_price
    roots: list[float] = []
    slopes: list[float] = []

    # Grid points that hit the target exactly. The zero floor on the price
    # makes whole runs of them when the target is 0; each run is one root,
    # represented by its point nearest the hint (its first without one).
    hit = d == 0.0
    i = 0
    while i < scan_points:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < scan_points and hit[j + 1]:
            j += 1
        run = grid[i:j + 1]
        k = 0 if sigma_hint is None else int(np.argmin(np.abs(run - sigma_hint)))
        roots.append(float(run[k]))
        a, b = max(i - 1, 0), min(j + 1, scan_points - 1)
        slopes.append(float((prices[b] - prices[a]) / (grid[b] - grid[a])))
        i = j + 1

    # Cells whose two ends lie strictly on opposite sides of the target.
    cells = np.nonzero((d[:-1] * d[1:]) < 0.0)[0]
    iterations = 0
    if cells.size:
        a = grid[cells].astype(np.float64)
        b = grid[cells + 1].astype(np.float64)
        fa = d[cells].astype(np.float64)
        done = np.zeros(cells.size, dtype=bool)
        slopes_cells = (d[cells + 1] - d[cells]) / (b - a)
        while iterations < max_iter and not done.all():
            live = ~done
            mid = 0.5 * (a[live] + b[live])
            fm = price_many(mid) - target_price
            iterations += 1
            idx = np.nonzero(live)[0]
            for n, m, f in zip(idx, mid, fm):
                if abs(f) <= tol:
                    a[n] = b[n] = m
                    done[n] = True
                elif (fa[n] < 0) == (f < 0):
                    a[n], fa[n] = m, f
                else:
                    b[n] = m
                if b[n] - a[n] <= 1e-9:
                    done[n] = True
        roots.extend(float(x) for x in 0.5 * (a + b))
        slopes.extend(float(s) for s in slopes_cells)

    order = np.argsort(roots)
    roots = [roots[k] for k in order]
    slopes = [slopes[k] for k in order]
    chosen = _pick(roots, slopes, sigma_hint)
    sigma = roots[chosen]
    return VolSolution(sigma, float(price_many(np.array([sigma]))[0]),
                       target_price, iterations, True, p_min, p_max, lo, hi,
                       tuple(roots))
