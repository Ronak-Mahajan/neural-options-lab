"""Levy (1992) moment matching for the discretely monitored arithmetic Asian.

Arithmetic Asian options have had fast closed-form approximations since the
early 1990s, so a surrogate has two baselines to be measured against: the Monte
Carlo it replaces and a formula that runs in microseconds. This module
implements the standard formula. asian_approx.py adds Curran (1994), and
scripts/benchmark_approximations.py scores both and the served ensemble against
control-variate Monte Carlo.

Levy (1992) / Turnbull & Wakeman (1991) moment matching
-------------------------------------------------------
Approximate the arithmetic average A as lognormal by matching its first two exact
moments. With t_i = i*T/n:

    M1 = E[A]   = (S0/n) * sum_i e^{r t_i}
    M2 = E[A^2] = (S0^2/n^2) * sum_i sum_j e^{r(t_i + t_j) + sigma^2 min(t_i, t_j)}

using E[S(t_i) S(t_j)] = S0^2 exp(r(t_i+t_j) + sigma^2 min(t_i,t_j)). Then treat A as
lognormal with variance v = ln M2 - 2 ln M1 and apply a Black-Scholes-shaped formula:

    C = e^{-rT} [ M1 * N(d1) - K * N(d2) ],
    d1 = (ln(M1/K) + v/2)/sqrt(v),  d2 = d1 - sqrt(v)

The approximation is exact in the first two moments and degrades where the true
distribution of A is far from lognormal: the high-volatility, long-maturity
corner of the parameter box.

Measured accuracy
-----------------
`python scripts/benchmark_approximations.py --lhs` scores this formula on 300
Latin-hypercube points over the trained box against 200,000-path
control-variate references, with a fixed seed. RMSE, bias, tail error and
latency, and RMSE by volatility band and by price level, are stored in
docs/approximation_benchmark.json under "box_lhs" and rendered in
docs/approximation_benchmark.md.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["levy_asian_call", "levy_asian_price"]


def levy_asian_call(spot: float, strike: float, maturity: float, sigma: float,
                    rate: float, n_steps: int = 50) -> float:
    """Moment-matched arithmetic Asian call price (Levy 1992).

    Discrete monitoring at t_i = i*T/n, i = 1..n, matching the convention used
    everywhere else in this package.
    """
    if maturity <= 0.0 or sigma <= 0.0:
        return max(spot - strike, 0.0)
    t = np.arange(1, n_steps + 1) * (maturity / n_steps)
    m1 = spot * np.exp(rate * t).sum() / n_steps
    ti, tj = t[:, None], t[None, :]
    m2 = (spot ** 2) * np.exp(rate * (ti + tj)
                              + sigma ** 2 * np.minimum(ti, tj)).sum() / n_steps ** 2
    v = float(np.log(m2) - 2.0 * np.log(m1))
    disc = float(np.exp(-rate * maturity))
    if v <= 1e-14:                       # degenerate: A is deterministic
        return max(disc * (m1 - strike), 0.0)
    sd = np.sqrt(v)
    d1 = (np.log(m1 / strike) + 0.5 * v) / sd
    d2 = d1 - sd
    return float(disc * (m1 * norm.cdf(d1) - strike * norm.cdf(d2)))


def levy_asian_price(spot: float, strike: float, maturity: float, sigma: float,
                     rate: float, n_steps: int = 50,
                     option_type: str = "call") -> float:
    """Levy call, plus the put via exact Asian parity C - P = e^{-rT}(E[A] - K)."""
    call = levy_asian_call(spot, strike, maturity, sigma, rate, n_steps)
    if option_type == "call":
        return call
    t = np.arange(1, n_steps + 1) * (maturity / n_steps)
    ea = spot * np.exp(rate * t).sum() / n_steps
    return float(call - np.exp(-rate * maturity) * (ea - strike))
