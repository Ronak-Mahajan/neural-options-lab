"""Classical fast approximations for the arithmetic Asian, and why the surrogate earns its keep.

The obvious objection
---------------------
A neural surrogate that replaces Monte Carlo is only interesting if Monte Carlo is
the relevant alternative. It usually isn't. Arithmetic Asian options have had fast
closed-form approximations since the early 1990s, and any quant will ask why a
133k-parameter network beats a formula that runs in microseconds. This module
implements the standard one so the question has a measured answer rather than a
hand-wave.

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
distribution of A is far from lognormal, which is precisely the high-volatility,
long-maturity corner of the parameter box.

Measured result
---------------
300 Latin-hypercube points over the trained box, referenced against 200,000-path
control-variate Monte Carlo (all figures in bps of strike):

                        RMSE     MAE     bias    p95|e|   max|e|   latency
    neural ensemble     1.551   1.258   +1.102    2.743    8.032   714 us p50
    Levy moment-match  44.105  21.330  +19.793  101.672  227.006    56 us
    MC 200k paths           (reference)                            357,000 us

By true price magnitude, the surrogate wins everywhere except the near-worthless
bucket:

    true price [0,1) bps      Levy 0.077  vs  NN 1.050   -> Levy
    true price [1,10)         Levy 1.078  vs  NN 1.024   -> NN
    true price [10,100)       Levy 4.504  vs  NN 1.108   -> NN
    true price [100,1000)     Levy 6.522  vs  NN 1.311   -> NN
    true price [1000,inf)     Levy 53.243 vs  NN 1.690   -> NN

So the surrogate is ~28x more accurate than the closed form and ~500x faster than
Monte Carlo: a genuine point on the speed/accuracy frontier that neither alternative
occupies. The single regime where Levy wins is where the true price is essentially
zero — Levy returns ~0 correctly, while the Softplus output floor of the surrogate
cannot. That is the same architectural bias documented in scripts/fullscale_ablation.py,
observed here from an independent direction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["levy_asian_call", "levy_asian_price"]


def levy_asian_call(spot: float, strike: float, maturity: float, sigma: float,
                    rate: float, n_steps: int = 50) -> float:
    """Moment-matched arithmetic Asian CALL price (Levy 1992).

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
