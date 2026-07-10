"""Monte Carlo engine for arithmetic-average Asian options under GBM.

Pricing methodology
-------------------
The underlying follows geometric Brownian motion under the risk-neutral
measure:

    dS_t = r S_t dt + sigma S_t dW_t

The option pays on the *discrete arithmetic average* of the spot observed at
n equally spaced monitoring dates t_i = i * T / n, i = 1..n:

    Asian call payoff:  max(A - K, 0),   A = (1/n) * sum_i S(t_i)
    Asian put  payoff:  max(K - A, 0)

There is no closed form for the arithmetic Asian, but the *geometric* Asian
(G = (prod_i S(t_i))^(1/n)) is lognormal and admits a Black-Scholes-style
closed form (Kemna & Vorst, 1990; discrete-monitoring variant). Because G is
highly correlated with A, it makes an excellent control variate: we simulate
both payoffs on the same paths and correct the arithmetic estimate by the
known geometric bias. Combined with antithetic sampling this typically cuts
the standard error by 1-2 orders of magnitude at the same path budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Closed-form building blocks
# ---------------------------------------------------------------------------

def expected_arithmetic_average(spot: float, rate: float, maturity: float,
                                n_steps: int) -> float:
    """E[A] for the discrete arithmetic average under the risk-neutral measure.

    E[A] = (S0 / n) * sum_{i=1..n} exp(r * t_i), which telescopes to a
    geometric series. This powers exact Asian put-call parity:
        C - P = exp(-rT) * (E[A] - K)
    """
    if abs(rate) < 1e-12:
        return spot
    dt = maturity / n_steps
    g = math.exp(rate * dt)
    return spot * g * (g ** n_steps - 1.0) / (n_steps * (g - 1.0))


def geometric_asian_price(spot: float, strike: float, maturity: float,
                          sigma: float, rate: float, n_steps: int,
                          option_type: str = "call") -> float:
    """Closed form for the discretely monitored geometric Asian option.

    ln G is Gaussian with
        mean = ln S0 + (r - sigma^2/2) * dt * (n+1)/2
        var  = sigma^2 * dt * (n+1)(2n+1) / (6n)
    and the price follows from the standard lognormal expectation.
    """
    n = n_steps
    dt = maturity / n
    mu = math.log(spot) + (rate - 0.5 * sigma ** 2) * dt * (n + 1) / 2.0
    var = sigma ** 2 * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    sd = math.sqrt(max(var, 1e-16))
    disc = math.exp(-rate * maturity)
    d1 = (mu + var - math.log(strike)) / sd
    d2 = d1 - sd
    fwd = math.exp(mu + 0.5 * var)
    if option_type == "call":
        return disc * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    return disc * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int


def price_asian_mc(spot: float, strike: float, maturity: float, sigma: float,
                   rate: float, n_paths: int = 50_000, n_steps: int = 50,
                   option_type: str = "call", seed: int | None = None,
                   control_variate: bool = True) -> MCResult:
    """Price a discrete arithmetic Asian option by Monte Carlo.

    Uses antithetic variates always, and the geometric Asian control variate
    unless disabled (the plain estimator is kept around for benchmarking the
    variance reduction itself).
    """
    rng = np.random.default_rng(seed)
    half = max(n_paths // 2, 1)
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)

    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)                    # antithetic
    log_paths = math.log(spot) + np.cumsum(drift + vol * z, axis=1)

    arith = np.exp(log_paths).mean(axis=1)                 # A per path
    disc = math.exp(-rate * maturity)
    sign = 1.0 if option_type == "call" else -1.0
    x = disc * np.maximum(sign * (arith - strike), 0.0)

    if control_variate:
        geo = np.exp(log_paths.mean(axis=1))               # G per path
        y = disc * np.maximum(sign * (geo - strike), 0.0)
        ey = geometric_asian_price(spot, strike, maturity, sigma, rate,
                                   n_steps, option_type)
        cov = np.cov(x, y)
        beta = cov[0, 1] / max(cov[1, 1], 1e-16)
        x = x - beta * (y - ey)

    n = x.shape[0]
    price = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n))
    return MCResult(price=price, std_error=se,
                    ci_low=price - 1.96 * se, ci_high=price + 1.96 * se,
                    n_paths=n, n_steps=n_steps)
