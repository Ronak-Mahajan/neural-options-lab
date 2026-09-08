"""Closed-form approximations for the discretely monitored arithmetic Asian option.

Two standard approximations, sharing one convention with the rest of the
package: the average is taken over n equally spaced monitoring dates
t_i = i*T/n, i = 1..n, and puts come from the exact Asian parity

    C - P = e^{-rT} (E[A] - K),   E[A] = expected_arithmetic_average(...)

so that every put here satisfies parity to floating-point precision.

1. Turnbull & Wakeman (1991) / Levy (1992) moment matching
   Already implemented in ``benchmarks.levy_asian_call`` with the exact first
   two moments of the discrete average, so this module does NOT re-derive it:
   ``turnbull_wakeman_price`` delegates to that function. What this module
   adds is ``arithmetic_average_moments``, the moments themselves as a
   reusable, testable object. With E[S_i S_j] = S0^2 e^{r(t_i+t_j)} e^{sigma^2 min(t_i,t_j)}:

       M1 = E[A]   = (S0/n)   sum_i e^{r t_i}
       M2 = E[A^2] = (S0^2/n^2) sum_i sum_j e^{r(t_i+t_j) + sigma^2 min(t_i,t_j)}

   and the matched lognormal has log-variance v = ln M2 - 2 ln M1. For n = 1
   these are the Black-Scholes moments (M1 = S0 e^{rT}, v = sigma^2 T), and
   the price is exactly Black-Scholes; that is tested.

2. Curran (1994) conditioning on the geometric average
   G = (prod_i S_i)^{1/n} is lognormal and A >= G pathwise (AM-GM), so on the
   event {G >= K} the call pays A - K with certainty and E[(A - K) 1{G >= g}]
   has a closed form because (ln S_i, ln G) is bivariate normal. Curran writes

       E[(A-K)^+] = E[(A-K)^+ ; G >= K] + E[(A-K)^+ ; G < K]

   takes the first term exactly, and in the second replaces A by its
   conditional mean E[A | G]. The result is a single closed-form expression
   with the indicator threshold moved from K down to the g_hat solving
   E[A | G = g_hat] = K:

       C = e^{-rT} [ (1/n) sum_i e^{mu_i + s_i^2/2} N((mu_G + c_i - ln g_hat)/s_G)
                     - K N((mu_G - ln g_hat)/s_G) ]

   with mu_i, s_i^2 the mean and variance of ln S_i, mu_G, s_G^2 those of
   ln G, and c_i = Cov(ln S_i, ln G) = sigma^2 (1/n) sum_j min(t_i, t_j).

   Because E[A|G] is increasing in G, this is exactly the Jensen lower bound
   E[(E[A|G] - K)^+] <= E[(A-K)^+] (Rogers & Shi 1995), so the Curran price
   is a rigorous LOWER bound on the true call, and it lies above the
   geometric Asian price since E[A|G] >= G. Both orderings are tested.

   Curran's paper solves for g_hat by a first-order expansion around K,
   g_hat ~= 2K - E[A | G = K], which is the form reproduced in Haug's
   handbook; ``threshold="linear"`` gives that. The default
   ``threshold="exact"`` solves E[A | G = g_hat] = K by Newton's method on a
   convex function (a sum of exponentials in ln g), which converges from
   ln K in a handful of steps and is what makes the lower-bound property
   exact rather than approximate. The two thresholds agree to well under a
   basis point of strike near the money; the difference is reported by
   scripts/benchmark_approximations.py, not assumed.

Neither approximation is a substitute for the reference Monte Carlo; they are
the fast baselines the neural surrogate has to beat, and the benchmark script
measures both against 400,000-path control-variate references.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .benchmarks import levy_asian_call, levy_asian_price
from .monte_carlo import expected_arithmetic_average

__all__ = [
    "monitoring_times",
    "arithmetic_average_moments",
    "turnbull_wakeman_price",
    "curran_call",
    "curran_price",
]

# Below this log-variance the average is treated as deterministic. Same
# threshold benchmarks.levy_asian_call uses for its degenerate branch.
_DEGENERATE_VAR = 1e-14


def monitoring_times(maturity: float, n_steps: int) -> np.ndarray:
    """t_i = i*T/n for i = 1..n: the project's fixed averaging protocol."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    return np.arange(1, n_steps + 1, dtype=float) * (maturity / n_steps)


def arithmetic_average_moments(spot: float, maturity: float, sigma: float,
                               rate: float, n_steps: int = 50
                               ) -> tuple[float, float]:
    """Exact (M1, M2) = (E[A], E[A^2]) of the discrete arithmetic average.

    The double sum uses the full GBM covariance structure
    E[S_i S_j] = S0^2 exp(r(t_i + t_j)) exp(sigma^2 min(t_i, t_j)); no
    continuous-averaging shortcut, so it matches the monitored payoff the
    Monte Carlo engine prices. Identical to the inline computation inside
    benchmarks.levy_asian_call (tested), exposed here so the moments can be
    checked on their own.
    """
    t = monitoring_times(maturity, n_steps)
    n = n_steps
    m1 = spot * float(np.exp(rate * t).sum()) / n
    ti, tj = t[:, None], t[None, :]
    m2 = spot ** 2 * float(np.exp(rate * (ti + tj)
                                  + sigma ** 2 * np.minimum(ti, tj)).sum()) / n ** 2
    return m1, m2


def turnbull_wakeman_price(spot: float, strike: float, maturity: float,
                           sigma: float, rate: float, n_steps: int = 50,
                           option_type: str = "call") -> float:
    """Turnbull-Wakeman / Levy moment-matched price, call or put.

    A named alias for benchmarks.levy_asian_price: the two papers give the
    same lognormal-matching formula once the exact discrete moments are used,
    and the project already implements it there. The put comes from exact
    Asian parity inside that function.
    """
    return levy_asian_price(spot, strike, maturity, sigma, rate, n_steps,
                            option_type)


# ---------------------------------------------------------------------------
# Curran (1994)
# ---------------------------------------------------------------------------

def _conditioning_geometry(spot: float, maturity: float, sigma: float,
                           rate: float, n_steps: int):
    """Gaussian parameters of (ln S_i)_i and ln G, and their covariances.

    Returns mu_i (n,), var_i (n,), mu_G, var_G, cov_iG (n,).
    """
    t = monitoring_times(maturity, n_steps)
    log_s0 = math.log(spot)
    mu_i = log_s0 + (rate - 0.5 * sigma ** 2) * t
    var_i = sigma ** 2 * t
    mu_g = log_s0 + (rate - 0.5 * sigma ** 2) * float(t.mean())
    ti, tj = t[:, None], t[None, :]
    min_tt = np.minimum(ti, tj)
    cov_ig = sigma ** 2 * min_tt.mean(axis=1)          # (1/n) sum_j min(t_i,t_j)
    var_g = sigma ** 2 * float(min_tt.mean())          # (1/n^2) sum_ij min(t_i,t_j)
    return mu_i, var_i, mu_g, var_g, cov_ig


def _conditional_mean_terms(mu_i, var_i, mu_g, var_g, cov_ig):
    """E[S_i | ln G = x] = a_i * exp(b_i * x): return (a_i, b_i), each (n,)."""
    b = cov_ig / var_g
    log_a = mu_i - b * mu_g + 0.5 * (var_i - cov_ig ** 2 / var_g)
    return np.exp(log_a), b


def _solve_threshold(a: np.ndarray, b: np.ndarray, strike: float,
                     n: int, x_start: float) -> float:
    """Solve (1/n) sum_i a_i exp(b_i x) = K for x by Newton's method.

    h(x) = (1/n) sum_i a_i e^{b_i x} - K is increasing and convex (all a_i,
    b_i > 0). Starting from x_start = ln K, where h >= 0 because
    E[A | G = K] >= K by AM-GM, Newton iterates decrease monotonically to the
    root and cannot overshoot it. Convergence is quadratic; 50 iterations is
    a ceiling never reached in practice, and the assertion below is the
    guard, not a silent fallback.
    """
    x = float(x_start)
    for _ in range(50):
        e = a * np.exp(b * x)
        h = float(e.sum()) / n - strike
        dh = float((b * e).sum()) / n
        step = h / dh
        x -= step
        if abs(step) < 1e-14 * max(1.0, abs(x)):
            return x
    raise RuntimeError("Curran threshold did not converge")   # pragma: no cover


def curran_call(spot: float, strike: float, maturity: float, sigma: float,
                rate: float, n_steps: int = 50,
                threshold: str = "exact") -> float:
    """Curran (1994) conditioning approximation for the arithmetic Asian CALL.

    threshold="exact"  solves E[A | G = g_hat] = K exactly (default); the
                       result is the Rogers-Shi lower bound conditioned on G.
    threshold="linear" uses Curran's first-order g_hat = 2K - E[A | G = K].
    """
    if threshold not in ("exact", "linear"):
        raise ValueError(f"unknown threshold {threshold!r}")
    if maturity <= 0.0:
        return max(spot - strike, 0.0)
    n = n_steps
    disc = math.exp(-rate * maturity)
    if sigma <= 0.0:
        # No randomness: A equals its mean and the call is the discounted
        # forward intrinsic, e^{-rT}(E[A] - K)^+, not (S0 - K)^+.
        return max(disc * (expected_arithmetic_average(spot, rate, maturity, n)
                           - strike), 0.0)
    mu_i, var_i, mu_g, var_g, cov_ig = _conditioning_geometry(
        spot, maturity, sigma, rate, n)
    if var_g <= _DEGENERATE_VAR:                       # A is deterministic
        m1, _ = arithmetic_average_moments(spot, maturity, sigma, rate, n)
        return max(disc * (m1 - strike), 0.0)

    a, b = _conditional_mean_terms(mu_i, var_i, mu_g, var_g, cov_ig)
    log_k = math.log(strike)
    if threshold == "linear":
        cond_mean_at_k = float((a * np.exp(b * log_k)).sum()) / n
        g_hat = 2.0 * strike - cond_mean_at_k
        if g_hat <= 0.0:
            # The expansion has left its domain (deep in the money at long
            # maturity). Curran's own remedy is to fall back to the exact
            # solve; do that rather than return a NaN.
            x_hat = _solve_threshold(a, b, strike, n, log_k)
        else:
            x_hat = math.log(g_hat)
    else:
        x_hat = _solve_threshold(a, b, strike, n, log_k)

    sd_g = math.sqrt(var_g)
    fwd_i = np.exp(mu_i + 0.5 * var_i)               # E[S_i]
    term = float((fwd_i * norm.cdf((mu_g + cov_ig - x_hat) / sd_g)).sum()) / n
    call = disc * (term - strike * norm.cdf((mu_g - x_hat) / sd_g))
    return max(float(call), 0.0)


def curran_price(spot: float, strike: float, maturity: float, sigma: float,
                 rate: float, n_steps: int = 50, option_type: str = "call",
                 threshold: str = "exact") -> float:
    """Curran call, plus the put via exact Asian parity C - P = e^{-rT}(E[A] - K)."""
    call = curran_call(spot, strike, maturity, sigma, rate, n_steps, threshold)
    if option_type == "call":
        return call
    if option_type != "put":
        raise ValueError(f"unknown option_type {option_type!r}")
    ea = expected_arithmetic_average(spot, rate, maturity, n_steps)
    return float(call - math.exp(-rate * maturity) * (ea - strike))
