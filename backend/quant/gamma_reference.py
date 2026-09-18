"""High-precision gamma for the discrete arithmetic-average call.

Price and the first-order Greeks have pathwise Monte Carlo estimators, which
is why `dataset._simulate_chunk` returns delta and vega alongside the price
and why `evaluate.py` can report their held-out error. Gamma has no pathwise
estimator: differentiating the pathwise delta a second time differentiates an
indicator, which produces a Dirac mass rather than something you can average.
So gamma was the one served Greek with no accuracy number anywhere in the
project. This module supplies the reference that closes that.

The construction rests on one exact fact about this contract. The arithmetic
average is LINEAR in the spot:

    A = (1/n) sum_i S_i,   S_i = m exp(sum_{j<=i} (drift + vol z_j))

so A = m * Atilde with Atilde independent of m. All of the spot dependence
therefore sits in the payoff, and for the unit-strike call

    C(m)  = e^{-rT} E[(m Atilde - 1)^+]
    dC/dm = e^{-rT} E[Atilde 1{Atilde > 1/m}]
    d2C/dm2 = e^{-rT} p(1/m) / m^3

where p is the density of Atilde. Gamma is the discounted density of the
scaled average AT the strike, so measuring gamma is estimating one density at
one point - and that is a problem with an exact, bandwidth-free answer.

CONDITIONAL DENSITY (the estimator used here). The FIRST increment is the one
to integrate out, because every fixing is proportional to S_1:

    Atilde = exp(drift + vol z_1) * G,
    G = (1/n) sum_{i=1..n} exp(sum_{2<=j<=i} (drift + vol z_j)),

with G a function of z_2..z_n alone. Conditioning on those, Atilde is a
strictly increasing lognormal function of z_1 with full support, its
conditional density is closed-form, and z_1 integrates out exactly:

    p(a) = E[ phi(z*) ] / (vol a),   z* = (log(a/G) - drift) / vol,

so gamma reduces to a bounded average:

    d2C/dm2 = e^{-rT} E[phi(z*)] / (vol m^2).

There is no kernel and no bandwidth, only a Monte Carlo average of an exact
conditional density, so the estimator is unbiased for the density of the
DISCRETE average the surrogate was trained on - the same n fixings, the same
discretisation, no continuous-time approximation anywhere.

Conditioning on the LAST increment instead is the obvious first attempt and
is much worse: the final fixing carries only 1/n of the average, so only the
paths whose partial average already sits within one fixing of the strike
contribute anything, and the standard error on a 200,000-path run comes out
near 3 % of the estimate. Conditioning on the first increment moves the whole
distribution, every path contributes, the integrand is bounded by phi(0), and
the same budget gives a standard error two orders of magnitude smaller.

It is also checkable against a closed form with no Monte Carlo error at all.
At n = 1 the average IS the terminal spot, the contract is a European call,
and the estimator collapses to the exact lognormal density: every path gives
the identical value and it equals the Black-Scholes gamma to machine
precision. `tests/test_gamma_reference.py` pins that, which is a stronger
check than any convergence study.

CROSS-CHECK. `gamma_finite_difference` differences the pathwise delta in the
spot on the SAME paths (common random numbers). It is a histogram estimate of
the same density with bin width tied to the step, so it is biased at O(h^2)
and noisier, but it is built from a different identity and agreeing with it
rules out an algebra error in the conditional derivation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

#: Paths per block. The estimators hold (block, n_steps) log-return matrices,
#: so this bounds peak memory at roughly block * n_steps * 8 bytes * 3.
PATH_BLOCK = 20_000


def _blocks(n_paths: int, block: int):
    """Antithetic-safe path blocks: every block is an even number of paths."""
    block = max(2, (block // 2) * 2)
    done = 0
    while done < n_paths:
        yield min(block, n_paths - done)
        done += block


def gamma_conditional(m: float, maturity: float, sigma: float, rate: float,
                      n_steps: int, n_paths: int = 200_000,
                      seed: int = 0, block: int = PATH_BLOCK) -> dict:
    """d2C/dm2 for the unit-strike call, by conditional Monte Carlo.

    Returns ``{"gamma", "se", "n_paths", "phi_mean"}``. `se` is the standard
    error across paths of the bounded average E[phi(z*)], carried through the
    same constant factor as the estimate itself.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    if vol <= 0.0:
        raise ValueError("gamma is a density and needs a positive volatility")
    a = 1.0 / m                      # the strike, in units of the scaled average

    total = 0.0
    total_sq = 0.0
    rng = np.random.default_rng(seed)
    for size in _blocks(n_paths, block):
        half = size // 2
        if n_steps == 1:
            # G is identically 1: the average is the single fixing. Every
            # path gives the same exact lognormal density, so one evaluation
            # is the answer and drawing normals would only add noise.
            g = np.ones(size)
        else:
            z = rng.standard_normal((half, n_steps - 1))
            z = np.concatenate([z, -z], axis=0)          # antithetic
            # U_1 = 0 (the first fixing is the factored-out one), then the
            # partial sums of the remaining increments.
            u = np.cumsum(drift + vol * z, axis=1)
            g = (1.0 + np.exp(u).sum(axis=1)) / n_steps
            del z, u

        z_star = (np.log(a / g) - drift) / vol
        phi = norm.pdf(z_star)
        total += float(phi.sum())
        total_sq += float((phi ** 2).sum())

    mean = total / n_paths
    var = max(total_sq / n_paths - mean ** 2, 0.0)
    se_phi = math.sqrt(var / n_paths)
    scale = math.exp(-rate * maturity) / (vol * m ** 2)
    return {
        "gamma": scale * mean,
        "se": scale * se_phi,
        "n_paths": int(n_paths),
        "phi_mean": mean,
    }


def scaled_average_density(a_grid: np.ndarray, maturity: float, sigma: float,
                           rate: float, n_steps: int,
                           n_paths: int = 200_000, seed: int = 0,
                           block: int = PATH_BLOCK) -> np.ndarray:
    """The density of the scaled average `Atilde` at each point of `a_grid`.

    Exposes the object `gamma_conditional` actually estimates, evaluated on
    one shared set of paths. Gamma at moneyness m is
    ``exp(-rT) * density(1/m) / m**3``.

    This is what makes the reference checkable without a second estimator: a
    density has two exact properties, and both are closed form here.
    ``integral p da`` is 1, and ``integral a p(a) da`` is the mean of the
    scaled average, ``(1/n) sum_i exp(r t_i)`` - the same geometric series
    the engine's put-call parity adjustment uses. Neither is a convergence
    study or a comparison against another approximation; they are identities
    the estimate either satisfies or does not.
    """
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    a = np.asarray(a_grid, dtype=np.float64).reshape(-1)
    total = np.zeros(a.size)
    rng = np.random.default_rng(seed)
    for size in _blocks(n_paths, block):
        half = size // 2
        if n_steps == 1:
            g = np.ones(size)
        else:
            z = rng.standard_normal((half, n_steps - 1))
            z = np.concatenate([z, -z], axis=0)
            u = np.cumsum(drift + vol * z, axis=1)
            g = (1.0 + np.exp(u).sum(axis=1)) / n_steps
            del z, u
        # (paths, grid): phi of the z_1 that lands the average on each a.
        z_star = (np.log(a[None, :] / g[:, None]) - drift) / vol
        total += norm.pdf(z_star).sum(axis=0)
    return total / (n_paths * vol * a)


def mean_scaled_average(maturity: float, rate: float, n_steps: int) -> float:
    """E[Atilde] = (1/n) sum_i exp(r t_i), in closed form."""
    dt = maturity / n_steps
    t = dt * np.arange(1, n_steps + 1)
    return float(np.exp(rate * t).mean())


def gamma_finite_difference(m: float, maturity: float, sigma: float,
                            rate: float, n_steps: int,
                            n_paths: int = 200_000, seed: int = 0,
                            rel_step: float = 0.01,
                            block: int = PATH_BLOCK) -> dict:
    """d2C/dm2 by differencing the PATHWISE delta on common random numbers.

    An independent route to the same quantity, used to corroborate
    `gamma_conditional` rather than to serve numbers. Because the average is
    linear in the spot, the same draw of `Atilde` prices every spot, so the
    two deltas differ only through which paths finish in the money - the
    difference is a histogram count of the density over a bin of width
    2 h / m^2, biased O(h^2) and much noisier than the conditional estimator.
    """
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    h = rel_step * m
    lo, hi = m - h, m + h
    if lo <= 0.0:
        raise ValueError("rel_step too large: the down-shifted spot is not positive")

    delta_hi = 0.0
    delta_lo = 0.0
    rng = np.random.default_rng(seed)
    for size in _blocks(n_paths, block):
        half = size // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
        atilde = np.exp(np.cumsum(drift + vol * z, axis=1)).mean(axis=1)
        del z
        # Pathwise delta at spot s is E[Atilde 1{s Atilde > 1}].
        delta_hi += float((atilde * (hi * atilde > 1.0)).sum())
        delta_lo += float((atilde * (lo * atilde > 1.0)).sum())

    disc = math.exp(-rate * maturity)
    d_hi = disc * delta_hi / n_paths
    d_lo = disc * delta_lo / n_paths
    return {"gamma": (d_hi - d_lo) / (2.0 * h), "rel_step": rel_step,
            "n_paths": int(n_paths)}


def black_scholes_gamma(spot: float, strike: float, maturity: float,
                        sigma: float, rate: float) -> float:
    """Closed-form gamma, for the n = 1 case where the Asian IS a European."""
    sd = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / sd
    return float(norm.pdf(d1) / (spot * sd))
