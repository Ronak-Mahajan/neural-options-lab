"""High-precision gamma for the discrete arithmetic-average call.

Price, delta and vega have pathwise Monte Carlo estimators
(`dataset._simulate_chunk`). Gamma has none: differentiating the pathwise
delta again differentiates an indicator and leaves a Dirac mass, which cannot
be averaged. This module supplies the gamma reference that `evaluate.py`
measures the surrogate against.

The arithmetic average is linear in the spot,

    A = (1/n) sum_i S_i,   S_i = m exp(sum_{j<=i} (drift + vol z_j)),

so A = m * Atilde with Atilde independent of m. All of the spot dependence
sits in the payoff, and for the unit-strike call

    C(m)  = e^{-rT} E[(m Atilde - 1)^+]
    dC/dm = e^{-rT} E[Atilde 1{Atilde > 1/m}]
    d2C/dm2 = e^{-rT} p(1/m) / m^3

where p is the density of Atilde. Gamma is the discounted density of the
scaled average at the strike.

Conditional density (the estimator used here). Every fixing is proportional
to S_1, so the first increment is the one to integrate out:

    Atilde = exp(drift + vol z_1) * G,
    G = (1/n) sum_{i=1..n} exp(sum_{2<=j<=i} (drift + vol z_j)),

with G a function of z_2..z_n alone. Given those, Atilde is a strictly
increasing lognormal function of z_1 with full support, its conditional
density is closed-form, and z_1 integrates out exactly:

    p(a) = E[ phi(z*) ] / (vol a),   z* = (log(a/G) - drift) / vol,
    d2C/dm2 = e^{-rT} E[phi(z*)] / (vol m^2).

The estimator is a Monte Carlo average of an exact conditional density, with
no kernel and no bandwidth. It is unbiased for the density of the discrete
average the surrogate is trained on: the same n fixings and no
continuous-time approximation. Every path contributes and the integrand is
bounded by phi(0).

Conditioning on the last increment is also unbiased but less efficient. The
final fixing carries 1/n of the average, so only paths whose partial average
sits within one fixing of the strike contribute. At n = 50 on 200,000 paths
(seed 0) with m = 1, T = 1, sigma = 0.25, r = 0.04, its standard error is
2.7% of the estimate against 0.4% for the first increment, and the ratio is
6x to 9x over three contracts across the box
(`tests/test_quant_core.py` holds the comparison).

At n = 1 the average is the terminal spot and the contract is a European
call. G is identically 1, every path returns the same lognormal density, and
the estimate equals the Black-Scholes gamma to machine precision with no
Monte Carlo error. `tests/test_gamma_reference.py` pins that identity.

Cross-check. `gamma_finite_difference` differences the pathwise delta in the
spot on common random numbers. It is a histogram estimate of the same density
with bin width tied to the step, so it is biased at O(h^2) and noisier. It
rests on a different identity, so agreement with it rules out an algebra
error in the conditional derivation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

#: Paths per block. The estimators hold (block, n_steps) log-return matrices,
#: so this bounds peak memory at roughly block * n_steps * 8 bytes * 3.
PATH_BLOCK = 20_000


def _blocks(n_paths: int, block: int):
    """Path block sizes for antithetic sampling; n_paths must be even.

    The block size is rounded down to an even number, so every block is even
    when n_paths is. An odd remainder would lose one path to `size // 2` in
    the caller while its mean still divides by n_paths. `gamma_conditional`
    rounds n_paths down to even; the other estimators take it as given.
    """
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
    error of the bounded average E[phi(z*)] over antithetic pairs, carried
    through the same constant factor as the estimate itself.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    if vol <= 0.0:
        raise ValueError("gamma is a density and needs a positive volatility")
    a = 1.0 / m                      # the strike, in units of the scaled average
    n_paths = (int(n_paths) // 2) * 2          # antithetic pairs, so even
    n_pairs = n_paths // 2

    # The sampling unit is the antithetic pair: z and -z are dependent draws,
    # so the standard error is the spread of pair means over n_pairs
    # independent pairs. Treating the two halves as separate samples would
    # count each pair twice and misstate the error in either direction.
    total = 0.0
    total_sq = 0.0
    rng = np.random.default_rng(seed)
    for size in _blocks(n_paths, block):
        half = size // 2
        if n_steps == 1:
            # G is identically 1: the average is the single fixing, there are
            # no later increments to draw, and every path returns the same
            # exact lognormal density.
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
        pair_mean = 0.5 * (phi[:half] + phi[half:])
        total += float(pair_mean.sum())
        total_sq += float((pair_mean ** 2).sum())

    mean = total / n_pairs
    var = max(total_sq / n_pairs - mean ** 2, 0.0)
    se_phi = math.sqrt(var / n_pairs)
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

    This is the quantity `gamma_conditional` estimates, evaluated on one
    shared set of paths. Gamma at moneyness m is
    ``exp(-rT) * density(1/m) / m**3``.

    Two closed-form identities check the reference without a second
    estimator. ``integral p da`` is 1, and ``integral a p(a) da`` is the mean
    of the scaled average, ``(1/n) sum_i exp(r t_i)``, the geometric series
    the engine's put-call parity adjustment uses.
    `tests/test_gamma_reference.py` checks both.
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
    """d2C/dm2 by differencing the pathwise delta on common random numbers.

    An independent route to the same quantity, used only to corroborate
    `gamma_conditional`. Because the average is linear in the spot, the same
    draw of `Atilde` prices every spot, so the two deltas differ only through
    which paths finish in the money. The difference is a histogram count of
    the density over a bin of width 2 h / m^2, biased O(h^2) and much noisier
    than the conditional estimator.
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
    """Closed-form gamma, for the n = 1 case where the Asian is a European."""
    sd = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / sd
    return float(norm.pdf(d1) / (spot * sd))
