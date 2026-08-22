"""Training-set generation for the neural Asian-option pricer.

Key design decisions
--------------------
1. **Moneyness parameterization.** The Asian price is homogeneous of degree 1
   in (S, K): price(S, K, ...) = K * price(S/K, 1, ...). We therefore train
   the network on f(m, T, sigma, r) = price(m, 1, T, sigma, r) with
   m = S/K, collapsing one input dimension and guaranteeing exact
   generalization across all strike levels.

2. **Latin Hypercube sampling** of the parameter box gives far better space
   coverage than i.i.d. uniform draws at the same sample count.

3. **Noisy labels are fine.** Each label is a control-variate MC estimate
   with a modest path budget. The noise is (asymptotically) unbiased, and
   least-squares regression averages it out across the dataset - so we spend
   the simulation budget on *many parameter points* rather than ultra-precise
   labels at few points.

4. **Chunked, fully vectorized simulation.** Paths for a whole chunk of
   parameter sets are generated in one (chunk, paths, steps) tensor,
   amortizing numpy overhead.

5. **Pathwise differentials (Differential ML).** Alongside each price label
   we compute the *pathwise* sensitivities of the discounted payoff
   X = e^{-rT}(A - K)+ on the same paths - essentially free once the paths
   exist (Huge & Savine, "Differential Machine Learning", 2020):

       dX/dS0    = e^{-rT} 1{A>K} A/S0            (S_i is linear in S0)
       dX/dsigma = e^{-rT} 1{A>K} (1/n) sum_i S_i (-sigma t_i + sqrt(dt) W_i)

   where W_i is the cumulated Gaussian driver up to t_i. Both estimators are
   unbiased because the payoff is Lipschitz in A. Training the network to
   match these differentials (via a combined loss) teaches it the *shape*
   of the pricing function, not just point values.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import qmc

from .monte_carlo import geometric_asian_price

# Parameter box the model is trained on (and should be served within).
PARAM_RANGES = {
    "moneyness": (0.5, 2.0),    # S / K
    "maturity":  (0.05, 2.0),   # years
    "sigma":     (0.05, 0.80),  # annualized vol
    "rate":      (0.00, 0.10),  # risk-free rate
}
N_MONITORING_STEPS = 50  # fixed averaging protocol for the whole project


def _simulate_chunk(params: np.ndarray, n_paths: int, n_steps: int,
                    rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV Monte Carlo call prices + pathwise (delta, vega) for a (B, 4)
    chunk of (m, T, sigma, r).

    Prices the unit-strike call at spot=m. Antithetic everywhere; the
    geometric control variate is applied to the *price* estimator (the
    differential estimators are left plain - their noise is averaged out by
    the differential regression, per Huge & Savine).
    Returns (price, dprice/dm, dprice/dsigma), each shape (B,).
    """
    m, mat, sig, r = (params[:, i][:, None] for i in range(4))
    b = params.shape[0]
    half = n_paths // 2
    dt = mat / n_steps                                     # (B, 1)
    drift = (r - 0.5 * sig ** 2) * dt
    vol = sig * np.sqrt(dt)

    z = rng.standard_normal((b, half, n_steps))
    z = np.concatenate([z, -z], axis=1)                    # (B, P, S)
    cum_z = np.cumsum(z, axis=2)  # Gaussian driver W_i
    log_paths = np.log(m)[:, :, None] + np.cumsum(
        drift[:, :, None] + vol[:, :, None] * z, axis=2)
    del z

    paths = np.exp(log_paths)                              # S_i, (B, P, S)
    arith = paths.mean(axis=2)                             # A, (B, P)
    geo = np.exp(log_paths.mean(axis=2))
    del log_paths
    disc = np.exp(-r * mat)                                # (B, 1)
    in_money = arith > 1.0                                 # (B, P)

    # -- pathwise differentials -------------------------------------------
    delta = (disc * np.where(in_money, arith, 0.0) / m).mean(axis=1)

    t_grid = dt[:, :, None] * np.arange(1, n_steps + 1)    # t_i, (B, 1, S)
    dlnS_dsig = -sig[:, :, None] * t_grid \
        + np.sqrt(dt)[:, :, None] * cum_z                  # (B, P, S)
    dA_dsig = (paths * dlnS_dsig).mean(axis=2)             # (B, P)
    del paths, dlnS_dsig, cum_z
    vega = (disc * np.where(in_money, dA_dsig, 0.0)).mean(axis=1)

    # -- control-variate price --------------------------------------------
    x = disc * np.maximum(arith - 1.0, 0.0)
    y = disc * np.maximum(geo - 1.0, 0.0)
    ey = np.array([
        geometric_asian_price(float(params[i, 0]), 1.0, float(params[i, 1]),
                              float(params[i, 2]), float(params[i, 3]),
                              n_steps, "call")
        for i in range(b)
    ])
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    beta = (xc * yc).mean(axis=1) / np.maximum((yc * yc).mean(axis=1), 1e-16)
    price = (x - beta[:, None] * (y - ey[:, None])).mean(axis=1)
    return price, delta, vega                              # each (B,)


def generate_dataset(n_samples: int = 40_000, n_paths: int = 2_000,
                     n_steps: int = N_MONITORING_STEPS, seed: int = 7,
                     chunk_size: int = 64, verbose: bool = True
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, dydx): LHS parameters (N, 4), CV-MC call prices/K (N,)
    and pathwise differentials (N, 2) - columns (dprice/dm, dprice/dsigma).
    """
    lows = np.array([lo for lo, _ in PARAM_RANGES.values()])
    highs = np.array([hi for _, hi in PARAM_RANGES.values()])

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    X = lows + sampler.random(n_samples) * (highs - lows)

    rng = np.random.default_rng(seed + 1)
    y = np.empty(n_samples)
    dydx = np.empty((n_samples, 2))
    t0 = time.perf_counter()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        price, delta, vega = _simulate_chunk(X[start:end], n_paths,
                                             n_steps, rng)
        y[start:end] = price
        dydx[start:end, 0] = delta
        dydx[start:end, 1] = vega
        if verbose and (start // chunk_size) % 50 == 0:
            done = end / n_samples
            elapsed = time.perf_counter() - t0
            eta = elapsed / max(done, 1e-9) * (1 - done)
            print(f"  dataset: {end:>7,}/{n_samples:,} "
                  f"({done:5.1%})  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s",
                  flush=True)
    return X, y, dydx
