"""CUDA label generator for the Asian-option surrogate, in double precision.

Why this module exists
----------------------
Label generation dominates the cost of this project: the shipped checkpoint used
500,000 parameter points x 5,000 paths x 50 monitoring steps = 1.25e11 path-steps,
generated entirely in numpy on the CPU. The computation is elementwise arithmetic
plus two cumulative sums over a (batch, paths, steps) tensor, which is exactly the
shape of work a GPU is built for. Porting it moves dataset generation from roughly
an hour to a few minutes and makes full-scale ablations affordable.

Why float64 and not float32
---------------------------
Measured, not assumed. Running this kernel at fp32 and fp64 through *identical*
generator seeds (so the only difference is rounding) gives:

    paths    price RMSE    delta RMSE    vega RMSE     (fp32 vs fp64, bps)
     5,000      10.03         54.75        161.81
    50,000       3.12         15.58         49.71

The surrogate targets ~1 bp of pricing error, so fp32 injects several times the
entire error budget before the network sees a single label. The culprit is the
control variate: it forms x - beta * (y - E[y]) where x and y are deliberately
near-identical (that is the whole point of a control variate), so the subtraction
is catastrophic cancellation, compounded by a 50-step cumulative sum. fp64 costs
about 2x the throughput of fp32 here and is not optional.

Numerics are otherwise identical to dataset._simulate_chunk: antithetic sampling,
a geometric-Asian control variate with in-sample beta, and pathwise differentials
for delta and vega (Huge & Savine, "Differential Machine Learning", 2020).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import qmc

from .dataset import PARAM_RANGES, N_MONITORING_STEPS

__all__ = ["geometric_asian_call_torch", "simulate_chunk_gpu",
           "generate_dataset_gpu", "default_device"]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def geometric_asian_call_torch(m: torch.Tensor, mat: torch.Tensor,
                               sig: torch.Tensor, r: torch.Tensor,
                               n_steps: int) -> torch.Tensor:
    """Closed-form discretely-monitored geometric Asian CALL at unit strike.

    Differentiable, batched, and device/dtype-agnostic - this is the same
    Kemna & Vorst (1990) formula implemented in monte_carlo.geometric_asian_price,
    rewritten in torch so it can serve two roles: the control-variate expectation
    here, and the analytic baseline of the residual surrogate in the ablation.

    ln G is Gaussian with
        mean = ln m + (r - sigma^2/2) * dt * (n+1)/2
        var  = sigma^2 * dt * (n+1)(2n+1) / (6n)
    """
    n = n_steps
    dt = mat / n
    mu = torch.log(m) + (r - 0.5 * sig ** 2) * dt * (n + 1) / 2.0
    var = sig ** 2 * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    sd = torch.sqrt(torch.clamp(var, min=1e-300))
    disc = torch.exp(-r * mat)
    d1 = (mu + var) / sd                      # strike = 1 so ln K = 0
    d2 = d1 - sd
    fwd = torch.exp(mu + 0.5 * var)
    return disc * (fwd * torch.special.ndtr(d1) - torch.special.ndtr(d2))


def simulate_chunk_gpu(params: torch.Tensor, n_paths: int, n_steps: int,
                       generator: torch.Generator
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Control-variate MC price plus pathwise (delta, vega) for a (B, 4) chunk.

    params columns are (moneyness, maturity, sigma, rate); the unit-strike call is
    priced at spot = moneyness. Mirrors dataset._simulate_chunk exactly.
    Returns three (B,) tensors.
    """
    m, mat, sig, r = (params[:, i:i + 1] for i in range(4))       # (B,1)
    dev, dt_ = params.device, params.dtype
    b = params.shape[0]
    half = max(n_paths // 2, 1)
    dt = mat / n_steps
    drift = (r - 0.5 * sig ** 2) * dt
    vol = sig * torch.sqrt(dt)

    z = torch.randn(b, half, n_steps, device=dev, dtype=dt_, generator=generator)
    z = torch.cat([z, -z], dim=1)                                 # antithetic
    cum_z = torch.cumsum(z, dim=2)                                # Gaussian driver
    log_paths = torch.log(m).unsqueeze(-1) + torch.cumsum(
        drift.unsqueeze(-1) + vol.unsqueeze(-1) * z, dim=2)
    del z

    paths = torch.exp(log_paths)
    arith = paths.mean(dim=2)                                     # A per path
    geo = torch.exp(log_paths.mean(dim=2))                        # G per path
    del log_paths
    disc = torch.exp(-r * mat)
    itm = arith > 1.0
    zero = torch.zeros((), device=dev, dtype=dt_)

    # -- pathwise differentials (S_i is linear in S0, so dA/dS0 = A/S0) --------
    delta = (disc * torch.where(itm, arith, zero) / m).mean(dim=1)

    t_grid = dt.unsqueeze(-1) * torch.arange(1, n_steps + 1, device=dev,
                                             dtype=dt_)
    dlnS_dsig = -sig.unsqueeze(-1) * t_grid \
        + torch.sqrt(dt).unsqueeze(-1) * cum_z
    dA_dsig = (paths * dlnS_dsig).mean(dim=2)
    del paths, dlnS_dsig, cum_z
    vega = (disc * torch.where(itm, dA_dsig, zero)).mean(dim=1)

    # -- control-variate price ------------------------------------------------
    x = disc * torch.clamp(arith - 1.0, min=0.0)
    y = disc * torch.clamp(geo - 1.0, min=0.0)
    ey = geometric_asian_call_torch(m, mat, sig, r, n_steps)       # (B,1)
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    beta = (xc * yc).mean(dim=1) / torch.clamp((yc * yc).mean(dim=1), min=1e-300)
    price = (x - beta.unsqueeze(1) * (y - ey)).mean(dim=1)
    return price, delta, vega


def generate_dataset_gpu(n_samples: int = 500_000, n_paths: int = 5_000,
                         n_steps: int = N_MONITORING_STEPS, seed: int = 7,
                         chunk_size: int = 256, device: torch.device | None = None,
                         verbose: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latin-hypercube parameters plus CV-MC labels and pathwise differentials.

    Returns (X, y, dydx) as float64 numpy arrays shaped (N,4), (N,), (N,2),
    matching dataset.generate_dataset so the two are drop-in interchangeable.
    chunk_size trades memory for throughput; each chunk holds several
    (chunk, paths, steps) float64 tensors, i.e. roughly
    chunk * paths * steps * 8 bytes * 4 in flight.
    """
    device = device or default_device()
    lows = np.array([lo for lo, _ in PARAM_RANGES.values()])
    highs = np.array([hi for _, hi in PARAM_RANGES.values()])
    X = lows + qmc.LatinHypercube(d=4, seed=seed).random(n_samples) * (highs - lows)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    Xt = torch.as_tensor(X, device=device, dtype=torch.float64)

    y = torch.empty(n_samples, device=device, dtype=torch.float64)
    dydx = torch.empty((n_samples, 2), device=device, dtype=torch.float64)

    t0 = time.perf_counter()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        price, delta, vega = simulate_chunk_gpu(Xt[start:end], n_paths,
                                                n_steps, gen)
        y[start:end] = price
        dydx[start:end, 0] = delta
        dydx[start:end, 1] = vega
        if verbose and (start // chunk_size) % 200 == 0:
            done = end / n_samples
            el = time.perf_counter() - t0
            print(f"  gpu dataset: {end:>8,}/{n_samples:,} ({done:5.1%})  "
                  f"elapsed {el:6.1f}s  eta {el/max(done,1e-9)*(1-done):6.1f}s",
                  flush=True)
    if verbose:
        steps = n_samples * n_paths * n_steps
        el = time.perf_counter() - t0
        print(f"  gpu dataset complete: {el:.1f}s  "
              f"({steps/el/1e9:.2f} G path-steps/s, float64)", flush=True)
    return (X, y.cpu().numpy(), dydx.cpu().numpy())
