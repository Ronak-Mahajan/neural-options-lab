"""Rough Bergomi Monte Carlo engine for short-dated options.

Implements the Riemann-Liouville (Volterra) driver of Bayer, Friz & Gatheral
(2016) and a PyTorch Monte Carlo pricer, used to generate ground truth for the
0DTE neural surrogate.

The kernel this module used to use was the wrong one
----------------------------------------------------
`generate_fbm_covariance` built

    C_ij = 0.5 * (t_i^2H + t_j^2H - |t_i - t_j|^2H)

which is the Type-I (Mandelbrot-Van Ness) fractional Brownian motion
covariance. Rough Bergomi is not driven by Type-I fBm. It is driven by the
Riemann-Liouville Volterra process

    W~_t = sqrt(2H) * int_0^t (t - s)^{H - 1/2} dW_s,
    E[W~_u W~_v] = 2H * int_0^{u ^ v} (u - s)^{H-1/2} (v - s)^{H-1/2} ds.

The two agree on the DIAGONAL - both give Var[W~_t] = t^{2H}, so the
-0.5 eta^2 t^{2H} drift correction and the martingale property were unaffected,
which is why this survived undetected. They agree nowhere else. Measured at
H = 0.1172, n_steps = 50: maximum off-diagonal relative difference 4.93,
corr(W~_t1, W~_t50) = +0.320 under the old kernel versus +0.054 under the true
Volterra process.

There is a second, subtler consequence. The discretised Volterra process is
W~_i = sum_{j<=i} K_ij dW_j with K lower triangular and positive on the
diagonal, so C = K K^T and - by uniqueness of the Cholesky factorisation -
K IS chol(C) and the Gaussian vector feeding it IS the driving Brownian
increment. That makes the existing leverage construction
Z_spot = rho Z_vol + sqrt(1 - rho^2) Z_indep exactly right, but ONLY once C is
the Volterra covariance. Under the old Type-I matrix, chol(C) was not the
Volterra kernel and Z_vol was not dW, so the correlation was being applied to
the wrong object: corr(Z_vol_1, W~_t1) was +1.0000 where the truth is +0.7844.
Fixing the covariance therefore fixes the leverage too, with no change to the
simulation code below.

Closed form used here
---------------------
For u <= v, substituting s = u x and applying Euler's integral representation,

    E[W~_u W~_v] = (2H / (H + 1/2)) * u^{H+1/2} * v^{H-1/2}
                   * 2F1(1/2 - H, 1; H + 3/2; u/v)

with u/v in [0, 1], so the hypergeometric series is evaluated inside its disc of
convergence everywhere, including the diagonal (where Gauss's theorem gives
2F1(...; 1) = (H + 1/2) / (2H) and the expression collapses to t^{2H}).

The covariance is homogeneous of degree 2H, so it is computed once on the unit
grid t_i = i and rescaled by dt^{2H}. That makes it cacheable across batch
elements, which also removes the redundant per-element refactorisation.

CONSEQUENCE FOR THE SHIPPED ARTIFACT: artifacts/model_0dte.pt was trained
against the OLD kernel, so it is a surrogate for a process that is not rough
Bergomi. It must be regenerated (dataset_0dte.py, then train_0dte.py) before its
prices mean what the README says they mean.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from scipy.special import hyp2f1

# Paths simulated per block inside rough_bergomi_mc. At 25,000 the per-block
# working set is ~25 MB of float32 tensors regardless of the requested
# n_paths; see the block comment inside rough_bergomi_mc for why this exists.
_CHUNK_PATHS = 25_000


@lru_cache(maxsize=64)
def _volterra_covariance_unit(n_steps: int, H: float) -> tuple:
    """E[W~_i W~_j] on the unit grid t_i = i, i = 1..n_steps.

    Returned as a nested tuple so lru_cache can hold it; callers convert to a
    tensor. Homogeneity of degree 2H means the physical covariance is this
    matrix times dt^{2H}.
    """
    i = np.arange(1, n_steps + 1, dtype=np.float64)
    u = np.minimum(i[:, None], i[None, :])          # min(t_i, t_j)
    v = np.maximum(i[:, None], i[None, :])          # max(t_i, t_j)
    pref = 2.0 * H / (H + 0.5)
    c = pref * u ** (H + 0.5) * v ** (H - 0.5) \
        * hyp2f1(0.5 - H, 1.0, H + 1.5, u / v)
    return tuple(map(tuple, c))


def volterra_covariance(n_steps: int, dt: float, H: float,
                        device: torch.device) -> torch.Tensor:
    """Exact covariance of the Riemann-Liouville driver at t_i = i*dt."""
    c = np.asarray(_volterra_covariance_unit(n_steps, float(H)),
                   dtype=np.float64) * (dt ** (2.0 * H))
    return torch.as_tensor(c, dtype=torch.float32, device=device)


@lru_cache(maxsize=64)
def _joint_factor_unit(n_steps: int, H: float) -> tuple:
    """Cholesky factor of the JOINT law of (dW / sqrt(dt), W~ / dt^H).

    Why a joint covariance rather than just factorising C
    ----------------------------------------------------
    W~_{t_i} = sqrt(2H) int_0^{t_i} (t_i - s)^{H-1/2} dW_s is a continuous
    stochastic integral. It is NOT a linear function of the n coarse increments
    dW_1..dW_n - there is residual randomness inside each step - so C is not
    K K^T for any n x n K built from those increments, and therefore chol(C) is
    NOT the Volterra kernel and its Gaussian input is NOT dW.

    Concretely, factorising C alone forces corr(Z_1, W~_{t_1}) = 1 by
    construction (the first row of a lower-triangular factor has one entry),
    whereas the true value is

        corr(dW_1, W~_{t_1}) = sqrt(2H) / (H + 1/2)

    = 0.7844 at H = 0.1172. Applying the leverage correlation rho to the wrong
    object is the second half of the kernel bug: it makes the effective
    spot/vol correlation too strong at the short end, which is exactly where a
    0DTE skew fit is identified.

    The fix is the standard exact scheme: simulate the 2n-dimensional Gaussian
    (dW_1..dW_n, W~_{t_1}..W~_{t_n}) directly, using the exact cross-covariance

        E[dW_j W~_{t_i}] = sqrt(2H)/(H+1/2)
                           * [ (t_i - t_{j-1})^{H+1/2} - (t_i - t_j)^{H+1/2} ]
                                                                    for t_j <= t_i
                         = 0                                        otherwise.

    Both blocks are dimensionless once dW is scaled by sqrt(dt) and W~ by dt^H,
    so the factor depends only on (n_steps, H) and is cached.
    """
    n = n_steps
    i = np.arange(1, n + 1, dtype=np.float64)
    c = np.asarray(_volterra_covariance_unit(n, H), dtype=np.float64)

    # D[i, j] = E[ (dW_j/sqrt(dt)) * (W~_{t_i}/dt^H) ] on the unit grid.
    lag_hi = i[:, None] - (i[None, :] - 1.0)        # t_i - t_{j-1}
    lag_lo = i[:, None] - i[None, :]                # t_i - t_j
    mask = lag_lo >= 0.0                            # only j <= i contributes
    d = np.zeros((n, n))
    d[mask] = (math.sqrt(2.0 * H) / (H + 0.5)) * (
        np.power(lag_hi[mask], H + 0.5) - np.power(lag_lo[mask], H + 0.5))

    sigma = np.empty((2 * n, 2 * n))
    sigma[:n, :n] = np.eye(n)
    sigma[n:, :n] = d
    sigma[:n, n:] = d.T
    sigma[n:, n:] = c
    # W~ is very nearly determined by the increments on a fine grid, so the
    # joint law is close to singular; jitter only the diagonal.
    sigma[np.diag_indices(2 * n)] += 1e-10
    return tuple(map(tuple, np.linalg.cholesky(sigma)))


def joint_factor(n_steps: int, dt: float, H: float,
                 device: torch.device) -> torch.Tensor:
    """Lower-triangular factor L with (Z, W~_scaled) = L @ noise, shape (2n, 2n)."""
    lf = np.asarray(_joint_factor_unit(n_steps, float(H)), dtype=np.float64)
    return torch.as_tensor(lf, dtype=torch.float32, device=device)


def generate_fbm_covariance(*args, **kwargs):  # pragma: no cover
    """Removed: this built the Type-I fBm covariance, which is the wrong driver.

    Kept as a loud failure rather than deleted so that any caller still relying
    on it stops instead of silently simulating the wrong process.
    """
    raise NotImplementedError(
        "generate_fbm_covariance built the Mandelbrot-Van Ness fBm covariance, "
        "which is not the rough Bergomi driver. Use volterra_covariance(); see "
        "the module docstring.")

def rough_bergomi_mc(spot: torch.Tensor, strike: torch.Tensor, maturity: torch.Tensor,
                     xi: torch.Tensor, eta: torch.Tensor, rho: torch.Tensor, rate: torch.Tensor,
                     n_paths: int = 50000, n_steps: int = 50, H: float = 0.1,
                     seed: int | None = None,
                     jumps: tuple[float, float, float] | None = None,
                     return_std_error: bool = False):
    """Prices European Call options using the Rough Bergomi model.
    
    Args:
        spot, strike, maturity: (B,) tensors for standard option parameters
        xi: (B,) initial forward variance (similar to sigma^2)
        eta: (B,) volatility of volatility
        rho: (B,) correlation between spot and variance
        rate: (B,) risk-free rate
        n_paths: number of MC paths per contract
        n_steps: number of time steps (high resolution for intraday)
        H: Hurst parameter (H < 0.5 is rough, typically ~0.1 in markets)
        jumps: optional (lam, mu_j, sig_j) adding lognormal (Merton) jumps to
            the spot: N_T ~ Poisson(lam*T) jumps over the horizon, each with
            log-size ~ Normal(mu_j, sig_j^2), drift-compensated by
            -lam*(e^{mu_j+sig_j^2/2}-1)*T so E[S_T] is unchanged EXACTLY.
            None (the default) draws nothing extra, so seeded results are
            bit-identical to the pure-diffusion model - and because the jump
            draws happen after the diffusion draws, two calls with the same
            seed and different jump parameters share the SAME diffusion
            sample (common random numbers across the jump axis).

            Why jumps at all: measured on live SPY (fit_diagnostics), the
            diffusive model sits ~2.1 vol points BELOW the market at 2-3
            sigma into the put wing at EVERY maturity. At 2-11 days a
            continuous-path model cannot make the left tail fat enough,
            because crash risk does not scale with sqrt(tau). Jumps are the
            standard mechanism, and only S_T matters for a European payoff,
            so their placement within the horizon is irrelevant - one
            compensated Poisson draw per path is exact, not a scheme.
        
    Returns:
        prices: (B,) tensor of European call prices
    """
    B = spot.shape[0]
    device = spot.device
    prices = torch.zeros(B, device=device)
    std_errors = torch.zeros(B, device=device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    # Group by the parameters that determine the TERMINAL DISTRIBUTION. Strike
    # is not one of them: every contract sharing (spot, T, xi, eta, rho, rate)
    # is priced against the same S_T and differs only in where the payoff is
    # struck. This used to loop per contract, drawing an independent path set
    # for every strike on a smile.
    #
    # Two consequences, both bad. Cost: a calibration objective over 439 quotes
    # in 6 expiries ran 439 simulations where 6 suffice, which measured at
    # 15.82 s per objective evaluation and 2,788 s for a single live SPY fit.
    # Accuracy: independent draws per strike inject noise into the SMILE SHAPE,
    # and shape is exactly what identifies rho and eta. Sharing one sample
    # across a smile makes it monotone in the strike by construction, which is
    # the common-random-numbers argument applied where it actually matters.
    groups: dict[tuple, list[int]] = {}
    for b in range(B):
        key = (float(spot[b]), float(maturity[b]), float(xi[b]),
               float(eta[b]), float(rho[b]), float(rate[b]))
        groups.setdefault(key, []).append(b)

    for (spot_g, T, xi_g, eta_g, rho_g, rate_g), members in groups.items():
        dt = T / n_steps

        # Joint exact simulation of the driving increments and the Volterra
        # process. The factor is cached on (n_steps, H) and dimensionless, so
        # the 100x100 factorisation is built once rather than per batch element.
        Lj = joint_factor(n_steps, dt, H, device)
        t = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device) * dt
        t_2H = t ** (2 * H)
        eta_b = eta_g

        # Paths are simulated in blocks of _CHUNK_PATHS; each path depends only
        # on its own rows of noise, so blocks are exact, and only the 1-D S_T
        # survives a block. The one-shot version held ~2,200 bytes/path of
        # (n_paths, n_steps)-shaped tensors simultaneously (~880 MB at 400k
        # paths), which OOM-killed the 512 MB container serving this. Jump
        # draws stay AFTER all diffusion draws, preserving the documented
        # common-random-numbers property across the jump axis. For
        # n_paths <= _CHUNK_PATHS the draw order matches the old code exactly;
        # above it the seed stream is laid out per block, so large seeded runs
        # are deterministic but not bit-identical to the previous version.
        def _terminal_spot_chunk(n_c: int) -> torch.Tensor:
            noise = torch.randn(n_c, 2 * n_steps, device=device, generator=gen)
            joint = noise @ Lj.T
            del noise
            # .clone() unpins joint's full 2n-wide buffer (a slice is a view).
            Z_vol = joint[:, :n_steps].clone()           # dW / sqrt(dt), the
                                                         # actual driving BM
            W_tilde = joint[:, n_steps:] * (dt ** H)     # W~_{t_i}
            del joint

            # Volatility process (Rough Bergomi)
            V = xi_g * torch.exp(eta_b * W_tilde - 0.5 * (eta_b ** 2) * t_2H)
            del W_tilde

            # Left-point (predictable) variance: the variance applied over
            # step i must not contain step i's own innovation. Using the
            # right-endpoint V_i against a spot shock built from the same
            # normal breaks the martingale property whenever rho != 0 (the
            # -V/2 dt correction no longer offsets E[exp(sqrt(V) dW)]), which
            # collapses prices at negative correlation. This is the discrete
            # analogue of the Ito integral being left-point by construction.
            V = torch.cat([torch.full((n_c, 1), xi_g, device=device),
                           V[:, :-1]], dim=1)

            # Standard driving normals for the spot process (correlated with
            # Z_vol)
            Z_indep = torch.randn(n_c, n_steps, device=device, generator=gen)
            Z_spot = rho_g * Z_vol + math.sqrt(1 - rho_g ** 2) * Z_indep
            del Z_vol, Z_indep
            dW_spot = Z_spot * math.sqrt(dt)
            del Z_spot

            # Euler scheme for the log spot process
            integral_drift = torch.sum((rate_g - 0.5 * V) * dt, dim=1)
            integral_vol = torch.sum(torch.sqrt(V) * dW_spot, dim=1)
            return spot_g * torch.exp(integral_drift + integral_vol)

        st_chunks = []
        remaining = n_paths
        while remaining > 0:
            n_c = min(_CHUNK_PATHS, remaining)
            st_chunks.append(_terminal_spot_chunk(n_c))
            remaining -= n_c
        S_T = st_chunks[0] if len(st_chunks) == 1 else torch.cat(st_chunks)
        del st_chunks
        if jumps is not None:
            lam, mu_j, sig_j = (float(v) for v in jumps)
            if lam > 0.0:
                n_jumps = torch.poisson(
                    torch.full((n_paths,), lam * T, device=device),
                    generator=gen)
                eps_j = torch.randn(n_paths, device=device, generator=gen)
                total_j = n_jumps * mu_j + torch.sqrt(n_jumps) * sig_j * eps_j
                kappa_bar = math.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
                S_T = S_T * torch.exp(total_j - lam * kappa_bar * T)
        disc = math.exp(-rate_g * T)

        # Every strike in this group is evaluated against the SAME S_T sample.
        for b in members:
            payoff = torch.clamp(S_T - strike[b], min=0.0)
            prices[b] = payoff.mean() * disc
            std_errors[b] = payoff.std() * disc / math.sqrt(n_paths)

    if return_std_error:
        return prices, std_errors
    return prices

if __name__ == "__main__":
    # Quick sanity check
    spot = torch.tensor([100.0])
    strike = torch.tensor([100.0])
    maturity = torch.tensor([5.0 / 252.0]) # 5 Days to expiry
    xi = torch.tensor([0.25 ** 2])         # 25% initial vol
    eta = torch.tensor([1.5])              # high vol of vol
    rho = torch.tensor([-0.7])             # strong negative correlation (leverage effect)
    rate = torch.tensor([0.05])
    
    import time
    t0 = time.perf_counter()
    price = rough_bergomi_mc(spot, strike, maturity, xi, eta, rho, rate, n_paths=100000, n_steps=50)
    print(f"0DTE Rough Vol Price: ${price[0].item():.4f} (computed in {time.perf_counter() - t0:.2f}s)")
