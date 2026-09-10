"""Heston (1993) semi-analytic reference: characteristic function + COS pricing.

Why this module exists
----------------------
The project's short-dated engine is rough Bergomi, priced by Monte Carlo and by
a neural map of that Monte Carlo. Both need a classical benchmark that is
(a) exact to floating-point precision, (b) fast enough to calibrate in seconds
on a CPU, and (c) the textbook stochastic-volatility model whose *failure* on
the short-dated skew is the reason rough volatility exists. Heston is all
three. This module is numpy only - no torch - and is importable from anywhere
in the backend.

Model (risk-neutral, dividend yield q):

    dS_t / S_t = (r - q) dt + sqrt(v_t) dW_t
    dv_t       = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dZ_t,   d<W, Z> = rho dt

Characteristic function
-----------------------
`heston_cf` returns E[exp(i u ln(S_T / S_0))] in the numerically stable form of
Albrecher, Mayer, Schoutens & Tistaert (2007), "The little Heston trap": with
beta = kappa - rho sigma_v i u and d = sqrt(beta^2 + sigma_v^2 (u^2 + i u)) on
the principal branch, the exponent uses (beta - d) and g = (beta - d)/(beta + d),
which never crosses the branch cut of the complex log for any maturity (the
original Heston form with (beta + d) does, and produces discontinuities for
T beyond a few years).

One further rewrite, ours: the trap-free form still divides by sigma_v^2, so it
cannot be evaluated at or near the Black-Scholes limit sigma_v -> 0 (the
coefficient kappa theta / sigma_v^2 blows up while (beta - d) -> 0 by
cancellation). The identity

    beta - d = -sigma_v^2 (u^2 + i u) / (beta + d)

is exact and has no cancellation, so (beta - d)/sigma_v^2 and g/sigma_v^2 are
computed directly, and the log term is written with log1p so its sigma_v^2
factor also divides out analytically. The result is one formula that is
smooth all the way down to sigma_v = 0, where it reduces to the Black-Scholes
characteristic function with the deterministic variance integral
theta T + (v0 - theta)(1 - e^{-kappa T})/kappa. The BS-limit test in
tests/test_heston.py checks this to 1e-8.

COS pricing
-----------
`heston_call` / `heston_put` price European options by the COS method of Fang &
Oosterlee (2008): the density of the log return z = ln(S_T/S_0) is expanded in
cosines on a truncation range [a, b], and the payoff's cosine coefficients are
known in closed form, so

    price = e^{-rT} sum'_{k=0}^{N-1} Re{ phi(u_k) e^{-i u_k a} } V_k(K),
    u_k = k pi / (b - a),

with the prime meaning the k = 0 term is halved. The range is the standard
cumulant rule [a, b] = c1 -+ L sqrt(c2) (c4 = 0 for Heston, as in FO2008),
default L = 12, and the payoff coefficients V_k are vectorised over strikes:
phi is evaluated once per maturity and the (N x n_strikes) coefficient matrix
carries the strike dependence through the kink at ln(K / S_0). N = 2^10 is the
default. Measured against the reference value of FO2008 Table 3 (see the
tests): 8.0e-9 at their own L = 12 from N = 2^8 on, and 1.6e-8 once the range
is widened until the martingale identity holds to 1e-10 (below) - the residual
is the published reference's own truncation, not this pricer's.

The cumulant rule alone is not safe. For FO2008's own Feller-violating test
parameters at T = 1, L = 12 leaves 4e-7 of the exponential moment E[e^z]
outside the range, which shows up as a 4e-5 error in the put (the call is
insensitive to the left tail, so their call reference does not see it); for
the parameters a short-dated SPY calibration selects (sigma_v ~ 5, kappa ~ 100)
the tails are heavier still. `_cos_terms` therefore checks the expanded
density against E[e^z] = e^{mu T} - exactly the identity put-call parity
tests - and widens L by 1.5 (scaling N with it) until the defect is below
RANGE_TOL = 1e-10. `cos_range_report` shows what was used.

`heston_implied_vol` prices the OUT-OF-THE-MONEY instrument at each strike
(put below the forward, call above) and inverts it with a vectorised Black-76
bisection, which is the numerically well-conditioned way to get a smile:
inverting a deep in-the-money call recovers a tiny extrinsic value from a
large number and loses digits in exactly the wing the calibration cares about.

Calibration
-----------
`calibrate_heston` fits (v0, kappa, theta, sigma_v, rho) to a list of
`calibrate.Quote` by bounded least squares (scipy.optimize.least_squares, TRF)
on implied vols in vol points, from several starting points. Everything is
priced ON THE FORWARD of each expiry (S = F = fwd_pv e^{r tau}, r = q = 0,
undiscounted), the convention of backend/quant/surface.py, so model and market
share one forward by construction and the market's own Black-76 implied vol is
the target. Quotes whose model price falls below the Black-76 no-arbitrage
floor have no implied vol; they are scored by the vega-linearised price error
(P_model - P_mid) / vega, the same continuation the project's
calibrate.iv_fit_report uses, so the worst-fitting quotes are never dropped
from the metric.

The Feller condition 2 kappa theta > sigma_v^2 is REPORTED (`feller`), never
imposed: the little-trap characteristic function is valid either way, and a
short-dated index smile routinely wants it violated.

ATM skew
--------
`heston_atm_skew` returns psi(T) = d sigma_imp / dk at k = 0 by the same
five-point stencil convention as scripts/atm_skew_term_structure.py (central
difference at steps h and 2h, Richardson combination, |psi_h - psi_2h|/3 as
the truncation estimate), so Heston's term structure can be laid over that
document's market and rough-Bergomi curves. The classical short-maturity limit
`heston_short_skew_limit` = rho sigma_v / (4 sqrt(v0)) is the textbook result
(Gatheral 2006, ch. 3): finite as T -> 0, i.e. a log-log slope of ZERO, where
rough volatility predicts H - 1/2.

References
----------
Heston (1993), Rev. Financial Studies 6(2), 327-343.
Albrecher, Mayer, Schoutens & Tistaert (2007), "The little Heston trap",
    Wilmott Magazine, Jan 2007, 83-92.
Fang & Oosterlee (2008), "A novel pricing method for European options based
    on Fourier-cosine series expansions", SIAM J. Sci. Comput. 31(2), 826-848.
Lord, Koekkoek & van Dijk (2010), "A comparison of biased simulation schemes
    for stochastic volatility models", Quant. Finance 10(2), 177-194
    (full-truncation Euler, used only for the Monte Carlo cross-check).
Gatheral (2006), "The Volatility Surface", Wiley, ch. 3.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.stats import norm

__all__ = [
    "HESTON_BOUNDS", "HestonFit", "black76_implied_vol", "black76_price",
    "black76_vega", "calibrate_heston", "cos_range_report", "feller_ratio",
    "heston_atm_skew", "heston_call", "heston_cf", "heston_cumulants",
    "heston_implied_vol", "heston_mc_call", "heston_put",
    "heston_short_skew_limit", "heston_smile_residuals",
]

#: COS defaults. L = 12 is the value FO2008 used for their Heston reference;
#: N = 2^10 is far past convergence for every maturity in this project.
DEFAULT_N = 1024
DEFAULT_L = 12.0
#: Black-76 inversion bracket and bisection steps (vol units): 24 steps leave
#: a 3e-7 bracket, and the three Newton polishes that follow take it to
#: double precision.
IV_LO, IV_HI, IV_ITERS = 1e-4, 5.0, 24
#: Vega floor for the linearised continuation of the implied-vol error, the
#: same value calibrate.VEGA_FLOOR uses (forward vega here, undiscounted).
VEGA_FLOOR = 1e-4
#: Calibration bounds. kappa and sigma_v are judged for bound-pinning on a log
#: scale, v0 and theta in vol (sqrt) space, rho linearly - see HestonFit.
HESTON_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (1e-4, 1.0), "kappa": (1e-2, 100.0), "theta": (1e-4, 1.0),
    "sigma_v": (1e-2, 10.0), "rho": (-0.999, 0.5),
}
PARAM_NAMES = ("v0", "kappa", "theta", "sigma_v", "rho")
PIN_FRAC = 0.02
#: five-point strike stencil, in multiples of the step h (same layout as
#: scripts/atm_skew_term_structure.STENCIL)
STENCIL = (-2.0, -1.0, 0.0, 1.0, 2.0)


# ── characteristic function ──────────────────────────────────────────────
def _log1p_c(z: np.ndarray) -> np.ndarray:
    """log(1 + z) for complex z with full relative precision at tiny |z|.

    numpy's complex log1p is log(1 + z) evaluated naively: measured here,
    log1p(-1e-12 - 1e-13j) came back with a 2e-5 RELATIVE error, which
    destroyed the sigma_v -> 0 limit of the characteristic function. Below
    |z| = 1e-4 the four-term Taylor series is exact to 2e-17 relative; above
    it the classic corrected form (Higham 2002, sec. 1.14.1) is used, where
    the rounding error of forming w = 1 + z cancels in z * log(w) / (w - 1).
    (The corrected form alone is not enough: a denormal imaginary part of z
    survives w - 1 and overflows the division - observed at u ~ 1500 on the
    FO2008 parameters, where e^{-dT} ~ 1e-308.)
    """
    z = np.asarray(z, dtype=np.complex128)
    small = np.abs(z) < 1e-4
    zs = np.where(small, z, 0.0)
    series = zs * (1.0 - zs * (0.5 - zs * (1.0 / 3.0 - 0.25 * zs)))
    zb = np.where(small, 0.5, z)
    w = 1.0 + zb
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        higham = np.log(w) * zb / (w - 1.0)
    return np.where(small, series, higham)


def heston_cf(u: np.ndarray | float, T: float, v0: float, kappa: float,
              theta: float, sigma_v: float, rho: float, mu: float = 0.0
              ) -> np.ndarray:
    """E[exp(i u ln(S_T / S_0))] under Heston with drift mu = r - q.

    Little-Heston-trap form (Albrecher et al. 2007) rewritten so that nothing
    is divided by sigma_v^2: valid and smooth for sigma_v = 0 exactly (the
    Black-Scholes limit with the deterministic variance integral). `u` may be
    real or complex and of any shape.
    """
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    q2 = u * u + iu                                  # u^2 + i u
    beta = kappa - rho * sigma_v * iu
    d = np.sqrt(beta * beta + (sigma_v ** 2) * q2)   # principal branch
    bpd = beta + d                                   # Re > 0 always
    A_s2 = -q2 / bpd                                 # (beta - d) / sigma_v^2
    G = -q2 / (bpd * bpd)                            # g / sigma_v^2
    g = (sigma_v ** 2) * G
    E = np.exp(-d * T)
    # ln((1 - g E) / (1 - g)) / g  ->  (1 - E) as g -> 0; the precise log1p
    # keeps full relative precision when g is tiny.
    with np.errstate(divide="ignore", invalid="ignore"):
        L = _log1p_c(-g * E) - _log1p_c(-g)
        ratio = np.where(np.abs(g) > 0.0, L / np.where(g == 0, 1.0, g), 1.0 - E)
    C = kappa * theta * (A_s2 * T - 2.0 * G * ratio)
    D = v0 * A_s2 * (1.0 - E) / (1.0 - g * E)
    return np.exp(iu * mu * T + C + D)


def heston_cumulants(T: float, v0: float, kappa: float, theta: float,
                     sigma_v: float, rho: float, mu: float = 0.0
                     ) -> tuple[float, float]:
    """First two cumulants of ln(S_T / S_0).

    Derived from the CIR moments rather than copied: with I = int_0^T v dt and
    M = int_0^T sqrt(v) dW,  X = mu T - I/2 + M, so

        c1 = mu T - E[I]/2,
        c2 = Var[I]/4 + E[I] - E[I M],

    E[I] = theta T + (v0 - theta)(1 - e)/kappa,
    Var[I] = (sigma_v^2/kappa^2) { v0 [(1 - e^2)/kappa - 2 T e]
             + theta [T (1 + 2 e) - 2 (1 - e)/kappa - (1 - e^2)/(2 kappa)] },
    E[I M] = rho sigma_v { theta (T - (1 - e)/kappa)/kappa
             + (v0 - theta)(1 - e (1 + kappa T))/kappa^2 },   e = e^{-kappa T}.

    The version printed as FO2008 Table 11 that this module first carried had
    "theta (6 e - 7)" where the derivation gives "theta (4 e - 5)"; the
    numerical second derivative of ln phi at u = 0 sides with the derivation
    (tests/test_heston.py checks both cumulants against the cf to 1e-6).
    """
    e = math.exp(-kappa * T)
    s = sigma_v
    EI = theta * T + (v0 - theta) * (1.0 - e) / kappa
    varI = (s * s / kappa ** 2) * (
        v0 * ((1.0 - e * e) / kappa - 2.0 * T * e)
        + theta * (T * (1.0 + 2.0 * e) - 2.0 * (1.0 - e) / kappa
                   - (1.0 - e * e) / (2.0 * kappa)))
    EIM = rho * s * (theta * (T - (1.0 - e) / kappa) / kappa
                     + (v0 - theta) * (1.0 - e * (1.0 + kappa * T)) / kappa ** 2)
    c1 = mu * T - 0.5 * EI
    c2 = 0.25 * varI + EI - EIM
    return float(c1), float(c2)


def feller_ratio(kappa: float, theta: float, sigma_v: float) -> float:
    """2 kappa theta / sigma_v^2; > 1 means the variance cannot reach zero."""
    return 2.0 * kappa * theta / (sigma_v ** 2)


# ── COS pricing ───────────────────────────────────────────────────────────
#: Truncation self-check: the cosine expansion of the density on [a, b] must
#: reproduce E[e^z] = e^{mu T} (the martingale condition, which is exactly
#: what put-call parity measures) to this relative tolerance, else the range
#: is widened. FO2008's L = 12 passes it at every maturity this project
#: prices; it fails at T = 1 for their own Feller-violating test parameters
#: (measured defect 4.2e-7, i.e. a 4e-5 put error at K = 100), which is why
#: the check exists.
RANGE_TOL = 1e-10
RANGE_WIDEN = 1.5
RANGE_MAX_STEPS = 4
N_MAX = 1 << 14
#: the defect check conflates truncation with resolution: below this many
#: terms per unit of L the expansion cannot resolve the density and widening
#: the range would only make that worse, so the range is left alone
#: (explicit small-N convergence studies stay at their nominal L).
MIN_TERMS_PER_L = 16


def _cos_terms(T: float, v0: float, kappa: float, theta: float, sigma_v: float,
               rho: float, mu: float, N: int, L: float,
               tol: float = RANGE_TOL) -> dict[str, Any]:
    """Density side of the COS sum: range [a, b], frequencies u_k and the
    weights F_k = Re{phi(u_k) e^{-i u_k a}} (k = 0 halved), with the range
    widened by RANGE_WIDEN (N scaled with it, capped at N_MAX) until the
    expanded density's first exponential moment matches e^{mu T} to `tol`."""
    c1, c2 = heston_cumulants(T, v0, kappa, theta, sigma_v, rho, mu)
    s = math.sqrt(max(c2, 1e-16))
    L_eff, N_eff = float(L), int(N)
    for step in range(RANGE_MAX_STEPS + 1):
        a, b = c1 - L_eff * s, c1 + L_eff * s
        u = np.arange(N_eff, dtype=float) * (math.pi / (b - a))
        phi = heston_cf(u, T, v0, kappa, theta, sigma_v, rho, mu)
        Fk = np.real(phi * np.exp(-1j * u * a))
        Fk[0] *= 0.5
        # int_a^b e^z cos(u_k (z - a)) dz = ((-1)^k e^b - e^a) / (1 + u_k^2)
        sign = np.where(np.arange(N_eff) % 2 == 0, 1.0, -1.0)
        m1 = (2.0 / (b - a)) * float(Fk @ ((sign * math.exp(b) - math.exp(a)) / (1.0 + u * u)))
        defect = abs(m1 * math.exp(-mu * T) - 1.0)
        if (defect < tol or step == RANGE_MAX_STEPS
                or N_eff < MIN_TERMS_PER_L * L_eff):
            break
        L_eff *= RANGE_WIDEN
        N_eff = min(N_MAX, int(64 * math.ceil(N_eff * RANGE_WIDEN / 64)))
    return {"a": a, "b": b, "u": u, "Fk": Fk, "defect": defect,
            "L": L_eff, "N": N_eff, "c1": c1, "c2": c2}


def _payoff_coefficients(S: float, K: np.ndarray, a: float, b: float,
                         u: np.ndarray, kind: str) -> np.ndarray:
    """V_k(K) = (2/(b-a)) int_a^b payoff(z) cos(u_k (z - a)) dz, shape (N, nK).

    The kink z* = ln(K/S) is clipped to [a, b]: a call struck beyond b is
    worth 0, one struck below a is worth the whole truncated expectation.
    The trig terms at the fixed edge are (N,) vectors; only the kink's are
    (N x nK)."""
    z = np.clip(np.log(K / S), a, b)
    uu = u[:, None]
    ez = np.exp(z)[None, :]
    cz = np.cos(uu * (z - a)[None, :])
    sz = np.sin(uu * (z - a)[None, :])
    one_u2 = 1.0 / (1.0 + u * u)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_u = np.where(u > 0, 1.0 / np.where(u > 0, u, 1.0), 0.0)
    if kind == "call":
        # chi(z*, b) and psi(z*, b)
        cb = np.where(np.arange(u.size) % 2 == 0, 1.0, -1.0) * math.exp(b)  # cos(u_k(b-a)) e^b
        chi = (cb[:, None] - cz * ez - uu * sz * ez) * one_u2[:, None]
        psi = (0.0 - sz) * inv_u[:, None]           # sin(u_k (b-a)) = 0
        psi[0] = b - z
        V = S * chi - K[None, :] * psi
    elif kind == "put":
        # chi(a, z*) and psi(a, z*)
        chi = (cz * ez - math.exp(a) + uu * sz * ez) * one_u2[:, None]
        psi = sz * inv_u[:, None]
        psi[0] = z - a
        V = K[None, :] * psi - S * chi
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return V * (2.0 / (b - a))


def _cos_vanilla(S: float, K, T: float, r: float, q: float, v0: float,
                 kappa: float, theta: float, sigma_v: float, rho: float,
                 N: int, L: float, kind: str,
                 range_tol: float = RANGE_TOL) -> np.ndarray:
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if T <= 0.0:
        intrinsic = (S - K) if kind == "call" else (K - S)
        return np.maximum(intrinsic, 0.0)
    t = _cos_terms(T, v0, kappa, theta, sigma_v, rho, r - q, N, L, range_tol)
    V = _payoff_coefficients(S, K, t["a"], t["b"], t["u"], kind)
    return math.exp(-r * T) * (t["Fk"] @ V)


def _otm_forward_prices(F: float, K: np.ndarray, T: float, is_call: np.ndarray,
                        v0: float, kappa: float, theta: float, sigma_v: float,
                        rho: float, N: int, L: float,
                        range_tol: float = RANGE_TOL) -> np.ndarray:
    """Undiscounted OTM prices on the forward: one COS density evaluation
    serves both wings (the payoff coefficients differ, the density does not)."""
    if T <= 0.0:
        return np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    t = _cos_terms(T, v0, kappa, theta, sigma_v, rho, 0.0, N, L, range_tol)
    p = np.empty_like(K)
    if is_call.any():
        p[is_call] = t["Fk"] @ _payoff_coefficients(F, K[is_call], t["a"], t["b"], t["u"], "call")
    if (~is_call).any():
        p[~is_call] = t["Fk"] @ _payoff_coefficients(F, K[~is_call], t["a"], t["b"], t["u"], "put")
    return p


def cos_range_report(T: float, v0: float, kappa: float, theta: float,
                     sigma_v: float, rho: float, mu: float = 0.0,
                     N: int = DEFAULT_N, L: float = DEFAULT_L,
                     range_tol: float = RANGE_TOL) -> dict[str, float]:
    """What truncation range the pricer actually used for these inputs, and
    the martingale defect it achieved: {a, b, L, N, defect, c1, c2}."""
    t = _cos_terms(T, v0, kappa, theta, sigma_v, rho, mu, N, L, range_tol)
    return {k: float(t[k]) for k in ("a", "b", "L", "N", "defect", "c1", "c2")}


def heston_call(S: float, K, T: float, r: float, q: float, v0: float,
                kappa: float, theta: float, sigma_v: float, rho: float,
                N: int = DEFAULT_N, L: float = DEFAULT_L,
                via_parity: bool = False,
                range_tol: float = RANGE_TOL) -> np.ndarray:
    """European call prices under Heston by the COS method.

    Parameters
    ----------
    S : spot. K : strike, scalar or array (vectorised). T : years.
    r, q : continuously compounded rate and dividend yield.
    v0, kappa, theta, sigma_v, rho : Heston parameters (v0, theta are
        VARIANCES; sigma_v is the vol of variance).
    N : cosine terms (2^8 .. 2^12 are all converged for this project's use).
    L : truncation half-width in standard deviations, [a, b] = c1 -+ L sqrt(c2).
        The range is widened automatically (and N scaled with it) until the
        expanded density satisfies the martingale identity to `range_tol`;
        pass range_tol=inf to pin L for a convergence study.
    via_parity : price the put by COS and map it with put-call parity, which
        FO2008 recommend when the range is wide (long maturities); the direct
        call is the default and the two agree to ~1e-11 once the range check
        passes.

    Returns an array shaped like `K` (float64).
    """
    if via_parity:
        put = _cos_vanilla(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, N, L,
                           "put", range_tol)
        K = np.atleast_1d(np.asarray(K, dtype=float))
        return put + S * math.exp(-q * T) - K * math.exp(-r * T)
    return _cos_vanilla(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, N, L,
                        "call", range_tol)


def heston_put(S: float, K, T: float, r: float, q: float, v0: float,
               kappa: float, theta: float, sigma_v: float, rho: float,
               N: int = DEFAULT_N, L: float = DEFAULT_L,
               range_tol: float = RANGE_TOL) -> np.ndarray:
    """European put prices under Heston by the COS method (see heston_call)."""
    return _cos_vanilla(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, N, L,
                        "put", range_tol)


# ── Black-76 helpers (undiscounted, on the forward), vectorised ───────────
def _is_call(kind, shape) -> np.ndarray:
    if isinstance(kind, str):
        if kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
        return np.full(shape, kind == "call")
    return np.broadcast_to(np.asarray(kind, dtype=bool), shape)


def _black76_core(F, K, logFK, sq_T, sigma, sgn) -> np.ndarray:
    """Undiscounted Black-76 as sgn (F N(sgn d1) - K N(sgn d2)), sgn = +1 call,
    -1 put, with log(F/K) and sqrt(T) precomputed; ndtr avoids the
    argument-handling overhead of norm.cdf (this sits inside the bisection)."""
    sd = np.maximum(sigma, 1e-12) * sq_T
    d1 = (logFK + 0.5 * sd * sd) / sd
    return sgn * (F * ndtr(sgn * d1) - K * ndtr(sgn * (d1 - sd)))


def black76_price(F, K, T, sigma, kind) -> np.ndarray:
    """Undiscounted Black (1976) value; `kind` is 'call'/'put' or a boolean
    array (True = call). F, K, T, sigma all broadcast."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    shape = np.broadcast_shapes(F.shape, K.shape, T.shape, sigma.shape)
    sgn = np.where(_is_call(kind, shape), 1.0, -1.0)
    return _black76_core(F, K, np.log(F / K), np.sqrt(T), sigma, sgn)


def black76_vega(F, K, T, sigma) -> np.ndarray:
    """d price / d sigma (per 1.00 of vol), undiscounted; same for call and put."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sq_T = np.sqrt(np.asarray(T, dtype=float))
    sd = np.maximum(np.asarray(sigma, dtype=float), 1e-12) * sq_T
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return F * norm.pdf(d1) * sq_T


def black76_implied_vol(price, F, K, T, kind, lo: float = IV_LO,
                        hi: float = IV_HI, iters: int = IV_ITERS) -> np.ndarray:
    """Vectorised inversion of black76_price: `iters` bisection steps on
    [lo, hi] followed by three Newton polishes (vega > 0 on the bracket, and
    the bracket is 5 * 2^-iters wide, so Newton converges quadratically).

    NaN where the price lies outside the no-arbitrage range
    (intrinsic, F or K) or below the price at sigma = `lo`: such a quote has
    no implied vol, and the caller decides how to score it (see
    heston_smile_residuals). T may be an array (one maturity per quote).
    """
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    shape = np.broadcast_shapes(price.shape, F.shape, K.shape, T.shape)
    price, F, K, T = (np.broadcast_to(x, shape) for x in (price, F, K, T))
    is_call = _is_call(kind, shape)
    sgn = np.where(is_call, 1.0, -1.0)
    sq_T = np.sqrt(T)
    logFK = np.log(F / K)
    intrinsic = np.maximum(sgn * (F - K), 0.0)
    cap = np.where(is_call, F, K)
    p_lo = _black76_core(F, K, logFK, sq_T, lo, sgn)
    ok = (price > intrinsic) & (price < cap) & (price >= p_lo) & np.isfinite(price)
    a = np.full(shape, lo)
    b = np.full(shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        below = _black76_core(F, K, logFK, sq_T, m, sgn) < price
        a = np.where(below, m, a)
        b = np.where(below, b, m)
    s = 0.5 * (a + b)
    for _ in range(3):
        sd = s * sq_T
        d1 = (logFK + 0.5 * sd * sd) / sd
        vega = F * norm.pdf(d1) * sq_T
        step = (_black76_core(F, K, logFK, sq_T, s, sgn) - price) / np.where(vega > 0, vega, np.inf)
        s = np.clip(s - step, a, b)
    return np.where(ok, s, np.nan)


def heston_implied_vol(S: float, K, T: float, r: float, q: float, v0: float,
                       kappa: float, theta: float, sigma_v: float, rho: float,
                       N: int = DEFAULT_N, L: float = DEFAULT_L,
                       range_tol: float = RANGE_TOL) -> np.ndarray:
    """Black-Scholes implied vols of Heston prices, vectorised over strikes.

    Prices the out-of-the-money instrument at each strike on the forward
    F = S e^{(r - q) T} (put for K < F, call for K >= F, undiscounted) and
    inverts it with black76_implied_vol. This IS the Black-Scholes implied
    vol of the corresponding spot-space option (the discounting and the
    dividend cancel between price and inversion), computed where it is best
    conditioned. NaN where the model price sits below the no-arb floor
    (numerically zero far in a wing).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * math.exp((r - q) * T)
    is_call = K >= F
    prices = _otm_forward_prices(F, K, T, is_call, v0, kappa, theta, sigma_v,
                                 rho, N, L, range_tol)
    return black76_implied_vol(prices, F, K, T, is_call)


# ── ATM skew ──────────────────────────────────────────────────────────────
def heston_short_skew_limit(v0: float, sigma_v: float, rho: float) -> float:
    """lim_{T -> 0} d sigma_imp / dk at k = 0 = rho sigma_v / (4 sqrt(v0))."""
    return rho * sigma_v / (4.0 * math.sqrt(v0))


def heston_atm_skew(T: float, v0: float, kappa: float, theta: float,
                    sigma_v: float, rho: float, h: float | None = None,
                    h_scale: float = 0.25, h_floor: float = 0.002,
                    N: int = 4096, L: float = DEFAULT_L) -> dict[str, float]:
    """psi(T) = d sigma_imp / dk at k = ln(K/F) = 0 by the five-point stencil
    convention of scripts/atm_skew_term_structure.py.

    Step h defaults to max(h_scale sqrt(v0) sqrt(T), h_floor) - the same rule
    that document applied with its forward variance xi in place of v0 - and
    the strikes are F e^{k} for k in STENCIL * h on a unit forward (the skew
    is scale free). Returns psi (central difference at h), psi_2h, the
    Richardson combination (4 psi_h - psi_2h)/3, the truncation estimate
    |psi_h - psi_2h|/3, the ATM implied vol, and h.
    """
    if h is None:
        h = max(h_scale * math.sqrt(v0) * math.sqrt(T), h_floor)
    ks = np.asarray(STENCIL, dtype=float) * h
    ivs = heston_implied_vol(1.0, np.exp(ks), T, 0.0, 0.0, v0, kappa, theta,
                             sigma_v, rho, N=N, L=L)
    psi = float((ivs[3] - ivs[1]) / (2.0 * h))
    psi_2h = float((ivs[4] - ivs[0]) / (4.0 * h))
    return {"T": float(T), "h": float(h), "ivs": ivs.tolist(),
            "atm_iv": float(ivs[2]), "psi": psi, "psi_2h": psi_2h,
            "psi_richardson": (4.0 * psi - psi_2h) / 3.0,
            "truncation": abs(psi - psi_2h) / 3.0}


# ── Monte Carlo cross-check (full-truncation Euler) ───────────────────────
def heston_mc_call(S: float, K, T: float, r: float, q: float, v0: float,
                   kappa: float, theta: float, sigma_v: float, rho: float,
                   n_paths: int = 200_000, n_steps: int = 200,
                   seed: int | None = 0, chunk: int = 50_000
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Independent check of the COS prices: full-truncation Euler
    (Lord, Koekkoek & van Dijk 2010) - the variance may go negative in the
    state but enters every coefficient as max(v, 0). Returns (price, standard
    error), each shaped like K. Discretisation bias is O(dt); the tests use
    enough steps that it sits well inside the statistical error."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sq_dt = math.sqrt(dt)
    rho_perp = math.sqrt(1.0 - rho * rho)
    sums = np.zeros(K.shape)
    sums2 = np.zeros(K.shape)
    done = 0
    while done < n_paths:
        n = min(chunk, n_paths - done)
        x = np.full(n, math.log(S))
        v = np.full(n, v0)
        for _ in range(n_steps):
            zv = rng.standard_normal(n)
            zs = rho * zv + rho_perp * rng.standard_normal(n)
            vp = np.maximum(v, 0.0)
            x += (r - q - 0.5 * vp) * dt + np.sqrt(vp) * sq_dt * zs
            v += kappa * (theta - vp) * dt + sigma_v * np.sqrt(vp) * sq_dt * zv
        ST = np.exp(x)
        pay = np.maximum(ST[:, None] - K[None, :], 0.0)
        sums += pay.sum(axis=0)
        sums2 += (pay * pay).sum(axis=0)
        done += n
    disc = math.exp(-r * T)
    mean = sums / n_paths
    var = (sums2 / n_paths - mean * mean) * n_paths / (n_paths - 1)
    return disc * mean, disc * np.sqrt(var / n_paths)


# ── calibration to market quotes ──────────────────────────────────────────
def _groups(quotes: Sequence[Any], rate: float) -> list[dict[str, Any]]:
    """Per-expiry arrays from calibrate.Quote objects, priced on the forward."""
    by: dict[str, list[Any]] = {}
    for qt in quotes:
        by.setdefault(qt.expiry, []).append(qt)
    out = []
    for expiry, qs in sorted(by.items(), key=lambda kv: kv[1][0].tau):
        tau = float(np.median([qt.tau for qt in qs]))
        fwd_pv = float(np.median([qt.fwd_pv for qt in qs]))
        F = fwd_pv * math.exp(rate * tau)
        K = np.array([qt.strike for qt in qs], dtype=float)
        iv = np.array([qt.iv for qt in qs], dtype=float)
        hs = np.array([getattr(qt, "half_spread_iv", float("nan")) for qt in qs],
                      dtype=float)
        is_call = K >= F
        out.append({
            "expiry": expiry, "tau": tau, "F": F, "K": K, "iv": iv,
            "k": np.log(K / F), "is_call": is_call, "half_spread_iv": hs,
            "price_mkt": black76_price(F, K, tau, iv, is_call),
            "vega_fwd": np.maximum(black76_vega(F, K, tau, iv), VEGA_FLOOR),
            "n": len(qs),
        })
    return out


def heston_smile_residuals(groups: Sequence[dict[str, Any]], params: Sequence[float],
                           N: int = DEFAULT_N, L: float = DEFAULT_L
                           ) -> tuple[np.ndarray, np.ndarray, int]:
    """(residuals in vol points, model IVs, number unpriceable), concatenated
    over expiries in `groups` order (see _groups).

    Residual = 100 (iv_model - iv_mkt) where the model price inverts, else
    100 (P_model - P_mkt) / vega_mkt - the first-order continuation of the
    implied-vol error past the no-arb boundary (calibrate.iv_fit_report).
    Prices are computed per expiry (one COS density each); the inversion
    runs once over every quote."""
    v0, kappa, theta, sigma_v, rho = (float(p) for p in params)
    p = np.concatenate([
        _otm_forward_prices(g["F"], g["K"], g["tau"], g["is_call"], v0, kappa,
                            theta, sigma_v, rho, N, L) for g in groups])
    F = np.concatenate([np.full(g["n"], g["F"]) for g in groups])
    K = np.concatenate([g["K"] for g in groups])
    tau = np.concatenate([np.full(g["n"], g["tau"]) for g in groups])
    is_call = np.concatenate([g["is_call"] for g in groups])
    iv_mkt = np.concatenate([g["iv"] for g in groups])
    p_mkt = np.concatenate([g["price_mkt"] for g in groups])
    vega = np.concatenate([g["vega_fwd"] for g in groups])
    iv_m = black76_implied_vol(p, F, K, tau, is_call)
    bad = ~np.isfinite(iv_m)
    r = np.where(bad, (p - p_mkt) / vega, iv_m - iv_mkt)
    return 100.0 * r, iv_m, int(bad.sum())


@dataclass
class HestonFit:
    """Result of calibrate_heston. `params` are the fitted values, `se` the
    nominal standard errors from the Jacobian at the optimum (s^2 (J'J)^-1
    with s^2 = RSS / (n - p); they assume independent, homoscedastic
    residuals, which smile residuals are not - read them as a curvature
    scale, not a confidence interval)."""
    params: dict[str, float]
    se: dict[str, float]
    rmse_volpts: float                  # sqrt(mean r^2), all quotes, vol points
    rmse_weighted_volpts: float         # with the fit's weights (== rmse when uniform)
    n_quotes: int
    n_unpriceable: int
    per_expiry: list[dict[str, float]]
    seconds: float
    n_evals: int
    starts: list[dict[str, Any]]        # every start's x0, final cost, rmse
    pinned: list[str]
    feller: bool
    feller_ratio: float
    loss: str
    weights: str
    cost: float                         # scipy's 0.5 sum rho(f^2)
    residuals: np.ndarray = field(repr=False)
    model_ivs: np.ndarray = field(repr=False)
    N: int = DEFAULT_N
    L: float = DEFAULT_L
    fixed: dict[str, float] = field(default_factory=dict)   # held, not fitted

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("residuals", "model_ivs")}
        d["starts"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in s.items()} for s in self.starts]
        return d


def _pinned(x: Sequence[float], bounds: dict[str, tuple[float, float]],
            skip: Sequence[str] = ()) -> list[str]:
    out = []
    for name, val in zip(PARAM_NAMES, x):
        if name in skip:
            continue
        lo, hi = bounds[name]
        if name in ("kappa", "sigma_v"):
            f = lambda t: math.log(t)          # noqa: E731
        elif name in ("v0", "theta"):
            f = math.sqrt
        else:
            f = lambda t: t                    # noqa: E731
        tol = PIN_FRAC * (f(hi) - f(lo))
        if not f(lo) + tol < f(val) < f(hi) - tol:
            out.append(f"{name} = {val:.4g} is pinned at its bound [{lo:g}, {hi:g}]")
    return out


def _default_starts(groups: Sequence[dict[str, Any]], n_random: int, seed: int,
                    bounds: dict[str, tuple[float, float]]) -> list[np.ndarray]:
    """Six structured starts spanning slow/fast mean reversion and small/large
    vol-of-vol, seeded from the shortest and longest ATM vols, plus
    `n_random` uniform draws in (sqrt, log, sqrt, log, linear) space."""
    def atm_var(g):
        return float(g["iv"][np.argmin(np.abs(g["k"]))]) ** 2
    vs, vl = atm_var(groups[0]), atm_var(groups[-1])
    vm = 0.5 * (vs + vl)
    starts = [
        [vs, 2.0, vl, 0.5, -0.6],
        [vs, 10.0, vl, 1.5, -0.7],
        [vs, 30.0, vl, 3.0, -0.8],
        [vm, 0.5, vm, 0.3, -0.3],
        [vm, 5.0, vm, 1.0, -0.9],
        [vm, 60.0, vm, 5.0, -0.5],
    ]
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        x = []
        for name in PARAM_NAMES:
            lo, hi = bounds[name]
            if name in ("kappa", "sigma_v"):
                x.append(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            elif name in ("v0", "theta"):
                x.append(rng.uniform(math.sqrt(lo), math.sqrt(0.6)) ** 2)
            else:
                x.append(rng.uniform(lo, hi))
        starts.append(x)
    lo = np.array([bounds[n][0] for n in PARAM_NAMES])
    hi = np.array([bounds[n][1] for n in PARAM_NAMES])
    return [np.clip(np.asarray(s, dtype=float), lo * 1.01, hi * 0.99) for s in starts]


def calibrate_heston(quotes: Sequence[Any], rate: float, *,
                     loss: str = "linear", huber_delta_volpts: float = 2.0,
                     weights: str = "uniform", hs_floor_volpts: float = 0.05,
                     n_random_starts: int = 4, seed: int = 0,
                     N: int = DEFAULT_N, L: float = DEFAULT_L,
                     bounds: dict[str, tuple[float, float]] | None = None,
                     fixed: dict[str, float] | None = None,
                     max_nfev: int = 400) -> HestonFit:
    """Fit (v0, kappa, theta, sigma_v, rho) to `quotes` (calibrate.Quote).

    loss     'linear' - plain least squares on the vol-point residuals, so the
             number minimised IS the RMSE reported; 'huber' - scipy's Huber
             with f_scale = `huber_delta_volpts` (2 vp, the project's
             calibrate.HUBER_DELTA), the objective the rough-Bergomi map
             calibration in artifacts/intraday_params.json was fitted with.
    weights  'uniform', or 'half_spread' for 1 / max(half_spread_iv, floor)
             normalised to mean 1 (quotes the market itself resolves better
             count more). The reported rmse_volpts is always unweighted.
    fixed    parameters held at the given values and excluded from the fit,
             e.g. {"kappa": 3.0} to profile the smile RMSE against the
             mean-reversion speed. Held parameters get se = 0 and are never
             reported as pinned.
    Several starts (see _default_starts); the lowest final cost wins.
    """
    bounds = bounds or HESTON_BOUNDS
    fixed = {k: float(v) for k, v in (fixed or {}).items()}
    for name in fixed:
        if name not in PARAM_NAMES:
            raise ValueError(f"unknown Heston parameter {name!r} in fixed")
    free = [i for i, n in enumerate(PARAM_NAMES) if n not in fixed]
    if not free:
        raise ValueError("every parameter is fixed; nothing to fit")
    groups = _groups(quotes, rate)
    if not groups or sum(g["n"] for g in groups) < 8:
        raise ValueError("need at least 8 quotes to calibrate Heston")
    if weights == "uniform":
        w = np.ones(sum(g["n"] for g in groups))
    elif weights == "half_spread":
        hs = np.concatenate([g["half_spread_iv"] for g in groups])
        hs = np.where(np.isfinite(hs) & (hs > 0), hs, np.nanmedian(hs))
        w = 1.0 / np.maximum(hs, hs_floor_volpts)
        w /= w.mean()
    else:
        raise ValueError("weights must be 'uniform' or 'half_spread'")
    sqrt_w = np.sqrt(w)
    n_evals = 0
    template = np.array([fixed.get(n, np.nan) for n in PARAM_NAMES])

    def full(xf: np.ndarray) -> np.ndarray:
        x = template.copy()
        x[free] = xf
        return x

    def fun(xf: np.ndarray) -> np.ndarray:
        nonlocal n_evals
        n_evals += 1
        r, _, _ = heston_smile_residuals(groups, full(xf), N, L)
        return sqrt_w * r

    lo = np.array([bounds[n][0] for n in PARAM_NAMES])[free]
    hi = np.array([bounds[n][1] for n in PARAM_NAMES])[free]
    x_scale = np.array([0.01, 10.0, 0.01, 1.0, 0.3])[free]
    t0 = time.perf_counter()
    best = None
    starts = []
    for x0_full in _default_starts(groups, n_random_starts, seed, bounds):
        x0 = x0_full[free]
        try:
            res = least_squares(
                fun, x0, bounds=(lo, hi), method="trf", loss=loss,
                f_scale=huber_delta_volpts, x_scale=x_scale,
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev)
        except Exception as exc:                      # noqa: BLE001
            starts.append({"x0": full(x0), "cost": float("nan"), "error": repr(exc)})
            continue
        r_plain, _, _ = heston_smile_residuals(groups, full(res.x), N, L)
        starts.append({"x0": full(x0), "x": full(res.x), "cost": float(res.cost),
                       "rmse_volpts": float(np.sqrt(np.mean(r_plain ** 2))),
                       "nfev": int(res.nfev), "status": int(res.status)})
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("every Heston start failed: " + repr(starts))
    seconds = time.perf_counter() - t0

    x = full(best.x)
    r_plain, ivs, n_unpr = heston_smile_residuals(groups, x, N, L)
    n = r_plain.size
    # nominal covariance from a fresh finite-difference Jacobian of the
    # unweighted residuals at the optimum, over the FREE parameters
    J = np.empty((n, len(free)))
    for c, j in enumerate(free):
        step = 1e-5 * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        J[:, c] = (heston_smile_residuals(groups, xp, N, L)[0]
                   - heston_smile_residuals(groups, xm, N, L)[0]) / (2 * step)
    s2 = float(np.sum(r_plain ** 2) / max(n - len(free), 1))
    se = np.zeros(len(x))
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
        se[free] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se[free] = float("nan")

    per_expiry = []
    off = 0
    for g in groups:
        rr = r_plain[off:off + g["n"]]
        body = np.abs(g["k"]) <= 2.0 * float(g["iv"][np.argmin(np.abs(g["k"]))]) * math.sqrt(g["tau"])
        per_expiry.append({
            "expiry": g["expiry"], "tau": g["tau"], "n": int(g["n"]),
            "rmse_volpts": float(np.sqrt(np.mean(rr ** 2))),
            "rmse_body_volpts": float(np.sqrt(np.mean(rr[body] ** 2))) if body.any() else float("nan"),
            "n_body": int(body.sum()),
            "bias_volpts": float(np.mean(rr)),
            "max_abs_volpts": float(np.max(np.abs(rr))),
        })
        off += g["n"]
    params = dict(zip(PARAM_NAMES, (float(v) for v in x)))
    return HestonFit(
        params=params, se=dict(zip(PARAM_NAMES, (float(v) for v in se))),
        rmse_volpts=float(np.sqrt(np.mean(r_plain ** 2))),
        rmse_weighted_volpts=float(np.sqrt(np.sum(w * r_plain ** 2) / np.sum(w))),
        n_quotes=int(n), n_unpriceable=int(n_unpr), per_expiry=per_expiry,
        seconds=seconds, n_evals=n_evals, starts=starts,
        pinned=_pinned(x, bounds, skip=tuple(fixed)),
        feller=feller_ratio(params["kappa"], params["theta"], params["sigma_v"]) > 1.0,
        feller_ratio=feller_ratio(params["kappa"], params["theta"], params["sigma_v"]),
        loss=loss, weights=weights, cost=float(best.cost),
        residuals=r_plain, model_ivs=ivs, N=N, L=L, fixed=fixed)
