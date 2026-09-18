"""Static-arbitrage audit of the served Asian pricer, in price space.

What this script does
---------------------
`artifacts/model.pt` is the 5-member ensemble `PricingEngine` routes every
maturity above 12/252 to: it outputs the discretely monitored **arithmetic
average** Asian call price per unit strike as a function of (m = S/K, T, sigma,
r) over its trained box (`backend.quant.dataset.PARAM_RANGES`, 50 monitoring
dates).  Nothing in its training constrains that surface to be free of static
arbitrage.  This script measures how far from arbitrage-free it is, on a dense
lattice over the trained box, for several (sigma, r) slices.

Everything is done in PRICE space.  The 0DTE audit
(`scripts/no_arbitrage_surface.py`) inverts prices to Black-Scholes implied
vols and evaluates Durrleman's g; neither step transfers to an arithmetic
average, whose law is not lognormal and whose Black-Scholes implied vol is not
a parameter of anything.  The conditions checked here are the ones that survive
the change of contract:

    dC/dK in [-e^{-rT}, 0]        strike monotonicity and its slope bound
    d2C/dK2 >= 0                  non-negative risk-neutral density of A,
                                  plus a discrete 1%-wide butterfly
    C >= e^{-rT} (E[A] - K)+      the ASIAN floor (Jensen), not (S - K)+
    C <= e^{-rT} E[A] <= S        upper bounds
    C >= 0                        positivity
    gamma = d2C/dS2 >= 0          convexity in spot
    vega  = dC/dsigma >= 0        monotonicity in volatility
    delta = dC/dS in [0, e^{-rT} E[A]/S]

E[A] is the engine's own: fixings t_i = i T/n for i = 1..n with n = 50, so
E[A] = (S/n) sum_i e^{r t_i}, the geometric series
`_parity_adjustment_torch` and `monte_carlo.expected_arithmetic_average` both
use.  `expected_average()` below is the vectorised form of exactly that
expression and `tests/test_asian_audit.py` pins it against all three.

Deliberately NOT checked, because they are not no-arbitrage conditions for this
contract: Black-Scholes implied-vol inversion and the Durrleman function (the
average is not lognormal); the European intrinsic floor max(S - K e^{-rT}, 0)
(strictly above the Asian floor, so it manufactures violations); and calendar
monotonicity dC/dT >= 0 (the averaging window moves with T, so a longer-dated
contract is a different average, not the same one held longer).

Reference.  Curran (1994) (`backend.quant.asian_approx.curran_call`) prices the
same contract to 0.104 bps mean / 0.345 bps max error against a 400,000-path
Monte Carlo on the benchmark grid (`docs/approximation_benchmark.md`).  It is
evaluated on the whole lattice as an independent arbiter and checked against
fresh Monte Carlo at the box corners, at every worst-violation location and at
its own worst disagreement with the network.

Resolution.  A shape statistic is only informative where the contract has more
time value than the surrogate has price error.  Mirroring the vega floor of the
0DTE audit, every statistic is reported over the full box AND over the region
where the reference's own vega per unit strike per unit vol - a central
difference of Curran at sigma +- 1e-3, in which Curran's level bias cancels to
first order - is at least `VEGA_FLOOR` = 0.02.  At that floor the ensemble's
measured 1.33 bps price RMSE (`artifacts/eval.json`, 600 held-out contracts
against 200,000-path references) is worth 0.7 vol points, and below it the
contract's whole sensitivity to volatility is smaller than the error bar on its
price.

Outputs: docs/asian_arbitrage_audit.json (the markdown is written from it).

    python -m scripts.asian_arbitrage_audit              # full run
    python -m scripts.asian_arbitrage_audit --quick      # smoke run to a temp dir
    python -m scripts.asian_arbitrage_audit --no-mc      # skip the Monte Carlo arbiter
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.asian_approx import curran_call  # noqa: E402
from backend.quant.dataset import PARAM_RANGES, N_MONITORING_STEPS  # noqa: E402
from backend.quant.engine import PricingEngine, ZERO_DTE_CUTOFF  # noqa: E402
from backend.quant.monte_carlo import price_asian_mc  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"

#: Reference vega (per unit strike, per unit vol) below which the ensemble's
#: own price error dominates the contract's sensitivity.  Same number and same
#: meaning as VEGA_FLOOR in backend/quant/iv_surface.py.
VEGA_FLOOR = 0.02

#: Central-difference step used for the reference vega.
VEGA_STEP = 1e-3

#: Half-width of the discrete butterfly, as a fraction of the strike.
BUTTERFLY_WIDTH = 0.01

#: Lattice defaults.  Uniform in m and T over the trained box.
N_M = 151                                   # m = S/K, step 0.01
N_T = 40                                    # T, step 0.05 years
SIGMAS = (0.05, 0.10, 0.20, 0.40, 0.80)
RATES = (0.0, 0.02, 0.04, 0.10)

_CHUNK = 8_192


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
#  The contract: E[A], the Asian floor, and the geometric companion
# --------------------------------------------------------------------------- #

def expected_average(spot: np.ndarray | float, maturity: np.ndarray | float,
                     rate: np.ndarray | float,
                     n_steps: int = N_MONITORING_STEPS) -> np.ndarray:
    """E[A] of the discrete arithmetic average, vectorised over the lattice.

    Fixings are t_i = i T / n for i = 1..n, so

        E[A] = (S / n) sum_{i=1..n} e^{r t_i}
             = S e^{r dt} (e^{rT} - 1) / (n (e^{r dt} - 1)),   dt = T / n,

    which is the expression `engine._parity_adjustment_torch` and
    `monte_carlo.expected_arithmetic_average` both evaluate.  The r -> 0 limit
    of that series is exactly S (every fixing has forward S), and is taken here
    where r dt underflows the closed form, so the audit evaluates the same
    quantity the serving path does at every rate including zero.
    """
    spot = np.asarray(spot, dtype=np.float64)
    maturity = np.asarray(maturity, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    dt = maturity / n_steps
    small = np.abs(rate * dt) < 1e-12
    denom = np.where(small, 1.0, np.expm1(rate * dt))
    series = np.where(small, 1.0,
                      np.exp(rate * dt) * np.expm1(rate * maturity)
                      / (n_steps * denom))
    return spot * series


def asian_floor(spot, strike, maturity, rate,
                n_steps: int = N_MONITORING_STEPS) -> np.ndarray:
    """e^{-rT} (E[A] - K)+ : the lower bound Jensen gives for the ARITHMETIC
    average call.  The European intrinsic max(S - K e^{-rT}, 0) is strictly
    larger whenever r > 0 and is not a bound on this contract."""
    ea = expected_average(spot, maturity, rate, n_steps)
    strike = np.asarray(strike, dtype=np.float64)
    maturity = np.asarray(maturity, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    return np.exp(-rate * maturity) * np.maximum(ea - strike, 0.0)


def curran_batch(m: np.ndarray, T: np.ndarray, sigma: np.ndarray,
                 rate: np.ndarray, n_steps: int = N_MONITORING_STEPS
                 ) -> np.ndarray:
    """Curran (1994) call price per unit strike, point by point (spot = m, K = 1)."""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    rate = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    out = np.empty(m.size, dtype=np.float64)
    for i in range(m.size):
        out[i] = curran_call(float(m[i]), 1.0, float(T[i]), float(sigma[i]),
                             float(rate[i]), n_steps)
    return out


def curran_vega(m, T, sigma, rate, n_steps: int = N_MONITORING_STEPS,
                h: float = VEGA_STEP) -> np.ndarray:
    """dC/dsigma of the ARITHMETIC contract, per unit strike per unit vol.

    A central difference of Curran at sigma +- h.  Curran's own level error is
    a smooth function of sigma (0.104 bps mean, 0.345 bps max against a
    400,000-path Monte Carlo on the benchmark grid), so it cancels to first
    order in the difference: what survives is h^2 d3C/dsigma3 plus the
    derivative of the bias, both far below the 0.02 this is thresholded at.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    up = curran_batch(m, T, sigma + h, rate, n_steps)
    dn = curran_batch(m, T, np.maximum(sigma - h, 1e-8), rate, n_steps)
    return (up - dn) / (2.0 * h)


# --------------------------------------------------------------------------- #
#  Network evaluation: prices through price_batch, derivatives through autograd
# --------------------------------------------------------------------------- #

def unit_strike_price(engine: PricingEngine, m: np.ndarray, T: np.ndarray,
                      sigma: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """C(S = m, K = 1) through the served batch path, per unit strike."""
    ones = np.ones_like(m, dtype=np.float64)
    return engine.price_batch(m.astype(np.float64), ones, T.astype(np.float64),
                              sigma.astype(np.float64),
                              rate.astype(np.float64)).astype(np.float64)


def strike_price(engine: PricingEngine, m: np.ndarray, T: np.ndarray,
                 sigma: np.ndarray, rate: np.ndarray,
                 strike: np.ndarray) -> np.ndarray:
    """C(S = m, K = strike) through the served batch path."""
    return engine.price_batch(m.astype(np.float64), strike.astype(np.float64),
                              T.astype(np.float64), sigma.astype(np.float64),
                              rate.astype(np.float64)).astype(np.float64)


def _t32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def autograd_derivatives(engine: PricingEngine, m: np.ndarray, T: np.ndarray,
                         sigma: np.ndarray, rate: np.ndarray,
                         chunk: int = _CHUNK) -> dict[str, np.ndarray]:
    """dC/dK, d2C/dK2, delta, gamma and vega at (S = m, K = 1), by autograd.

    Reverse mode through `engine._call_price_torch`, the same graph the served
    Greeks are taken from, in the served float32.  Every lattice row is priced
    independently, so differentiating the sum gives the per-row derivatives.
    """
    out = {k: np.empty(m.size, dtype=np.float64) for k in
           ("price", "dC_dK", "d2C_dK2", "delta", "gamma", "vega")}
    for lo in range(0, m.size, chunk):
        sl = slice(lo, min(lo + chunk, m.size))
        mt, Tt = _t32(m[sl]), _t32(T[sl])
        st, rt = _t32(sigma[sl]), _t32(rate[sl])

        # (a) strike as the free variable, spot fixed: C(K) = K f(S/K).
        spot = mt.detach()
        K = torch.ones_like(spot).requires_grad_(True)
        C = K * engine._call_price_torch(spot / K, Tt, st, rt)
        (dC_dK,) = torch.autograd.grad(C.sum(), K, create_graph=True)
        (d2C_dK2,) = torch.autograd.grad(dC_dK.sum(), K)

        # (b) spot as the free variable, strike fixed at 1: C(S) = f(S).
        S = spot.clone().requires_grad_(True)
        C2 = engine._call_price_torch(S, Tt, st, rt)
        (delta,) = torch.autograd.grad(C2.sum(), S, create_graph=True)
        (gamma,) = torch.autograd.grad(delta.sum(), S)

        # (c) volatility as the free variable.
        sg = st.clone().requires_grad_(True)
        C3 = engine._call_price_torch(spot, Tt, sg, rt)
        (vega,) = torch.autograd.grad(C3.sum(), sg)

        for key, val in (("price", C), ("dC_dK", dC_dK), ("d2C_dK2", d2C_dK2),
                         ("delta", delta), ("gamma", gamma), ("vega", vega)):
            out[key][sl] = val.detach().numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #

def _loc(coords: dict[str, np.ndarray], i: int) -> dict[str, float]:
    return {"m": float(coords["m"][i]), "T": float(coords["T"][i]),
            "sigma": float(coords["sigma"][i]), "rate": float(coords["rate"][i]),
            "sigma_sqrt_T": float(coords["sigma"][i] * np.sqrt(coords["T"][i]))}


def _bucket_fractions(viol: np.ndarray, m: np.ndarray, T: np.ndarray,
                      sig: np.ndarray) -> dict[str, Any]:
    """Where the violations sit: violation rate and share of the total per bucket."""
    out: dict[str, Any] = {}
    root = sig * np.sqrt(T)
    groups = {
        "by_T": {"0.05-0.25y": T < 0.25, "0.25-0.5y": (T >= 0.25) & (T < 0.5),
                 "0.5-1y": (T >= 0.5) & (T < 1.0), "1-2y": T >= 1.0},
        "by_moneyness": {"otm_m<0.95": m < 0.95,
                         "atm_|m-1|<=0.05": np.abs(m - 1.0) <= 0.05,
                         "itm_m>1.05": m > 1.05},
        "by_sigma_sqrt_T": {"<0.05": root < 0.05,
                            "0.05-0.10": (root >= 0.05) & (root < 0.10),
                            "0.10-0.20": (root >= 0.10) & (root < 0.20),
                            ">=0.20": root >= 0.20},
    }
    for name, gs in groups.items():
        out[name] = {g: {"n": int(msk.sum()),
                         "violation_fraction": (float(viol[msk].mean())
                                                if msk.any() else None),
                         "share_of_violations": float(viol[msk].sum()
                                                      / max(viol.sum(), 1))}
                     for g, msk in gs.items()}
    out["by_sigma"] = {}
    for s in np.unique(sig):
        msk = sig == s
        out["by_sigma"][f"{s:.2f}"] = {
            "n": int(msk.sum()), "violation_fraction": float(viol[msk].mean()),
            "share_of_violations": float(viol[msk].sum() / max(viol.sum(), 1))}
    return out


def region_stats(margin: np.ndarray, mask: np.ndarray,
                 coords: dict[str, np.ndarray],
                 list_violations: int = 0) -> dict[str, Any]:
    """Violation rate, worst margin and its location, on `mask`.

    Every condition is written as `margin >= 0`, so the worst point is the
    minimum of the margin whether or not anything is violated.
    """
    viol = mask & (margin < 0.0)
    out: dict[str, Any] = {"n_points": int(mask.sum()),
                           "n_violations": int(viol.sum()),
                           "violation_fraction": float(viol.sum()
                                                       / max(mask.sum(), 1))}
    if mask.any():
        idx = np.flatnonzero(mask)
        i = int(idx[np.argmin(margin[mask])])
        out["worst_value"] = float(margin[i])
        out["worst_at"] = _loc(coords, i)
        out["where"] = _bucket_fractions(viol[mask], coords["m"][mask],
                                         coords["T"][mask], coords["sigma"][mask])
    if list_violations and viol.any():
        order = np.flatnonzero(viol)[np.argsort(margin[viol])]
        out["violations"] = [{**_loc(coords, int(j)), "margin": float(margin[j])}
                             for j in order[:list_violations]]
        out["violations_listed"] = int(min(order.size, list_violations))
    return out


# --------------------------------------------------------------------------- #
#  The audit
# --------------------------------------------------------------------------- #

#: (key, statement, units, margin builder).  Every margin must be >= 0.
CONDITION_ORDER = (
    "monotone_dC_dK_le_0",
    "slope_dC_dK_ge_-discount",
    "convexity_d2C_dK2_ge_0",
    "butterfly_1pct_bps",
    "asian_floor_bps",
    "upper_bound_discounted_mean_bps",
    "upper_bound_spot_bps",
    "positivity_bps",
    "gamma_ge_0",
    "vega_ge_0",
    "delta_ge_0",
    "delta_le_forward",
)

CONDITION_META = {
    "monotone_dC_dK_le_0": ("dC/dK <= 0", "price per unit strike, per unit strike"),
    "slope_dC_dK_ge_-discount": ("dC/dK >= -e^{-rT}",
                                 "price per unit strike, per unit strike"),
    "convexity_d2C_dK2_ge_0": ("d2C/dK2 >= 0", "density, per unit strike"),
    "butterfly_1pct_bps": ("C(K(1-h)) - 2C(K) + C(K(1+h)) >= 0, h = 1%",
                           "bps of strike"),
    "asian_floor_bps": ("C >= e^{-rT} (E[A] - K)+", "bps of strike"),
    "upper_bound_discounted_mean_bps": ("C <= e^{-rT} E[A]", "bps of strike"),
    "upper_bound_spot_bps": ("C <= S", "bps of strike"),
    "positivity_bps": ("C >= 0", "bps of strike"),
    "gamma_ge_0": ("d2C/dS2 >= 0", "per unit strike, per unit spot squared"),
    "vega_ge_0": ("dC/dsigma >= 0", "per unit strike, per unit vol"),
    "delta_ge_0": ("dC/dS >= 0", "dimensionless"),
    "delta_le_forward": ("dC/dS <= e^{-rT} E[A]/S", "dimensionless"),
}


def build_lattice(n_m: int = N_M, n_T: int = N_T, sigmas=SIGMAS, rates=RATES
                  ) -> dict[str, np.ndarray]:
    """The trained box, uniformly in m and T, at each (sigma, r) slice."""
    m_lo, m_hi = PARAM_RANGES["moneyness"]
    T_lo, T_hi = PARAM_RANGES["maturity"]
    m_axis = np.linspace(m_lo, m_hi, n_m)
    T_axis = np.linspace(T_lo, T_hi, n_T)
    MM, TT = np.meshgrid(m_axis, T_axis, indexing="ij")
    cols = {k: [] for k in ("m", "T", "sigma", "rate")}
    for s in sigmas:
        for rr in rates:
            cols["m"].append(MM.ravel())
            cols["T"].append(TT.ravel())
            cols["sigma"].append(np.full(MM.size, float(s)))
            cols["rate"].append(np.full(MM.size, float(rr)))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["m_axis"], out["T_axis"] = m_axis, T_axis
    return out


def audit(engine: PricingEngine, coords: dict[str, np.ndarray],
          vega_floor: float = VEGA_FLOOR, butterfly_width: float = BUTTERFLY_WIDTH,
          n_steps: int = N_MONITORING_STEPS) -> dict[str, Any]:
    """Evaluate every transferable condition on the lattice."""
    m, T = coords["m"], coords["T"]
    sig, r = coords["sigma"], coords["rate"]
    n = m.size
    t0 = time.perf_counter()

    # The audited model must be the Asian one at every point: PricingEngine
    # routes maturities at or below 12/252 to the 0DTE European surrogate.
    assert float(T.min()) > ZERO_DTE_CUTOFF, (
        f"lattice reaches T = {T.min():.4f} <= the 0DTE cutoff "
        f"{ZERO_DTE_CUTOFF:.4f}; those points would be priced by "
        "model_0dte.pt, not model.pt")

    _log(f"  prices on {n:,} points ...")
    ones = np.ones(n)
    price = unit_strike_price(engine, m, T, sig, r)
    t_price = time.perf_counter() - t0

    _log("  butterfly wings ...")
    h = butterfly_width
    c_dn = strike_price(engine, m, T, sig, r, ones * (1.0 - h))
    c_up = strike_price(engine, m, T, sig, r, ones * (1.0 + h))
    butterfly = c_dn - 2.0 * price + c_up

    _log("  autograd derivatives ...")
    t1 = time.perf_counter()
    d = autograd_derivatives(engine, m, T, sig, r)
    t_grad = time.perf_counter() - t1

    _log("  bounds and the reference ...")
    ea = expected_average(m, T, r, n_steps)               # E[A], spot = m, K = 1
    disc = np.exp(-r * T)
    floor = disc * np.maximum(ea - 1.0, 0.0)
    t2 = time.perf_counter()
    ref = curran_batch(m, T, sig, r, n_steps)
    ref_vega = curran_vega(m, T, sig, r, n_steps)
    t_ref = time.perf_counter() - t2
    resolved = ref_vega >= vega_floor

    # The 1% butterfly reads the network at K(1 +- h), i.e. at m/(1 +- h); at
    # the two ends of the m axis one wing leaves the trained box by up to 1%.
    m_lo, m_hi = PARAM_RANGES["moneyness"]
    butterfly_mask = (m / (1.0 - h) <= m_hi + 1e-12) & (m / (1.0 + h) >= m_lo - 1e-12)

    margins = {
        "monotone_dC_dK_le_0": -d["dC_dK"],
        "slope_dC_dK_ge_-discount": d["dC_dK"] + disc,
        "convexity_d2C_dK2_ge_0": d["d2C_dK2"],
        "butterfly_1pct_bps": butterfly * 1e4,
        "asian_floor_bps": (price - floor) * 1e4,
        "upper_bound_discounted_mean_bps": (disc * ea - price) * 1e4,
        "upper_bound_spot_bps": (m - price) * 1e4,
        "positivity_bps": price * 1e4,
        "gamma_ge_0": d["gamma"],
        "vega_ge_0": d["vega"],
        "delta_ge_0": d["delta"],
        "delta_le_forward": disc * ea / m - d["delta"],
    }
    base_masks = {k: np.ones(n, dtype=bool) for k in margins}
    base_masks["butterfly_1pct_bps"] = butterfly_mask

    conditions: dict[str, Any] = {}
    for key in CONDITION_ORDER:
        statement, units = CONDITION_META[key]
        base = base_masks[key]
        conditions[key] = {
            "statement": statement, "units": units,
            "all": region_stats(margins[key], base, coords),
            # The resolved-region violations are few enough to name, and a
            # claim about where they sit should be checkable point by point.
            "resolved": region_stats(margins[key], base & resolved, coords,
                                     list_violations=64),
        }

    # Cross-checks.  (i) homogeneity of degree one makes d2C/dK2 = m^2 gamma
    # exactly, so the two autograd paths must agree in sign everywhere and in
    # value to float32 roundoff; (ii) the autograd graph must return the same
    # price the served batch path does.
    same_sign = float(np.mean(np.sign(d["d2C_dK2"]) == np.sign(d["gamma"])))
    gap = np.abs(d["d2C_dK2"] - m ** 2 * d["gamma"])
    big = np.abs(d["d2C_dK2"]) >= 1e-3
    price_gap = float(np.max(np.abs(d["price"] - price)))

    report: dict[str, Any] = {
        "protocol": {
            "n_points": int(n),
            "n_m": int(coords["m_axis"].size), "m_min": float(coords["m_axis"].min()),
            "m_max": float(coords["m_axis"].max()),
            "n_T": int(coords["T_axis"].size), "T_min": float(coords["T_axis"].min()),
            "T_max": float(coords["T_axis"].max()),
            "sigmas": sorted(float(x) for x in np.unique(sig)),
            "rates": sorted(float(x) for x in np.unique(r)),
            "n_monitoring_steps": int(n_steps),
            "butterfly_width": float(butterfly_width),
            "vega_floor": float(vega_floor),
            "vega_floor_measure": ("dC/dsigma of the Curran reference, central "
                                   "difference at sigma +- 1e-3, per unit "
                                   "strike per unit vol"),
            "normalisation": "strike = 1, spot = m; prices per unit strike",
            "zero_dte_cutoff": float(ZERO_DTE_CUTOFF),
            "dtype": "float32 (the served path)",
        },
        "coverage": {
            "resolved_fraction": float(resolved.mean()),
            "resolved_points": int(resolved.sum()),
            "min_reference_vega": float(ref_vega.min()),
            "max_reference_vega": float(ref_vega.max()),
            "median_reference_vega": float(np.median(ref_vega)),
        },
        "conditions": conditions,
        "cross_checks": {
            "sign_agreement_convexity_vs_gamma": same_sign,
            "max_abs_gap_d2C_dK2_vs_m2_gamma": float(gap.max()),
            "max_relative_gap_d2C_dK2_vs_m2_gamma_where_abs_ge_1e-3":
                float((gap[big] / np.abs(d["d2C_dK2"][big])).max())
                if big.any() else None,
            "max_abs_price_gap_autograd_vs_price_batch": price_gap,
        },
        "timings_s": {"price": t_price, "autograd": t_grad, "reference": t_ref},
    }

    err_bps = (price - ref) * 1e4
    report["reference"] = curran_summary(err_bps, resolved, coords)

    report["_margins"] = margins
    report["_resolved"] = resolved
    report["_price"] = price
    report["elapsed_s"] = time.perf_counter() - t0
    return report


def curran_summary(err_bps: np.ndarray, resolved: np.ndarray,
                   coords: dict[str, np.ndarray]) -> dict[str, Any]:
    def block(mask: np.ndarray) -> dict[str, Any]:
        e = err_bps[mask]
        i = int(np.flatnonzero(mask)[np.argmax(np.abs(e))])
        return {"n_points": int(mask.sum()), "rmse_bps": float(np.sqrt((e ** 2).mean())),
                "mae_bps": float(np.abs(e).mean()), "bias_bps": float(e.mean()),
                "p95_abs_bps": float(np.percentile(np.abs(e), 95)),
                "max_abs_bps": float(np.abs(e).max()), "max_abs_at": _loc(coords, i)}
    allm = np.ones(err_bps.size, dtype=bool)
    out = {"reference": "curran_call (1994), exact threshold, 50 fixings",
           "all": block(allm), "resolved": block(resolved)}
    out["by_sigma"] = {f"{s:.2f}": block(coords["sigma"] == s)
                       for s in np.unique(coords["sigma"])}
    return out


# --------------------------------------------------------------------------- #
#  Monte Carlo arbiter
# --------------------------------------------------------------------------- #

MC_SEEDS = (11, 12, 13, 14)


def _mc_prices(m: float, T: float, sigma: float, rate: float, n_paths: int,
               n_steps: int, seeds=MC_SEEDS, strike: float = 1.0) -> np.ndarray:
    return np.array([price_asian_mc(float(m), float(strike), float(T),
                                    float(sigma), float(rate), n_paths=n_paths,
                                    n_steps=n_steps, option_type="call",
                                    seed=s, control_variate=True).price
                     for s in seeds], dtype=np.float64)


def mc_reference(m: float, T: float, sigma: float, rate: float,
                 n_paths: int, n_steps: int, seeds=MC_SEEDS) -> dict[str, float]:
    """Mean and across-seed standard error of the project's MC pricer."""
    arr = _mc_prices(m, T, sigma, rate, n_paths, n_steps, seeds)
    return {"price_bps": float(arr.mean() * 1e4),
            "se_bps": float(arr.std(ddof=1) / np.sqrt(arr.size) * 1e4),
            "n_paths": int(n_paths), "n_seeds": len(seeds)}


def mc_vega(m: float, T: float, sigma: float, rate: float, n_paths: int,
            n_steps: int, h: float = VEGA_STEP, seeds=MC_SEEDS
            ) -> dict[str, float]:
    """dC/dsigma by common-random-number central difference on the MC pricer.

    The same seeds price sigma + h and sigma - h, so the path noise cancels in
    the difference and the standard error of the derivative is orders of
    magnitude below the standard error of either price.  This is the check that
    the resolution measure - a Curran central difference at the same step - is
    the contract's vega and not Curran's bias.
    """
    up = _mc_prices(m, T, sigma + h, rate, n_paths, n_steps, seeds)
    dn = _mc_prices(m, T, max(sigma - h, 1e-8), rate, n_paths, n_steps, seeds)
    d = (up - dn) / (2.0 * h)
    return {"mc_vega": float(d.mean()),
            "mc_vega_se": float(d.std(ddof=1) / np.sqrt(d.size)),
            "curran_vega": float(curran_vega(m, T, sigma, rate, n_steps, h)[0])}


def arbitrate(engine: PricingEngine, points: list[dict[str, Any]],
              n_paths: int, n_steps: int) -> list[dict[str, Any]]:
    """Network, Curran and Monte Carlo at a handful of named points."""
    out = []
    for p in points:
        m, T, s, r = p["m"], p["T"], p["sigma"], p["rate"]
        net = float(unit_strike_price(engine, np.array([m]), np.array([T]),
                                      np.array([s]), np.array([r]))[0])
        cur = curran_call(m, 1.0, T, s, r, n_steps)
        mc = mc_reference(m, T, s, r, n_paths, n_steps)
        row = {"label": p["label"], "m": m, "T": T, "sigma": s, "rate": r,
               "sigma_sqrt_T": float(s * np.sqrt(T)),
               "network_bps": net * 1e4, "curran_bps": cur * 1e4,
               "asian_floor_bps": float(asian_floor(m, 1.0, T, r, n_steps) * 1e4),
               "european_floor_bps": float(max(m - np.exp(-r * T), 0.0) * 1e4),
               **{f"mc_{k}": v for k, v in mc.items()}}
        row["curran_minus_mc_bps"] = row["curran_bps"] - row["mc_price_bps"]
        row["network_minus_mc_bps"] = row["network_bps"] - row["mc_price_bps"]
        out.append(row)
    return out


def butterfly_arbitration(engine: PricingEngine, points: list[dict[str, Any]],
                          n_paths: int, n_steps: int,
                          h: float = BUTTERFLY_WIDTH) -> list[dict[str, Any]]:
    """The network's 1%-wide butterfly against a common-random-number MC one.

    The three strikes are priced on the same paths, so the butterfly's Monte
    Carlo standard error is far smaller than any one leg's: a second
    difference of prices that share their noise.  This is what decides whether
    a butterfly the network prices negative is negative in the model.
    """
    out = []
    for p in points:
        m, T, s, r = p["m"], p["T"], p["sigma"], p["rate"]
        legs = {}
        for name, K in (("dn", 1.0 - h), ("mid", 1.0), ("up", 1.0 + h)):
            legs[name] = _mc_prices(m, T, s, r, n_paths, n_steps, strike=K)
        fly = legs["dn"] - 2.0 * legs["mid"] + legs["up"]
        net = strike_price(engine, np.full(3, m), np.full(3, T), np.full(3, s),
                           np.full(3, r),
                           np.array([1.0 - h, 1.0, 1.0 + h]))
        out.append({"label": p["label"], "m": m, "T": T, "sigma": s, "rate": r,
                    "network_butterfly_bps": float((net[0] - 2 * net[1] + net[2])
                                                   * 1e4),
                    "mc_butterfly_bps": float(fly.mean() * 1e4),
                    "mc_butterfly_se_bps": float(fly.std(ddof=1)
                                                 / np.sqrt(fly.size) * 1e4),
                    "n_paths": int(n_paths), "n_seeds": len(MC_SEEDS)})
    return out


def vega_arbitration(points: list[dict[str, Any]], n_paths: int, n_steps: int
                     ) -> list[dict[str, Any]]:
    """The reference vega against a common-random-number Monte Carlo vega."""
    out = []
    for p in points:
        row = {"label": p["label"], "m": p["m"], "T": p["T"],
               "sigma": p["sigma"], "rate": p["rate"]}
        row.update(mc_vega(p["m"], p["T"], p["sigma"], p["rate"], n_paths,
                           n_steps))
        row["gap"] = row["curran_vega"] - row["mc_vega"]
        out.append(row)
    return out


def corner_points() -> list[dict[str, Any]]:
    m_lo, m_hi = PARAM_RANGES["moneyness"]
    T_lo, T_hi = PARAM_RANGES["maturity"]
    s_lo, s_hi = PARAM_RANGES["sigma"]
    pts = []
    for m in (m_lo, 1.0, m_hi):
        for T in (T_lo, T_hi):
            for s in (s_lo, s_hi):
                pts.append({"label": f"corner m={m} T={T} sigma={s}", "m": m,
                            "T": T, "sigma": s, "rate": 0.04})
    return pts


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    hsh = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hsh.update(block)
    return hsh.hexdigest()


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "model.pt")
    ap.add_argument("--out-dir", type=Path, default=DOCS)
    ap.add_argument("--quick", action="store_true",
                    help="coarse lattice, fewer slices, temp dir")
    ap.add_argument("--no-mc", action="store_true",
                    help="skip the Monte Carlo arbiter")
    ap.add_argument("--mc-paths", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(tempfile.mkdtemp(prefix="asian_audit_")) if args.quick \
        else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    _log(f"Loading {args.checkpoint} ...")
    engine = PricingEngine(args.checkpoint)

    n_m, n_T = (31, 8) if args.quick else (N_M, N_T)
    sigmas = (0.05, 0.20, 0.80) if args.quick else SIGMAS
    rates = (0.0, 0.04) if args.quick else RATES
    coords = build_lattice(n_m, n_T, sigmas, rates)
    _log(f"Lattice: {coords['m'].size:,} points "
         f"({n_m} m x {n_T} T x {len(sigmas)} sigma x {len(rates)} r)")

    rep = audit(engine, coords)

    if not args.no_mc:
        pts = corner_points()
        seen = {(p["m"], p["T"], p["sigma"], p["rate"]) for p in pts}
        interesting = [("worst " + key, rep["conditions"][key]["all"].get("worst_at"))
                       for key in ("asian_floor_bps", "convexity_d2C_dK2_ge_0",
                                   "butterfly_1pct_bps", "gamma_ge_0",
                                   "vega_ge_0", "slope_dC_dK_ge_-discount")]
        interesting += [("largest gap to the reference",
                         rep["reference"]["all"]["max_abs_at"]),
                        ("largest gap to the reference, resolved region",
                         rep["reference"]["resolved"]["max_abs_at"])]
        for label, at in interesting:
            if at is None:
                continue
            tup = (at["m"], at["T"], at["sigma"], at["rate"])
            if tup in seen:
                continue
            seen.add(tup)
            pts.append({"label": label,
                        **{k: at[k] for k in ("m", "T", "sigma", "rate")}})
        _log(f"  Monte Carlo arbiter on {len(pts)} points "
             f"({args.mc_paths:,} paths x {len(MC_SEEDS)} seeds) ...")
        rep["arbitration"] = arbitrate(engine, pts, args.mc_paths,
                                       N_MONITORING_STEPS)
        vega_pts = [p for p in pts if p["label"].startswith("corner")][:6] \
            + [p for p in pts if not p["label"].startswith("corner")][:4]
        _log(f"  Monte Carlo vega check on {len(vega_pts)} points ...")
        rep["vega_check"] = vega_arbitration(vega_pts, args.mc_paths,
                                             N_MONITORING_STEPS)

        fly_pts: list[dict[str, Any]] = []
        for label, at in (("worst butterfly, full box",
                           rep["conditions"]["butterfly_1pct_bps"]["all"]
                           .get("worst_at")),
                          ("worst butterfly, resolved region",
                           rep["conditions"]["butterfly_1pct_bps"]["resolved"]
                           .get("worst_at")),
                          ("worst convexity, resolved region",
                           rep["conditions"]["convexity_d2C_dK2_ge_0"]["resolved"]
                           .get("worst_at"))):
            if at is not None:
                fly_pts.append({"label": label,
                                **{k: at[k] for k in ("m", "T", "sigma", "rate")}})
        _log(f"  Monte Carlo butterfly check on {len(fly_pts)} points ...")
        rep["butterfly_check"] = butterfly_arbitration(engine, fly_pts,
                                                       args.mc_paths,
                                                       N_MONITORING_STEPS)

    meta = {k: engine.meta.get(k) for k in
            ("arm", "width", "blocks", "n_members", "n_samples",
             "mc_paths_per_label", "epochs", "seed", "n_monitoring_steps",
             "output_scale", "n_parameters")}
    rep["checkpoint"] = {"path": args.checkpoint.relative_to(ROOT).as_posix()
                         if args.checkpoint.is_relative_to(ROOT)
                         else args.checkpoint.as_posix(),
                         "sha256": _sha256(args.checkpoint),
                         "param_ranges": {k: list(v) for k, v in
                                          PARAM_RANGES.items()},
                         "meta": meta}
    rep["generated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    rep["environment"] = {"python": platform.python_version(),
                          "torch": torch.__version__,
                          "numpy": np.__version__,
                          "platform": platform.platform(),
                          "torch_threads": int(torch.get_num_threads())}
    rep["conditions_not_tested"] = {
        "black_scholes_implied_vol_and_durrleman_g":
            "the arithmetic average is not lognormal, so a Black-Scholes "
            "implied vol read off this price is not a parameter of the model "
            "and g(k) is not a statement about its density",
        "european_intrinsic_floor_max(S - K e^{-rT}, 0)":
            "strictly above the Asian floor e^{-rT}(E[A] - K)+ whenever r > 0, "
            "because averaging replaces the terminal forward with the mean of "
            "the forwards; applying it would report violations that are not",
        "calendar_dC_dT_ge_0":
            "the 50 fixings are t_i = i T / n, so a longer maturity is a "
            "different averaging window rather than the same contract held "
            "longer; dC/dT carries no no-arbitrage sign for this contract",
    }
    rep["total_elapsed_s"] = time.perf_counter() - t_start

    out_json = out_dir / "asian_arbitrage_audit.json"
    out_json.write_text(json.dumps(_strip_private(rep), indent=2),
                        encoding="utf-8")
    _log(f"Wrote {out_json}  ({rep['total_elapsed_s']:.1f} s)")

    _log("")
    _log(f"{'condition':<34} {'all':>10} {'resolved':>10} {'worst (all)':>14}")
    for key in CONDITION_ORDER:
        c = rep["conditions"][key]
        _log(f"{key:<34} {c['all']['violation_fraction']:>9.3%} "
             f"{c['resolved']['violation_fraction']:>9.3%} "
             f"{c['all'].get('worst_value', float('nan')):>14.5g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
