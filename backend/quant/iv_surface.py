"""No-arbitrage implied-volatility surface for the 0DTE regime, and a static-arbitrage
audit of the served 0DTE price surrogate.

1. ``TeacherSurface`` wraps ``PricingEngine`` (the served 5-member ensemble in
   ``artifacts/model_0dte.pt``, which outputs European call prices per unit strike as a
   function of (S/K, T, sigma, r)) and exposes it as a differentiable total-variance
   surface w(k, T; sigma, r) = sigma_imp^2 T.  The price is inverted to a Black-Scholes
   implied vol by a vectorised bisection (no gradient) followed by two Newton steps
   that are differentiated: at a converged root the Newton map has zero derivative
   with respect to its starting point, so autograd through two steps returns the
   exact implicit first and second derivatives of sigma_imp with respect to k and T
   (one step is exact only to first order; see ``_newton_refine``).

2. ``IVSurface`` is a small MLP that models the total variance directly, following
   Ackerer, Tagasovska & Vatter, "Deep Smoothing of the Implied Volatility Surface"
   (NeurIPS 2020):

        w(k, T; sigma, r) = sigma^2 T * softplus(NN(k, T, sigma, r) + c0),

   a flat-vol prior (arbitrage-free by itself: g = 1, dw/dT = sigma^2 > 0) times a
   strictly positive multiplier that starts at exactly 1.  It is trained on the
   teacher's prices with the loss

        L = 2 * mean[Huber_delta((P_nn - P_teacher) / max(vega, vega_floor))]
            + lambda_but * mean[relu(eps - g)^2]
            + lambda_cal * mean[relu(eps - (dw/dT) / sigma^2)^2]
            + lambda_lee * mean[relu(|dw/dk| - 2)^2]

   (2 * Huber equals the squared residual inside delta = 0.02, calibrate.py's delta)
   with the penalties evaluated by autograd on a fresh random batch every step.

Conventions
-----------
* k = ln(K / F) is forward log-moneyness, F = S e^{rT} (the engine drifts the spot at
  the rate with no dividend).  This is the variable in which the Durrleman butterfly
  function and the calendar condition dw/dT >= 0 are stated (Gatheral & Jacquier
  2014).  The engine's moneyness is m = S/K = exp(-(k + rT)); with r <= 0.10 and
  T <= 12/252 the shift rT is at most 0.0048.  ``K_BOX`` is the largest k interval
  whose image lies inside the trained moneyness box [0.85, 1.15] for every (r, T).
* Prices are per unit strike (C/K), as the engine returns them.
* Total variance w = sigma_imp^2 T.  The Durrleman function is

      g(k) = (1 - k w'/(2w))^2 - (w'^2 / 4) (1/w + 1/4) + w''/2,

  and the surface is free of butterfly arbitrage on a slice iff g >= 0 there; the
  risk-neutral density in k is g(k) exp(-d_-^2 / 2) / sqrt(2 pi w), so g < 0 is a
  negative density.  Calendar-spread arbitrage is absent iff dw/dT >= 0 at fixed k.
* "Resolved" points: a price error dP moves the implied vol by dP / vega.  The
  reference price error is 1.33 bps of strike, the main pricer's ensemble price
  RMSE over 600 held-out points (artifacts/eval.json; that report does not
  cover the 0DTE ensemble).  Where the BS vega per unit strike per unit vol
  falls below ``VEGA_FLOOR`` = 0.02, an error of that size is already
  0.67 vol points and the implied vol of the price surrogate carries no
  information about the smile.  Every statistic below is reported on the full box
  and on the resolved sub-region.

Nothing here imports matplotlib; the figure is drawn by scripts/no_arbitrage_surface.py.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .engine import PricingEngine, ZERO_DTE_CUTOFF

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SURFACE_CHECKPOINT = ARTIFACTS / "iv_surface_0dte.pt"

TRADING_DAYS = 252.0
#: forward log-moneyness box: exp(-(k + rT)) stays inside [0.85, 1.15] for every
#: r in [0, 0.10] and T in [1/252, 12/252]  (-ln 1.15 = -0.1398, ln(1/0.85) = 0.1625)
K_BOX = (-0.139, 0.157)
T_BOX = (1.0 / TRADING_DAYS, 12.0 / TRADING_DAYS)
SIGMA_BOX = (0.05, 0.80)
RATE_BOX = (0.0, 0.10)
#: BS vega per unit strike per unit vol below which an implied vol read off the
#: price surrogate is not resolved (1.33 bp of price = 0.67 vol point at the floor)
VEGA_FLOOR = 0.02
#: Huber switch on the vega-normalised residual, in units of vol (2 vol points),
#: the same delta calibrate.py uses against market quotes
HUBER_DELTA = 0.02
#: bisection bracket for the implied vol; a price needing more than 500% vol is
#: reported as "capped" rather than inverted
IV_LO, IV_HI = 1e-4, 5.0
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# Black-Scholes in torch (all per unit strike)

def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _npdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT2PI


def bs_call_unit(m: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
                 r: torch.Tensor) -> torch.Tensor:
    """European call price / K for spot-over-strike m = S/K (no dividends)."""
    sd = sigma.clamp_min(1e-12) * torch.sqrt(T)
    d1 = (torch.log(m) + (r + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    return m * _ncdf(d1) - torch.exp(-r * T) * _ncdf(d2)


def bs_vega_unit(m: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
                 r: torch.Tensor) -> torch.Tensor:
    """d(C/K)/d sigma  (per unit vol, so a vol point is 0.01 of this)."""
    sd = sigma.clamp_min(1e-12) * torch.sqrt(T)
    d1 = (torch.log(m) + (r + 0.5 * sigma * sigma) * T) / sd
    return m * _npdf(d1) * torch.sqrt(T)


def bs_call_from_w(k: torch.Tensor, T: torch.Tensor, w: torch.Tensor,
                   r: torch.Tensor) -> torch.Tensor:
    """Call price / K from forward log-moneyness k = ln(K/F) and total variance w.

    C/K = e^{-rT} [ e^{-k} N(d1) - N(d2) ],  d1 = (-k + w/2)/sqrt(w),  d2 = d1 - sqrt(w).
    """
    sw = torch.sqrt(w.clamp_min(1e-16))
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    return torch.exp(-r * T) * (torch.exp(-k) * _ncdf(d1) - _ncdf(d2))


def price_bounds(m: torch.Tensor, T: torch.Tensor, r: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """No-arbitrage bounds on C/K: (max(m - e^{-rT}, 0), m)."""
    return (m - torch.exp(-r * T)).clamp_min(0.0), m


def implied_vol_torch(price: torch.Tensor, m: torch.Tensor, T: torch.Tensor,
                      r: torch.Tensor, *, iters: int = 64,
                      lo: float = IV_LO, hi: float = IV_HI
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised bisection inverse of bs_call_unit (no gradient).

    Returns (sigma, defined, capped): `defined` is False where the price lies outside
    the no-arbitrage bounds (no implied vol exists; sigma is NaN there), `capped` is
    True where the price needs more than `hi` vol (sigma is returned at `hi`).
    """
    with torch.no_grad():
        lower, upper = price_bounds(m, T, r)
        defined = (price > lower + 1e-12) & (price < upper - 1e-12)
        lo_t = torch.full_like(price, lo)
        hi_t = torch.full_like(price, hi)
        capped = bs_call_unit(m, T, hi_t, r) < price
        for _ in range(iters):
            mid = 0.5 * (lo_t + hi_t)
            below = bs_call_unit(m, T, mid, r) < price
            lo_t = torch.where(below, mid, lo_t)
            hi_t = torch.where(below, hi_t, mid)
        sigma = 0.5 * (lo_t + hi_t)
        sigma = torch.where(defined, sigma, torch.full_like(sigma, float("nan")))
    return sigma, defined, capped


def _newton_refine(price: torch.Tensor, m: torch.Tensor, T: torch.Tensor,
                   r: torch.Tensor, sigma0: torch.Tensor, steps: int = 2,
                   vega_min: float = 1e-10) -> torch.Tensor:
    """Differentiable Newton steps from a converged (detached) root sigma0.

    With N(s) = s - (BS(s) - P)/vega(s), N'(s*) = 0 at the root, so d(N o N)/dk
    equals the implicit derivative d sigma*/dk exactly and d^2(N o N)/dk^2 equals the
    implicit second derivative exactly (the N_{ss} sigma'^2 term that a single step
    misses is produced by the outer step).  Where vega < vega_min the root is
    returned unchanged (no gradient); those points are outside any resolution.
    """
    s = sigma0.detach()
    for _ in range(steps):
        c = bs_call_unit(m, T, s, r)
        v = bs_vega_unit(m, T, s, r)
        ok = v.detach() > vega_min
        step = (c - price) / v.clamp_min(vega_min)
        s = torch.where(ok, s - step, s)
    return s


# Durrleman / calendar diagnostics by autograd

def durrleman_g(w: torch.Tensor, w_k: torch.Tensor, w_kk: torch.Tensor,
                k: torch.Tensor) -> torch.Tensor:
    """g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  (Gatheral-Jacquier 2014)."""
    ws = w.clamp_min(1e-16)
    a = 1.0 - k * w_k / (2.0 * ws)
    return a * a - 0.25 * w_k * w_k * (1.0 / ws + 0.25) + 0.5 * w_kk


def surface_derivatives(w_fn: Callable[..., torch.Tensor], k: torch.Tensor,
                        T: torch.Tensor, sigma: torch.Tensor, r: torch.Tensor,
                        *, create_graph: bool = False) -> dict[str, torch.Tensor]:
    """w, dw/dk, d2w/dk2, dw/dT and g at each point, via autograd through `w_fn`.

    `w_fn(k, T, sigma, r)` must return a differentiable tensor of total variances.
    `create_graph=True` keeps the graph so the result can itself be differentiated
    (needed when g is a training penalty).
    """
    k = k.detach().clone().requires_grad_(True)
    T = T.detach().clone().requires_grad_(True)
    w = w_fn(k, T, sigma, r)
    (w_k, w_T) = torch.autograd.grad(w.sum(), (k, T), create_graph=True,
                                     allow_unused=True)
    if w_T is None:                    # a single slice that ignores T
        w_T = torch.zeros_like(w)
    if w_k is None:                    # flat in k (e.g. the sigma^2 T prior alone)
        w_k = torch.zeros_like(w)
        w_kk = torch.zeros_like(w)
    else:
        (w_kk,) = torch.autograd.grad(w_k.sum(), k, create_graph=create_graph,
                                      allow_unused=True)
        if w_kk is None:               # linear in k
            w_kk = torch.zeros_like(w)
    g = durrleman_g(w, w_k, w_kk, k)
    if not create_graph:
        w, w_k, w_kk, w_T, g = (t.detach() for t in (w, w_k, w_kk, w_T, g))
    return {"w": w, "w_k": w_k, "w_kk": w_kk, "w_T": w_T, "g": g}


def svi_total_variance(k: torch.Tensor, a: float, b: float, rho: float,
                       m: float, sigma: float) -> torch.Tensor:
    """Raw SVI slice w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))."""
    x = k - m
    return a + b * (rho * x + torch.sqrt(x * x + sigma * sigma))


class TeacherSurface:
    """The served 0DTE ensemble, viewed as an implied-volatility surface.

    The forward pass reproduces ``PricingEngine._zero_dte_call`` (the path
    ``_call_price_torch`` takes for every T <= ZERO_DTE_CUTOFF) on float64 copies of
    the same weights, so second derivatives are not polluted by float32 rounding;
    ``test_iv_surface.py`` checks the copies agree with the served float32 path to
    float32 resolution.
    """

    def __init__(self, engine: PricingEngine | None = None,
                 dtype: torch.dtype = torch.float64):
        self.engine = engine or PricingEngine()
        if not self.engine.has_0dte:
            raise RuntimeError("PricingEngine has no 0DTE ensemble loaded "
                               "(artifacts/model_0dte.pt missing)")
        self.dtype = dtype
        self.members = [copy.deepcopy(m).to(dtype).eval()
                        for m in self.engine._0dte_members]
        for mem in self.members:
            for p in mem.parameters():
                p.requires_grad_(False)
        self.lows = self.engine._0dte_lows.to(dtype)
        self.highs = self.engine._0dte_highs.to(dtype)
        self.meta = dict(self.engine.meta_0dte)

    def price_m(self, m: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
                r: torch.Tensor) -> torch.Tensor:
        """C/K as a function of moneyness m = S/K (mirrors engine._zero_dte_call)."""
        x = torch.stack([m, T, sigma, r], dim=-1).to(self.dtype)
        xn = 2.0 * (x - self.lows) / (self.highs - self.lows) - 1.0
        return torch.stack([net(xn) for net in self.members]).mean(dim=0)

    def price(self, k: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
              r: torch.Tensor) -> torch.Tensor:
        """C/K at forward log-moneyness k = ln(K/F)."""
        m = torch.exp(-(k + r * T))
        return self.price_m(m, T, sigma, r)

    def implied_vol(self, k: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
                    r: torch.Tensor, *, differentiable: bool = True
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """sigma_imp(k, T) plus masks {defined, capped, vega, price}."""
        k, T, sigma, r = (t.to(self.dtype) for t in (k, T, sigma, r))
        m = torch.exp(-(k + r * T))
        p = self.price_m(m, T, sigma, r)
        s0, defined, capped = implied_vol_torch(p.detach(), m.detach(), T.detach(),
                                                r.detach())
        s_fill = torch.where(defined, s0, torch.full_like(s0, float("nan")))
        if differentiable:
            s = _newton_refine(p, m, T, r, torch.nan_to_num(s0, nan=1.0))
            s = torch.where(defined, s, s_fill)
        else:
            s = s_fill
        with torch.no_grad():
            vega = bs_vega_unit(m, T, torch.nan_to_num(s0, nan=1.0), r)
            vega = torch.where(defined, vega, torch.zeros_like(vega))
        return s, {"defined": defined, "capped": capped, "vega": vega, "price": p}

    def total_variance(self, k: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor,
                       r: torch.Tensor) -> torch.Tensor:
        s, _ = self.implied_vol(k, T, sigma, r)
        return s * s * T.to(self.dtype)

    def price_space_conditions(self, k: torch.Tensor, T: torch.Tensor,
                               sigma: torch.Tensor, r: torch.Tensor,
                               butterfly_width: float = 0.01) -> dict[str, torch.Tensor]:
        """dC/dK, d2C/dK2, d(C/S)/dT at fixed k, plus a discrete 1%-wide butterfly.

        The derivatives are taken by autograd directly through the network.
        Prices are for S = 1 so K = e^{k + rT}; C(K) = K f(1/K).  Returned:
          dC_dK      : must lie in [-e^{-rT}, 0]
          d2C_dK2    : must be >= 0 (equals the density)
          dc_dT      : d/dT of C/S at fixed forward moneyness k; must be >= 0
                       (equivalent to the calendar condition dw/dT >= 0)
          dC_dT_fixK : d/dT of C at fixed strike (the classical call-calendar check)
          butterfly  : [C(K(1-h)) - 2 C(K) + C(K(1+h))] / K, in units of K (>= 0)
          below_intrinsic, above_spot : price outside the no-arbitrage bounds, as
                       positive shortfalls per unit strike
        """
        k = k.to(self.dtype).detach().clone().requires_grad_(True)
        T = T.to(self.dtype).detach().clone().requires_grad_(True)
        sigma = sigma.to(self.dtype)
        r = r.to(self.dtype)
        K = torch.exp(k + r * T)
        Kf = K.detach().clone().requires_grad_(True)     # strike as free variable
        Tf = T.detach().clone().requires_grad_(True)

        # (a) strike derivatives at fixed T
        C = Kf * self.price_m(1.0 / Kf, T.detach(), sigma, r)
        (dC_dK,) = torch.autograd.grad(C.sum(), Kf, create_graph=True)
        (d2C_dK2,) = torch.autograd.grad(dC_dK.sum(), Kf)
        # (b) maturity derivative at fixed strike
        C2 = Kf.detach() * self.price_m(1.0 / Kf.detach(), Tf, sigma, r)
        (dC_dT_fixK,) = torch.autograd.grad(C2.sum(), Tf)
        # (c) maturity derivative at fixed forward log-moneyness (C/S, S = 1)
        C3 = K * self.price_m(1.0 / K, T, sigma, r)
        (dc_dT,) = torch.autograd.grad(C3.sum(), T)
        with torch.no_grad():
            h = butterfly_width
            Kd = K.detach()
            c0 = Kd * self.price_m(1.0 / Kd, T.detach(), sigma, r)
            cm = Kd * (1 - h) * self.price_m(1.0 / (Kd * (1 - h)), T.detach(), sigma, r)
            cp = Kd * (1 + h) * self.price_m(1.0 / (Kd * (1 + h)), T.detach(), sigma, r)
            butterfly = (cm - 2.0 * c0 + cp) / Kd
            m = 1.0 / Kd
            lower, upper = price_bounds(m, T.detach(), r)
            p = self.price_m(m, T.detach(), sigma, r)
            below = (lower - p).clamp_min(0.0)
            above = (p - upper).clamp_min(0.0)
        return {"dC_dK": dC_dK.detach(), "d2C_dK2": d2C_dK2.detach(),
                "dC_dT_fixK": dC_dT_fixK.detach(), "dc_dT": dc_dT.detach(),
                "butterfly": butterfly, "below_intrinsic": below,
                "above_spot": above, "price": p}


# The constrained surface

_C0 = math.log(math.e - 1.0)      # softplus(_C0) == 1 exactly


class IVSurfaceNet(nn.Module):
    """Total variance w = sigma^2 T * softplus(MLP(features) + c0).

    Features: affine-normalised (k, T, sigma, r) plus asinh(k / (sigma sqrt T)) / 3.
    The standardised moneyness z = k / (sigma sqrt T) is the coordinate in which a
    stochastic-volatility smile is nearly stationary; without it a 1-day, 5%-vol
    smile is 0.003 wide in k and a width-64 MLP in raw k cannot resolve it.  The
    squashing function is asinh because it stays monotone and never saturates.
    """

    def __init__(self, width: int = 64, depth: int = 4,
                 k_box=K_BOX, T_box=T_BOX, sigma_box=SIGMA_BOX, rate_box=RATE_BOX):
        super().__init__()
        self.register_buffer("lows", torch.tensor(
            [k_box[0], T_box[0], sigma_box[0], rate_box[0]], dtype=torch.float32))
        self.register_buffer("highs", torch.tensor(
            [k_box[1], T_box[1], sigma_box[1], rate_box[1]], dtype=torch.float32))
        layers: list[nn.Module] = [nn.Linear(5, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)     # start exactly at the prior
        nn.init.zeros_(self.head.bias)

    def multiplier(self, k, T, sigma, r) -> torch.Tensor:
        x = torch.stack([k, T, sigma, r], dim=-1)
        xn = 2.0 * (x - self.lows) / (self.highs - self.lows) - 1.0
        z = k / (sigma * torch.sqrt(T))
        feats = torch.cat([xn, (torch.asinh(z) / 3.0).unsqueeze(-1)], dim=-1)
        return nn.functional.softplus(self.head(self.body(feats)).squeeze(-1) + _C0)

    def forward(self, k, T, sigma, r) -> torch.Tensor:
        return sigma * sigma * T * self.multiplier(k, T, sigma, r)


class IVSurface:
    """Loadable no-arbitrage implied-volatility surface for the 0DTE regime.

    ``IVSurface.load(path)`` restores ``artifacts/iv_surface_0dte.pt``; ``grid``
    evaluates a (sigma, rate) slice on a k x T lattice with the butterfly and
    calendar diagnostics; ``iv``/``price`` are vectorised numpy conveniences.
    """

    def __init__(self, net: IVSurfaceNet, meta: dict[str, Any] | None = None):
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: Path | str = SURFACE_CHECKPOINT) -> "IVSurface":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No constrained surface at {path}. Train one with "
                "'python -m scripts.no_arbitrage_surface'.")
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:                                  # pragma: no cover
            blob = torch.load(path, map_location="cpu", weights_only=False)
        arch = blob["meta"]["architecture"]
        net = IVSurfaceNet(width=int(arch["width"]), depth=int(arch["depth"]))
        net.load_state_dict(blob["state_dict"])
        return cls(net, blob["meta"])

    def save(self, path: Path | str = SURFACE_CHECKPOINT) -> None:
        torch.save({"state_dict": self.net.state_dict(), "meta": self.meta}, path)

    def total_variance(self, k, T, sigma, r) -> torch.Tensor:
        return self.net(k.float(), T.float(), sigma.float(), r.float())

    def implied_vol_torch(self, k, T, sigma, r) -> torch.Tensor:
        return torch.sqrt(self.total_variance(k, T, sigma, r) / T.float())

    def price_torch(self, k, T, sigma, r) -> torch.Tensor:
        return bs_call_from_w(k.float(), T.float(),
                              self.total_variance(k, T, sigma, r), r.float())

    @staticmethod
    def _bcast(*arrs) -> list[torch.Tensor]:
        b = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrs])
        return [torch.from_numpy(np.ascontiguousarray(x)).float() for x in b]

    def iv(self, k, T, sigma, rate) -> np.ndarray:
        with torch.no_grad():
            kt, Tt, st, rt = self._bcast(k, T, sigma, rate)
            return self.implied_vol_torch(kt, Tt, st, rt).numpy().astype(np.float64)

    def price(self, k, T, sigma, rate) -> np.ndarray:
        """European call price per unit strike at forward log-moneyness k."""
        with torch.no_grad():
            kt, Tt, st, rt = self._bcast(k, T, sigma, rate)
            return self.price_torch(kt, Tt, st, rt).numpy().astype(np.float64)

    def grid(self, sigma: float, rate: float, k_axis: np.ndarray,
             T_axis: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate one (sigma, rate) slice on the lattice T_axis x k_axis.

        Returns numpy arrays of shape (len(T_axis), len(k_axis)):
          iv, total_variance, g (Durrleman), calendar (dw/dT),
        the axes (k, T), and the scalars g_min, calendar_min with their
        (T, k) locations as 0-d / length-2 arrays.
        """
        k_axis = np.asarray(k_axis, dtype=np.float64)
        T_axis = np.asarray(T_axis, dtype=np.float64)
        TT, KK = np.meshgrid(T_axis, k_axis, indexing="ij")
        kt = torch.from_numpy(KK.ravel()).float()
        Tt = torch.from_numpy(TT.ravel()).float()
        st = torch.full_like(kt, float(sigma))
        rt = torch.full_like(kt, float(rate))
        d = surface_derivatives(self.total_variance, kt, Tt, st, rt)
        shape = TT.shape
        w = d["w"].numpy().astype(np.float64).reshape(shape)
        g = d["g"].numpy().astype(np.float64).reshape(shape)
        cal = d["w_T"].numpy().astype(np.float64).reshape(shape)
        ig = np.unravel_index(int(np.argmin(g)), shape)
        ic = np.unravel_index(int(np.argmin(cal)), shape)
        return {"k": k_axis, "T": T_axis, "iv": np.sqrt(w / TT),
                "total_variance": w, "g": g, "calendar": cal,
                "g_min": np.float64(g.min()),
                "g_min_at": np.array([T_axis[ig[0]], k_axis[ig[1]]]),
                "calendar_min": np.float64(cal.min()),
                "calendar_min_at": np.array([T_axis[ic[0]], k_axis[ic[1]]])}


# Audit

def default_k_axis(n: int = 149) -> np.ndarray:
    return np.linspace(K_BOX[0], K_BOX[1], n)


def default_T_axis(step_days: float = 0.25) -> np.ndarray:
    days = np.arange(1.0, 12.0 + 1e-9, step_days)
    return days / TRADING_DAYS


def _loc(T: float, k: float, sigma: float, rate: float) -> dict[str, float]:
    return {"T_days": float(T * TRADING_DAYS), "k": float(k),
            "sigma": float(sigma), "rate": float(rate)}


def _bucket_fractions(viol: np.ndarray, k: np.ndarray, T: np.ndarray, sig: np.ndarray
                      ) -> dict[str, Any]:
    """Share of violated points per maturity, log-moneyness and sigma bucket."""
    out: dict[str, Any] = {}
    Td = T * TRADING_DAYS
    tb = {"1-2d": (Td < 2), "2-4d": (Td >= 2) & (Td < 4),
          "4-8d": (Td >= 4) & (Td < 8), "8-12d": Td >= 8}
    kb = {"put_wing_k<-0.05": k < -0.05, "centre_|k|<=0.05": np.abs(k) <= 0.05,
          "call_wing_k>0.05": k > 0.05}
    for name, groups in (("by_T", tb), ("by_k", kb)):
        out[name] = {g: {"n": int(msk.sum()),
                         "violation_fraction": float(viol[msk].mean()) if msk.any() else float("nan"),
                         "share_of_violations": float(viol[msk].sum() / max(viol.sum(), 1))}
                     for g, msk in groups.items()}
    out["by_sigma"] = {}
    for s in np.unique(sig):
        msk = sig == s
        out["by_sigma"][f"{s:.2f}"] = {
            "n": int(msk.sum()), "violation_fraction": float(viol[msk].mean()),
            "share_of_violations": float(viol[msk].sum() / max(viol.sum(), 1))}
    return out


def arbitrage_audit(engine_or_surface: PricingEngine | TeacherSurface | IVSurface,
                    *, sigmas=(0.05, 0.10, 0.20, 0.40, 0.80),
                    rates=(0.0, 0.05, 0.10), k_axis: np.ndarray | None = None,
                    T_axis: np.ndarray | None = None, vega_floor: float = VEGA_FLOOR,
                    price_space: bool = True, return_grids: bool = False,
                    chunk: int = 8192) -> dict[str, Any]:
    """Static-arbitrage audit on the lattice sigmas x rates x T_axis x k_axis.

    Accepts the served ``PricingEngine`` (or a ``TeacherSurface`` around it), or an
    ``IVSurface``.  Returns a JSON-serialisable summary:

      iv_space.butterfly / .calendar : fraction of grid points violating, worst
          magnitude (min g, min dw/dT) and its location, on the IV-defined points
          and on the vega-resolved sub-region, with per-bucket shares;
      price_space (engine only): fractions and worst values for dC/dK in
          [-e^{-rT}, 0], d2C/dK2 >= 0, the 1%-wide butterfly in bps of strike,
          calendar monotonicity of C/S at fixed k and of C at fixed K, and the
          price bounds (below intrinsic / above spot) in bps of strike;
      grids (optional): the raw per-slice arrays.
    """
    k_axis = default_k_axis() if k_axis is None else np.asarray(k_axis, dtype=np.float64)
    T_axis = default_T_axis() if T_axis is None else np.asarray(T_axis, dtype=np.float64)
    if isinstance(engine_or_surface, PricingEngine):
        surf: TeacherSurface | IVSurface = TeacherSurface(engine_or_surface)
    else:
        surf = engine_or_surface
    is_teacher = isinstance(surf, TeacherSurface)
    t0 = time.perf_counter()

    TT, KK = np.meshgrid(T_axis, k_axis, indexing="ij")
    n_slice = TT.size
    cols: dict[str, list[np.ndarray]] = {key: [] for key in
                                         ("k", "T", "sigma", "rate", "w", "g", "w_T",
                                          "defined", "capped", "vega", "price")}
    pcols: dict[str, list[np.ndarray]] = {}
    grids: list[dict[str, Any]] = []
    for s in sigmas:
        for rr in rates:
            kt = torch.from_numpy(KK.ravel())
            Tt = torch.from_numpy(TT.ravel())
            st = torch.full_like(kt, float(s))
            rt = torch.full_like(kt, float(rr))
            parts: dict[str, list[torch.Tensor]] = {}
            for lo in range(0, n_slice, chunk):
                sl = slice(lo, lo + chunk)
                if is_teacher:
                    d = surface_derivatives(surf.total_variance, kt[sl], Tt[sl], st[sl], rt[sl])
                    _, info = surf.implied_vol(kt[sl], Tt[sl], st[sl], rt[sl],
                                               differentiable=False)
                    d.update({"defined": info["defined"], "capped": info["capped"],
                              "vega": info["vega"], "price": info["price"].detach()})
                    if price_space:
                        pc = surf.price_space_conditions(kt[sl], Tt[sl], st[sl], rt[sl])
                        for key, val in pc.items():
                            parts.setdefault("ps_" + key, []).append(val)
                else:
                    d = surface_derivatives(surf.total_variance, kt[sl], Tt[sl], st[sl], rt[sl])
                    with torch.no_grad():
                        w = d["w"]
                        m = torch.exp(-(kt[sl] + rt[sl] * Tt[sl])).float()
                        siv = torch.sqrt(w / Tt[sl].float())
                        vega = bs_vega_unit(m, Tt[sl].float(), siv, rt[sl].float())
                        d.update({"defined": torch.ones_like(w, dtype=torch.bool),
                                  "capped": torch.zeros_like(w, dtype=torch.bool),
                                  "vega": vega,
                                  "price": bs_call_from_w(kt[sl].float(), Tt[sl].float(),
                                                          w, rt[sl].float())})
                for key in ("w", "g", "w_T", "defined", "capped", "vega", "price"):
                    parts.setdefault(key, []).append(d[key])
            slice_arrays = {key: torch.cat(v).numpy().astype(np.float64)
                            for key, v in parts.items()}
            for key in ("w", "g", "w_T", "defined", "capped", "vega", "price"):
                cols[key].append(slice_arrays[key])
            for key, v in slice_arrays.items():
                if key.startswith("ps_"):
                    pcols.setdefault(key, []).append(v)
            cols["k"].append(KK.ravel()); cols["T"].append(TT.ravel())
            cols["sigma"].append(np.full(n_slice, s)); cols["rate"].append(np.full(n_slice, rr))
            if return_grids:
                grids.append({"sigma": float(s), "rate": float(rr),
                              **{key: v.reshape(TT.shape) for key, v in slice_arrays.items()}})

    A = {key: np.concatenate(v) for key, v in cols.items()}
    P = {key[3:]: np.concatenate(v) for key, v in pcols.items()}
    defined = A["defined"] > 0.5
    capped = A["capped"] > 0.5
    usable = defined & ~capped & np.isfinite(A["g"]) & np.isfinite(A["w_T"])
    resolved = usable & (A["vega"] >= vega_floor)
    n = int(A["k"].size)

    def _region(mask: np.ndarray, key: str, thresh: float = 0.0) -> dict[str, Any]:
        vals = A[key]
        viol = mask & (vals < thresh)
        out: dict[str, Any] = {"n_points": int(mask.sum()),
                               "n_violations": int(viol.sum()),
                               "violation_fraction": float(viol.sum() / max(mask.sum(), 1))}
        if mask.any():
            i = int(np.flatnonzero(mask)[np.argmin(vals[mask])])
            out["worst_value"] = float(vals[i])
            out["worst_at"] = _loc(A["T"][i], A["k"][i], A["sigma"][i], A["rate"][i])
            out["where"] = _bucket_fractions(viol[mask], A["k"][mask], A["T"][mask],
                                             A["sigma"][mask])
        return out

    report: dict[str, Any] = {
        "surface": "teacher_price_surrogate" if is_teacher else "constrained_iv_surface",
        "protocol": {"sigmas": [float(x) for x in sigmas], "rates": [float(x) for x in rates],
                     "n_k": int(k_axis.size), "k_min": float(k_axis.min()),
                     "k_max": float(k_axis.max()), "n_T": int(T_axis.size),
                     "T_days_min": float(T_axis.min() * TRADING_DAYS),
                     "T_days_max": float(T_axis.max() * TRADING_DAYS),
                     "n_points": n, "vega_floor": float(vega_floor),
                     "k_convention": "k = ln(K/F), F = S exp(rT)"},
        "coverage": {
            "iv_defined_fraction": float(defined.mean()),
            "iv_capped_fraction": float(capped.mean()),
            "usable_fraction": float(usable.mean()),
            "resolved_fraction": float(resolved.mean()),
            "unresolved_fraction_of_usable": float((usable & ~resolved).sum() / max(usable.sum(), 1)),
        },
        "iv_space": {
            "butterfly_all_defined": _region(usable, "g"),
            "butterfly_resolved": _region(resolved, "g"),
            "calendar_all_defined": _region(usable, "w_T"),
            "calendar_resolved": _region(resolved, "w_T"),
        },
    }
    if is_teacher:
        pb = A["price"]
        below = np.where(defined, 0.0, np.maximum(
            np.maximum(np.exp(-(A["k"] + A["rate"] * A["T"])) - np.exp(-A["rate"] * A["T"]), 0.0) - pb, 0.0))
        report["coverage"]["below_intrinsic_fraction"] = float((below > 0).mean())
        report["coverage"]["below_intrinsic_worst_bps"] = float(below.max() * 1e4)
        i = int(np.argmax(below))
        report["coverage"]["below_intrinsic_worst_at"] = _loc(A["T"][i], A["k"][i], A["sigma"][i], A["rate"][i])
    if is_teacher and P:
        allm = np.ones(n, dtype=bool)
        ps: dict[str, Any] = {}
        A_ps = {"d2C_dK2": P["d2C_dK2"], "dc_dT": P["dc_dT"],
                "dC_dT_fixK": P["dC_dT_fixK"], "butterfly_bps": P["butterfly"] * 1e4,
                "neg_dC_dK": -P["dC_dK"],
                "slope_margin": P["dC_dK"] + np.exp(-A["rate"] * A["T"]),
                "not_below_intrinsic_bps": -P["below_intrinsic"] * 1e4,
                "not_above_spot_bps": -P["above_spot"] * 1e4}
        saved = dict(A)
        A.update(A_ps)
        ps["convexity_d2C_dK2"] = _region(allm, "d2C_dK2")
        ps["convexity_resolved"] = _region(resolved, "d2C_dK2")
        ps["butterfly_1pct_bps"] = _region(allm, "butterfly_bps")
        ps["monotone_dC_dK_le_0"] = _region(allm, "neg_dC_dK")
        ps["slope_dC_dK_ge_-discount"] = _region(allm, "slope_margin")
        ps["calendar_fixed_k"] = _region(allm, "dc_dT")
        ps["calendar_fixed_k_resolved"] = _region(resolved, "dc_dT")
        ps["calendar_fixed_strike"] = _region(allm, "dC_dT_fixK")
        ps["price_below_intrinsic_bps"] = _region(allm, "not_below_intrinsic_bps")
        ps["price_above_spot_bps"] = _region(allm, "not_above_spot_bps")
        # agreement between the two formulations of the same condition
        both = usable
        sg = np.sign(A["g"][both]); sc = np.sign(P["d2C_dK2"][both])
        ps["sign_agreement_butterfly_vs_convexity"] = float((sg == sc).mean())
        sg2 = np.sign(A["w_T"][both]); sc2 = np.sign(P["dc_dT"][both])
        ps["sign_agreement_calendar_w_vs_price"] = float((sg2 == sc2).mean())
        A = saved
        report["price_space"] = ps
    report["elapsed_s"] = float(time.perf_counter() - t0)
    if return_grids:
        report["grids"] = grids
        report["arrays"] = A
        if P:
            report["price_arrays"] = P
    return report


# Training

def _sample_box(n: int, gen: torch.Generator, *, k_box=K_BOX, T_box=T_BOX,
                sigma_box=SIGMA_BOX, rate_box=RATE_BOX, z_share: float = 0.5,
                z_max: float = 8.0) -> tuple[torch.Tensor, ...]:
    """Random points in the box.  A `z_share` fraction is drawn uniformly in the
    standardised moneyness z = k/(sigma sqrt T) (clipped into the k box) so the
    narrow 1-day / 5%-vol smiles are sampled as densely as the wide ones."""
    u = torch.rand(n, 4, generator=gen, dtype=torch.float64)
    T = T_box[0] + (T_box[1] - T_box[0]) * u[:, 1]
    sigma = sigma_box[0] + (sigma_box[1] - sigma_box[0]) * u[:, 2]
    rate = rate_box[0] + (rate_box[1] - rate_box[0]) * u[:, 3]
    k_uniform = k_box[0] + (k_box[1] - k_box[0]) * u[:, 0]
    z = (2.0 * torch.rand(n, generator=gen, dtype=torch.float64) - 1.0) * z_max
    k_z = (z * sigma * torch.sqrt(T)).clamp(k_box[0], k_box[1])
    use_z = torch.rand(n, generator=gen, dtype=torch.float64) < z_share
    k = torch.where(use_z, k_z, k_uniform)
    return k, T, sigma, rate


def teacher_labels(teacher: TeacherSurface, k, T, sigma, r, chunk: int = 16384
                   ) -> dict[str, torch.Tensor]:
    """Teacher prices, implied vols and vegas (no gradient), in float64."""
    out: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for lo in range(0, k.shape[0], chunk):
            sl = slice(lo, lo + chunk)
            s, info = teacher.implied_vol(k[sl], T[sl], sigma[sl], r[sl],
                                          differentiable=False)
            for key, v in (("iv", s), ("price", info["price"]), ("vega", info["vega"]),
                           ("defined", info["defined"]), ("capped", info["capped"])):
                out.setdefault(key, []).append(v)
    return {key: torch.cat(v) for key, v in out.items()}


def train_iv_surface(teacher: TeacherSurface, *, n_train: int = 262_144,
                     n_val: int = 32_768, steps: int = 6000, batch: int = 4096,
                     penalty_batch: int = 4096, lr: float = 2e-3, width: int = 64,
                     depth: int = 4, lambda_but: float = 20.0, lambda_cal: float = 20.0,
                     lambda_lee: float = 1.0, margin: float = 2e-3,
                     vega_floor: float = VEGA_FLOOR, huber_delta: float = HUBER_DELTA,
                     penalty_margin_frac: float = 0.1, seed: int = 20260909,
                     log_every: int = 500, threads: int | None = 8,
                     log: Callable[[str], None] | None = print
                     ) -> tuple[IVSurface, dict[str, Any]]:
    """Fit the constrained surface to the teacher.  Returns (surface, history).

    `threads`: torch intra-op threads for the run (restored afterwards).  Measured on
    the 16-thread CPU this was developed on, a 4096 + 4096 step costs 58 ms at 8
    threads and 925 ms at 16.  The third-order autograd on a width-64 MLP is all
    small matmuls, and the extra threads only synchronise.
    """
    t0 = time.perf_counter()
    prev_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        return _train(teacher, t0, n_train=n_train, n_val=n_val, steps=steps, batch=batch,
                      penalty_batch=penalty_batch, lr=lr, width=width, depth=depth,
                      lambda_but=lambda_but, lambda_cal=lambda_cal, lambda_lee=lambda_lee,
                      margin=margin, vega_floor=vega_floor, huber_delta=huber_delta,
                      penalty_margin_frac=penalty_margin_frac, seed=seed,
                      log_every=log_every, log=log)
    finally:
        torch.set_num_threads(prev_threads)


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber loss with the quadratic/linear switch at |x| = delta (Huber 1964)."""
    a = x.abs()
    return torch.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def _train(teacher: TeacherSurface, t0: float, *, n_train, n_val, steps, batch,
           penalty_batch, lr, width, depth, lambda_but, lambda_cal, lambda_lee, margin,
           vega_floor, huber_delta, penalty_margin_frac, seed, log_every, log
           ) -> tuple[IVSurface, dict[str, Any]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    k, T, s, r = _sample_box(n_train, gen)
    kv, Tv, sv, rv = _sample_box(n_val, gen)
    lab = teacher_labels(teacher, k, T, s, r)
    labv = teacher_labels(teacher, kv, Tv, sv, rv)
    keep = lab["defined"] & ~lab["capped"]
    k, T, s, r = (t[keep].float() for t in (k, T, s, r))
    P = lab["price"][keep].float()
    n_keep = int(keep.sum())
    if log:
        log(f"labels: {n_train} sampled, {n_keep} with a price inside the "
            f"no-arbitrage bounds ({n_train - n_keep} dropped), "
            f"{time.perf_counter() - t0:.1f}s")

    net = IVSurfaceNet(width=width, depth=depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 5e-3)
    # penalty domain: the box widened by penalty_margin_frac on k and T so the
    # constraints hold up to and slightly beyond the edges that get audited
    dk = (K_BOX[1] - K_BOX[0]) * penalty_margin_frac
    dT = (T_BOX[1] - T_BOX[0]) * penalty_margin_frac
    pen_boxes = dict(k_box=(K_BOX[0] - dk, K_BOX[1] + dk),
                     T_box=(max(T_BOX[0] - dT, 0.5 / TRADING_DAYS), T_BOX[1] + dT))
    hist: list[dict[str, float]] = []
    vf = float(vega_floor)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_keep, (batch,), generator=gen)
        kb, Tb, sb, rb, Pb = k[idx], T[idx], s[idx], r[idx], P[idx]
        w = net(kb, Tb, sb, rb)
        Pn = bs_call_from_w(kb, Tb, w, rb)
        with torch.no_grad():
            m = torch.exp(-(kb + rb * Tb))
            vega = bs_vega_unit(m, Tb, torch.sqrt(w / Tb), rb).clamp_min(vf)
        # Huber on the vega-normalised residual (calibrate.py's convention).  Where
        # the teacher is wrong by more than `huber_delta` vol-point-equivalents
        # (its below-intrinsic kink region at low sigma sqrt(T)), the label's
        # influence is linear.  2 * Huber equals the squared residual inside the
        # delta, so `fit` reads as a mean squared vol-point error there.
        fit = 2.0 * _huber((Pn - Pb) / vega, huber_delta).mean()

        kp, Tp, sp, rp = (t.float() for t in _sample_box(penalty_batch, gen, **pen_boxes))
        d = surface_derivatives(net, kp, Tp, sp, rp, create_graph=True)
        but = (torch.relu(margin - d["g"]) ** 2).mean()
        cal = (torch.relu(margin - d["w_T"] / (sp * sp)) ** 2).mean()
        lee = (torch.relu(d["w_k"].abs() - 2.0) ** 2).mean()
        loss = fit + lambda_but * but + lambda_cal * cal + lambda_lee * lee
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1 or step == steps:
            with torch.no_grad():
                g_min = float(d["g"].min()); c_min = float((d["w_T"] / (sp * sp)).min())
            rec = {"step": step, "loss": float(loss.detach()), "fit": float(fit.detach()),
                   "butterfly_pen": float(but.detach()), "calendar_pen": float(cal.detach()),
                   "lee_pen": float(lee.detach()), "batch_g_min": g_min,
                   "batch_cal_min_over_sigma2": c_min,
                   "elapsed_s": time.perf_counter() - t0}
            hist.append(rec)
            if log:
                log(f"step {step:>5}/{steps}  fit {rec['fit']:.3e}  but {rec['butterfly_pen']:.2e} "
                    f"cal {rec['calendar_pen']:.2e} lee {rec['lee_pen']:.1e}  "
                    f"batch min g {g_min:+.4f}  min dTw/s2 {c_min:+.4f}  "
                    f"({rec['elapsed_s']:.0f}s)")

    surface = IVSurface(net)
    # fit metrics on the held-out sample, in the units of
    # docs/no_arbitrage_surface.md (vol points and bps of strike)
    val = _fit_metrics(surface, labv, kv, Tv, sv, rv, vega_floor)
    train_m = _fit_metrics(surface, lab, *(t.double() for t in (k, T, s, r)), vega_floor,
                           defined_mask=torch.ones(n_keep, dtype=torch.bool),
                           labels_already_kept=keep)
    surface.meta = {
        "kind": "iv_surface_0dte", "architecture": {"width": width, "depth": depth,
                                                    "features": "k,T,sigma,r,asinh(k/(sigma sqrtT))/3",
                                                    "prior": "sigma^2 T", "output": "softplus"},
        "ranges": {"k": list(K_BOX), "T": list(T_BOX), "sigma": list(SIGMA_BOX),
                   "rate": list(RATE_BOX), "k_convention": "ln(K/F), F = S exp(rT)",
                   "moneyness_box": [0.85, 1.15]},
        "dynamics": {key: teacher.meta.get(key) for key in ("H", "eta", "rho", "kernel",
                                                            "calibrated", "calibration_note")},
        "teacher": "artifacts/model_0dte.pt (5-member ensemble, European call price / K)",
        "penalties": {"lambda_butterfly": lambda_but, "lambda_calendar": lambda_cal,
                      "lambda_lee": lambda_lee, "margin": margin,
                      "calendar_normalisation": "dw/dT / sigma^2",
                      "penalty_domain": {key: [float(x) for x in v] for key, v in pen_boxes.items()},
                      "penalty_batch": penalty_batch, "resampled_every_step": True},
        "loss": ("Huber(delta=%.3f) on the vega-normalised price error, vega floored "
                 "at %.3f" % (huber_delta, vega_floor)),
        "training": {"n_train_sampled": n_train, "n_train_kept": n_keep, "n_val": n_val,
                     "steps": steps, "batch": batch, "lr": lr, "seed": seed,
                     "wall_s": time.perf_counter() - t0, "history": hist},
        "fit_metrics": {"validation": val, "train": train_m},
    }
    return surface, {"history": hist, "validation": val, "train": train_m,
                     "wall_s": time.perf_counter() - t0}


def _fit_metrics(surface: IVSurface, lab: dict[str, torch.Tensor], k, T, s, r,
                 vega_floor: float, defined_mask: torch.Tensor | None = None,
                 labels_already_kept: torch.Tensor | None = None) -> dict[str, float]:
    """IV RMSE (vol points) on resolved points, price RMSE (bps of strike) on all
    points with a defined teacher price, and the vega-normalised loss."""
    with torch.no_grad():
        kf, Tf, sf, rf = (t.float() for t in (k, T, s, r))
        w = surface.total_variance(kf, Tf, sf, rf).double()
        Pn = bs_call_from_w(k.double(), T.double(), w, r.double())
        iv_n = torch.sqrt(w / T.double())
        if labels_already_kept is not None:
            P_t = lab["price"][labels_already_kept]
            iv_t = lab["iv"][labels_already_kept]
            vega_t = lab["vega"][labels_already_kept]
            defined = torch.ones_like(P_t, dtype=torch.bool)
            capped = lab["capped"][labels_already_kept]
        else:
            P_t, iv_t, vega_t = lab["price"], lab["iv"], lab["vega"]
            defined, capped = lab["defined"], lab["capped"]
        usable = defined & ~capped
        resolved = usable & (vega_t >= vega_floor)
        well = usable & (vega_t >= 2.5 * vega_floor)
        dp = (Pn - P_t) * 1e4
        div = (iv_n - iv_t) * 100.0
        m = torch.exp(-(k.double() + r.double() * T.double()))
        vega_n = bs_vega_unit(m, T.double(), iv_n, r.double()).clamp_min(vega_floor)
        out = {
            "n": int(k.shape[0]), "n_usable": int(usable.sum()),
            "n_resolved": int(resolved.sum()), "n_well_resolved": int(well.sum()),
            "price_rmse_bps": float(torch.sqrt((dp[usable] ** 2).mean())),
            "price_mae_bps": float(dp[usable].abs().mean()),
            "price_p95_abs_bps": float(dp[usable].abs().quantile(0.95)),
            "price_max_abs_bps": float(dp[usable].abs().max()),
            "iv_rmse_volpts_resolved": float(torch.sqrt((div[resolved] ** 2).mean())),
            "iv_mae_volpts_resolved": float(div[resolved].abs().mean()),
            "iv_p95_abs_volpts_resolved": float(div[resolved].abs().quantile(0.95)),
            "iv_max_abs_volpts_resolved": float(div[resolved].abs().max()),
            "iv_rmse_volpts_well_resolved": float(torch.sqrt((div[well] ** 2).mean())),
            "vega_normalised_rmse_volpts": float(
                100.0 * torch.sqrt((((Pn - P_t) / vega_n)[usable] ** 2).mean())),
        }
    return out
