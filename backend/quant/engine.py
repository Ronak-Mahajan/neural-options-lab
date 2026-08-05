"""Serving engine: neural pricing, autograd Greeks, and benchmarking.

The network prices the *unit-strike call* as a function of
(m = S/K, T, sigma, r). Everything else is exact math on top:

- Any strike:      C(S, K, ...) = K * f(S/K, ...)          (homogeneity)
- Puts:            P = C - exp(-rT) * (E[A] - K)           (Asian parity)
- Greeks:          reverse-mode autograd through f *and* the parity term,
                   so delta/gamma/vega/theta/rho are analytic derivatives of
                   the surrogate, not finite differences.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TypeVar, Any, Callable, cast

import numpy as np
import torch

from .model import AsianPricerNet
from .monte_carlo import price_asian_mc, MCResult

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

# Maturity at or below which the 0DTE (European, rough-Bergomi) surrogate serves.
ZERO_DTE_CUTOFF = 12.0 / 252.0


class PricingEngine:
    def __init__(self, checkpoint: Path | None = None):
        checkpoint = checkpoint or ARTIFACTS / "model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No model checkpoint at {checkpoint}. "
                "Train one first: python -m backend.quant.train")
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        # New checkpoints hold an ensemble; old single-model ones still load.
        states = blob["members"] if "members" in blob else [blob["state_dict"]]
        self.members: list[AsianPricerNet] = []
        for state in states:
            model = AsianPricerNet(width=self.meta["width"],
                                   n_blocks=self.meta["blocks"])
            model.load_state_dict(state)
            model.eval()
            self.members.append(model)
        self.n_members = len(self.members)
        self.n_steps = self.meta["n_monitoring_steps"]
        ranges = self.meta["param_ranges"]
        self._lows = torch.tensor([lo for lo, _ in ranges.values()],
                                  dtype=torch.float32)
        self._highs = torch.tensor([hi for _, hi in ranges.values()],
                                   dtype=torch.float32)

        # Load 0DTE surrogate if available
        self.has_0dte = False
        ckpt_0dte = ARTIFACTS / "model_0dte.pt"
        if ckpt_0dte.exists():
            self.has_0dte = True
            blob_0dte = torch.load(ckpt_0dte, map_location="cpu", weights_only=False)
            # dynamics this surrogate was trained under; the API's MC
            # benchmark must simulate the same measure
            self.meta_0dte = blob_0dte.get("meta", {})
            self._0dte_members = []
            for state in blob_0dte["members"]:
                m = AsianPricerNet(width=128, n_blocks=4)
                m.load_state_dict(state)
                m.eval()
                self._0dte_members.append(m)
            self._0dte_lows = torch.tensor([0.85, 1/252.0, 0.05, 0.0], dtype=torch.float32)
            self._0dte_highs = torch.tensor([1.15, 12/252.0, 0.80, 0.10], dtype=torch.float32)

    # ------------------------------------------------------------- internals
    def _call_price_torch(self, m, mat, sig, r,
                          member: int | None = None) -> torch.Tensor:
        """Unit-strike call price/K as a differentiable torch graph.

        member=None averages the full ensemble (the production path);
        member=i evaluates a single member (used for error-attribution
        comparisons). Autograd flows through the mean either way.
        """
        if not self.has_0dte:
            return self._asian_call(m, mat, sig, r, member)

        mask = mat <= ZERO_DTE_CUTOFF
        if bool(mask.all()):
            return self._zero_dte_call(m, mat, sig, r, member)
        if not bool(mask.any()):
            return self._asian_call(m, mat, sig, r, member)

        # Mixed batch. This used to gate on .all(), so a batch spanning the
        # cutoff sent EVERY element through the Asian net — including
        # maturities below its 0.05 training floor, silently extrapolated —
        # while price_batch then applied EUROPEAN parity to the short-dated
        # ones. Two different parity relations on one price vector. Route
        # per element instead, so each maturity is priced by the model that
        # owns it and receives the matching parity relation.
        return torch.where(mask,
                           self._zero_dte_call(m, mat, sig, r, member),
                           self._asian_call(m, mat, sig, r, member))

    def _asian_call(self, m, mat, sig, r, member: int | None = None):
        x = torch.stack([m, mat, sig, r], dim=-1)
        xn = 2.0 * (x - self._lows) / (self._highs - self._lows) - 1.0
        if member is not None:
            return self.members[member](xn)
        return torch.stack([net(xn) for net in self.members]).mean(dim=0)

    def _zero_dte_call(self, m, mat, sig, r, member: int | None = None):
        x = torch.stack([m, mat, sig, r], dim=-1)
        xn = 2.0 * (x - self._0dte_lows) \
            / (self._0dte_highs - self._0dte_lows) - 1.0
        if member is not None:
            idx = min(member, len(self._0dte_members) - 1)
            return self._0dte_members[idx](xn).squeeze(-1)
        return torch.stack([net(xn).squeeze(-1)
                            for net in self._0dte_members]).mean(dim=0)

    def _parity_adjustment_torch(self, m, mat, r) -> torch.Tensor:
        """exp(-rT) * (E[A]/K - 1) with spot=m, strike=1, differentiable.

        E[A]/K = (m/n) * sum_i exp(r t_i) =
        m * e^{r dt} expm1(rT) / (n expm1(r dt)).
        r is clamped away from 0 to keep the geometric series well-defined;
        the induced rho error below r=1e-6 is negligible.
        """
        n = self.n_steps
        r_safe = torch.clamp(r, min=1e-6)
        dt = mat / n
        ea = m * torch.exp(r_safe * dt) * torch.expm1(r_safe * mat) \
            / (n * torch.expm1(r_safe * dt))
        return torch.exp(-r_safe * mat) * (ea - 1.0)

    # ------------------------------------------------------------ public API
    def price_with_greeks(self, spot: float, strike: float, maturity: float,
                          sigma: float, rate: float,
                          option_type: str = "call",
                          member: int | None = None) -> dict:
        """Price + full first-order Greeks (and gamma) via autograd."""
        m = torch.tensor(spot / strike, requires_grad=True)
        mat = torch.tensor(float(maturity), requires_grad=True)
        sig = torch.tensor(float(sigma), requires_grad=True)
        r = torch.tensor(float(rate), requires_grad=True)

        f = self._call_price_torch(m, mat, sig, r, member=member)
        if option_type == "put":
            if self.has_0dte and float(maturity) <= ZERO_DTE_CUTOFF:
                # European Put Parity: P/K = C/K - S/K + e^-rT
                f = f - m + torch.exp(-r * mat)
            else:
                f = f - self._parity_adjustment_torch(m, mat, r)
        price = strike * f

        # First-order sensitivities; keep the graph alive for gamma.
        (df_dm,) = torch.autograd.grad(f, m, create_graph=True)
        (d2f_dm2,) = torch.autograd.grad(df_dm, m, retain_graph=True)
        df_dmat, df_dsig, df_dr = torch.autograd.grad(f, (mat, sig, r))

        return {
            "price": max(price.item(), 0.0),
            "greeks": {
                # dC/dS = K * f_m * dm/dS = f_m
                "delta": df_dm.item(),
                "gamma": d2f_dm2.item() / strike,
                "vega": strike * df_dsig.item() / 100.0,  # per vol point
                "theta": -strike * df_dmat.item() / 365.0,  # per calendar day
                "rho": strike * df_dr.item() / 100.0,  # per rate point
            },
        }

    @torch.no_grad()
    def price_batch(self, spots: np.ndarray, strikes: np.ndarray,
                    maturities: np.ndarray, sigmas: np.ndarray,
                    rates: np.ndarray, option_type: str = "call",
                    member: int | None = None) -> np.ndarray:
        m = torch.from_numpy((spots / strikes).astype(np.float32))
        mat = torch.from_numpy(maturities.astype(np.float32))
        sig = torch.from_numpy(sigmas.astype(np.float32))
        r = torch.from_numpy(rates.astype(np.float32))
        f = self._call_price_torch(m, mat, sig, r, member=member)
        if option_type == "put":
            if self.has_0dte:
                mask_0dte = mat <= ZERO_DTE_CUTOFF
                adj_asian = self._parity_adjustment_torch(m, mat, r)
                adj_euro = m - torch.exp(-r * mat) # P = C - (S - Ke^-rT)
                adj = torch.where(mask_0dte, adj_euro, adj_asian)
                f = f - adj
            else:
                f = f - self._parity_adjustment_torch(m, mat, r)
        return np.maximum((torch.from_numpy(strikes.astype(np.float32)) * f)
                          .numpy(), 0.0)

    def mc_price(self, spot: float, strike: float, maturity: float,
                 sigma: float, rate: float, n_paths: int,
                 option_type: str = "call", seed: int | None = None,
                 control_variate: bool = True) -> MCResult:
        return price_asian_mc(spot, strike, maturity, sigma, rate,
                              n_paths=n_paths, n_steps=self.n_steps,
                              option_type=option_type, seed=seed,
                              control_variate=control_variate)

    def in_domain(self, spot: float, strike: float, maturity: float,
                  sigma: float, rate: float) -> bool:
        """Is this request inside the trained box of whichever model serves it?

        Previously this checked only the Asian box, so a legitimate 0DTE query
        (maturity below 12/252, which the 0DTE surrogate covers down to 1/252)
        was reported out-of-domain. It also went uncalled by price_batch, so
        nothing actually guarded extrapolation.

        Note the gap between the two boxes: maturities in (12/252, 0.05) are
        above the 0DTE cutoff but below the Asian surrogate's training floor,
        so NO model is valid there and this returns False.
        """
        x = np.array([spot / strike, maturity, sigma, rate])
        if self.has_0dte and maturity <= ZERO_DTE_CUTOFF:
            lo, hi = self._0dte_lows.numpy(), self._0dte_highs.numpy()
        else:
            lo, hi = self._lows.numpy(), self._highs.numpy()
        return bool(np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9))


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

T = TypeVar('T')


def time_call(fn: Callable[..., T], *args: Any, repeats: int = 3,
              **kwargs: Any) -> tuple[float, T]:
    """Best-of-N wall time in ms, plus the last return value."""
    best = math.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, cast(T, out)
