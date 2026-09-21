"""Serving engine: neural pricing, autograd Greeks, and benchmarking.

The network prices the unit-strike call as a function of
(m = S/K, T, sigma, r). Strikes, puts and Greeks follow from exact identities
applied to that output:

- Any strike:      C(S, K, ...) = K * f(S/K, ...)          (homogeneity)
- Puts:            P = C - exp(-rT) * (E[A] - K)           (Asian parity;
                   European parity at or below ZERO_DTE_CUTOFF)
- Greeks:          reverse-mode autograd through f and the parity term,
                   so delta/gamma/vega/theta/rho are analytic derivatives of
                   the surrogate.
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

# Days per year on the engine's clock. Maturities are in trading days
# throughout: the Asian fixings are trading days, ZERO_DTE_CUTOFF is twelve of
# them, and the dashboard shows T * 252. Theta is quoted per trading day so it
# shares that clock (a 365-day theta would differ by 365/252 = 1.45x).
# Calendar time enters only in calibrate.py, which reads ACT/365 timestamps.
TRADING_DAYS_PER_YEAR = 252.0


class PricingEngine:
    def __init__(self, checkpoint: Path | None = None):
        checkpoint = checkpoint or ARTIFACTS / "model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No model checkpoint at {checkpoint}. "
                "Train one first: python -m backend.quant.train")
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        # A checkpoint holds an ensemble under "members" or one "state_dict".
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

        # Fixed output scale. The Softplus head emits a quantity of order 1
        # and the price magnitude is carried here. An unscaled head starts at
        # softplus(0) = 0.693, i.e. 6,930 bps against a mean price of 3,664
        # bps; with the scale the systematic price bias is +0.467 bps against
        # +0.985 bps. Legacy checkpoints carry no output_scale and get 1.0.
        self._output_scale = float(self.meta.get("output_scale", 1.0))

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

        # Mixed batch: route per element, so each maturity is priced by the
        # model that owns it and price_batch applies that model's parity
        # relation (Asian above the cutoff, European at or below). Pinned by
        # tests/test_regression.py::test_mixed_maturity_batch_matches_scalar.
        return torch.where(mask,
                           self._zero_dte_call(m, mat, sig, r, member),
                           self._asian_call(m, mat, sig, r, member))

    def _asian_call(self, m, mat, sig, r, member: int | None = None):
        x = torch.stack([m, mat, sig, r], dim=-1)
        xn = 2.0 * (x - self._lows) / (self._highs - self._lows) - 1.0
        if member is not None:
            return self._output_scale * self.members[member](xn)
        # The scale is shared by every member, so scaling the ensemble mean is
        # identical to scaling each member first, and autograd is unaffected
        # (a constant factor passes straight through to the Greeks).
        return self._output_scale * torch.stack(
            [net(xn) for net in self.members]).mean(dim=0)

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

        E[A]/K = (m/n) * sum_{i=1..n} exp(r t_i), t_i = i * T / n, evaluated
        as the sum. Its geometric-series closed form
        m * e^{r dt} * expm1(rT) / (n * expm1(r dt)) is 0/0 at r = 0, a rate
        the API accepts, and a guard that holds r away from zero (a clamp, a
        floor, an epsilon) has zero derivative there, which removes the
        parity contribution from rho and leaves a put reporting its call's
        rho. The sum is analytic in r at every rate. At r = 0 the term
        reduces to m - 1 and its derivative in r is
        -T * (m - 1) + m * T * (n + 1) / (2n): the discount factor's
        sensitivity, which vanishes at the money, plus that of E[A]. That
        derivative is the gap between a put's rho and a call's. Accumulated
        in float64 and cast back, so summing n terms costs no precision
        against the closed form.
        """
        n = self.n_steps
        m64, mat64, r64 = (m.to(torch.float64), mat.to(torch.float64),
                           r.to(torch.float64))
        steps = torch.arange(1, n + 1, dtype=torch.float64, device=m.device)
        t = (mat64 / n).unsqueeze(-1) * steps          # fixing dates t_i
        ea = m64 * torch.exp(r64.unsqueeze(-1) * t).mean(dim=-1)
        return (torch.exp(-r64 * mat64) * (ea - 1.0)).to(m.dtype)

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
                # per trading day, the unit every maturity here is quoted in
                "theta": -strike * df_dmat.item() / TRADING_DAYS_PER_YEAR,
                "rho": strike * df_dr.item() / 100.0,  # per rate point
            },
        }

    # Rows fed through the ensemble per block in price_batch. Every row is
    # priced independently, so blocking is exact. Unblocked, a 50,000-point
    # batch transiently holds ~110 MB of (batch, width) float32 activations
    # across the 5 members, most of the request headroom on the 512 MB
    # container this serves from.
    _BATCH_CHUNK = 8_192

    @torch.no_grad()
    def price_batch(self, spots: np.ndarray, strikes: np.ndarray,
                    maturities: np.ndarray, sigmas: np.ndarray,
                    rates: np.ndarray, option_type: str = "call",
                    member: int | None = None) -> np.ndarray:
        out = np.empty(spots.shape[0], dtype=np.float32)
        for lo in range(0, spots.shape[0], self._BATCH_CHUNK):
            hi = lo + self._BATCH_CHUNK
            m = torch.from_numpy((spots[lo:hi] / strikes[lo:hi])
                                 .astype(np.float32))
            mat = torch.from_numpy(maturities[lo:hi].astype(np.float32))
            sig = torch.from_numpy(sigmas[lo:hi].astype(np.float32))
            r = torch.from_numpy(rates[lo:hi].astype(np.float32))
            f = self._call_price_torch(m, mat, sig, r, member=member)
            if option_type == "put":
                if self.has_0dte:
                    mask_0dte = mat <= ZERO_DTE_CUTOFF
                    adj_asian = self._parity_adjustment_torch(m, mat, r)
                    adj_euro = m - torch.exp(-r * mat)  # P = C - (S - Ke^-rT)
                    adj = torch.where(mask_0dte, adj_euro, adj_asian)
                    f = f - adj
                else:
                    f = f - self._parity_adjustment_torch(m, mat, r)
            out[lo:hi] = (torch.from_numpy(strikes[lo:hi].astype(np.float32))
                          * f).numpy()
        return np.maximum(out, 0.0)

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
        """True when the request lies inside the serving model's trained box.

        (m, T, sigma, r) is checked against the 0DTE box, which reaches down
        to 1/252, for maturities at or below ZERO_DTE_CUTOFF, and against the
        Asian box for longer ones. Maturities in (12/252, 0.05) are above the
        0DTE cutoff and below the Asian surrogate's training floor. Neither
        model covers them and this returns False.

        The pricing methods do not call this. Requests are gated by the
        caller (validate_domain in backend/api/main.py).
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
