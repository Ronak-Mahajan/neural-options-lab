"""Explainable AI: Integrated Gradients through the pricing ensemble.

Attribution of the neural price to its inputs (spot/moneyness, maturity,
volatility, rate), relative to a "minimal option" baseline: at-the-money,
minimum maturity and volatility, zero rate. Integrated Gradients
(Sundararajan et al., 2017) integrates the model gradient along the straight
line from baseline to target:

    IG_j = (x_j - b_j) * integral_0^1  dF/dx_j (b + a (x - b))  da

approximated with a midpoint Riemann sum, batched into a single forward pass
through the ensemble. IG satisfies *completeness* - the attributions sum to
F(x) - F(b) - which we compute and return as the compliance check
(`completeness_error` should be ~1e-3 of the price or less).

Everything runs through the engine's normalized forward (and the put-parity
term for puts), so attributions are exact for the model actually served.
"""

from __future__ import annotations

import torch

from .dataset import PARAM_RANGES
from .engine import PricingEngine

# Baseline: the least-optiony option in the trained domain.
_BASELINE = {
    "moneyness": 1.0,
    "maturity": PARAM_RANGES["maturity"][0],   # 0.05y
    "sigma": PARAM_RANGES["sigma"][0],         # 5% vol
    "rate": 0.0,
}


def integrated_gradients(engine: PricingEngine, spot: float, strike: float,
                         maturity: float, sigma: float, rate: float,
                         option_type: str = "call", steps: int = 96) -> dict:
    # The engine routes T <= 12/252 to the 0DTE rough-vol surrogate. The IG
    # integration path must stay inside one regime (attributions across a
    # model switch are meaningless), so the baseline moves with the target:
    # 0DTE targets get a 1-day ATM baseline, everything else the 0.05y one.
    is_0dte = maturity <= 12.0 / 252.0 and getattr(engine, "has_0dte", False)
    baseline = dict(_BASELINE)
    if is_0dte:
        baseline["maturity"] = 1.0 / 252.0

    target = torch.tensor([spot / strike, maturity, sigma, rate],
                          dtype=torch.float32)
    base = torch.tensor([baseline["moneyness"], baseline["maturity"],
                         baseline["sigma"], baseline["rate"]],
                        dtype=torch.float32)

    def price_of(x: torch.Tensor) -> torch.Tensor:
        """Ensemble price/K (incl. put parity) for a (..., 4) input batch."""
        m, mat, sig, r = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        f = engine._call_price_torch(m, mat, sig, r)
        if option_type == "put":
            if is_0dte:  # European parity in the 0DTE regime
                f = f - m + torch.exp(-r * mat)
            else:
                f = f - engine._parity_adjustment_torch(m, mat, r)
        return f

    # Midpoint Riemann sum over the straight-line path, one batched pass.
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    path = base + alphas[:, None] * (target - base)        # (steps, 4)
    path.requires_grad_(True)
    (grads,) = torch.autograd.grad(price_of(path).sum(), path)
    avg_grads = grads.mean(dim=0)                          # (4,)
    attributions = (target - base) * avg_grads * strike

    with torch.no_grad():
        target_price = float(price_of(target)) * strike
        base_price = float(price_of(base)) * strike

    attr = {
        "spot": round(float(attributions[0]), 6),
        "maturity": round(float(attributions[1]), 6),
        "sigma": round(float(attributions[2]), 6),
        "rate": round(float(attributions[3]), 6),
    }
    return {
        "attributions": attr,
        "baseline": baseline,
        "regime": "0dte_rough_bergomi" if is_0dte else "asian_gbm",
        "baseline_price": round(base_price, 6),
        "target_price": round(max(target_price, 0.0), 6),
        "completeness_error": round(
            sum(attr.values()) - (target_price - base_price), 6),
        "ig_steps": steps,
    }
