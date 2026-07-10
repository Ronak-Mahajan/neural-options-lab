"""Deep Hedging (Buehler, Gonon, Teichmann & Wood, 2019).

We hedge a SHORT 30-day at-the-money European call, rebalancing daily in the
underlying only, under proportional transaction costs. Two agents run on the
same simulated paths:

    Standard delta hedge — holds the exact Black-Scholes delta each day
                           (the textbook strategy; optimal only in the
                           frictionless, continuous limit)
    Deep hedge           — a policy network state -> holding, trained to
                           minimize CVaR_95 of the terminal hedging loss,
                           transaction costs inside the objective

Policy state is (tau, S/K, h_prev, sigma, r, cost): the network is trained
*conditionally* over a box of (sigma, r, cost) sampled per path, so one
offline training run serves any ticker's live parameters — and lets the
dashboard expose a transaction-cost slider.

CVaR is optimized in the Rockafellar-Uryasev form

    CVaR_a(L) = min_w  w + E[(L - w)+] / (1 - a)

with w produced by a small head conditioned on (sigma, r, cost), which keeps
the whole objective a single differentiable expectation.

P&L accounting (strike units, S0 = K = 1, N daily steps of dt = 1/252):
    cash_0 = premium - h_0 S_0 - c |h_0| S_0
    cash_{i+1} = cash_i e^{r dt} - (h_{i+1} - h_i) S_{i+1}
                 - c |h_{i+1} - h_i| S_{i+1}
    PL = cash_N e^{r dt} + h_N S_N - c |h_N| S_N - (S_N - K)+

Train:  python -m backend.quant.hedging          (~2 min CPU)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch
import torch.nn as nn
from scipy.stats import norm

from backend.quant.generative import PathGenerator, risk_neutralize

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

N_STEPS = 30                 # daily rebalances over the 30-day horizon
DT = 1.0 / 252.0
MATURITY = N_STEPS * DT
CVAR_ALPHA = 0.95
TRAIN_BOX = {"sigma": (0.08, 0.65), "rate": (0.0, 0.09),
             "cost": (0.0, 0.02)}


# ---------------------------------------------------------------------------
# Black-Scholes building blocks (exact baseline)
# ---------------------------------------------------------------------------

def bs_call_price(spot, strike, tau, sigma, rate):
    tau = np.maximum(tau, 1e-12)
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    d2 = d1 - sd
    return spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)


def bs_call_delta(spot, strike, tau, sigma, rate):
    tau = np.maximum(tau, 1e-12)
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class HedgePolicy(nn.Module):
    """(tau, S/K, h_prev, sigma, r, cost) -> holding in [0, 1.5]."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return 1.5 * torch.sigmoid(self.net(state)).squeeze(-1)


class CVaRHead(nn.Module):
    """(sigma, r, cost) -> the RU quantile variable w."""

    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.SiLU(),
                                 nn.Linear(width, 1))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).squeeze(-1)


# ---------------------------------------------------------------------------
# Differentiable hedging episode
# ---------------------------------------------------------------------------

def _torch_bs_premium(sigma: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """ATM BS call premium (S=K=1) for per-path (sigma, rate), in torch."""
    sd = sigma * math.sqrt(MATURITY)
    d1 = (rate + 0.5 * sigma ** 2) * MATURITY / sd
    d2 = d1 - sd
    n = torch.distributions.Normal(0.0, 1.0)
    return n.cdf(d1) - torch.exp(-rate * MATURITY) * n.cdf(d2)


def simulate_pl(policy: HedgePolicy, log_returns: torch.Tensor, sigma: torch.Tensor,
                rate: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """Terminal hedging P&L for a batch of paths (differentiable).

    log_returns: (B, N) log returns; sigma/rate/cost: (B,).
    """
    b = log_returns.shape[0]
    growth = torch.exp(rate * DT)
    premium = _torch_bs_premium(sigma, rate)

    spot = torch.ones(b)
    cash = premium.clone()
    h = torch.zeros(b)
    for i in range(N_STEPS):
        tau = torch.full((b,), (N_STEPS - i) * DT)
        state = torch.stack([tau / MATURITY, spot, h, sigma, rate, cost],
                            dim=-1)
        h_new = policy(state)
        trade = h_new - h
        cash = cash - trade * spot - cost * trade.abs() * spot
        h = h_new
        spot = spot * torch.exp(log_returns[:, i])
        cash = cash * growth
    payoff = torch.clamp(spot - 1.0, min=0.0)
    return cash + h * spot - cost * h.abs() * spot - payoff


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(iters: int = 6000, batch: int = 2048, lr: float = 1e-3,
          seed: int = 21) -> None:
    torch.manual_seed(seed)
    policy = HedgePolicy()
    head = CVaRHead()
    opt = torch.optim.AdamW(list(policy.parameters())
                            + list(head.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    generator = PathGenerator()
    gen_ckpt = ARTIFACTS / "generator.pt"
    if gen_ckpt.exists():
        blob = torch.load(gen_ckpt, map_location="cpu", weights_only=True)
        generator.load_state_dict(blob["generator"])
        print("Loaded WGAN-GP market simulator.")
    else:
        print("Warning: WGAN-GP not found, using untrained generator.")
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False

    t0 = time.perf_counter()
    for it in range(1, iters + 1):
        # One (sigma, rate, cost) triple per mini-batch: the cross-sectional
        # standardization in risk_neutralize is only exact for a homogeneous
        # batch (mixing regimes leaves per-path residual drift the policy
        # would learn to exploit). Conditional coverage of the training box
        # comes from iterating thousands of batches.
        sigma = torch.full((batch,),
                           float(torch.empty(1).uniform_(*TRAIN_BOX["sigma"])))
        rate = torch.full((batch,),
                          float(torch.empty(1).uniform_(*TRAIN_BOX["rate"])))
        cost = torch.full((batch,),
                          float(torch.empty(1).uniform_(*TRAIN_BOX["cost"])))
        z = torch.randn(batch, generator.noise_dim)

        # GAN paths, drift/vol-corrected onto the pricing measure — the
        # policy must learn hedging skill, not the generator's drift bias.
        with torch.no_grad():
            raw = generator(z, sigma.unsqueeze(-1), rate.unsqueeze(-1))
        log_returns = risk_neutralize(raw, sigma.unsqueeze(-1),
                                      rate.unsqueeze(-1))

        pl = simulate_pl(policy, log_returns, sigma, rate, cost)
        loss_var = -pl                                    # hedging shortfall
        w = head(torch.stack([sigma, rate, cost], dim=-1))
        cvar = (w + torch.clamp(loss_var - w, min=0.0)
                / (1.0 - CVAR_ALPHA)).mean()

        opt.zero_grad(set_to_none=True)
        cvar.backward()
        opt.step()
        sched.step()

        if it % 500 == 0 or it == 1:
            with torch.no_grad():
                print(f"iter {it:>5}/{iters}  "
                      f"CVaR objective {cvar.item():+.5f}  "
                      f"mean PL {pl.mean().item():+.5f}  "
                      f"({time.perf_counter() - t0:5.1f}s)", flush=True)

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "meta": {"n_steps": N_STEPS, "maturity": MATURITY,
                 "cvar_alpha": CVAR_ALPHA, "train_box": TRAIN_BOX,
                 "iters": iters, "batch": batch,
                 "train_seconds": round(time.perf_counter() - t0, 1)},
    }, ARTIFACTS / "hedger.pt")
    print(f"saved {ARTIFACTS / 'hedger.pt'}")


# ---------------------------------------------------------------------------
# Inference: deep hedge vs delta hedge on common paths
# ---------------------------------------------------------------------------

class HedgingEngine:
    def __init__(self, checkpoint: Path | None = None):
        checkpoint = checkpoint or ARTIFACTS / "hedger.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No hedger checkpoint at {checkpoint}. "
                "Train one first: python -m backend.quant.hedging")
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        self.policy = HedgePolicy()
        self.policy.load_state_dict(blob["policy"])
        self.policy.eval()

        self.generator = PathGenerator()
        gen_blob = torch.load(ARTIFACTS / "generator.pt", map_location="cpu", weights_only=True)
        self.generator.load_state_dict(gen_blob["generator"])
        self.generator.eval()

    @torch.no_grad()
    def compare(self, sigma: float, rate: float, cost: float,
                n_paths: int = 3000, seed: int = 17) -> dict:
        """Run both strategies over the same GBM paths; report P&L in
        strike units (multiply by spot/strike outside for dollars)."""
        sigma_c = float(np.clip(sigma, *TRAIN_BOX["sigma"]))
        rate_c = float(np.clip(rate, *TRAIN_BOX["rate"]))
        cost_c = float(np.clip(cost, *TRAIN_BOX["cost"]))

        premium = float(bs_call_price(1.0, 1.0, MATURITY, sigma_c, rate_c))
        growth = math.exp(rate_c * DT)

        # simulate spot paths once (shared by both strategies) using the
        # GAN, drift/vol-corrected onto the pricing measure; seeded so the
        # dashboard comparison is reproducible
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n_paths, self.generator.noise_dim, generator=gen)
        sigma_t = torch.full((n_paths, 1), sigma_c, dtype=torch.float32)
        rate_t = torch.full((n_paths, 1), rate_c, dtype=torch.float32)
        with torch.no_grad():
            raw = self.generator(z, sigma_t, rate_t)
            log_incr = risk_neutralize(raw, sigma_t, rate_t) \
                .numpy().astype(np.float64)

        spots = np.empty((n_paths, N_STEPS + 1))
        spots[:, 0] = 1.0
        spots[:, 1:] = np.exp(np.cumsum(log_incr, axis=1))

        def run(holdings_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            cash = np.full(n_paths, premium)
            h = np.zeros(n_paths)
            hist = np.empty((n_paths, N_STEPS))
            costs = np.zeros(n_paths)
            for i in range(N_STEPS):
                tau = (N_STEPS - i) * DT
                h_new = holdings_fn(i, tau, spots[:, i], h)
                trade_cost = cost_c * np.abs(h_new - h) * spots[:, i]
                cash -= (h_new - h) * spots[:, i] + trade_cost
                costs += trade_cost
                h = h_new
                hist[:, i] = h
                cash *= growth
            final_cost = cost_c * np.abs(h) * spots[:, -1]
            costs += final_cost
            pl = cash + h * spots[:, -1] - final_cost \
                - np.maximum(spots[:, -1] - 1.0, 0.0)
            return pl, hist, costs

        def deep_fn(i, tau, s, h):
            state = torch.from_numpy(np.stack([
                np.full_like(s, tau / MATURITY), s, h,
                np.full_like(s, sigma_c), np.full_like(s, rate_c),
                np.full_like(s, cost_c)], axis=-1).astype(np.float32))
            return self.policy(state).numpy().astype(np.float64)

        def delta_fn(i, tau, s, h):
            return bs_call_delta(s, 1.0, tau, sigma_c, rate_c)

        deep_pl, deep_h, deep_costs = run(deep_fn)
        delta_pl, delta_h, delta_costs = run(delta_fn)

        def stats(pl: np.ndarray, costs: np.ndarray) -> dict:
            losses = np.sort(-pl)
            k = int(math.ceil(CVAR_ALPHA * n_paths))
            return {
                "mean": float(pl.mean()), "std": float(pl.std()),
                "cvar95": float(losses[k:].mean()),
                "p5": float(np.percentile(pl, 5)),
                "p95": float(np.percentile(pl, 95)),
                "mean_costs": float(costs.mean()),
            }

        # one illustrative path (median terminal spot) for the UI
        idx = int(np.argsort(spots[:, -1])[n_paths // 2])
        return {
            "premium": premium,
            "sigma": sigma_c, "rate": rate_c, "cost": cost_c,
            "clamped": bool(sigma_c != sigma or rate_c != rate
                            or cost_c != cost),
            "n_paths": n_paths, "n_steps": N_STEPS,
            "cvar_alpha": CVAR_ALPHA,
            "deep": {"pnl": np.round(deep_pl, 6).tolist(),
                     **stats(deep_pl, deep_costs)},
            "delta": {"pnl": np.round(delta_pl, 6).tolist(),
                      **stats(delta_pl, delta_costs)},
            "example_path": {
                "spot": np.round(spots[idx], 5).tolist(),
                "deep_holdings": np.round(deep_h[idx], 5).tolist(),
                "delta_holdings": np.round(delta_h[idx], 5).tolist(),
            },
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch", type=int, default=2048)
    args = p.parse_args()
    train(iters=args.iters, batch=args.batch)
