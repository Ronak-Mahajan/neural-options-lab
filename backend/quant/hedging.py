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

P&L accounting (strike units, S0 = K = 1, N decisions at i = 0..N-1,
dt = 1/252, holdings chosen at S_i and financed at S_i):

    cash_0 = (premium - h_0 S_0 - c |h_0| S_0) e^{r dt}
    cash_i = (cash_{i-1} - (h_i - h_{i-1}) S_i
              - c |h_i - h_{i-1}| S_i) e^{r dt}          i = 1 .. N-1
    PL     = cash_{N-1} + h_{N-1} S_N - c |h_{N-1}| S_N - (S_N - K)+

(The previous version of this docstring traded at S_{i+1} and applied one extra
period of growth at expiry; neither matched the code below.)

READ THIS BEFORE QUOTING ANY NUMBER FROM compare()
--------------------------------------------------
An earlier version of this module trained the policy on the WGAN measure and
then evaluated it on that same measure, and reported that the deep hedger cut
CVaR_95 by ~30% versus a delta hedge. That result does not survive out of
sample, for two independent reasons, both measured:

  1. The delta baseline was handicapped. It hedged at the caller's sigma while
     the paths realized 1.28x that volatility, which is a known way to lose
     money in the tail. Vol-matching the baseline closed ~83% of the reported
     gap on its own.
  2. The generator is mode-collapsed (participation ratio 4.66 of 30 factors),
     which makes its paths forecastable: regressing the remaining log return on
     the realized ones gives R^2 = 0.8755 on GAN paths versus 0.0006 on GBM. A
     spot-conditioned holding is therefore a directional bet, and minimizing
     CVaR under that measure rewards market timing rather than hedging.

compare() now reports BOTH measures side by side and defaults its headline to
the out-of-sample GBM one. Item (2) is a property of the shipped generator that
this module cannot fix; it is documented rather than papered over.

Train:  python -m backend.quant.hedging --iters 6000     (see train() for the
        runtime actually observed on this machine)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from backend.quant.generative import (PathGenerator, gbm_log_returns,
                                      risk_neutralize)

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


def bs_call_gamma(spot, strike, tau, sigma, rate):
    """d2C/dS2 — needed for the Whalley-Wilmott no-trade band."""
    tau = np.maximum(tau, 1e-12)
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    return norm.pdf(d1) / (spot * sd)


def cvar(pl: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Mean of the worst (1-alpha) tail of the LOSS distribution.

    Guards the empty-tail case: ceil(alpha*n) == n whenever
    n < 1/(1-alpha) (n <= 19 at alpha=0.95), which previously sliced an empty
    array and returned NaN.
    """
    if pl.size == 0:
        return float("nan")
    losses = np.sort(-pl)
    k = int(math.ceil(alpha * losses.size))
    tail = losses[k:] if k < losses.size else losses[-1:]
    return float(tail.mean())


def cvar_bootstrap_se(pl: np.ndarray, alpha: float = CVAR_ALPHA,
                      n_boot: int = 500, seed: int = 0) -> float:
    """Bootstrap standard error of the CVaR estimate.

    CVaR is a tail statistic: only ceil((1-alpha)*n) paths enter it (150 of
    3000 at alpha=0.95), so the sampling error is far larger than the P&L
    standard deviation suggests. Reporting the point estimate alone — as this
    module previously did, from a single hard-coded seed — overstates precision.
    """
    if pl.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, pl.size, size=(n_boot, pl.size))
    return float(np.std([cvar(pl[i], alpha) for i in idx], ddof=1))


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
          seed: int = 21, measure: str = "gan",
          out_name: str | None = None) -> None:
    """Train the CVaR policy under `measure` ('gan' or 'gbm').

    The measure is recorded in the checkpoint meta, because a policy trained
    under one measure and evaluated under another is not a meaningful test —
    and the shipped hedger.pt was trained under a risk_neutralize that has
    since been corrected (it was neither a martingale nor correctly scaled).
    """
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

        # Paths on the pricing measure — the policy must learn hedging skill,
        # not the generator's drift bias. NOTE: risk_neutralize now enforces
        # the martingale condition and the terminal variance, which the version
        # this checkpoint family was originally trained under did not.
        if measure == "gbm":
            log_returns = gbm_log_returns(batch, float(sigma[0]),
                                          float(rate[0]), N_STEPS)
        else:
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
    out = ARTIFACTS / (out_name or "hedger.pt")
    torch.save({
        "policy": policy.state_dict(),
        "meta": {"n_steps": N_STEPS, "maturity": MATURITY,
                 "cvar_alpha": CVAR_ALPHA, "train_box": TRAIN_BOX,
                 "iters": iters, "batch": batch, "lr": lr, "seed": seed,
                 "train_measure": measure,
                 "martingale_enforced": True,
                 "train_seconds": round(time.perf_counter() - t0, 1)},
    }, out)
    print(f"saved {out}  ({time.perf_counter() - t0:.0f}s, measure={measure})")


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

        gen_ckpt = ARTIFACTS / "generator.pt"
        if not gen_ckpt.exists():
            raise FileNotFoundError(
                f"No WGAN checkpoint at {gen_ckpt}. The hedging comparison "
                "needs it for the in-sample measure. Train one first: "
                "python -m backend.quant.generative")
        self.generator = PathGenerator()
        gen_blob = torch.load(gen_ckpt, map_location="cpu", weights_only=True)
        self.generator.load_state_dict(gen_blob["generator"])
        self.generator.eval()

    # ------------------------------------------------------------- measures
    @torch.no_grad()
    def _spots(self, measure: str, sigma: float, rate: float, n_paths: int,
               seed: int) -> np.ndarray:
        """Spot paths (n_paths, N_STEPS+1) under the requested measure."""
        if measure == "gbm":
            gen = torch.Generator().manual_seed(seed)
            log_incr = gbm_log_returns(n_paths, sigma, rate, N_STEPS,
                                       generator=gen).numpy().astype(np.float64)
        elif measure == "gan":
            gen = torch.Generator().manual_seed(seed)
            z = torch.randn(n_paths, self.generator.noise_dim, generator=gen)
            st = torch.full((n_paths, 1), sigma, dtype=torch.float32)
            rt = torch.full((n_paths, 1), rate, dtype=torch.float32)
            raw = self.generator(z, st, rt)
            log_incr = risk_neutralize(raw, st, rt).numpy().astype(np.float64)
        else:
            raise ValueError(f"unknown measure {measure!r}; use 'gbm' or 'gan'")
        spots = np.empty((n_paths, N_STEPS + 1))
        spots[:, 0] = 1.0
        spots[:, 1:] = np.exp(np.cumsum(log_incr, axis=1))
        return spots

    # ------------------------------------------------------------- the book
    @staticmethod
    def _run_book(spots, holdings_fn, premium, cost, rate):
        n = spots.shape[0]
        growth = math.exp(rate * DT)
        cash = np.full(n, premium)
        h = np.zeros(n)
        hist = np.empty((n, N_STEPS))
        costs = np.zeros(n)
        for i in range(N_STEPS):
            tau = (N_STEPS - i) * DT
            h_new = holdings_fn(i, tau, spots[:, i], h)
            trade_cost = cost * np.abs(h_new - h) * spots[:, i]
            cash -= (h_new - h) * spots[:, i] + trade_cost
            costs += trade_cost
            h = h_new
            hist[:, i] = h
            cash *= growth
        final_cost = cost * np.abs(h) * spots[:, -1]
        costs += final_cost
        pl = (cash + h * spots[:, -1] - final_cost
              - np.maximum(spots[:, -1] - 1.0, 0.0))
        return pl, hist, costs

    # ---------------------------------------------------------- strategies
    def _deep_fn(self, sigma, rate, cost):
        def f(i, tau, s, h):
            state = torch.from_numpy(np.stack([
                np.full_like(s, tau / MATURITY), s, h,
                np.full_like(s, sigma), np.full_like(s, rate),
                np.full_like(s, cost)], axis=-1).astype(np.float32))
            with torch.no_grad():
                return self.policy(state).numpy().astype(np.float64)
        return f

    @staticmethod
    def _delta_fn(sigma, rate):
        return lambda i, tau, s, h: bs_call_delta(s, 1.0, tau, sigma, rate)

    @staticmethod
    def _whalley_wilmott_fn(sigma, rate, cost, risk_aversion):
        """Delta hedge inside a no-trade band (Whalley & Wilmott, 1997).

        Half-width  W = ( 3/2 * c * e^{-r tau} * S * Gamma^2 / gamma )^{1/3};
        trade only to the nearest band edge when |h - Delta| > W. This is the
        standard cost-aware baseline. The deep policy receives `cost` in its
        state, so comparing it against a cost-BLIND delta hedge is not a fair
        fight — this is.
        """
        def f(i, tau, s, h):
            d = bs_call_delta(s, 1.0, tau, sigma, rate)
            if cost <= 0.0 or risk_aversion <= 0.0:
                return d
            g = bs_call_gamma(s, 1.0, tau, sigma, rate)
            w = np.cbrt(1.5 * cost * np.exp(-rate * tau) * s * g ** 2
                        / risk_aversion)
            lo, hi = d - w, d + w
            return np.clip(h, lo, hi)
        return f

    # -------------------------------------------------------------- compare
    @torch.no_grad()
    def compare(self, sigma: float, rate: float, cost: float,
                n_paths: int = 3000, seeds: tuple[int, ...] = (17, 18, 19, 20, 21),
                primary: str = "gbm", seed: int | None = None) -> dict:
        """Deep hedge vs delta and Whalley-Wilmott, on BOTH measures.

        The headline numbers ("deep"/"delta" at the top level) come from
        `primary`, which defaults to out-of-sample GBM. The in-sample GAN
        numbers are still returned under `by_measure` so the difference is
        visible rather than hidden.

        Every statistic is averaged over `seeds` and carries a bootstrap
        standard error, because CVaR is a tail statistic estimated from only
        ceil((1-alpha)*n) paths.
        """
        if seed is not None:               # back-compat with the old signature
            seeds = (seed,)
        sigma_c = float(np.clip(sigma, *TRAIN_BOX["sigma"]))
        rate_c = float(np.clip(rate, *TRAIN_BOX["rate"]))
        cost_c = float(np.clip(cost, *TRAIN_BOX["cost"]))
        premium_bs = float(bs_call_price(1.0, 1.0, MATURITY, sigma_c, rate_c))

        by_measure: dict[str, dict] = {}
        example: dict[str, dict] = {}
        for measure in ("gbm", "gan"):
            # Book the premium at the option's value UNDER THE MEASURE BEING
            # SIMULATED, not at Black-Scholes. On GBM the two agree; on the
            # fat-tailed GAN measure they differ, and booking the BS premium
            # would leave both hedgers short a mispriced option.
            probe = self._spots(measure, sigma_c, rate_c, max(n_paths, 20_000),
                                seeds[0] + 9_000)
            premium_mc = float(np.exp(-rate_c * MATURITY)
                               * np.maximum(probe[:, -1] - 1.0, 0.0).mean())
            realized_vol = float(np.std(np.log(probe[:, -1]))
                                 / math.sqrt(MATURITY))

            # Tune the Whalley-Wilmott risk-aversion once, on the first seed.
            # This is in-sample FOR THE BASELINE, i.e. deliberately generous to
            # it: we want the strongest honest baseline the deep hedger must beat.
            tune = self._spots(measure, sigma_c, rate_c, n_paths, seeds[0])
            best_g, best_c = 1.0, math.inf
            for ra in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
                pl, _, _ = self._run_book(
                    tune, self._whalley_wilmott_fn(realized_vol, rate_c,
                                                   cost_c, ra),
                    premium_mc, cost_c, rate_c)
                c = cvar(pl)
                if c < best_c:
                    best_g, best_c = ra, c

            strategies = {
                "deep": self._deep_fn(sigma_c, rate_c, cost_c),
                # vol-matched: hedge at the vol the paths actually realize
                "delta": self._delta_fn(realized_vol, rate_c),
                "delta_naive": self._delta_fn(sigma_c, rate_c),
                "whalley_wilmott": self._whalley_wilmott_fn(
                    realized_vol, rate_c, cost_c, best_g),
            }
            acc = {k: {"pl": [], "costs": []} for k in strategies}
            for sd in seeds:
                spots = self._spots(measure, sigma_c, rate_c, n_paths, sd)
                for name, fn in strategies.items():
                    pl, hist, cst = self._run_book(spots, fn, premium_mc,
                                                   cost_c, rate_c)
                    acc[name]["pl"].append(pl)
                    acc[name]["costs"].append(cst)
                    if sd == seeds[0] and name in ("deep", "delta"):
                        idx = int(np.argsort(spots[:, -1])[n_paths // 2])
                        example.setdefault(measure, {"spot": np.round(
                            spots[idx], 5).tolist()})
                        example[measure][f"{name}_holdings"] = np.round(
                            hist[idx], 5).tolist()

            out = {}
            for name in strategies:
                pl = np.concatenate(acc[name]["pl"])
                cst = np.concatenate(acc[name]["costs"])
                out[name] = {
                    "mean": float(pl.mean()), "std": float(pl.std()),
                    "cvar95": cvar(pl),
                    "cvar95_se": cvar_bootstrap_se(pl, seed=1),
                    "p5": float(np.percentile(pl, 5)),
                    "p95": float(np.percentile(pl, 95)),
                    "mean_costs": float(cst.mean()),
                    "pnl": np.round(acc[name]["pl"][0], 6).tolist(),
                }
            ratio = (out["deep"]["cvar95"] / out["delta"]["cvar95"]
                     if out["delta"]["cvar95"] else float("nan"))
            by_measure[measure] = {
                "premium_mc": premium_mc, "premium_bs": premium_bs,
                "realized_vol": realized_vol,
                "whalley_wilmott_risk_aversion": best_g,
                "deep_over_delta_cvar95": ratio,
                "deep_beats_delta": bool(ratio < 1.0),
                **out,
            }

        p = by_measure[primary]
        return {
            "premium": p["premium_mc"], "premium_bs": premium_bs,
            "sigma": sigma_c, "rate": rate_c, "cost": cost_c,
            "clamped": bool(sigma_c != sigma or rate_c != rate
                            or cost_c != cost),
            "n_paths": n_paths * len(seeds), "n_paths_per_seed": n_paths,
            "seeds": list(seeds), "n_steps": N_STEPS,
            "cvar_alpha": CVAR_ALPHA,
            "measure": primary,
            "measure_note": (
                "Headline numbers are out-of-sample (risk-neutral GBM). The "
                "policy was TRAINED on the 'gan' measure, so those numbers are "
                "in-sample and are reported alongside for comparison only."),
            "deep": p["deep"], "delta": p["delta"],
            "delta_naive": p["delta_naive"],
            "whalley_wilmott": p["whalley_wilmott"],
            "deep_over_delta_cvar95": p["deep_over_delta_cvar95"],
            "deep_beats_delta": p["deep_beats_delta"],
            "by_measure": by_measure,
            "example_path": {
                "spot": example[primary]["spot"],
                "deep_holdings": example[primary]["deep_holdings"],
                "delta_holdings": example[primary]["delta_holdings"],
            },
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch", type=int, default=2048)
    args = p.parse_args()
    train(iters=args.iters, batch=args.batch)
