"""Deep Hedging (Buehler, Gonon, Teichmann & Wood, 2019).

We hedge a SHORT 30-day at-the-money European call, rebalancing daily in the
underlying only, under proportional transaction costs. Two agents run on the
same simulated paths:

    Standard delta hedge - holds the exact Black-Scholes delta each day
                           (the textbook strategy; optimal only in the
                           frictionless, continuous limit)
    Deep hedge           - a policy network state -> holding, trained to
                           minimize CVaR_95 of the terminal hedging loss,
                           transaction costs inside the objective

Policy state is (tau, S/K, h_prev, sigma, r, cost): the network is trained
*conditionally* over a box of (sigma, r, cost) sampled per path, so one
offline training run serves any ticker's live parameters - and lets the
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

Measures and baselines beyond the original pair
------------------------------------------------
GBM is a complete market with small costs, where a static delta hedge is
near-optimal and the negative result above is EXPECTED. The question worth
asking is whether the learned policy wins where a static delta cannot:
stochastic rough volatility with spot-vol correlation ('rbergomi'), jumps
('rbergomi_jumps', an incomplete market), and transaction costs. Both rough
measures are wired into train() and HedgingEngine._spots() at the SPY
calibration read from disk (rough_measure_params). A third baseline, the
Ruf-Wang linear-regression hedge (fit_linear_hedge), is a holding rule linear
in Black-Scholes features with coefficients fitted by OLS to minimise the
variance of terminal P&L on separate training paths. The designed experiment
lives in scripts/deep_hedging_regimes.py and its result in
docs/deep_hedging_regimes.md.

Train:  python -m backend.quant.hedging --iters 6000     (see train() for the
        runtime actually observed on this machine)
        python -m backend.quant.hedging --measure rbergomi --out hedger_rbergomi.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from backend.quant.generative import (PathGenerator, gbm_log_returns,
                                      risk_neutralize)
from backend.quant.rough_vol import rough_bergomi_log_returns

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

N_STEPS = 30                 # daily rebalances over the 30-day horizon
DT = 1.0 / 252.0
MATURITY = N_STEPS * DT
CVAR_ALPHA = 0.95
TRAIN_BOX = {"sigma": (0.08, 0.65), "rate": (0.0, 0.09),
             "cost": (0.0, 0.02)}

#: Simulated measures the hedgers can be trained and evaluated under.
#:   gbm            exact risk-neutral geometric Brownian motion (complete
#:                  market: delta hedging is near-optimal, the control)
#:   gan            the WGAN market simulator after risk_neutralize (see the
#:                  module docstring for why it is not a sound measure)
#:   rbergomi       risk-neutral rough Bergomi at the SPY calibration in
#:                  artifacts/rough_calibration.json (eta, rho, H; the
#:                  forward variance xi = sigma^2 is set by the caller's
#:                  sigma so the training box still covers a vol range, and
#:                  rough_measure_params() exposes the CALIBRATED xi and
#:                  rate so an experiment can evaluate at that level)
#:   rbergomi_jumps rough Bergomi plus compensated Merton jumps at the jump
#:                  fit in scripts/_fit_jumps_last.json
#: Under both rough measures the market is incomplete (stochastic vol with
#: spot-vol correlation, and jumps), which is where a learned policy can in
#: principle do what a static delta cannot.
ROUGH_MEASURES = ("rbergomi", "rbergomi_jumps")
MEASURES = ("gbm", "gan") + ROUGH_MEASURES
ROUGH_CALIBRATION_FILE = ARTIFACTS / "rough_calibration.json"
JUMP_FIT_FILE = ROOT / "scripts" / "_fit_jumps_last.json"


@lru_cache(maxsize=4)
def rough_measure_params(measure: str) -> dict:
    """Calibrated rough-vol parameters for a rough measure, read from disk.

    'rbergomi' takes (eta, rho, H, xi, rate) from the SPY calibration
    artifact. 'rbergomi_jumps' takes theta = [eta, rho, H, xi, lam, mu_j,
    sig_j] from the paired jump fit (which carries no rate, so the rate is the
    calibration's). Nothing is hardcoded: change the files and every measure
    follows. The returned dict has keys eta, rho, H, xi, rate, jumps
    (None or (lam, mu_j, sig_j)) and source.
    """
    if measure not in ROUGH_MEASURES:
        raise ValueError(f"{measure!r} is not a rough measure; "
                         f"use one of {ROUGH_MEASURES}")
    cal = json.loads(ROUGH_CALIBRATION_FILE.read_text(encoding="utf-8"))
    rate = float(cal["rate"])
    if measure == "rbergomi":
        return {"eta": float(cal["eta"]), "rho": float(cal["rho"]),
                "H": float(cal["H"]), "xi": float(cal["xi"]), "rate": rate,
                "jumps": None, "source": str(ROUGH_CALIBRATION_FILE.name)}
    fit = json.loads(JUMP_FIT_FILE.read_text(encoding="utf-8"))
    eta, rho, H, xi, lam, mu_j, sig_j = (float(v)
                                         for v in fit["jumps"]["theta"])
    return {"eta": eta, "rho": rho, "H": H, "xi": xi, "rate": rate,
            "jumps": (lam, mu_j, sig_j),
            "source": f"{JUMP_FIT_FILE.name} (rate from "
                      f"{ROUGH_CALIBRATION_FILE.name})"}


def rough_log_returns(measure: str, n_paths: int, sigma: float, rate: float,
                      seed: int | None = None) -> torch.Tensor:
    """(n_paths, N_STEPS) log returns under a rough measure at vol level sigma.

    sigma sets the forward variance xi = sigma^2 (E[V_t] = xi for every t, so
    sigma is the model's flat forward vol); eta, rho, H and the jump
    parameters come from rough_measure_params(measure).
    """
    p = rough_measure_params(measure)
    return rough_bergomi_log_returns(
        n_paths, N_STEPS, DT, xi=sigma ** 2, eta=p["eta"], rho=p["rho"],
        H=p["H"], rate=rate, seed=seed, jumps=p["jumps"])


def measure_log_returns(measure: str, n_paths: int, sigma: float,
                        rate: float, seed: int | None = None,
                        generator: PathGenerator | None = None
                        ) -> torch.Tensor:
    """(n_paths, N_STEPS) log returns under any measure in MEASURES.

    The 'gan' measure needs the WGAN `generator`; the others do not.
    """
    if measure == "gbm":
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return gbm_log_returns(n_paths, sigma, rate, N_STEPS, generator=gen)
    if measure in ROUGH_MEASURES:
        return rough_log_returns(measure, n_paths, sigma, rate, seed)
    if measure == "gan":
        if generator is None:
            raise ValueError("the 'gan' measure needs the WGAN generator")
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(n_paths, generator.noise_dim, generator=gen)
        st = torch.full((n_paths, 1), float(sigma), dtype=torch.float32)
        rt = torch.full((n_paths, 1), float(rate), dtype=torch.float32)
        with torch.no_grad():
            raw = generator(z, st, rt)
        return risk_neutralize(raw, st, rt)
    raise ValueError(f"unknown measure {measure!r}; use one of {MEASURES}")


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
    """d2C/dS2 - needed for the Whalley-Wilmott no-trade band."""
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
    standard deviation suggests. Reporting the point estimate alone - as this
    module previously did, from a single hard-coded seed - overstates precision.
    """
    if pl.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, pl.size, size=(n_boot, pl.size))
    return float(np.std([cvar(pl[i], alpha) for i in idx], ddof=1))


def paired_cvar_bootstrap(pls: dict[str, np.ndarray],
                          alpha: float = CVAR_ALPHA, n_boot: int = 2000,
                          seed: int = 1, max_elems: int = 1_500_000) -> dict:
    """Bootstrap every hedger's CVaR on ONE resampling of the shared paths.

    Every strategy in `compare` is run on the same simulated paths, in the
    same order, so `pls[a][i]` and `pls[b][i]` are the same path hedged two
    ways. That pairing is information, and `cvar_bootstrap_se` throws it
    away: it gives each hedger's own sampling error, from which the error of
    a DIFFERENCE can only be recovered as hypot(se_a, se_b), and that formula
    assumes the two are independent. They are the opposite of independent -
    a path that is bad for one hedger is usually bad for all of them - so
    hypot overstates the error of the difference, sometimes by a lot, and a
    real ranking gets reported as a tie.

    Here one bootstrap index draw is shared by every strategy in a replicate,
    so the common path risk cancels inside each replicate difference and what
    is left is the sampling error of the comparison itself. The marginal
    standard errors are read off the same replicates, so the individual
    errors and the difference errors are one estimate of one thing rather
    than two that have to be reconciled.

    Returns ``{"names", "n_boot", "alpha", "cvar", "se", "pairs"}``. `cvar`
    and `se` are per strategy; `pairs` is keyed ``"a|b"`` and carries the
    difference ``cvar[a] - cvar[b]`` in loss units (NEGATIVE means `a` has
    the smaller loss, i.e. `a` is the better hedger), its paired standard
    error, a 2.5/97.5 percentile bootstrap interval, the bootstrap bias, the
    fraction of replicates in which `a` wins, the correlation between the two
    hedgers' replicate CVaRs, and `hypot_se` - what the unpaired formula
    would have claimed - so the difference the pairing makes stays visible.

    Bounded memory by construction: replicates are drawn in blocks sized so
    that no intermediate exceeds `max_elems` entries, because this runs in
    the same 512 MB container that serves the dashboard. Block size changes
    how the random stream is consumed, so `max_elems` is part of the
    reproducibility contract along with `seed` and `n_boot`: the point CVaRs
    are unaffected, the replicate draw is not. Fewer than two replicates
    leaves every error and correlation as NaN rather than inventing one.
    """
    names = list(pls)
    if not names:
        return {"names": [], "n_boot": 0, "alpha": alpha, "cvar": {},
                "se": {}, "pairs": {}}
    arrays = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in pls.items()}
    n = arrays[names[0]].size
    if any(a.size != n for a in arrays.values()):
        raise ValueError("paired bootstrap needs one P&L per path per "
                         "strategy, in the same path order: got sizes "
                         + repr({k: a.size for k, a in arrays.items()}))
    if n == 0:
        nan = float("nan")
        return {"names": names, "n_boot": 0, "alpha": alpha,
                "cvar": {k: nan for k in names},
                "se": {k: nan for k in names}, "pairs": {}}

    losses = {k: -a for k, a in arrays.items()}      # CVaR acts on losses
    k_tail = int(math.ceil(alpha * n))
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot, dtype=np.float64) for k in names}
    block = max(1, int(max_elems // n))
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        for k in names:
            s = losses[k][idx]
            if k_tail < n:
                reps[k][done:done + b] = np.partition(
                    s, k_tail, axis=1)[:, k_tail:].mean(axis=1)
            else:
                reps[k][done:done + b] = s.max(axis=1)
        done += b

    nan = float("nan")
    spread = n_boot >= 2          # one replicate measures no spread at all
    point = {k: cvar(arrays[k], alpha) for k in names}
    se = {k: float(reps[k].std(ddof=1)) if spread else nan for k in names}
    pairs: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            d = reps[a] - reps[b_]
            diff = point[a] - point[b_]
            lo, hi = (float(x) for x in np.percentile(d, (2.5, 97.5)))
            if spread and d.std() > 0.0:
                corr = float(np.corrcoef(reps[a], reps[b_])[0, 1])
            else:
                # Identical replicates: the two hedgers move together
                # exactly, which is a correlation of one, not undefined.
                corr = 1.0 if spread else nan
            pairs[f"{a}|{b_}"] = {
                "diff": diff,
                "se": float(d.std(ddof=1)) if spread else nan,
                "ci_low": lo, "ci_high": hi,
                "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "bias": float(d.mean() - diff),
                "p_first_better": float((d < 0.0).mean()),
                "corr": corr,
                "hypot_se": (float(math.hypot(se[a], se[b_])) if spread
                             else nan),
            }
    return {"names": names, "n_boot": int(n_boot), "alpha": alpha,
            "cvar": point, "se": se, "pairs": pairs}


# ---------------------------------------------------------------------------
# Linear-regression hedge (Ruf & Wang, JBES 2022)
# ---------------------------------------------------------------------------

LINEAR_FEATURES = ("const", "delta", "delta_1m_delta", "vega_norm")


def linear_hedge_features(spot, tau, sigma: float, rate: float,
                          n_features: int = 3) -> np.ndarray:
    """Black-Scholes features of the linear hedge, shape (n, n_features).

    Column order follows LINEAR_FEATURES: 1, delta, delta*(1-delta) and,
    with n_features=4, vega/(S*sqrt(tau)) = phi(d1). The third and fourth
    are both bell-shaped in d1 and nearly collinear, which is why the
    default fits three.
    """
    if n_features not in (3, 4):
        raise ValueError("n_features must be 3 or 4")
    spot = np.asarray(spot, dtype=np.float64)
    d = bs_call_delta(spot, 1.0, tau, sigma, rate)
    cols = [np.ones_like(d), d, d * (1.0 - d)]
    if n_features == 4:
        tau_c = np.maximum(tau, 1e-12)
        sd = sigma * np.sqrt(tau_c)
        d1 = (np.log(spot) + (rate + 0.5 * sigma ** 2) * tau_c) / sd
        cols.append(norm.pdf(d1))
    return np.stack(cols, axis=-1)


def fit_linear_hedge(spots: np.ndarray, sigma: float, rate: float,
                     n_features: int = 3) -> dict:
    """OLS fit of h_i = c0 + c1*delta + c2*delta*(1-delta) [+ c3*phi(d1)]
    that minimises the VARIANCE of terminal P&L, costs ignored.

    Why OLS is exact here. With the book's accounting (docstring at the top
    of the module) and no costs, terminal P&L is linear in the holdings:

        PL = premium*g^N - (S_N - K)+ + sum_i h_i G_i,
        G_i = S_{i+1} g^{N-i-1} - S_i g^{N-i},   g = e^{r dt},

    G_i being the forward value at T of one share bought at S_i and sold at
    S_{i+1}. With h_i = sum_k c_k f_k(S_i, tau_i) that is
    PL = const - payoff + sum_k c_k X_k, X_k = sum_i f_k(i) G_i, so
    argmin_c Var[PL] is the ordinary least-squares regression of the payoff
    on the aggregated feature gains X_k with an intercept (which absorbs the
    mean; the premium is irrelevant to the fit). No iteration, no
    hyper-parameters, and the same information the delta hedge uses: the
    fit is cost-blind, so it is a min-variance baseline rather than a
    cost-aware one, and the coefficients are fitted on a SEPARATE set of
    training paths from the same measure, then evaluated with costs on the
    test paths.

    On a complete-market GBM measure the min-variance holding is (up to the
    daily discretisation) the Black-Scholes delta, so the fit should return
    c1 ~ 1 and c0, c2 ~ 0; that is a test. Under rho < 0 the literature
    (Hull & White 2017, Ruf & Wang 2022) finds the min-variance delta of a
    call BELOW the Black-Scholes delta, which shows up here as c1 < 1 or a
    negative c2.

    spots: (n, N_STEPS+1) with spots[:, 0] = 1 = K. Returns a dict with
    `coef` (n_features,), `features`, `r2` of the payoff regression, and
    `n_paths`.
    """
    spots = np.asarray(spots, dtype=np.float64)
    n, n_steps_plus = spots.shape
    if n_steps_plus != N_STEPS + 1:
        raise ValueError(f"expected {N_STEPS + 1} columns, got {n_steps_plus}")
    g = math.exp(rate * DT)
    X = np.zeros((n, n_features))
    for i in range(N_STEPS):
        tau = (N_STEPS - i) * DT
        gain = spots[:, i + 1] * g ** (N_STEPS - i - 1) \
            - spots[:, i] * g ** (N_STEPS - i)
        X += linear_hedge_features(spots[:, i], tau, sigma, rate,
                                   n_features) * gain[:, None]
    y = np.maximum(spots[:, -1] - 1.0, 0.0)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1.0 - float(resid.var()) / float(y.var()) if y.var() > 0 else 0.0
    return {"coef": beta[1:], "features": LINEAR_FEATURES[:n_features],
            "r2": r2, "n_paths": int(n)}


def linear_hedge_fn(coef: np.ndarray, sigma: float, rate: float):
    """Holdings rule for _run_book from a fit_linear_hedge() result."""
    coef = np.asarray(coef, dtype=np.float64)
    k = coef.size

    def f(i, tau, s, h):
        return linear_hedge_features(s, tau, sigma, rate, k) @ coef
    return f


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

def train_box_for(measure: str) -> dict:
    """The (sigma, rate, cost) box a policy is trained over under `measure`.

    gbm / gan: TRAIN_BOX, so one policy serves any ticker's live parameters.
    Rough measures: sigma and rate are PINNED at the calibrated forward vol
    sqrt(xi) and rate, and only the cost is sampled. The rough measure is a
    calibration, not a family (eta = 3.9 with H = 0.26 at sigma = 0.65 would
    be a market nobody calibrated), and the experiment evaluates at the
    calibrated level, so a rough policy spends its capacity there. This is
    also the setup most favourable to the learned hedger, which is the right
    way to look for the regime where it wins.
    """
    if measure in ROUGH_MEASURES:
        p = rough_measure_params(measure)
        s = math.sqrt(p["xi"])
        return {"sigma": (s, s), "rate": (p["rate"], p["rate"]),
                "cost": TRAIN_BOX["cost"]}
    return dict(TRAIN_BOX)


def train(iters: int = 6000, batch: int = 2048, lr: float = 1e-3,
          seed: int = 21, measure: str = "gan",
          out_name: str | None = None, box: dict | None = None,
          log_every: int = 500) -> dict:
    """Train the CVaR policy under `measure` (any of MEASURES).

    The measure is recorded in the checkpoint meta, because a policy trained
    under one measure and evaluated under another is not a meaningful test -
    and the shipped hedger.pt was trained under a risk_neutralize that has
    since been corrected (it was neither a martingale nor correctly scaled).
    `box` overrides train_box_for(measure). Returns the checkpoint meta.
    """
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; use one of {MEASURES}")
    box = dict(box or train_box_for(measure))
    torch.manual_seed(seed)
    policy = HedgePolicy()
    head = CVaRHead()
    opt = torch.optim.AdamW(list(policy.parameters())
                            + list(head.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    generator = None
    if measure == "gan":
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
                           float(torch.empty(1).uniform_(*box["sigma"])))
        rate = torch.full((batch,),
                          float(torch.empty(1).uniform_(*box["rate"])))
        cost = torch.full((batch,),
                          float(torch.empty(1).uniform_(*box["cost"])))

        # Paths on the pricing measure - the policy must learn hedging skill,
        # not the generator's drift bias. NOTE: risk_neutralize now enforces
        # the martingale condition and the terminal variance, which the version
        # this checkpoint family was originally trained under did not. The
        # rough measures are risk-neutral by construction (left-point
        # variance, per-step jump compensation). Unseeded here: the global
        # torch seed set above makes the run reproducible.
        log_returns = measure_log_returns(measure, batch, float(sigma[0]),
                                          float(rate[0]), generator=generator)

        pl = simulate_pl(policy, log_returns, sigma, rate, cost)
        loss_var = -pl                                    # hedging shortfall
        w = head(torch.stack([sigma, rate, cost], dim=-1))
        cvar = (w + torch.clamp(loss_var - w, min=0.0)
                / (1.0 - CVAR_ALPHA)).mean()

        opt.zero_grad(set_to_none=True)
        cvar.backward()
        opt.step()
        sched.step()

        if log_every and (it % log_every == 0 or it == 1):
            with torch.no_grad():
                print(f"iter {it:>5}/{iters}  "
                      f"CVaR objective {cvar.item():+.5f}  "
                      f"mean PL {pl.mean().item():+.5f}  "
                      f"({time.perf_counter() - t0:5.1f}s)", flush=True)

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / (out_name or "hedger.pt")
    meta = {"n_steps": N_STEPS, "maturity": MATURITY,
            "cvar_alpha": CVAR_ALPHA, "train_box": box,
            "iters": iters, "batch": batch, "lr": lr, "seed": seed,
            "train_measure": measure,
            "martingale_enforced": True,
            "train_seconds": round(time.perf_counter() - t0, 1)}
    if measure in ROUGH_MEASURES:
        meta["measure_params"] = rough_measure_params(measure)
    torch.save({"policy": policy.state_dict(), "meta": meta}, out)
    print(f"saved {out}  ({time.perf_counter() - t0:.0f}s, measure={measure})")
    return meta


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
        """Spot paths (n_paths, N_STEPS+1) under any measure in MEASURES.

        'gbm' and 'gan' are unchanged from before the rough measures were
        added (same generator, same draw order, so seeded paths are
        bit-identical). 'rbergomi' and 'rbergomi_jumps' use the calibrated
        rough dynamics at forward vol `sigma`; see rough_log_returns().
        """
        log_incr = measure_log_returns(
            measure, n_paths, sigma, rate, seed=seed,
            generator=self.generator).numpy().astype(np.float64)
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
        fight - this is.
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

    @staticmethod
    def _linear_fn(coef, sigma, rate):
        """Ruf-Wang linear-regression hedge; see fit_linear_hedge()."""
        return linear_hedge_fn(coef, sigma, rate)

    # -------------------------------------------------------------- compare
    @torch.no_grad()
    def compare(self, sigma: float, rate: float, cost: float,
                n_paths: int = 3000, seeds: tuple[int, ...] = (17, 18, 19, 20, 21),
                primary: str = "gbm", seed: int | None = None,
                measures: tuple[str, ...] = ("gbm", "gan")) -> dict:
        """Deep hedge vs delta, Whalley-Wilmott and the linear-regression
        hedge, on every measure in `measures`.

        The headline numbers ("deep"/"delta" at the top level) come from
        `primary`, which defaults to out-of-sample GBM. The in-sample GAN
        numbers are still returned under `by_measure` so the difference is
        visible rather than hidden. The rough measures ('rbergomi',
        'rbergomi_jumps') can be requested through `measures`; the default
        stays at the two the dashboard was built on so the served request
        keeps its memory and latency budget.

        Every statistic is averaged over `seeds` and carries a bootstrap
        standard error, because CVaR is a tail statistic estimated from only
        ceil((1-alpha)*n) paths.
        """
        if seed is not None:               # back-compat with the old signature
            seeds = (seed,)
        if primary not in measures:
            raise ValueError(f"primary {primary!r} must be one of {measures}")
        sigma_c = float(np.clip(sigma, *TRAIN_BOX["sigma"]))
        rate_c = float(np.clip(rate, *TRAIN_BOX["rate"]))
        cost_c = float(np.clip(cost, *TRAIN_BOX["cost"]))
        premium_bs = float(bs_call_price(1.0, 1.0, MATURITY, sigma_c, rate_c))

        by_measure: dict[str, dict] = {}
        example: dict[str, dict] = {}
        for measure in measures:
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

            # Tune the Whalley-Wilmott risk aversion once, on a dedicated
            # block drawn at seeds[0] + 5_000: disjoint from every evaluation
            # seed, and the same offset scripts/deep_hedging_regimes.py fits
            # its baselines on. The offset is shared, not the draw - the
            # script tunes on 20,000 rows and probes on 40,000 where the
            # endpoint uses n_paths and 20,000 - so the two agree on protocol
            # and not on digits. The tuning is in-sample FOR THE
            # BASELINE, i.e. deliberately generous to it: we want the
            # strongest honest baseline the deep hedger must beat. Generous
            # on its own block is a fair fight; generous on the block every
            # hedger is scored on is not, and it is the baseline that stands
            # to gain from any overlap.
            tune = self._spots(measure, sigma_c, rate_c, n_paths,
                               seeds[0] + 5_000)
            best_g, best_c = 1.0, math.inf
            for ra in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
                pl, _, _ = self._run_book(
                    tune, self._whalley_wilmott_fn(realized_vol, rate_c,
                                                   cost_c, ra),
                    premium_mc, cost_c, rate_c)
                c = cvar(pl)
                if c < best_c:
                    best_g, best_c = ra, c

            # Linear-regression hedge (Ruf & Wang): OLS on the same in-sample
            # baseline paths, costs ignored in the fit, evaluated with costs.
            lin_fit = fit_linear_hedge(tune, realized_vol, rate_c)

            strategies = {
                "deep": self._deep_fn(sigma_c, rate_c, cost_c),
                # vol-matched: hedge at the vol the paths actually realize
                "delta": self._delta_fn(realized_vol, rate_c),
                "delta_naive": self._delta_fn(sigma_c, rate_c),
                "whalley_wilmott": self._whalley_wilmott_fn(
                    realized_vol, rate_c, cost_c, best_g),
                "linear": self._linear_fn(lin_fit["coef"], realized_vol,
                                          rate_c),
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

            # Every strategy above ran on the same paths in the same order,
            # so these arrays are aligned path by path and can be compared
            # pairwise. One bootstrap draw serves all of them: the marginal
            # errors below and the difference errors in `paired` are then the
            # same estimate rather than two that could disagree.
            pl_all = {name: np.concatenate(acc[name]["pl"])
                      for name in strategies}
            paired = paired_cvar_bootstrap(pl_all, seed=1)

            out = {}
            for name in strategies:
                pl = pl_all[name]
                cst = np.concatenate(acc[name]["costs"])
                out[name] = {
                    "mean": float(pl.mean()), "std": float(pl.std()),
                    "cvar95": cvar(pl),
                    "cvar95_se": paired["se"][name],
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
                "linear_coef": np.round(lin_fit["coef"], 6).tolist(),
                "linear_features": list(lin_fit["features"]),
                "deep_over_delta_cvar95": ratio,
                "deep_beats_delta": bool(ratio < 1.0),
                "paired_bootstrap": paired,
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
            "linear": p["linear"],
            "paired_bootstrap": p["paired_bootstrap"],
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
    p.add_argument("--measure", choices=MEASURES, required=True,
                   help="training measure; the served checkpoints use "
                        "rbergomi_jumps and gbm")
    p.add_argument("--out", default=None,
                   help="checkpoint file name under artifacts/")
    args = p.parse_args()
    train(iters=args.iters, batch=args.batch, measure=args.measure,
          out_name=args.out)
