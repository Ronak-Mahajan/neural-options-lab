"""The sim-to-real test: hedge on HISTORY instead of on a model.

Every number the hedging module reports comes from simulated paths — GBM
(out-of-sample control) or the WGAN (in-sample). Both are models, and a
hedger that only ever beats baselines inside models has not been tested. This
replays the SAME hedgers over windows of real daily closes (free, via
yfinance): the deep policy, the Black-Scholes delta hedge, and the
Whalley-Wilmott no-trade band, on identical windows.

Protocol, chosen to be attackable in the right places:

  - windows of N_STEPS=30 consecutive trading-day returns, spot normalized to
    1.0 at entry — the contract the policy was trained on (short one ATM
    30-day call, daily rebalances);
  - the vol fed to every hedger is the trailing 60-day realized vol at entry,
    i.e. information available AT THE TIME, clipped to the policy's training
    box. Nobody gets the window's own realized vol: hedging with the answer
    is the classic backtest sin;
  - the premium is booked at Black-Scholes under that ex-ante vol — the price
    a desk quoting off this forecast would actually have collected — so P&L
    includes the vol-forecast error, as it does on a real desk;
  - windows overlap with a 5-day stride. Overlap inflates the effective
    sample: consecutive windows share 25 of 30 days, so the CVaR standard
    errors printed here are OPTIMISTIC and the honest unit of independence is
    closer to n_windows/6. Both counts are printed.

What this cannot show: one historical path per asset is a single draw — a
strategy can lose on a draw and still be right. The value here is the PAIRED
comparison (same windows, same information), not the absolute P&L.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from backend.quant.hedging import (DT, MATURITY, N_STEPS, TRAIN_BOX,
                                   HedgingEngine, bs_call_delta,
                                   bs_call_gamma, bs_call_price, cvar,
                                   cvar_bootstrap_se)

LOOKBACK = 60          # trading days of history behind each entry
STRIDE = 5             # entry every 5 trading days; windows overlap 25/30


def daily_closes(ticker: str, years: int = 8) -> np.ndarray:
    import yfinance as yf
    px = yf.Ticker(ticker).history(period=f"{years}y", interval="1d",
                                   auto_adjust=True)["Close"].dropna()
    if len(px) < LOOKBACK + N_STEPS + 10:
        raise SystemExit(f"{ticker}: only {len(px)} closes; not enough")
    return px.to_numpy(dtype=np.float64)


def build_windows(closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(windows, sigmas): (n, N_STEPS) log returns and the EX-ANTE vol."""
    logret = np.diff(np.log(closes))
    wins, sigs = [], []
    for start in range(LOOKBACK, len(logret) - N_STEPS + 1, STRIDE):
        wins.append(logret[start:start + N_STEPS])
        trail = logret[start - LOOKBACK:start]
        sigs.append(float(np.std(trail, ddof=1) * math.sqrt(252.0)))
    return np.array(wins), np.array(sigs)


def run_book(spots: np.ndarray, holdings_fn, premiums: np.ndarray,
             cost: float, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """hedging._run_book, generalised to per-path premium (each window books
    its own ex-ante price). Same bookkeeping otherwise."""
    n = spots.shape[0]
    growth = math.exp(rate * DT)
    cash = premiums.copy()
    h = np.zeros(n)
    costs = np.zeros(n)
    for i in range(N_STEPS):
        tau = (N_STEPS - i) * DT
        h_new = holdings_fn(i, tau, spots[:, i], h)
        trade_cost = cost * np.abs(h_new - h) * spots[:, i]
        cash -= (h_new - h) * spots[:, i] + trade_cost
        costs += trade_cost
        h = h_new
        cash *= growth
    final_cost = cost * np.abs(h) * spots[:, -1]
    pl = (cash + h * spots[:, -1] - final_cost
          - np.maximum(spots[:, -1] - 1.0, 0.0))
    return pl, costs + final_cost


def strategies(engine: HedgingEngine, sigmas: np.ndarray, rate: float,
               cost: float, ww_gamma: float = 1.0):
    """Per-path-sigma versions of the module's three hedgers."""
    def deep(i, tau, s, h):
        state = torch.from_numpy(np.stack([
            np.full_like(s, tau / MATURITY), s, h, sigmas,
            np.full_like(s, rate), np.full_like(s, cost)],
            axis=-1).astype(np.float32))
        with torch.no_grad():
            return engine.policy(state).numpy().astype(np.float64)

    def delta(i, tau, s, h):
        return bs_call_delta(s, 1.0, tau, sigmas, rate)

    def ww(i, tau, s, h):
        d = bs_call_delta(s, 1.0, tau, sigmas, rate)
        if cost <= 0.0:
            return d
        g = bs_call_gamma(s, 1.0, tau, sigmas, rate)
        w = np.cbrt(1.5 * cost * np.exp(-rate * tau) * s * g ** 2 / ww_gamma)
        return np.clip(h, d - w, d + w)

    return {"deep": deep, "delta": delta, "whalley_wilmott": ww}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default="SPY,BTC-USD")
    p.add_argument("--cost", type=float, default=0.001,
                   help="proportional cost per unit traded (10 bps default)")
    p.add_argument("--years", type=int, default=8)
    args = p.parse_args()

    engine = HedgingEngine()
    for ticker in [t.strip() for t in args.tickers.split(",") if t.strip()]:
        closes = daily_closes(ticker, args.years)
        wins, sigs_raw = build_windows(closes)
        sigs = np.clip(sigs_raw, *TRAIN_BOX["sigma"])
        clipped = int(np.sum(sigs != sigs_raw))
        rate = 0.04 if not ticker.upper().startswith("BTC") else 0.0
        n = len(wins)
        spots = np.empty((n, N_STEPS + 1))
        spots[:, 0] = 1.0
        spots[:, 1:] = np.exp(np.cumsum(wins, axis=1))
        premiums = bs_call_price(1.0, 1.0, MATURITY, sigs, rate)

        print(f"\n=== {ticker}: {n} windows from {len(closes)} closes "
              f"(stride {STRIDE}; ~{n // 6} independent), ex-ante vol "
              f"median {np.median(sigs_raw):.1%}"
              + (f", {clipped} clipped into the training box" if clipped
                 else "") + f", cost {args.cost:.4f} ===")
        print(f"  {'strategy':>16} {'mean P&L':>9} {'std':>7} "
              f"{'CVaR95':>8} {'(se)':>7} {'costs':>7}")
        rows = {}
        for name, fn in strategies(engine, sigs, rate, args.cost).items():
            pl, costs = run_book(spots, fn, premiums, args.cost, rate)
            rows[name] = pl
            cv = cvar(pl)
            se = cvar_bootstrap_se(pl)
            print(f"  {name:>16} {pl.mean():>+9.4f} {pl.std(ddof=1):>7.4f} "
                  f"{cv:>+8.4f} {se:>7.4f} {costs.mean():>7.4f}")
        d = rows["deep"] - rows["delta"]
        print(f"  paired deep-delta: mean {d.mean():+.4f}, "
              f"deep better on {np.mean(d > 0):.0%} of windows")


if __name__ == "__main__":
    main()
