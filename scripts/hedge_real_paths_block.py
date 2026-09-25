"""Moving-block bootstrap intervals for the real-history hedging comparison.

scripts/hedge_real_paths.py replays three hedgers over overlapping 30-day
windows of real daily closes and prints i.i.d. bootstrap standard errors.
Consecutive windows share 25 of 30 days, so those errors assume more
independence than the data has. This script rebuilds the same per-window P&L
with that script's own `build_windows`, `run_book` and `strategies`, then
resamples it with a moving-block bootstrap over consecutive windows (blocks of
1, 6, 12 and 24 windows, 4,000 resamples each). For each (asset, cost) it
reports CVaR95 standard errors per hedger, 95% intervals for the paired
deep-minus-delta CVaR95 and mean P&L differences and for the win rate, an
autocorrelation-based effective sample size, and t-statistics on the six
non-overlapping subsamples. A second pass shifts the start of the stride-5
window grid by 0 to 4 days on the same closes (the window phase).

Closes: the last 2,010 SPY and 2,922 BTC-USD adjusted daily closes up to
--end, the counts of the committed 2026-09-11 run (docs/hedging_real_paths.txt).
They are fetched through yfinance, or read from --closes, a JSON file of the
form {"SPY": {"dates": [...], "close": [...]}, "BTC-USD": {...}}. Yahoo revises
adjusted history, so a later fetch can differ slightly from the one behind the
committed JSON; the output records the first and last date of every series.
--shift 4 is the grid offset used for the committed intervals.

    python scripts/hedge_real_paths_block.py        # writes docs/hedging_real_paths_block.json

It runs in under a minute on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.hedging import (MATURITY, N_STEPS, TRAIN_BOX,  # noqa: E402
                                   HedgingEngine, bs_call_price, cvar,
                                   cvar_bootstrap_se)
from scripts.hedge_real_paths import build_windows, run_book, strategies  # noqa: E402

TARGET = {"SPY": 2010, "BTC-USD": 2922}
COSTS = (0.001, 0.005)
BLOCKS = (1, 6, 12, 24)
N_BOOT = 4000


def fetch_closes(years: int = 10) -> dict:
    import yfinance as yf
    out = {}
    for t in TARGET:
        px = yf.Ticker(t).history(period=f"{years}y", interval="1d", auto_adjust=True)["Close"].dropna()
        out[t] = {"dates": [str(d.date()) for d in px.index], "close": px.to_numpy().tolist()}
    return out


def acf_neff(x: np.ndarray, maxlag: int = 30) -> float:
    x = x - x.mean()
    n = len(x)
    v = np.dot(x, x) / n
    s = sum((1 - k / n) * (np.dot(x[:-k], x[k:]) / n / v) for k in range(1, maxlag + 1))
    return float(n / (1 + 2 * s))


def book(eng, closes: np.ndarray, rate: float, cost: float):
    wins, sraw = build_windows(closes)
    sig = np.clip(sraw, *TRAIN_BOX["sigma"])
    n = len(wins)
    spots = np.empty((n, N_STEPS + 1))
    spots[:, 0] = 1.0
    spots[:, 1:] = np.exp(np.cumsum(wins, 1))
    prem = bs_call_price(1.0, 1.0, MATURITY, sig, rate)
    pls = {name: run_book(spots, fn, prem, cost, rate)[0]
           for name, fn in strategies(eng, sig, rate, cost).items()}
    return pls, sraw, sig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--end", default="2026-09-11")
    p.add_argument("--shift", type=int, default=4)
    p.add_argument("--phases", type=int, default=5, help="grid offsets 0..phases-1 for the phase pass")
    p.add_argument("--closes", type=Path, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "hedging_real_paths_block.json")
    a = p.parse_args()

    C = json.loads(a.closes.read_text(encoding="utf-8")) if a.closes else fetch_closes()
    eng = HedgingEngine()
    rng = np.random.default_rng(a.seed)

    def mbb_idx(n, b):
        nb = math.ceil(n / b)
        starts = rng.integers(0, n - b + 1, size=(N_BOOT, nb))
        return (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(N_BOOT, -1)[:, :n]

    def cvar_rows(pl, idx):
        x = -pl[idx]
        k = int(math.ceil(0.05 * x.shape[1]))
        return np.sort(x, axis=1)[:, -k:].mean(1)

    res = {"command": "python scripts/hedge_real_paths_block.py", "end": a.end, "shift": a.shift,
           "n_boot": N_BOOT, "seed": a.seed, "block": {}, "phase": {}}
    for t, target in TARGET.items():
        d = np.array(C[t]["dates"])
        c = np.array(C[t]["close"], dtype=np.float64)
        keep = d <= a.end
        c, d = c[keep][-target:], d[keep][-target:]
        rate = 0.0 if t.startswith("BTC") else 0.04

        cs, ds = c[a.shift:], d[a.shift:]
        R = {"first_date": str(ds[0]), "last_date": str(ds[-1]), "n_closes": int(len(cs))}
        for cost in COSTS:
            pls, sraw, sig = book(eng, cs, rate, cost)
            n = len(sraw)
            R.update(n_windows=n, vol_median=float(np.median(sraw)), clipped=int((sig != sraw).sum()))
            row = {k: {"mean": float(v.mean()), "cvar95": cvar(v), "se_iid": cvar_bootstrap_se(v)}
                   for k, v in pls.items()}
            dlt = pls["deep"] - pls["delta"]
            row.update(paired_mean_diff=float(dlt.mean()), win_rate=float((dlt > 0).mean()),
                       paired_mean_diff_se_iid=float(dlt.std(ddof=1) / math.sqrt(n)),
                       neff_acf_diff=acf_neff(dlt), neff_acf_deep_pl=acf_neff(pls["deep"]),
                       neff_acf_winindicator=acf_neff((dlt > 0).astype(float)))
            for b in BLOCKS:
                idx = mbb_idx(n, b)
                cv = {k: cvar_rows(v, idx) for k, v in pls.items()}
                md, wr = dlt[idx].mean(1), (dlt[idx] > 0).mean(1)
                cvd = cv["deep"] - cv["delta"]
                row[f"mbb_b{b}"] = {
                    "cvar_se": {k: float(v.std(ddof=1)) for k, v in cv.items()},
                    "cvar_diff_deep_minus_delta": float(cvar(pls["deep"]) - cvar(pls["delta"])),
                    "cvar_diff_se": float(cvd.std(ddof=1)),
                    "cvar_diff_ci95": np.percentile(cvd, [2.5, 97.5]).tolist(),
                    "mean_diff_se": float(md.std(ddof=1)),
                    "mean_diff_ci95": np.percentile(md, [2.5, 97.5]).tolist(),
                    "win_rate_ci95": np.percentile(wr, [2.5, 97.5]).tolist()}
            sub = []
            for off in range(6):
                s = slice(off, None, 6)
                x = dlt[s]
                sub.append({"n": int(x.size), "mean_diff": float(x.mean()),
                            "t": float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size))),
                            "cvar_deep": cvar(pls["deep"][s]), "cvar_delta": cvar(pls["delta"][s])})
            row["nonoverlap_subsamples"] = sub
            R[f"cost_{cost}"] = row
            m = row["mbb_b6"]
            print(f"{t} cost {cost}: cvar diff {m['cvar_diff_deep_minus_delta']:+.4f} "
                  f"ci {np.round(m['cvar_diff_ci95'], 4)}  mean diff {row['paired_mean_diff']:+.4f} "
                  f"ci {np.round(m['mean_diff_ci95'], 4)}  win {row['win_rate']:.0%} "
                  f"ci {np.round(m['win_rate_ci95'], 2)}", flush=True)
        res["block"][t] = R

        rows = []
        for k in range(a.phases):
            r = {"shift": k}
            for cost in COSTS:
                pls, sraw, _ = book(eng, c[k:], rate, cost)
                r["n"] = len(sraw)
                r[f"{cost}"] = {nm: cvar(v) for nm, v in pls.items()}
                dl = pls["deep"] - pls["delta"]
                r[f"{cost}"]["pair_mean"] = float(dl.mean())
                r[f"{cost}"]["win"] = float((dl > 0).mean())
            rows.append(r)
        res["phase"][t] = rows
        lo = min(x["0.001"]["deep"] for x in rows)
        hi = max(x["0.001"]["deep"] for x in rows)
        print(f"{t} phase 0..{a.phases - 1}: 10 bp deep CVaR95 {lo:.4f} to {hi:.4f}", flush=True)

    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
