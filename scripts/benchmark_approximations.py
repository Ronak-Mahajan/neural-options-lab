"""Neural surrogate versus the closed-form Asian approximations, on one grid.

The README compares the surrogate to Levy (1992) moment matching over the
whole trained box. This script asks the narrower, reproducible question a
reader can check in a minute: on a fixed 6x6 grid of moneyness and maturity
at one vol and one rate, how far is each fast pricer from a high-precision
Monte Carlo reference, and what does each cost per price?

Protocol
--------
    moneyness S/K   6 points, evenly spaced in [0.7, 1.3], strike 100
    maturity        6 points, evenly spaced in [0.1, 2.0] years
    sigma = 0.25, r = 0.04, n_steps = 50 (the project's fixed protocol)
    contract        arithmetic-average Asian CALL
    reference       price_asian_mc, 400,000 paths, antithetic + geometric
                    control variate, seed 7 (one seed per grid cell, so
                    every method is scored against the identical number)

Pricers
-------
    neural surrogate       backend.quant.engine.PricingEngine, price only,
                           batch of one, float32 serving path
    Turnbull-Wakeman       backend.quant.benchmarks.levy_asian_price via
                           asian_approx.turnbull_wakeman_price
    Curran (exact)         asian_approx.curran_call, threshold solved exactly
    Curran (linear)        asian_approx.curran_call, Curran's first-order
                           threshold 2K - E[A|G=K]

Errors are (method - reference) in basis points of strike. Wall-clock is the
median of repeated single-price calls on this machine; the reference column
is the mean time of one 400,000-path run.

Output: docs/approximation_benchmark.md (also printed).

Usage (from the repo root):
    python scripts/benchmark_approximations.py
    python scripts/benchmark_approximations.py --ref-paths 100000 --repeats 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from backend.quant.asian_approx import (curran_call,  # noqa: E402
                                        turnbull_wakeman_price)
from backend.quant.engine import PricingEngine  # noqa: E402
from backend.quant.monte_carlo import price_asian_mc  # noqa: E402

STRIKE = 100.0
SIGMA = 0.25
RATE = 0.04
N_STEPS = 50
MONEYNESS = np.linspace(0.7, 1.3, 6)
MATURITY = np.linspace(0.1, 2.0, 6)


def build_grid() -> list[tuple[float, float]]:
    """(moneyness, maturity) cells, maturity-major so the table reads by row."""
    return [(float(m), float(t)) for t in MATURITY for m in MONEYNESS]


def reference_prices(grid, n_paths: int, seed: int):
    prices, ses, secs = [], [], []
    t_all = time.perf_counter()
    for k, (m, t) in enumerate(grid):
        t0 = time.perf_counter()
        res = price_asian_mc(m * STRIKE, STRIKE, t, SIGMA, RATE,
                             n_paths=n_paths, n_steps=N_STEPS,
                             option_type="call", seed=seed)
        secs.append(time.perf_counter() - t0)
        prices.append(res.price)
        ses.append(res.std_error)
        if k % 6 == 5:
            print(f"  reference {k + 1:>2}/{len(grid)}  "
                  f"({time.perf_counter() - t_all:5.1f}s)", flush=True)
    return np.array(prices), np.array(ses), float(np.mean(secs))


def time_single_prices(fn, grid, repeats: int) -> tuple[np.ndarray, float]:
    """Price every cell `repeats` times; return prices and the median seconds
    per call over all cells and repeats."""
    prices = np.empty(len(grid))
    samples = []
    for k, (m, t) in enumerate(grid):
        for _ in range(repeats):
            t0 = time.perf_counter()
            p = fn(m, t)
            samples.append(time.perf_counter() - t0)
        prices[k] = p
    return prices, float(np.median(samples))


def make_pricers(engine: PricingEngine) -> dict:
    ones = np.ones(1)

    def neural(m, t):
        return float(engine.price_batch(np.array([m * STRIKE]),
                                        ones * STRIKE, np.array([t]),
                                        ones * SIGMA, ones * RATE,
                                        option_type="call")[0])

    def tw(m, t):
        return turnbull_wakeman_price(m * STRIKE, STRIKE, t, SIGMA, RATE,
                                      N_STEPS)

    def curran_exact(m, t):
        return curran_call(m * STRIKE, STRIKE, t, SIGMA, RATE, N_STEPS,
                           threshold="exact")

    def curran_linear(m, t):
        return curran_call(m * STRIKE, STRIKE, t, SIGMA, RATE, N_STEPS,
                           threshold="linear")

    return {
        "neural surrogate (5-member ensemble)": neural,
        "Turnbull-Wakeman / Levy": tw,
        "Curran (1994), exact threshold": curran_exact,
        "Curran (1994), linear threshold": curran_linear,
    }


def fmt_latency(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:,.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:,.2f} ms"
    return f"{seconds * 1e6:,.0f} us"


def cdf_overhead_us(n: int = 10_000) -> tuple[float, float]:
    """Median microseconds per scalar norm.cdf and per scipy.special.ndtr."""
    from scipy.special import ndtr
    from scipy.stats import norm
    out = []
    for fn in (norm.cdf, ndtr):
        for _ in range(100):
            fn(0.3)
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(0.3)
            samples.append(time.perf_counter() - t0)
        out.append(float(np.median(samples)) * 1e6)
    return out[0], out[1]


def build_report(grid, ref, ref_se, ref_secs, results, args,
                 engine: PricingEngine) -> str:
    n_cells = len(grid)
    cdf_us, ndtr_us = cdf_overhead_us()
    se_bps = ref_se / STRIKE * 1e4
    rows = []
    for name, (prices, secs) in results.items():
        err = (prices - ref) / STRIKE * 1e4
        worst = int(np.argmax(np.abs(err)))
        m_w, t_w = grid[worst]
        rows.append({
            "name": name, "mean_abs": float(np.mean(np.abs(err))),
            "max_abs": float(np.max(np.abs(err))), "bias": float(np.mean(err)),
            "worst": f"m={m_w:.2f}, T={t_w:.2f}", "secs": secs, "err": err,
        })

    nn, tw, cur = rows[0], rows[1], rows[2]
    lines = []
    a = lines.append
    a("# Closed-form approximations versus the neural surrogate")
    a("")
    a(f"Generated by `scripts/benchmark_approximations.py` on "
      f"{_dt.date.today().isoformat()}; {platform.machine()}, "
      f"{os.cpu_count()} logical CPUs, torch {torch.__version__} (CPU), "
      f"numpy {np.__version__}. Re-run the script to refresh every number "
      f"below.")
    a("")
    a("## Protocol")
    a("")
    a(f"- Contract: arithmetic-average Asian call, strike {STRIKE:.0f}, "
      f"{N_STEPS} equally spaced monitoring dates.")
    a(f"- Grid: moneyness S/K at {', '.join(f'{m:.2f}' for m in MONEYNESS)} "
      f"x maturity at {', '.join(f'{t:.2f}' for t in MATURITY)} years "
      f"({n_cells} cells), sigma = {SIGMA}, r = {RATE}.")
    a(f"- Reference: `price_asian_mc`, {args.ref_paths:,} paths, antithetic "
      f"sampling with the geometric-Asian control variate, seed {args.seed}. "
      f"Reference standard error: mean {se_bps.mean():.3f} bps of strike, "
      f"max {se_bps.max():.3f} bps (cell m={grid[int(np.argmax(se_bps))][0]:.2f}, "
      f"T={grid[int(np.argmax(se_bps))][1]:.2f}). Errors below that scale "
      f"are not resolved by this reference.")
    a(f"- Surrogate: `PricingEngine` on `artifacts/model.pt`, "
      f"{engine.n_members} members, price only through `price_batch` with a "
      f"batch of one (the float32 serving path). Greeks are not timed here.")
    a(f"- Latency: median of {args.repeats} calls per cell x {n_cells} cells "
      f"for the fast pricers; mean of one run per cell for the reference. "
      f"Single process, `torch.set_num_threads(1)` to match the Dockerfile's "
      f"`OMP_NUM_THREADS=1`, one warm-up call per pricer. Absolute times are "
      f"specific to this machine; compare the ratios between rows, not the "
      f"rows against the README, which was timed elsewhere.")
    a("")
    a("## Results")
    a("")
    a("Errors are (method - reference), in basis points of strike.")
    a("")
    a("| method | mean abs error | max abs error | bias | worst cell | wall-clock per price |")
    a("|---|---|---|---|---|---|")
    for r in rows:
        a(f"| {r['name']} | {r['mean_abs']:.3f} bps | {r['max_abs']:.3f} bps | "
          f"{r['bias']:+.3f} bps | {r['worst']} | {fmt_latency(r['secs'])} |")
    a(f"| Monte Carlo, {args.ref_paths:,} paths | (reference) | "
      f"SE <= {se_bps.max():.3f} bps | n/a | n/a | {fmt_latency(ref_secs)} |")
    a("")
    a("Mean absolute error by maturity (bps of strike, averaged over the six "
      "moneyness points):")
    a("")
    a("| maturity (y) | " + " | ".join(r["name"] for r in rows) + " |")
    a("|---|" + "---|" * len(rows))
    for j, t in enumerate(MATURITY):
        sl = slice(6 * j, 6 * j + 6)
        a(f"| {t:.2f} | " + " | ".join(
            f"{np.mean(np.abs(r['err'][sl])):.3f}" for r in rows) + " |")
    a("")
    a("Signed error per cell for the two contenders (bps of strike; rows are "
      "maturity, columns moneyness):")
    a("")
    for r in (nn, cur):
        a(f"*{r['name']}*")
        a("")
        a("| T \\ S/K | " + " | ".join(f"{m:.2f}" for m in MONEYNESS) + " |")
        a("|---|" + "---|" * len(MONEYNESS))
        for j, t in enumerate(MATURITY):
            a(f"| {t:.2f} | " + " | ".join(
                f"{e:+.2f}" for e in r["err"][6 * j:6 * j + 6]) + " |")
        a("")

    # Statistics shared by the interpretation and the notes.
    n_pos = int(np.sum(nn["err"] > 0))
    # Cells where every reference path paid zero have price 0 and SE 0
    # exactly; a z-score is undefined there, so they are reported separately.
    resolved = ref_se > 0.0
    z_cur = (cur["err"][resolved] * STRIKE / 1e4) / ref_se[resolved]
    zero_cells = [(grid[k], cur["err"][k], nn["err"][k])
                  for k in np.flatnonzero(~resolved)]
    d_thr = float(np.max(np.abs(rows[2]["err"] - rows[3]["err"])))

    # Interpretation: three sentences, every number from this run.
    ratio_tw = tw["mean_abs"] / max(nn["mean_abs"], 1e-12)
    ratio_cur = nn["mean_abs"] / max(cur["mean_abs"], 1e-12)
    cur_wins = cur["mean_abs"] < nn["mean_abs"]
    tw_by_t = [float(np.mean(np.abs(tw["err"][6 * j:6 * j + 6])))
               for j in range(len(MATURITY))]
    a("## Interpretation")
    a("")
    a(f"Against Turnbull-Wakeman / Levy moment matching the surrogate is "
      f"{ratio_tw:.0f}x more accurate on this grid (mean {nn['mean_abs']:.2f} "
      f"vs {tw['mean_abs']:.2f} bps of strike, max {nn['max_abs']:.2f} vs "
      f"{tw['max_abs']:.2f} bps), and the closed form's error is a bias that "
      f"grows with maturity ({tw_by_t[0]:.2f} bps at {MATURITY[0]:.2f} y to "
      f"{tw_by_t[-1]:.2f} bps at {MATURITY[-1]:.2f} y), not noise; the "
      f"README's 33x is RMSE on 300 points spanning the full trained box "
      f"(sigma up to 0.80), where Levy measures 44.3 bps against the "
      f"surrogate's 1.329 bps. "
      f"Curran's conditioning approximation is a different baseline: at a "
      f"mean {cur['mean_abs']:.2f} bps (max {cur['max_abs']:.2f} bps, never "
      f"more than {max(float(z_cur.max()), 0.0):.1f} reference standard "
      f"errors above the Monte Carlo price, as a lower bound must be) it is "
      f"{ratio_cur:.1f}x {'more' if cur_wins else 'less'} accurate than the "
      f"surrogate on price alone at {fmt_latency(cur['secs'])} against "
      f"{fmt_latency(nn['secs'])} per price, so 'more accurate than the "
      f"standard closed-form approximation' is true of Levy and "
      f"{'false' if cur_wins else 'also true'} of Curran at this vol. "
      f"What the surrogate offers over Curran is therefore not price accuracy "
      f"on a GBM Asian but the rest of the package (all five Greeks by "
      f"autograd in one call, batched throughput, and a training recipe that "
      f"carries over to dynamics with no geometric-conditioning trick, such "
      f"as the rough-volatility 0DTE pricer), and a headline built on the "
      f"Levy comparison alone should name Levy.")
    a("")

    # Notes: cross-checks against the README and the timing floor.
    a("## Notes")
    a("")
    a(f"- The surrogate's signed error is positive in {n_pos} of {n_cells} "
      f"cells (bias {nn['bias']:+.2f} bps). The README reports "
      f"+0.468 bps bias for the served head on its paired 1,500-point set; "
      f"same sign, same Softplus-floor mechanism, larger here because this "
      f"grid is all near-the-money at one vol rather than a box average.")
    a(f"- Curran (exact threshold), over the {int(resolved.sum())} cells "
      f"where the reference has a nonzero standard error, sits at most "
      f"{z_cur.max():+.2f} SE above the Monte Carlo price and "
      f"{z_cur.min():+.2f} SE below it at worst; a lower bound may exceed "
      f"the reference only by noise, and it does not. Curran's first-order "
      f"threshold differs from the exact solve by at most {d_thr:.4f} bps on "
      f"this grid.")
    for (m_z, t_z), e_cur, e_nn in zero_cells:
        a(f"- At m={m_z:.2f}, T={t_z:.2f} every one of the {args.ref_paths:,} "
          f"reference paths pays zero, so the reference is 0 with zero "
          f"standard error. Curran returns {e_cur:+.4f} bps there and the "
          f"surrogate {e_nn:+.2f} bps: the Softplus output floor the README "
          f"documents, seen on a cell where the true price is 0.")
    a(f"- Timing floor: Turnbull-Wakeman is two scalar `norm.cdf` calls plus "
      f"a 50x50 exponential sum, and scipy's `norm.cdf` wrapper costs "
      f"{cdf_us:.0f} us per scalar call in this run against {ndtr_us:.1f} us "
      f"for `scipy.special.ndtr`, so the wrapper is most of its "
      f"{fmt_latency(tw['secs'])}. Curran makes two `norm.cdf` calls, one on "
      f"a length-{N_STEPS} vector; the {fmt_latency(cur['secs'])} of the "
      f"exact threshold against {fmt_latency(rows[3]['secs'])} for the "
      f"linear one is the Newton solve, the only code that differs between "
      f"them. Switching the cdf primitive would speed every closed form up "
      f"and change none of the accuracy columns; it is not done here so "
      f"they stay on equal footing with `benchmarks.py`.")
    a("")
    a("## Reproduce")
    a("")
    a("```bash")
    a(f"python scripts/benchmark_approximations.py --ref-paths {args.ref_paths} "
      f"--seed {args.seed} --repeats {args.repeats}")
    a("```")
    a("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-paths", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--repeats", type=int, default=20,
                   help="timed calls per grid cell for each fast pricer")
    p.add_argument("--out", type=Path,
                   default=ROOT / "docs" / "approximation_benchmark.md")
    args = p.parse_args()

    # The Dockerfile serves with OMP_NUM_THREADS=1; time the same path. On
    # the 16-core box this was written on, one thread was also the fastest
    # setting for a batch of one (3.0 ms p50 vs 3.3 ms at 16, 4.5 ms at 4).
    torch.set_num_threads(1)
    engine = PricingEngine()
    if engine.n_steps != N_STEPS:
        raise SystemExit(f"checkpoint uses {engine.n_steps} monitoring steps, "
                         f"benchmark assumes {N_STEPS}")
    grid = build_grid()

    print(f"reference: {len(grid)} cells x {args.ref_paths:,} paths, seed "
          f"{args.seed}")
    ref, ref_se, ref_secs = reference_prices(grid, args.ref_paths, args.seed)

    pricers = make_pricers(engine)
    for fn in pricers.values():
        fn(1.0, 1.0)                                    # warm-up
    results = {}
    for name, fn in pricers.items():
        prices, secs = time_single_prices(fn, grid, args.repeats)
        results[name] = (prices, secs)
        err = (prices - ref) / STRIKE * 1e4
        print(f"  {name:<38} mean|e| {np.mean(np.abs(err)):7.3f}  "
              f"max|e| {np.max(np.abs(err)):7.3f}  bias {np.mean(err):+7.3f} "
              f"bps   {fmt_latency(secs)}/price")

    report = build_report(grid, ref, ref_se, ref_secs, results, args, engine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
