"""Neural surrogate versus the closed-form Asian approximations.

Scores the served ensemble, Turnbull-Wakeman / Levy (1992) moment matching and
Curran (1994) conditioning against control-variate Monte Carlo on two point
sets, and times a single price from each. Every per-point price is written to
docs/approximation_benchmark.json, and docs/approximation_benchmark.md is
rendered from that file, so the page and the artifact hold the same digits.
Each mode rewrites its own key and leaves the other in place.

Box mode (--lhs), JSON key "box_lhs"
------------------------------------
    points          300 Latin-hypercube points over dataset.PARAM_RANGES, the
                    box the surrogate is trained on: S/K 0.5-2.0, maturity
                    0.05-2.0 y, sigma 0.05-0.80, r 0.00-0.10. Seed 1992; the
                    training, ablation, promotion and evaluation draws use
                    7, 20261, 606060 and 99.
    contract        arithmetic-average Asian call, strike 100, n_steps = 50
    reference       dataset._simulate_chunk, 200,000 paths, antithetic +
                    geometric control variate: the estimator
                    backend.quant.evaluate scores the surrogate against
    reference SE    _simulate_chunk returns no standard error, so each point
                    is priced a second time by price_asian_mc (the same
                    estimator on an independent draw), which reports one
                    over antithetic pairs. The two prices also check that
                    standard error: their difference in units of sqrt(2) SE
                    has an RMS near 1.
    statistics      RMSE, MAE, bias, 95th percentile and max of
                    (method - reference); the Levy / surrogate and
                    surrogate / Curran RMSE ratios with a paired percentile
                    bootstrap over the points; RMSE by sigma band and by
                    reference price level

Grid mode (default), JSON key "grid"
------------------------------------
    moneyness S/K   6 points, evenly spaced in [0.7, 1.3], strike 100
    maturity        6 points, evenly spaced in [0.1, 2.0] years
    sigma = 0.25, r = 0.04, n_steps = 50 (the project's fixed protocol)
    contract        arithmetic-average Asian call
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
median of repeated single-price calls on this machine; the Monte Carlo row is
the mean time of one reference-sized run. Seeded prices and errors reproduce
digit for digit; timings move with the machine, so each run also records the
system-wide CPU utilisation and the power source when psutil is installed.

Output: docs/approximation_benchmark.json and docs/approximation_benchmark.md
(also printed).

Usage (from the repo root):
    python scripts/benchmark_approximations.py --lhs            # box, ~8 min
    python scripts/benchmark_approximations.py                  # grid, ~1 min
    python scripts/benchmark_approximations.py --ref-paths 100000 --repeats 5
    python scripts/benchmark_approximations.py --render-only    # page from JSON
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scipy.stats import norm, qmc  # noqa: E402

try:                       # optional: records system load while timing
    import psutil  # noqa: E402
except ImportError:        # pragma: no cover
    psutil = None

from backend.quant.asian_approx import (curran_call,  # noqa: E402
                                        turnbull_wakeman_price)
from backend.quant.dataset import PARAM_RANGES, _simulate_chunk  # noqa: E402
from backend.quant.engine import PricingEngine  # noqa: E402
from backend.quant.evaluate import (SERVED_CHECKPOINT,  # noqa: E402
                                    checkpoint_fingerprint)
from backend.quant.monte_carlo import price_asian_mc  # noqa: E402

STRIKE = 100.0
SIGMA = 0.25
RATE = 0.04
N_STEPS = 50
MONEYNESS = np.linspace(0.7, 1.3, 6)
MATURITY = np.linspace(0.1, 2.0, 6)

# Box-mode defaults. The seed is one no training or model-selection draw uses
# (train 7, ablation evaluation 20261, promotion gate 606060, evaluate 99).
BOX_POINTS = 300
BOX_SEED = 1992
BOX_REF_PATHS = 200_000
GRID_SEED = 7
GRID_REF_PATHS = 400_000

# Reference-price bucket edges, in bps of strike.
PRICE_EDGES_BPS = (1.0, 10.0, 100.0, 1000.0)
PRICE_LABELS = ("below 1", "1 to 10", "10 to 100", "100 to 1,000",
                "1,000 and above")

JSON_PATH = ROOT / "docs" / "approximation_benchmark.json"
MD_PATH = ROOT / "docs" / "approximation_benchmark.md"


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


def time_single_prices(fn, points, repeats: int) -> tuple[np.ndarray, float]:
    """Price every point `repeats` times; return prices and the median seconds
    per call over all points and repeats. A point is the pricer's positional
    arguments: (m, T) on the grid, (m, T, sigma, r) in the box."""
    prices = np.empty(len(points))
    samples = []
    for k, pt in enumerate(points):
        for _ in range(repeats):
            t0 = time.perf_counter()
            p = fn(*pt)
            samples.append(time.perf_counter() - t0)
        prices[k] = p
    return prices, float(np.median(samples))


def make_pricers(engine: PricingEngine) -> dict:
    ones = np.ones(1)

    def neural(m, t, sig=SIGMA, r=RATE):
        return float(engine.price_batch(np.array([m * STRIKE]),
                                        ones * STRIKE, np.array([t]),
                                        ones * sig, ones * r,
                                        option_type="call")[0])

    def tw(m, t, sig=SIGMA, r=RATE):
        return turnbull_wakeman_price(m * STRIKE, STRIKE, t, sig, r, N_STEPS)

    def curran_exact(m, t, sig=SIGMA, r=RATE):
        return curran_call(m * STRIKE, STRIKE, t, sig, r, N_STEPS,
                           threshold="exact")

    def curran_linear(m, t, sig=SIGMA, r=RATE):
        return curran_call(m * STRIKE, STRIKE, t, sig, r, N_STEPS,
                           threshold="linear")

    return {
        f"neural surrogate ({engine.n_members}-member ensemble)": neural,
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


def environment() -> dict:
    return {"machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "torch": torch.__version__, "numpy": np.__version__}


def _env_line(env: dict) -> str:
    return (f"{env['machine']}, {env['logical_cpus']} logical CPUs, torch "
            f"{env['torch']} (CPU), numpy {env['numpy']}")


class TimingConditions:
    """System-wide CPU utilisation over a timed run, and the power source.

    A single-price latency on a shared machine moves with what else is
    running and with the CPU's power state, so the artifact records both next
    to the numbers they affect. Needs psutil; without it the fields are None.
    """

    def __enter__(self):
        if psutil is not None:
            psutil.cpu_percent(interval=None)          # start the window
        return self

    def __exit__(self, *exc) -> None:
        self.record = {"system_cpu_percent": None, "on_battery": None}
        if psutil is not None:
            self.record["system_cpu_percent"] = float(
                psutil.cpu_percent(interval=None))
            battery = getattr(psutil, "sensors_battery", lambda: None)()
            if battery is not None:
                self.record["on_battery"] = not battery.power_plugged


def _conditions_line(section: dict) -> str | None:
    cond = section.get("timing_conditions") or {}
    load = cond.get("system_cpu_percent")
    if load is None:
        return None
    cpus = section["environment"]["logical_cpus"]
    power = {True: " The machine ran on battery power.",
             False: " The machine ran on mains power.",
             None: ""}[cond.get("on_battery")]
    return (f"- Timing conditions: system-wide CPU utilisation averaged "
            f"{load:.0f}% of {cpus} logical CPUs over the run, of which this "
            f"benchmark is one thread.{power}")


def _span(bounds) -> str:
    return f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"


def _stored(x: np.ndarray) -> list[float]:
    """Prices as stored in the JSON: 1e-10 of a strike-100 price is 1e-8 bps,
    far below any figure reported from them."""
    return np.round(np.asarray(x, dtype=float), 10).tolist()


# ---------------------------------------------------------------------------
# Box mode: Latin-hypercube points over the trained box
# ---------------------------------------------------------------------------

def sample_box(n_points: int, seed: int) -> np.ndarray:
    """(n, 4) Latin-hypercube points (m, T, sigma, r) over PARAM_RANGES."""
    lows = np.array([lo for lo, _ in PARAM_RANGES.values()])
    highs = np.array([hi for _, hi in PARAM_RANGES.values()])
    unit = qmc.LatinHypercube(d=4, seed=seed).random(n_points)
    return lows + unit * (highs - lows)


def _point_seed(seed: int, stream: int, i: int) -> int:
    """Integer seed for point i of one stream. SeedSequence keeps the streams
    of one run disjoint, so the whole run is a function of --seed."""
    return int(np.random.SeedSequence([seed, stream, i]).generate_state(1)[0])


def box_references(X: np.ndarray, n_paths: int, seed: int) -> dict:
    """Reference price per point from _simulate_chunk, plus an independent
    price_asian_mc run of the same estimator for the standard error that
    _simulate_chunk does not return. Prices are at strike STRIKE."""
    n = len(X)
    price, price_b, se = np.empty(n), np.empty(n), np.empty(n)
    chunk_secs, mc_secs = [], []
    t_all = time.perf_counter()
    for i in range(n):
        m, t, sig, r = (float(v) for v in X[i])
        rng = np.random.default_rng(_point_seed(seed, 0, i))
        t0 = time.perf_counter()
        unit_price, _, _ = _simulate_chunk(X[i:i + 1], n_paths, N_STEPS, rng)
        chunk_secs.append(time.perf_counter() - t0)
        price[i] = unit_price[0] * STRIKE

        t0 = time.perf_counter()
        res = price_asian_mc(m * STRIKE, STRIKE, t, sig, r, n_paths=n_paths,
                             n_steps=N_STEPS, option_type="call",
                             seed=_point_seed(seed, 1, i))
        mc_secs.append(time.perf_counter() - t0)
        price_b[i], se[i] = res.price, res.std_error
        if i % 50 == 49:
            print(f"  reference {i + 1:>4}/{n}  "
                  f"({time.perf_counter() - t_all:5.1f}s)", flush=True)
    return {"price": price, "independent_price": price_b, "std_error": se,
            "simulate_chunk_mean_seconds": float(np.mean(chunk_secs)),
            "price_only_mean_seconds": float(np.mean(mc_secs))}


def summarize_bps(err_bps: np.ndarray) -> dict:
    a = np.abs(err_bps)
    return {"rmse_bps": float(np.sqrt(np.mean(err_bps ** 2))),
            "mae_bps": float(np.mean(a)),
            "bias_bps": float(np.mean(err_bps)),
            "p95_abs_bps": float(np.percentile(a, 95)),
            "max_abs_bps": float(np.max(a)),
            "worst_point": int(np.argmax(a))}


def rmse_ratio_interval(err_num: np.ndarray, err_den: np.ndarray,
                        n_boot: int, rng: np.random.Generator) -> dict:
    """RMSE(err_num) / RMSE(err_den) with a 95% percentile bootstrap interval.

    Points are resampled jointly, so both RMSEs in a resample are taken on the
    same points. Squared error over this box is concentrated in the
    high-volatility points, so a point estimate of the ratio moves with the
    draw; the interval is its spread over resamples of the points.
    """
    n = len(err_num)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = np.sqrt(np.mean(err_num[idx] ** 2, axis=1))
    den = np.sqrt(np.mean(err_den[idx] ** 2, axis=1))
    lo, hi = np.percentile(num / np.maximum(den, 1e-300), [2.5, 97.5])
    ratio = float(np.sqrt(np.mean(err_num ** 2))
                  / max(float(np.sqrt(np.mean(err_den ** 2))), 1e-300))
    return {"ratio": ratio, "ci95": [float(lo), float(hi)]}


def box_statistics(box: dict) -> dict:
    """Every summary figure of the box section, from its stored per-point
    prices. The run stores the result; tests recompute it from the JSON."""
    ref = np.array(box["reference"]["price"])
    se_bps = np.array(box["reference"]["std_error"]) / STRIKE * 1e4
    ref_bps = ref / STRIKE * 1e4
    sig = np.array(box["points"]["sigma"])
    names = list(box["methods"])
    err = {k: (np.array(v["price"]) - ref) / STRIKE * 1e4
           for k, v in box["methods"].items()}

    sig_lo, sig_hi = PARAM_RANGES["sigma"]
    sig_edges = np.linspace(sig_lo, sig_hi, 4)
    sig_band = np.digitize(sig, sig_edges[1:-1])
    price_band = np.digitize(ref_bps, PRICE_EDGES_BPS)

    def banded(band, labels):
        rows = []
        for j, label in enumerate(labels):
            sel = band == j
            if not sel.any():
                continue
            rows.append({"label": label, "n": int(sel.sum()),
                         "rmse_bps": {k: float(np.sqrt(np.mean(err[k][sel] ** 2)))
                                      for k in names}})
        return rows

    sig_labels = [f"{sig_edges[j]:.2f} to {sig_edges[j + 1]:.2f}"
                  for j in range(3)]
    resolved = se_bps > 0.0
    dz = ((ref - np.array(box["reference"]["independent_price"]))
          / STRIKE * 1e4)[resolved] / (np.sqrt(2.0) * se_bps[resolved])
    worst_se = int(np.argmax(se_bps))
    return {
        "summary": {k: summarize_bps(err[k]) for k in names},
        "by_sigma": banded(sig_band, sig_labels),
        "by_reference_price": banded(price_band, PRICE_LABELS),
        "reference_se_bps": {"mean": float(se_bps.mean()),
                             "rms": float(np.sqrt(np.mean(se_bps ** 2))),
                             "max": float(se_bps.max()),
                             "max_point": worst_se},
        "reference_cross_check": {"n_points": int(resolved.sum()),
                                  "rms_z": float(np.sqrt(np.mean(dz ** 2)))},
    }


def run_box(engine: PricingEngine, args) -> dict:
    X = sample_box(args.lhs, args.seed)
    print(f"reference: {args.lhs} Latin-hypercube points x "
          f"{args.ref_paths:,} paths, seed {args.seed}")
    points = [tuple(float(v) for v in row) for row in X]
    pricers = make_pricers(engine)
    for fn in pricers.values():
        fn(1.0, 1.0)                                    # warm-up
    methods = {}
    with TimingConditions() as conditions:
        ref = box_references(X, args.ref_paths, args.seed)
        for name, fn in pricers.items():
            prices, secs = time_single_prices(fn, points, args.repeats)
            methods[name] = {"price": _stored(prices), "median_seconds": secs}

    box = {
        "date": _dt.date.today().isoformat(),
        "environment": environment(),
        "command": (f"python scripts/benchmark_approximations.py --lhs "
                    f"{args.lhs} --seed {args.seed} --ref-paths "
                    f"{args.ref_paths} --repeats {args.repeats} --bootstrap "
                    f"{args.bootstrap}"),
        "timing_conditions": conditions.record,
        "checkpoint": checkpoint_fingerprint(SERVED_CHECKPOINT),
        "n_members": engine.n_members,
        "protocol": {"n_points": args.lhs, "seed": args.seed,
                     "ref_paths": args.ref_paths, "n_steps": N_STEPS,
                     "strike": STRIKE, "repeats": args.repeats,
                     "bootstrap_resamples": args.bootstrap,
                     "param_ranges": {k: list(v)
                                      for k, v in PARAM_RANGES.items()}},
        "points": {k: _stored(X[:, j]) for j, k in enumerate(PARAM_RANGES)},
        "reference": {
            "price": _stored(ref["price"]),
            "std_error": _stored(ref["std_error"]),
            "independent_price": _stored(ref["independent_price"]),
            "simulate_chunk_mean_seconds": ref["simulate_chunk_mean_seconds"],
            "price_only_mean_seconds": ref["price_only_mean_seconds"],
        },
        "methods": methods,
    }
    box.update(box_statistics(box))

    nn, tw, cur = list(methods)[:3]
    ref_price = np.array(box["reference"]["price"])
    err = {k: np.array(methods[k]["price"]) - ref_price for k in (nn, tw, cur)}
    rng = np.random.default_rng(_point_seed(args.seed, 2, 0))
    box["rmse_ratios"] = {
        "levy_over_surrogate": rmse_ratio_interval(err[tw], err[nn],
                                                   args.bootstrap, rng),
        "surrogate_over_curran": rmse_ratio_interval(err[nn], err[cur],
                                                     args.bootstrap, rng),
    }
    for name in methods:
        s = box["summary"][name]
        print(f"  {name:<38} RMSE {s['rmse_bps']:8.3f}  bias "
              f"{s['bias_bps']:+8.3f}  p95 {s['p95_abs_bps']:8.3f}  max "
              f"{s['max_abs_bps']:8.3f} bps   "
              f"{fmt_latency(methods[name]['median_seconds'])}/price")
    return box


def _net_of_reference(rmse: float, se_rms: float) -> str:
    """RMSE with the reference's RMS standard error removed in quadrature.
    The reference noise is independent of every pricer's error, so the two
    add in squares."""
    net2 = rmse ** 2 - se_rms ** 2
    if net2 <= 0.0:
        return "below the reference's resolution"
    return f"{np.sqrt(net2):.3f} bps"


def render_box(box: dict, grid: dict | None) -> list[str]:
    pr, stats, ref = box["protocol"], box["summary"], box["reference"]
    names = list(box["methods"])
    nn, tw, cur, cur_lin = names[:4]
    n = pr["n_points"]
    pts = box["points"]
    se = box["reference_se_bps"]
    ranges = pr["param_ranges"]
    lat = {k: box["methods"][k]["median_seconds"] for k in names}

    def where(i: int) -> str:
        return (f"m={pts['moneyness'][i]:.2f}, T={pts['maturity'][i]:.2f}, "
                f"sigma={pts['sigma'][i]:.2f}, r={pts['rate'][i]:.3f}")

    lines: list[str] = []
    a = lines.append
    a(f"## Trained box, {n} Latin-hypercube points")
    a("")
    a(f"Measured on {box['date']}; {_env_line(box['environment'])}. "
      f"Checkpoint `{box['checkpoint']['file']}`, sha256 "
      f"`{box['checkpoint']['sha256'][:16]}`. Stored under `box_lhs` in "
      f"`docs/approximation_benchmark.json`.")
    a("")
    a("### Protocol")
    a("")
    a(f"- Contract: arithmetic-average Asian call, strike {pr['strike']:.0f}, "
      f"{pr['n_steps']} equally spaced monitoring dates.")
    a(f"- Points: {n} Latin-hypercube points "
      f"(`scipy.stats.qmc.LatinHypercube`, seed {pr['seed']}) over the box "
      f"the surrogate is trained on: S/K in {_span(ranges['moneyness'])}, "
      f"maturity in {_span(ranges['maturity'])} years, sigma in "
      f"{_span(ranges['sigma'])}, r in {_span(ranges['rate'])}. No training "
      f"or model-selection draw uses this seed.")
    a(f"- Reference: `dataset._simulate_chunk`, {pr['ref_paths']:,} paths, "
      f"antithetic sampling with the geometric-Asian control variate, the "
      f"estimator `backend/quant/evaluate.py` scores the surrogate against.")
    a(f"- Reference standard error: `_simulate_chunk` returns none, so each "
      f"point is priced a second time by `price_asian_mc` (the same "
      f"estimator, an independent {pr['ref_paths']:,}-path draw), which "
      f"reports one over antithetic pairs. RMS {se['rms']:.3f} bps of "
      f"strike, mean {se['mean']:.3f} bps, max {se['max']:.3f} bps "
      f"({where(se['max_point'])}). Over the "
      f"{box['reference_cross_check']['n_points']} points with a nonzero "
      f"standard error, the two independent prices differ by an RMS of "
      f"{box['reference_cross_check']['rms_z']:.2f} in units of sqrt(2) "
      f"standard errors; a value near 1 means the standard error describes "
      f"the reference's noise.")
    a(f"- Surrogate: `PricingEngine` on `artifacts/model.pt`, "
      f"{box['n_members']} members, price only through `price_batch` with a "
      f"batch of one (the float32 serving path). Greeks are not timed here.")
    a(f"- Latency: median of {pr['repeats']} calls per point x {n} points "
      f"for the fast pricers; mean of one run per point for Monte Carlo. "
      f"Single process, `torch.set_num_threads(1)` to match the Dockerfile's "
      f"`OMP_NUM_THREADS=1`, one warm-up call per pricer. Absolute times are "
      f"specific to this machine and run; the ratios between rows are the "
      f"comparable quantity.")
    if _conditions_line(box) is not None:
        a(_conditions_line(box))
    a("")
    a("### Results")
    a("")
    a("Errors are (method - reference), in basis points of strike.")
    a("")
    a("| method | RMSE | mean abs error | bias | p95 abs error | max abs error | wall-clock per price |")
    a("|---|---|---|---|---|---|---|")
    for k in names:
        s = stats[k]
        a(f"| {k} | {s['rmse_bps']:.3f} bps | {s['mae_bps']:.3f} bps | "
          f"{s['bias_bps']:+.3f} bps | {s['p95_abs_bps']:.3f} bps | "
          f"{s['max_abs_bps']:.3f} bps | {fmt_latency(lat[k])} |")
    a(f"| Monte Carlo, {pr['ref_paths']:,} paths, price only "
      f"(`price_asian_mc`) | (reference) | RMS SE {se['rms']:.3f} bps | n/a | "
      f"n/a | SE <= {se['max']:.3f} bps | "
      f"{fmt_latency(ref['price_only_mean_seconds'])} |")
    a("")
    a(f"Worst points: surrogate at {where(stats[nn]['worst_point'])}; Levy at "
      f"{where(stats[tw]['worst_point'])}; Curran at "
      f"{where(stats[cur]['worst_point'])}.")
    a("")

    r_tw = box["rmse_ratios"]["levy_over_surrogate"]
    r_cur = box["rmse_ratios"]["surrogate_over_curran"]
    by_sig = box["by_sigma"]
    sig_ratios = [row["rmse_bps"][tw] / max(row["rmse_bps"][nn], 1e-300)
                  for row in by_sig]
    para = (
        f"Over the trained box the surrogate's price RMSE is "
        f"{stats[nn]['rmse_bps']:.3f} bps of strike and Levy's is "
        f"{stats[tw]['rmse_bps']:.3f} bps, a ratio of {r_tw['ratio']:.1f} "
        f"(95% interval {r_tw['ci95'][0]:.1f} to {r_tw['ci95'][1]:.1f}, "
        f"paired percentile bootstrap, {pr['bootstrap_resamples']:,} "
        f"resamples of the {n} points). By sigma band the ratio is "
        + ", ".join(f"{x:.1f}" for x in sig_ratios[:-1])
        + f" and {sig_ratios[-1]:.1f} (table below)")
    if grid is not None:
        g = grid["summary"]
        g_nn, g_tw = list(grid["methods"])[:2]
        para += (f", and on the 36-cell grid at sigma = "
                 f"{grid['protocol']['sigma']} it is "
                 f"{g[g_tw]['mean_abs_bps'] / g[g_nn]['mean_abs_bps']:.1f} in "
                 f"mean absolute error")
    para += (". Levy matches a lognormal to the first two moments of the "
             "average, and the distance between that lognormal and the true "
             "law of the average grows with sigma^2 T.")
    a(para)
    a("")
    cur_smaller = r_cur["ratio"] > 1.0

    def bands(rows, winner: str, loser: str, loser_name: str) -> str:
        return " and ".join(
            f"{row['label']} ({row['rmse_bps'][winner]:.3f} against "
            f"{loser_name} {row['rmse_bps'][loser]:.3f} bps)" for row in rows)

    cur_bands = [row for row in by_sig
                 if row["rmse_bps"][cur] < row["rmse_bps"][nn]]
    nn_bands = [row for row in by_sig if row not in cur_bands]
    para = (f"Curran's RMSE on the same points is "
            f"{stats[cur]['rmse_bps']:.3f} bps. The surrogate / Curran RMSE "
            f"ratio is {r_cur['ratio']:.1f} (95% interval "
            f"{r_cur['ci95'][0]:.1f} to {r_cur['ci95'][1]:.1f}), so over "
            f"the whole box {'Curran' if cur_smaller else 'the surrogate'} "
            f"has the lower RMSE.")
    if cur_bands and nn_bands:
        para += (f" The ordering depends on volatility. Curran has the lower "
                 f"RMSE for sigma in "
                 f"{bands(cur_bands, cur, nn, 'the surrogate at')}. "
                 f"The surrogate has the lower RMSE for sigma in "
                 f"{bands(nn_bands, nn, cur, 'Curran at')}. "
                 f"Curran is a lower bound whose gap to the true price "
                 f"widens with sigma^2 T; its bias over the box is "
                 f"{stats[cur]['bias_bps']:+.3f} bps.")
    a(para)
    a("")
    a(f"The reference's RMS standard error is {se['rms']:.3f} bps. With "
      f"that removed in quadrature the surrogate's RMSE is "
      f"{_net_of_reference(stats[nn]['rmse_bps'], se['rms'])}, Levy's "
      f"{_net_of_reference(stats[tw]['rmse_bps'], se['rms'])} and Curran's "
      f"{_net_of_reference(stats[cur]['rmse_bps'], se['rms'])}.")
    a("")
    speed = ref["price_only_mean_seconds"] / lat[nn]
    a(f"One surrogate price takes {fmt_latency(lat[nn])} at the median "
      f"against {fmt_latency(ref['price_only_mean_seconds'])} for one "
      f"{pr['ref_paths']:,}-path price-only Monte Carlo run, a ratio of "
      f"{speed:,.0f}. The two are at different accuracies. That Monte Carlo "
      f"price carries an RMS standard error of {se['rms']:.3f} bps against "
      f"the surrogate's {stats[nn]['rmse_bps']:.3f} bps RMSE. Levy costs "
      f"{fmt_latency(lat[tw])} per price and Curran "
      f"{fmt_latency(lat[cur])}, so the surrogate is "
      f"{lat[nn] / lat[tw]:.1f}x the cost of Levy and "
      f"{lat[nn] / lat[cur]:.1f}x the cost of Curran. The reference run "
      f"itself (`_simulate_chunk`, which also returns pathwise delta and "
      f"vega) takes {fmt_latency(ref['simulate_chunk_mean_seconds'])} per "
      f"point.")
    a("")
    a("### Error by volatility and by price level")
    a("")
    a("RMSE in bps of strike within each band.")
    a("")
    for title, rows in (("sigma", by_sig),
                        ("reference price (bps of strike)",
                         box["by_reference_price"])):
        a(f"| {title} | points | " + " | ".join(names[:3])
          + " | Levy / surrogate |")
        a("|---|---|" + "---|" * 4)
        for row in rows:
            e = row["rmse_bps"]
            a(f"| {row['label']} | {row['n']} | "
              + " | ".join(f"{e[k]:.3f}" for k in names[:3])
              + f" | {e[tw] / max(e[nn], 1e-300):.1f} |")
        a("")

    levy_wins = [row for row in box["by_reference_price"]
                 if row["rmse_bps"][tw] < row["rmse_bps"][nn]]
    if levy_wins:
        parts = [f"{_bps_label(row['label'])} "
                 f"({row['rmse_bps'][tw]:.3f} against "
                 f"{row['rmse_bps'][nn]:.3f} bps over {row['n']} "
                 f"point{'s' if row['n'] != 1 else ''})"
                 for row in levy_wins]
        text = ("Levy has the lower RMSE where the reference price is "
                + "; ".join(parts) + ".")
        if levy_wins[0]["label"] == PRICE_LABELS[0]:
            text += (" Where the true price is near zero Levy returns a "
                     "price near zero, and the surrogate's Softplus output "
                     "cannot emit zero, so its error there is a floor "
                     "(`scripts/fullscale_ablation.py` documents the same "
                     "floor).")
        a(text)
    else:
        a("The surrogate has the lower RMSE than Levy in every price band.")
    a("")
    a("### Notes")
    a("")
    ref_p = np.array(ref["price"])
    se_abs = np.array(ref["std_error"])
    ok = se_abs > 0.0
    z_cur = ((np.array(box["methods"][cur]["price"]) - ref_p)[ok]
             / se_abs[ok])
    d_thr = float(np.max(np.abs(np.array(box["methods"][cur]["price"])
                                - np.array(box["methods"][cur_lin]["price"])))
                  / STRIKE * 1e4)
    n_pos = int(np.sum(np.array(box["methods"][nn]["price"]) > ref_p))
    a(f"- The surrogate's signed error is positive at {n_pos} of {n} points "
      f"(bias {stats[nn]['bias_bps']:+.3f} bps). Levy's bias is "
      f"{stats[tw]['bias_bps']:+.3f} bps and Curran's "
      f"{stats[cur]['bias_bps']:+.3f} bps.")
    a(f"- Curran with the exact threshold is a lower bound on the true "
      f"price. Over the {int(ok.sum())} points with a nonzero reference "
      f"standard error it lies between {z_cur.min():+.2f} and "
      f"{z_cur.max():+.2f} standard errors of the reference. A lower bound "
      f"exceeds the reference only through reference noise, and the largest "
      f"of {int(ok.sum())} independent standard normal draws has median "
      f"{float(norm.ppf(0.5 ** (1.0 / int(ok.sum())))):.2f}. Curran's "
      f"first-order threshold differs from the exact solve by at most "
      f"{d_thr:.4f} bps over the box.")
    a(f"- RMSE over this box is heavy-tailed. The five largest squared "
      f"errors carry "
      f"{_top_share(box, tw, 5):.0f}% of Levy's total squared error and "
      f"{_top_share(box, nn, 5):.0f}% of the surrogate's, so the RMSE ratio "
      f"moves with the draw; the bootstrap interval above is its spread over "
      f"resamples of these {n} points.")
    a("")
    return lines


def _bps_label(label: str) -> str:
    return "below 1 bp" if label == PRICE_LABELS[0] else f"{label} bps"


def _top_share(box: dict, name: str, k: int) -> float:
    err = (np.array(box["methods"][name]["price"])
           - np.array(box["reference"]["price"]))
    sq = np.sort(err ** 2)[::-1]
    return float(sq[:k].sum() / max(sq.sum(), 1e-300) * 100.0)


# ---------------------------------------------------------------------------
# Grid mode: 36 cells at one vol and one rate
# ---------------------------------------------------------------------------

def run_grid(engine: PricingEngine, args) -> dict:
    grid = build_grid()
    print(f"reference: {len(grid)} cells x {args.ref_paths:,} paths, seed "
          f"{args.seed}")
    pricers = make_pricers(engine)
    for fn in pricers.values():
        fn(1.0, 1.0)                                    # warm-up
    methods, summary = {}, {}
    with TimingConditions() as conditions:
        ref, ref_se, ref_secs = reference_prices(grid, args.ref_paths,
                                                 args.seed)
        for name, fn in pricers.items():
            prices, secs = time_single_prices(fn, grid, args.repeats)
            methods[name] = {"price": _stored(prices), "median_seconds": secs}
    for name, res in methods.items():
        err = (np.array(res["price"]) - ref) / STRIKE * 1e4
        worst = int(np.argmax(np.abs(err)))
        summary[name] = {"mean_abs_bps": float(np.mean(np.abs(err))),
                         "max_abs_bps": float(np.max(np.abs(err))),
                         "bias_bps": float(np.mean(err)),
                         "worst_cell": list(grid[worst])}
        print(f"  {name:<38} mean|e| {np.mean(np.abs(err)):7.3f}  "
              f"max|e| {np.max(np.abs(err)):7.3f}  bias {np.mean(err):+7.3f} "
              f"bps   {fmt_latency(res['median_seconds'])}/price")
    cdf_us, ndtr_us = cdf_overhead_us()
    return {
        "date": _dt.date.today().isoformat(),
        "environment": environment(),
        "command": (f"python scripts/benchmark_approximations.py --ref-paths "
                    f"{args.ref_paths} --seed {args.seed} --repeats "
                    f"{args.repeats}"),
        "timing_conditions": conditions.record,
        "checkpoint": checkpoint_fingerprint(SERVED_CHECKPOINT),
        "n_members": engine.n_members,
        "protocol": {"strike": STRIKE, "sigma": SIGMA, "rate": RATE,
                     "n_steps": N_STEPS, "moneyness": MONEYNESS.tolist(),
                     "maturity": MATURITY.tolist(),
                     "ref_paths": args.ref_paths, "seed": args.seed,
                     "repeats": args.repeats},
        "reference": {"price": _stored(ref), "std_error": _stored(ref_se),
                      "mean_seconds": ref_secs},
        "methods": methods,
        "summary": summary,
        "cdf_overhead_us": {"norm_cdf": cdf_us, "ndtr": ndtr_us},
    }


def render_grid(g: dict, box: dict | None) -> list[str]:
    pr = g["protocol"]
    moneyness, maturity = pr["moneyness"], pr["maturity"]
    grid = [(m, t) for t in maturity for m in moneyness]
    n_cells = len(grid)
    n_m = len(moneyness)
    ref = np.array(g["reference"]["price"])
    ref_se = np.array(g["reference"]["std_error"])
    ref_secs = g["reference"]["mean_seconds"]
    cdf_us, ndtr_us = g["cdf_overhead_us"]["norm_cdf"], g["cdf_overhead_us"]["ndtr"]
    se_bps = ref_se / STRIKE * 1e4
    rows = []
    for name, res in g["methods"].items():
        err = (np.array(res["price"]) - ref) / STRIKE * 1e4
        worst = int(np.argmax(np.abs(err)))
        m_w, t_w = grid[worst]
        rows.append({
            "name": name, "mean_abs": float(np.mean(np.abs(err))),
            "max_abs": float(np.max(np.abs(err))), "bias": float(np.mean(err)),
            "worst": f"m={m_w:.2f}, T={t_w:.2f}",
            "secs": res["median_seconds"], "err": err,
        })

    nn, tw, cur = rows[0], rows[1], rows[2]
    lines: list[str] = []
    a = lines.append
    a(f"## Grid at sigma = {pr['sigma']}, {n_cells} cells")
    a("")
    a(f"Measured on {g['date']}; {_env_line(g['environment'])}. Checkpoint "
      f"`{g['checkpoint']['file']}`, sha256 "
      f"`{g['checkpoint']['sha256'][:16]}`. Stored under `grid` in "
      f"`docs/approximation_benchmark.json`.")
    a("")
    a("### Protocol")
    a("")
    a(f"- Contract: arithmetic-average Asian call, strike {pr['strike']:.0f}, "
      f"{pr['n_steps']} equally spaced monitoring dates.")
    a(f"- Grid: moneyness S/K at {', '.join(f'{m:.2f}' for m in moneyness)} "
      f"x maturity at {', '.join(f'{t:.2f}' for t in maturity)} years "
      f"({n_cells} cells), sigma = {pr['sigma']}, r = {pr['rate']}.")
    a(f"- Reference: `price_asian_mc`, {pr['ref_paths']:,} paths, antithetic "
      f"sampling with the geometric-Asian control variate, seed {pr['seed']}. "
      f"Reference standard error: mean {se_bps.mean():.3f} bps of strike, "
      f"max {se_bps.max():.3f} bps (cell m={grid[int(np.argmax(se_bps))][0]:.2f}, "
      f"T={grid[int(np.argmax(se_bps))][1]:.2f}). Errors below that scale "
      f"are not resolved by this reference.")
    a(f"- Surrogate: `PricingEngine` on `artifacts/model.pt`, "
      f"{g['n_members']} members, price only through `price_batch` with a "
      f"batch of one (the float32 serving path). Greeks are not timed here.")
    a(f"- Latency: median of {pr['repeats']} calls per cell x {n_cells} cells "
      f"for the fast pricers; mean of one run per cell for the reference. "
      f"Single process, `torch.set_num_threads(1)` to match the Dockerfile's "
      f"`OMP_NUM_THREADS=1`, one warm-up call per pricer. Absolute times are "
      f"specific to this machine and run; the ratios between rows are the "
      f"comparable quantity.")
    if _conditions_line(g) is not None:
        a(_conditions_line(g))
    a("")
    a("### Results")
    a("")
    a("Errors are (method - reference), in basis points of strike.")
    a("")
    a("| method | mean abs error | max abs error | bias | worst cell | wall-clock per price |")
    a("|---|---|---|---|---|---|")
    for r in rows:
        a(f"| {r['name']} | {r['mean_abs']:.3f} bps | {r['max_abs']:.3f} bps | "
          f"{r['bias']:+.3f} bps | {r['worst']} | {fmt_latency(r['secs'])} |")
    a(f"| Monte Carlo, {pr['ref_paths']:,} paths | (reference) | "
      f"SE <= {se_bps.max():.3f} bps | n/a | n/a | {fmt_latency(ref_secs)} |")
    a("")
    a(f"Mean absolute error by maturity (bps of strike, averaged over the "
      f"{n_m} moneyness points):")
    a("")
    a("| maturity (y) | " + " | ".join(r["name"] for r in rows) + " |")
    a("|---|" + "---|" * len(rows))
    for j, t in enumerate(maturity):
        sl = slice(n_m * j, n_m * j + n_m)
        a(f"| {t:.2f} | " + " | ".join(
            f"{np.mean(np.abs(r['err'][sl])):.3f}" for r in rows) + " |")
    a("")
    a("Signed error per cell for the surrogate and for Curran (bps of "
      "strike; rows are maturity, columns moneyness):")
    a("")
    for r in (nn, cur):
        a(f"{r['name']}")
        a("")
        a("| T \\ S/K | " + " | ".join(f"{m:.2f}" for m in moneyness) + " |")
        a("|---|" + "---|" * n_m)
        for j, t in enumerate(maturity):
            a(f"| {t:.2f} | " + " | ".join(
                f"{e:+.2f}" for e in r["err"][n_m * j:n_m * j + n_m]) + " |")
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

    # Interpretation: every number from this run.
    ratio_tw = tw["mean_abs"] / max(nn["mean_abs"], 1e-12)
    cur_wins = cur["mean_abs"] < nn["mean_abs"]
    ratio_cur = (nn["mean_abs"] / max(cur["mean_abs"], 1e-12) if cur_wins
                 else cur["mean_abs"] / max(nn["mean_abs"], 1e-12))
    tw_by_t = [float(np.mean(np.abs(tw["err"][n_m * j:n_m * j + n_m])))
               for j in range(len(maturity))]
    a("### Interpretation")
    a("")
    a(f"On this grid the surrogate's mean absolute error is "
      f"{nn['mean_abs']:.2f} bps of strike against {tw['mean_abs']:.2f} bps "
      f"for Turnbull-Wakeman / Levy moment matching, a ratio of "
      f"{ratio_tw:.1f} (max {nn['max_abs']:.2f} against {tw['max_abs']:.2f} "
      f"bps). The closed form's error is a bias that grows with maturity, "
      f"from {tw_by_t[0]:.2f} bps at {maturity[0]:.2f} y to "
      f"{tw_by_t[-1]:.2f} bps at {maturity[-1]:.2f} y.")
    a("")
    a(f"Curran's conditioning approximation has a mean absolute error of "
      f"{cur['mean_abs']:.2f} bps (max {cur['max_abs']:.2f} bps) at "
      f"{fmt_latency(cur['secs'])} per price against the surrogate's "
      f"{fmt_latency(nn['secs'])}. It is a lower bound on the true price, "
      f"and it sits at most {max(float(z_cur.max()), 0.0):.1f} reference "
      f"standard errors above the Monte Carlo price. On price alone at this "
      f"vol {'Curran' if cur_wins else 'the surrogate'} is the more accurate "
      f"pricer, by a factor of {ratio_cur:.1f} in mean absolute error. The "
      f"surrogate returns all five Greeks by automatic differentiation in "
      f"one call and prices in batches, and its training recipe carries over "
      f"to dynamics with no geometric-conditioning closed form, such as the "
      f"rough-volatility model behind the 0DTE pricer.")
    a("")

    a("### Notes")
    a("")
    note = (f"- The surrogate's signed error is positive in {n_pos} of "
            f"{n_cells} cells (bias {nn['bias']:+.2f} bps).")
    if box is not None:
        b_nn = list(box["methods"])[0]
        note += (f" Over the trained box the bias is "
                 f"{box['summary'][b_nn]['bias_bps']:+.2f} bps on "
                 f"{box['protocol']['n_points']} points.")
    a(note)
    a(f"- Curran with the exact threshold, over the {int(resolved.sum())} "
      f"cells where the reference has a nonzero standard error, lies between "
      f"{z_cur.min():+.2f} and {z_cur.max():+.2f} standard errors of the "
      f"Monte Carlo price. A lower bound exceeds the reference only through "
      f"reference noise. Curran's first-order threshold differs from the "
      f"exact solve by at most {d_thr:.4f} bps on this grid.")
    for (m_z, t_z), e_cur, e_nn in zero_cells:
        a(f"- At m={m_z:.2f}, T={t_z:.2f} every one of the "
          f"{pr['ref_paths']:,} reference paths pays zero, so the reference "
          f"is 0 with zero standard error. Curran returns {e_cur:+.4f} bps "
          f"there. The surrogate returns {e_nn:+.2f} bps because its "
          f"Softplus output cannot emit zero.")
    a(f"- Timing floor: Turnbull-Wakeman is two scalar `norm.cdf` calls plus "
      f"a 50x50 exponential sum, and scipy's `norm.cdf` wrapper costs "
      f"{cdf_us:.0f} us per scalar call in this run against {ndtr_us:.1f} us "
      f"for `scipy.special.ndtr`, so the two wrapper calls are about "
      f"{min(2.0 * cdf_us / (tw['secs'] * 1e6), 1.0) * 100:.0f}% of its "
      f"{fmt_latency(tw['secs'])}. Curran makes two `norm.cdf` calls, one on "
      f"a length-{pr['n_steps']} vector; the {fmt_latency(cur['secs'])} of "
      f"the exact threshold against {fmt_latency(rows[3]['secs'])} for the "
      f"linear one is the Newton solve, the only code that differs between "
      f"them. Switching the cdf primitive would speed up every closed form "
      f"and change none of the accuracy columns. The timings keep "
      f"`norm.cdf` because `backend/quant/benchmarks.py` and "
      f"`asian_approx.py` call it.")
    a("")
    return lines


def build_report(payload: dict) -> str:
    """Render docs/approximation_benchmark.md from the JSON payload. Pure in
    the payload, so tests can check the committed page against the artifact."""
    box, grid = payload.get("box_lhs"), payload.get("grid")
    lines: list[str] = []
    a = lines.append
    a("# Closed-form approximations versus the neural surrogate")
    a("")
    lede = []
    if box is not None:
        nn, tw, cur = list(box["methods"])[:3]
        s, pr = box["summary"], box["protocol"]
        rg = pr["param_ranges"]
        lede.append(
            f"On {pr['n_points']} Latin-hypercube points over the trained "
            f"box (S/K in {_span(rg['moneyness'])}, maturity in "
            f"{_span(rg['maturity'])} years, sigma in {_span(rg['sigma'])}, "
            f"r in {_span(rg['rate'])}), scored against {pr['ref_paths']:,}-path "
            f"control-variate Monte Carlo, the served ensemble has a price "
            f"RMSE of {s[nn]['rmse_bps']:.3f} bps of strike, Levy (1992) "
            f"moment matching {s[tw]['rmse_bps']:.3f} bps and Curran (1994) "
            f"conditioning {s[cur]['rmse_bps']:.3f} bps.")
    if grid is not None:
        nn, tw, cur = list(grid["methods"])[:3]
        s, pr = grid["summary"], grid["protocol"]
        lede.append(
            f"On a {len(pr['moneyness']) * len(pr['maturity'])}-cell grid at "
            f"sigma = {pr['sigma']} against {pr['ref_paths']:,}-path "
            f"references the mean absolute errors are "
            f"{s[nn]['mean_abs_bps']:.2f}, {s[tw]['mean_abs_bps']:.2f} and "
            f"{s[cur]['mean_abs_bps']:.2f} bps.")
    lede.append("Both measurements are produced by "
                "`scripts/benchmark_approximations.py` and stored point by "
                "point in `docs/approximation_benchmark.json`; this page is "
                "rendered from that file.")
    a(" ".join(lede))
    a("")
    if box is not None:
        lines.extend(render_box(box, grid))
    if grid is not None:
        lines.extend(render_grid(grid, box))
    a("## Reproduce")
    a("")
    a("```bash")
    for section in (box, grid):
        if section is not None:
            a(section["command"])
    a("```")
    a("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lhs", type=int, nargs="?", const=BOX_POINTS, default=None,
                   metavar="N",
                   help=f"box mode: N Latin-hypercube points over the trained "
                        f"box (default {BOX_POINTS}); without this flag the "
                        f"36-cell grid runs")
    p.add_argument("--ref-paths", type=int, default=None,
                   help=f"reference paths (default {GRID_REF_PATHS:,} on the "
                        f"grid, {BOX_REF_PATHS:,} in the box)")
    p.add_argument("--seed", type=int, default=None,
                   help=f"default {GRID_SEED} on the grid, {BOX_SEED} in the "
                        f"box")
    p.add_argument("--repeats", type=int, default=20,
                   help="timed calls per point for each fast pricer")
    p.add_argument("--bootstrap", type=int, default=10_000,
                   help="box mode: resamples for the RMSE-ratio intervals")
    p.add_argument("--render-only", action="store_true",
                   help="measure nothing; re-render the page from the JSON")
    p.add_argument("--out", type=Path, default=MD_PATH)
    p.add_argument("--json", type=Path, default=JSON_PATH)
    args = p.parse_args(argv)
    box_mode = args.lhs is not None
    if args.ref_paths is None:
        args.ref_paths = BOX_REF_PATHS if box_mode else GRID_REF_PATHS
    if args.seed is None:
        args.seed = BOX_SEED if box_mode else GRID_SEED

    payload = (json.loads(args.json.read_text(encoding="utf-8"))
               if args.json.exists() else {})
    payload["generated_by"] = "scripts/benchmark_approximations.py"
    if not args.render_only:
        # The Dockerfile serves with OMP_NUM_THREADS=1; time the same path.
        # On a 16-core box one thread is also the fastest setting for a batch
        # of one (3.0 ms p50 against 3.3 ms at 16 threads and 4.5 ms at 4).
        torch.set_num_threads(1)
        engine = PricingEngine()
        if engine.n_steps != N_STEPS:
            raise SystemExit(f"checkpoint uses {engine.n_steps} monitoring "
                             f"steps, benchmark assumes {N_STEPS}")
        if box_mode:
            payload["box_lhs"] = run_box(engine, args)
        else:
            payload["grid"] = run_grid(engine, args)

    report = build_report(payload)
    for path in (args.json, args.out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1) + "\n",
                         encoding="utf-8", newline="\n")
    args.out.write_text(report, encoding="utf-8", newline="\n")   # LF on Windows too
    print()
    print(report)
    print(f"wrote {args.json}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
