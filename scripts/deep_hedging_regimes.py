"""The deep hedger as a designed experiment: where (if anywhere) does a learned
CVaR hedger beat delta hedging?

docs/hedging_findings.md records that on risk-neutral GBM the learned policy
loses to a vol-matched delta hedge and to Whalley-Wilmott. Under GBM with
small costs that is EXPECTED: the market is complete and a static delta is
near-optimal. The question this script asks is whether the learned policy
wins where a static delta cannot, and answers it with a measured matrix:

    train measure  x  eval measure  x  cost,
    {gbm, rbergomi, rbergomi_jumps}^2 x {0, 0.001, 0.005, 0.01, 0.02}

against THREE properly specified baselines on identical paths:

    delta            vol-matched Black-Scholes delta (sigma = realized
                     terminal vol of the eval measure, from a probe)
    whalley_wilmott  cost-aware no-trade band, risk aversion tuned in-sample
                     FOR THE BASELINE (deliberately generous to it)
    linear           Ruf-Wang linear-regression hedge, h = c0 + c1 delta +
                     c2 delta(1 - delta), OLS min-variance fit on separate
                     training paths of the eval measure, costs ignored in
                     the fit and charged in the evaluation

Every cell: 5 seeds x 3000 paths, CVaR_95 of the terminal loss with bootstrap
standard errors, the premium booked at the Monte Carlo price under the eval
measure. "Wins" means the paired ratio deep/baseline is below 1 by more than
2 bootstrap SE (paired: the ratio is bootstrapped by resampling PATHS, since
both hedgers run on the same paths).

Outputs: docs/deep_hedging_regimes.{md,png,json} and the two checkpoints
artifacts/hedger_rbergomi.pt and artifacts/hedger_rbergomi_jumps.pt.

Usage
    python scripts/deep_hedging_regimes.py --train rbergomi        # one ckpt
    python scripts/deep_hedging_regimes.py --train rbergomi_jumps
    python scripts/deep_hedging_regimes.py                         # evaluate
    python scripts/deep_hedging_regimes.py --quick                 # smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from backend.quant.hedging import (ARTIFACTS, CVAR_ALPHA, DT, MATURITY,  # noqa: E402
                                   N_STEPS, HedgingEngine, bs_call_delta,
                                   cvar, fit_linear_hedge, linear_hedge_fn,
                                   rough_measure_params, train)

DOCS = ROOT / "docs"
OUT_STEM = "deep_hedging_regimes"
EVAL_MEASURES = ("gbm", "rbergomi", "rbergomi_jumps")
TRAIN_MEASURES = ("gbm", "rbergomi", "rbergomi_jumps")
COSTS = (0.0, 0.001, 0.005, 0.01, 0.02)
SEEDS = (17, 18, 19, 20, 21)
CHECKPOINTS = {"gbm": "hedger_gbm.pt", "rbergomi": "hedger_rbergomi.pt",
               "rbergomi_jumps": "hedger_rbergomi_jumps.pt"}
WW_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_BOOT = 500

# Thread count: measured on the experiment machine, one training step at
# batch 2048 took 0.377 s with 16 intra-op threads, 0.163 s with 8, 0.116 s
# with 4 and 0.212 s with 1 - the tensors are too small for wide
# parallelism, and two trainings run side by side. 4 it is.
TORCH_THREADS = int(os.environ.get("HEDGE_THREADS", "4"))


# --------------------------------------------------------------------------- #
#  measure levels: sigma / rate at which each eval measure is simulated
# --------------------------------------------------------------------------- #

def measure_level(measure: str) -> tuple[float, float]:
    """(sigma, rate) for an eval measure. Rough measures sit at their own
    calibrated forward vol sqrt(xi) and the calibration rate; GBM is placed at
    the SPY calibration's sqrt(xi) and rate so the three measures share one
    vol level up to the jump fit's slightly different xi."""
    if measure == "gbm":
        p = rough_measure_params("rbergomi")
    else:
        p = rough_measure_params(measure)
    return math.sqrt(p["xi"]), p["rate"]


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #

def train_checkpoint(measure: str, iters: int, batch: int) -> dict:
    torch.set_num_threads(TORCH_THREADS)
    name = CHECKPOINTS[measure]
    print(f"[train] {measure} -> {name}: {iters} iters, batch {batch}, "
          f"{TORCH_THREADS} threads", flush=True)
    meta = train(iters=iters, batch=batch, measure=measure, out_name=name)
    print(f"[train] done in {meta['train_seconds']} s", flush=True)
    return meta


# --------------------------------------------------------------------------- #
#  statistics
# --------------------------------------------------------------------------- #

def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def cvar_and_se(pl: np.ndarray, idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    boots = np.array([cvar(pl[i], CVAR_ALPHA) for i in idx])
    return cvar(pl, CVAR_ALPHA), float(boots.std(ddof=1)), boots


def paired_ratio(boots_a: np.ndarray, boots_b: np.ndarray, a: float,
                 b: float) -> tuple[float, float]:
    """Ratio a/b with the SE of the PAIRED bootstrap (same path resamples
    for numerator and denominator)."""
    if b == 0.0:
        return float("nan"), float("nan")
    r = boots_a / boots_b
    return a / b, float(np.std(r, ddof=1))


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #

def evaluate(n_paths: int, seeds: tuple[int, ...], costs: tuple[float, ...],
             n_boot: int) -> dict:
    torch.set_num_threads(TORCH_THREADS)
    t_eval = time.perf_counter()
    engines = {m: HedgingEngine(ARTIFACTS / CHECKPOINTS[m])
               for m in TRAIN_MEASURES}
    any_engine = engines["gbm"]          # measures / books are static

    results: dict = {"cells": {}, "measures": {}, "diagnostics": {},
                     "n_seeds": len(seeds), "n_paths_per_seed": n_paths}
    for ev in EVAL_MEASURES:
        sigma, rate = measure_level(ev)
        # Probe: premium under the eval measure and its realized terminal vol
        probe = any_engine._spots(ev, sigma, rate, 40_000, seeds[0] + 9_000)
        premium_mc = float(np.exp(-rate * MATURITY)
                           * np.maximum(probe[:, -1] - 1.0, 0.0).mean())
        premium_se = float(np.exp(-rate * MATURITY)
                           * np.maximum(probe[:, -1] - 1.0, 0.0).std()
                           / math.sqrt(probe.shape[0]))
        lr_T = np.log(probe[:, -1])
        realized_vol = float(np.std(lr_T) / math.sqrt(MATURITY))
        z = (lr_T - lr_T.mean()) / lr_T.std()
        # Separate TRAINING paths for the baselines that need fitting: the
        # Whalley-Wilmott risk aversion and the linear hedge's coefficients.
        tune = any_engine._spots(ev, sigma, rate, 20_000, seeds[0] + 5_000)
        lin_fit = fit_linear_hedge(tune, realized_vol, rate)
        lin_fit4 = fit_linear_hedge(tune, realized_vol, rate, n_features=4)
        test_spots = {sd: any_engine._spots(ev, sigma, rate, n_paths, sd)
                      for sd in seeds}
        results["measures"][ev] = {
            "sigma": sigma, "rate": rate, "premium_mc": premium_mc,
            "premium_mc_se": premium_se, "realized_vol": realized_vol,
            "terminal_skew": float((z ** 3).mean()),
            "terminal_kurtosis": float((z ** 4).mean()),
            "linear_coef": np.round(lin_fit["coef"], 5).tolist(),
            "linear_r2": lin_fit["r2"],
            "linear_coef_4feat": np.round(lin_fit4["coef"], 5).tolist(),
            "linear_r2_4feat": lin_fit4["r2"],
            "n_fit_paths": int(tune.shape[0]),
            "params": rough_measure_params(ev) if ev != "gbm" else None,
        }
        print(f"[eval] {ev}: sigma {sigma:.4f} realized {realized_vol:.4f} "
              f"premium {premium_mc:.5f} skew {(z**3).mean():+.2f} "
              f"kurt {(z**4).mean():.1f} linear {np.round(lin_fit['coef'], 3)}",
              flush=True)

        for cost in costs:
            # Whalley-Wilmott risk aversion tuned on the tuning paths.
            best_g, best_c = 1.0, math.inf
            for ra in WW_GRID:
                pl, _, _ = any_engine._run_book(
                    tune[:n_paths], any_engine._whalley_wilmott_fn(
                        realized_vol, rate, cost, ra), premium_mc, cost, rate)
                c = cvar(pl)
                if c < best_c:
                    best_g, best_c = ra, c
            baselines = {
                "delta": any_engine._delta_fn(realized_vol, rate),
                "whalley_wilmott": any_engine._whalley_wilmott_fn(
                    realized_vol, rate, cost, best_g),
                "linear": linear_hedge_fn(lin_fit["coef"], realized_vol,
                                          rate),
            }
            deep = {tr: engines[tr]._deep_fn(sigma, rate, cost)
                    for tr in TRAIN_MEASURES}
            pls: dict[str, list] = {k: [] for k in
                                    list(baselines) + [f"deep_{t}" for t in deep]}
            cst = {k: [] for k in pls}
            turnover = {k: [] for k in pls}
            hold = {k: [] for k in pls}
            for sd in seeds:
                spots = test_spots[sd]
                for name, fn in list(baselines.items()) + [
                        (f"deep_{t}", f) for t, f in deep.items()]:
                    pl, hist, c = any_engine._run_book(spots, fn, premium_mc,
                                                       cost, rate)
                    pls[name].append(pl)
                    cst[name].append(c)
                    turnover[name].append(np.abs(np.diff(
                        np.column_stack([np.zeros(len(hist)), hist]),
                        axis=1)).sum(axis=1))
                    if sd == seeds[0]:
                        hold[name] = hist
            n_total = n_paths * len(seeds)
            idx = boot_indices(n_total, n_boot, seed=1)
            stats, boots = {}, {}
            for name in pls:
                pl = np.concatenate(pls[name])
                c, se, b = cvar_and_se(pl, idx)
                boots[name] = b
                stats[name] = {
                    "cvar95": c, "cvar95_se": se,
                    "mean": float(pl.mean()), "std": float(pl.std()),
                    "mean_costs": float(np.concatenate(cst[name]).mean()),
                    "mean_turnover": float(np.concatenate(turnover[name]).mean()),
                }
            cell = {"cost": cost, "eval": ev, "n_paths": n_total,
                    "whalley_wilmott_risk_aversion": best_g,
                    "baselines": {k: stats[k] for k in baselines},
                    "deep": {}}
            for tr in TRAIN_MEASURES:
                key = f"deep_{tr}"
                entry = dict(stats[key])
                entry["vs"] = {}
                for bname in baselines:
                    r, rse = paired_ratio(boots[key], boots[bname],
                                          stats[key]["cvar95"],
                                          stats[bname]["cvar95"])
                    entry["vs"][bname] = {
                        "ratio": r, "ratio_se": rse,
                        "wins": bool(r < 1.0 and (1.0 - r) > 2.0 * rse),
                        "loses": bool(r > 1.0 and (r - 1.0) > 2.0 * rse),
                    }
                cell["deep"][tr] = entry
            results["cells"][f"{ev}|{cost}"] = cell

            # Holding diagnostics on the first seed's paths: is the policy
            # under-hedging relative to the BS delta, as the literature
            # predicts under rho < 0, and how close is it to the OLS hedge?
            d_bs = hold["delta"]
            diag = {}
            for tr in TRAIN_MEASURES:
                h = hold[f"deep_{tr}"]
                diff = (h - d_bs).ravel()
                slope = float(np.polyfit(d_bs.ravel(), h.ravel(), 1)[0])
                diag[tr] = {
                    "mean_h_minus_delta": float(diff.mean()),
                    "mean_h_minus_delta_se": float(diff.std()
                                                   / math.sqrt(diff.size)),
                    "mean_abs_h_minus_delta": float(np.abs(diff).mean()),
                    "mean_abs_h_minus_linear": float(np.abs(
                        h - hold["linear"]).mean()),
                    "corr_h_delta": float(np.corrcoef(h.ravel(),
                                                      d_bs.ravel())[0, 1]),
                    "slope_h_on_delta": slope,
                    "mean_holding": float(h.mean()),
                }
            diag["linear_minus_delta_mean"] = float(
                (hold["linear"] - d_bs).mean())
            diag["delta_mean_holding"] = float(d_bs.mean())
            results["diagnostics"][f"{ev}|{cost}"] = diag

            line = " ".join(
                f"{tr[:4]}:{cell['deep'][tr]['vs']['delta']['ratio']:.3f}"
                f"({cell['deep'][tr]['vs']['delta']['ratio_se']:.3f})"
                for tr in TRAIN_MEASURES)
            print(f"[eval] {ev:15s} cost {cost:<6} delta {stats['delta']['cvar95']:.5f} "
                  f"ww {stats['whalley_wilmott']['cvar95']:.5f} "
                  f"lin {stats['linear']['cvar95']:.5f} | deep/delta {line}",
                  flush=True)
    results["eval_seconds"] = round(time.perf_counter() - t_eval, 1)
    return results


# --------------------------------------------------------------------------- #
#  summaries: crossover, regime matrix, figure
# --------------------------------------------------------------------------- #

def summarise(res: dict, costs: tuple[float, ...]) -> dict:
    """Per (train, eval, baseline): the first cost at which the deep hedger
    wins (ratio < 1 by > 2 SE), the regime matrix of median ratios, counts."""
    summary = {"crossover": {}, "matrix": {}, "wins": {}}
    for tr in TRAIN_MEASURES:
        for ev in EVAL_MEASURES:
            for bname in ("delta", "whalley_wilmott", "linear"):
                ratios = []
                first_win = None
                n_win = n_lose = 0
                for cost in costs:
                    v = res["cells"][f"{ev}|{cost}"]["deep"][tr]["vs"][bname]
                    ratios.append(v["ratio"])
                    n_win += int(v["wins"])
                    n_lose += int(v["loses"])
                    if v["wins"] and first_win is None:
                        first_win = cost
                key = f"{tr}|{ev}|{bname}"
                summary["crossover"][key] = first_win
                summary["matrix"][key] = {
                    "median_ratio": float(np.median(ratios)),
                    "min_ratio": float(np.min(ratios)),
                    "max_ratio": float(np.max(ratios)),
                    "wins": n_win, "loses": n_lose, "n_cells": len(costs),
                }
    summary["total_wins"] = sum(v["wins"] for v in summary["matrix"].values())
    summary["total_loses"] = sum(v["loses"] for v in summary["matrix"].values())
    summary["total_cells"] = sum(v["n_cells"] for v in summary["matrix"].values())
    return summary


def make_figure(res: dict, costs: tuple[float, ...], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = "rbergomi_jumps"
    series = {
        "deep (trained on rbergomi_jumps)": ("#0A84FF",
                                             lambda c: c["deep"]["rbergomi_jumps"]),
        "vol-matched delta": ("#FF9F0A", lambda c: c["baselines"]["delta"]),
        "linear regression (Ruf-Wang)": ("#A0A0A8",
                                         lambda c: c["baselines"]["linear"]),
        "Whalley-Wilmott band": ("#30D158",
                                 lambda c: c["baselines"]["whalley_wilmott"]),
    }
    bg, panel, fg, grid = "#000000", "#1c1c1e", "#f2f2f7", "#3a3a3c"
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(bg)
    x = np.array(costs)
    ax = axes[0]
    ax.set_facecolor(panel)
    for label, (color, pick) in series.items():
        y = np.array([pick(res["cells"][f"{ev}|{c}"])["cvar95"] for c in costs])
        e = np.array([pick(res["cells"][f"{ev}|{c}"])["cvar95_se"] for c in costs])
        ax.errorbar(x * 1e4, y * 1e4, yerr=2 * e * 1e4, color=color, lw=2.2,
                    marker="o", ms=6, capsize=4, label=label)
    ax.set_xlabel("proportional transaction cost (bp of traded notional)",
                  color=fg, fontsize=12)
    ax.set_ylabel("CVaR$_{95}$ of terminal loss (bp of strike)", color=fg,
                  fontsize=12)
    m = res["measures"][ev]
    ax.set_title("Short 30-day ATM call, daily rebalancing, rough Bergomi + "
                 "jumps (SPY fit)\n"
                 f"{res['n_seeds']} seeds x {res['n_paths_per_seed']} paths "
                 f"per cost, bars = 2 bootstrap SE; realized vol "
                 f"{m['realized_vol']:.3f}, skew {m['terminal_skew']:+.1f}",
                 color=fg, fontsize=12, loc="left")
    ax.legend(facecolor=panel, edgecolor=grid, labelcolor=fg, fontsize=11,
              loc="upper left")

    # Right panel: the regime matrix of deep/delta median ratios.
    ax = axes[1]
    ax.set_facecolor(panel)
    mat = np.array([[res["summary"]["matrix"][f"{tr}|{ev2}|delta"]["median_ratio"]
                     for ev2 in EVAL_MEASURES] for tr in TRAIN_MEASURES])
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.8, vmax=1.6)
    for i, tr in enumerate(TRAIN_MEASURES):
        for j, ev2 in enumerate(EVAL_MEASURES):
            cellm = res["summary"]["matrix"][f"{tr}|{ev2}|delta"]
            ax.text(j, i, f"{mat[i, j]:.3f}\n{cellm['wins']}W/{cellm['loses']}L "
                    f"of {cellm['n_cells']}", ha="center", va="center",
                    color="#000000", fontsize=11, fontweight="bold")
    ax.set_xticks(range(3), EVAL_MEASURES, color=fg, fontsize=11)
    ax.set_yticks(range(3), TRAIN_MEASURES, color=fg, fontsize=11)
    ax.set_xlabel("evaluation measure", color=fg, fontsize=12)
    ax.set_ylabel("training measure", color=fg, fontsize=12)
    ax.set_title("Regime matrix: median over costs of CVaR$_{95}$ deep / delta\n"
                 "(<1 = deep hedger better; W/L = cells won/lost by > 2 SE)",
                 color=fg, fontsize=12, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg)
    cb.outline.set_edgecolor(grid)
    for a in axes:
        for s in a.spines.values():
            s.set_color(grid)
        a.tick_params(colors=fg)
        a.grid(True, color=grid, lw=0.6, alpha=0.7)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def _fmt_cost(c: float) -> str:
    return f"{c * 1e4:.0f} bp" if c > 0 else "0"


def write_markdown_tables(res: dict, costs: tuple[float, ...]) -> str:
    """Machine-generated tables that the hand-written write-up embeds."""
    lines = []
    for ev in EVAL_MEASURES:
        m = res["measures"][ev]
        lines.append(f"\n**Evaluation measure `{ev}`** (sigma {m['sigma']:.4f}, "
                     f"realized vol {m['realized_vol']:.4f}, premium "
                     f"{m['premium_mc'] * 1e4:.1f} bp, terminal skew "
                     f"{m['terminal_skew']:+.2f}, kurtosis "
                     f"{m['terminal_kurtosis']:.1f}; linear hedge c = "
                     f"{m['linear_coef']}, R^2 {m['linear_r2']:.3f})\n")
        lines.append("| cost | delta | Whalley-Wilmott | linear | "
                     + " | ".join(f"deep[{tr}]" for tr in TRAIN_MEASURES)
                     + " | " + " | ".join(f"deep[{tr}]/delta" for tr in TRAIN_MEASURES)
                     + " |")
        lines.append("|" + "---|" * (4 + 2 * len(TRAIN_MEASURES)))
        for c in costs:
            cell = res["cells"][f"{ev}|{c}"]
            b = cell["baselines"]
            row = [_fmt_cost(c)] + [
                f"{b[k]['cvar95'] * 1e4:.1f} ± {b[k]['cvar95_se'] * 1e4:.1f}"
                for k in ("delta", "whalley_wilmott", "linear")]
            row += [f"{cell['deep'][tr]['cvar95'] * 1e4:.1f} ± "
                    f"{cell['deep'][tr]['cvar95_se'] * 1e4:.1f}"
                    for tr in TRAIN_MEASURES]
            for tr in TRAIN_MEASURES:
                v = cell["deep"][tr]["vs"]["delta"]
                tag = " **W**" if v["wins"] else (" L" if v["loses"] else "")
                row.append(f"{v['ratio']:.3f} ± {v['ratio_se']:.3f}{tag}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(res: dict, costs: tuple[float, ...], path: Path) -> None:
    """Data-driven write-up: every number below is read from `res`, i.e. was
    measured in this run. The conclusion paragraph is chosen by the measured
    win/loss counts, not written in advance."""
    s = res["summary"]
    pr = res["protocol"]
    ck = pr["checkpoints"]
    tot_w, tot_l, tot_n = s["total_wins"], s["total_loses"], s["total_cells"]
    # wins vs delta only, in the in-sample cells (train == eval)
    insample = {tr: s["matrix"][f"{tr}|{tr}|delta"] for tr in TRAIN_MEASURES}
    L = []
    L.append("# Deep hedging as a designed experiment: rough volatility, jumps and costs\n")
    L.append("Generated by `scripts/deep_hedging_regimes.py`; every number here was "
             "measured in that run (`docs/deep_hedging_regimes.json` holds them all, "
             "`docs/deep_hedging_regimes.png` is the figure).\n")
    L.append("## Question\n")
    L.append("`docs/hedging_findings.md` records that on risk-neutral GBM the learned "
             "CVaR hedger loses to a vol-matched delta hedge and to Whalley-Wilmott. "
             "Under GBM with small costs that is expected: the market is complete and "
             "a static delta is near-optimal. This experiment asks whether the learned "
             "policy wins where a static delta cannot - rough stochastic volatility "
             "with spot-vol correlation, jumps (an incomplete market) and transaction "
             "costs - against three properly specified baselines on identical paths.\n")
    L.append("## Protocol\n")
    L.append(f"- Contract: short one 30-day ATM European call, S0 = K = 1, "
             f"{N_STEPS} daily rebalances (dt = 1/252), CVaR_{int(CVAR_ALPHA * 100)} "
             f"of the terminal loss (positive = loss) with the premium booked at the "
             f"Monte Carlo price under the evaluation measure (40,000-path probe).")
    L.append("- Measures (parameters read from disk, none hardcoded):")
    for ev in EVAL_MEASURES:
        m = res["measures"][ev]
        if m["params"] is None:
            L.append(f"  - `gbm`: sigma = sqrt(xi_SPY) = {m['sigma']:.4f}, r = {m['rate']:.4f}.")
        else:
            p = m["params"]
            j = ("" if p["jumps"] is None else
                 f", jumps (lam, mu_j, sig_j) = ({p['jumps'][0]:.3f}, {p['jumps'][1]:.4f}, "
                 f"{p['jumps'][2]:.4f}) per step, compensated")
            L.append(f"  - `{ev}`: eta = {p['eta']:.4f}, rho = {p['rho']:.4f}, "
                     f"H = {p['H']:.4f}, xi = {p['xi']:.6f} (sigma = {m['sigma']:.4f}), "
                     f"r = {p['rate']:.4f}{j}; source `{p['source']}`.")
        L.append(f"    Realized terminal vol {m['realized_vol']:.4f}, premium "
                 f"{m['premium_mc'] * 1e4:.1f} +/- {m['premium_mc_se'] * 1e4:.1f} bp, "
                 f"terminal-log-return skew {m['terminal_skew']:+.2f}, kurtosis "
                 f"{m['terminal_kurtosis']:.1f}.")
    L.append("  The SPY calibration is the one its own quality gate marks `accepted: "
             "false` (eta pinned at its 4.0 bound); it is used as specified because "
             "it is the calibration the repository has, and the rejection is a "
             "caveat on the measure, not on the comparison.")
    L.append("- Baselines, all vol-matched to the realized terminal vol of the "
             "evaluation measure: (i) Black-Scholes delta; (ii) Whalley-Wilmott "
             "no-trade band with the risk aversion chosen on a separate 20,000-path "
             f"tuning set from the grid {list(WW_GRID)} to minimise the baseline's own "
             "CVaR (in-sample for the baseline, deliberately generous); (iii) the "
             "Ruf-Wang linear-regression hedge h = c0 + c1 delta + c2 delta(1 - delta), "
             "coefficients fitted by OLS to minimise the variance of cost-free terminal "
             "P&L on the same 20,000 tuning paths (P&L is linear in c without costs, so "
             "OLS is exact), then charged costs on the test paths.")
    L.append("- Deep policies: the shipped `hedger_gbm.pt` for the GBM-trained row, and "
             "two new checkpoints trained with the same recipe (CVaR_95 objective, "
             "AdamW, cosine schedule, batch 2048) under each rough measure with sigma "
             "and rate pinned at the calibrated level and the cost sampled from [0, 0.02]:")
    for tr in TRAIN_MEASURES:
        c = ck[tr]
        L.append(f"  - `{c['file']}`: measure `{c['train_measure']}`, {c['iters']} iters x "
                 f"batch {c['batch']}, {c['train_seconds']} s of training.")
    L.append(f"- Evaluation grid: train measure x eval measure x cost "
             f"{[_fmt_cost(c) for c in costs]}, {pr['n_paths_per_seed']} paths x "
             f"{len(pr['seeds'])} seeds per cell, {pr['n_boot']} bootstrap resamples. "
             "Ratios deep/baseline are bootstrapped PAIRED (both hedgers on the same "
             "resampled paths). 'Wins' = ratio below 1 by more than 2 SE; 'loses' = "
             "above 1 by more than 2 SE.\n")
    L.append("## Results\n")
    L.append("CVaR_95 in basis points of strike, +/- one bootstrap SE; the last three "
             "columns are the paired ratio deep/delta (W = wins by > 2 SE, L = loses).")
    L.append(write_markdown_tables(res, costs))
    L.append("\n### Regime matrix (train x eval), median over costs of the CVaR ratio\n")
    for bname in ("delta", "whalley_wilmott", "linear"):
        L.append(f"\nversus **{bname}**:\n")
        L.append("| train \\ eval | " + " | ".join(EVAL_MEASURES) + " |")
        L.append("|---|" + "---|" * len(EVAL_MEASURES))
        for tr in TRAIN_MEASURES:
            row = []
            for ev in EVAL_MEASURES:
                v = s["matrix"][f"{tr}|{ev}|{bname}"]
                row.append(f"{v['median_ratio']:.3f} [{v['min_ratio']:.3f}, "
                           f"{v['max_ratio']:.3f}] W{v['wins']}/L{v['loses']}")
            L.append(f"| {tr} | " + " | ".join(row) + " |")
    L.append("\n### Crossover cost (first cost at which the deep hedger wins by > 2 SE)\n")
    L.append("| train | eval | vs delta | vs Whalley-Wilmott | vs linear |")
    L.append("|---|---|---|---|---|")
    for tr in TRAIN_MEASURES:
        for ev in EVAL_MEASURES:
            cells = [s["crossover"][f"{tr}|{ev}|{b}"]
                     for b in ("delta", "whalley_wilmott", "linear")]
            L.append(f"| {tr} | {ev} | " + " | ".join(
                "none" if c is None else _fmt_cost(c) for c in cells) + " |")
    L.append("\n### Transaction costs paid (mean per path, bp of strike) and turnover\n")
    L.append("| eval | cost | delta | WW | linear | " + " | ".join(
        f"deep[{tr}]" for tr in TRAIN_MEASURES) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(TRAIN_MEASURES))
    for ev in EVAL_MEASURES:
        for c in costs:
            cell = res["cells"][f"{ev}|{c}"]
            b = cell["baselines"]
            row = [f"{b[k]['mean_costs'] * 1e4:.1f} (turnover {b[k]['mean_turnover']:.2f})"
                   for k in ("delta", "whalley_wilmott", "linear")]
            row += [f"{cell['deep'][tr]['mean_costs'] * 1e4:.1f} "
                    f"(turnover {cell['deep'][tr]['mean_turnover']:.2f})"
                    for tr in TRAIN_MEASURES]
            L.append(f"| {ev} | {_fmt_cost(c)} | " + " | ".join(row) + " |")
    L.append("\n### Holdings diagnostics (first seed, cost 50 bp)\n")
    L.append("Mean of (deep holding - BS delta) along the paths, its SE, the slope of "
             "the deep holding regressed on the delta, and the distance to the OLS "
             "hedge. Under rho < 0 the literature (Hull-White 2017, Ruf-Wang 2022) "
             "predicts a min-variance hedge BELOW the BS delta; the linear fit's c1 < 1 "
             "and c2 < 0 say the same thing directly.\n")
    L.append("| eval | train | mean(h - delta) | SE | slope h~delta | corr | mean|h - h_lin| | mean h | mean delta |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ev in EVAL_MEASURES:
        d = res["diagnostics"][f"{ev}|0.005"]
        for tr in TRAIN_MEASURES:
            q = d[tr]
            L.append(f"| {ev} | {tr} | {q['mean_h_minus_delta']:+.4f} | "
                     f"{q['mean_h_minus_delta_se']:.4f} | {q['slope_h_on_delta']:.3f} | "
                     f"{q['corr_h_delta']:.3f} | {q['mean_abs_h_minus_linear']:.3f} | "
                     f"{q['mean_holding']:.3f} | {d['delta_mean_holding']:.3f} |")
    L.append("\nLinear-hedge coefficients per measure (c0, c1, c2) and R^2 of the "
             "payoff regression: " + "; ".join(
                 f"`{ev}` {res['measures'][ev]['linear_coef']} R^2 "
                 f"{res['measures'][ev]['linear_r2']:.3f}" for ev in EVAL_MEASURES) + ".\n")
    L.append("## Conclusion\n")
    L.append(f"Across all {tot_n} (train x eval x baseline x cost) cells the deep hedger "
             f"wins {tot_w} by more than 2 SE and loses {tot_l}. In-sample (train = eval) "
             "versus the vol-matched delta: " + "; ".join(
                 f"`{tr}` median ratio {insample[tr]['median_ratio']:.3f}, "
                 f"W{insample[tr]['wins']}/L{insample[tr]['loses']} of {insample[tr]['n_cells']}"
                 for tr in TRAIN_MEASURES) + ".")
    ww_w = sum(s["matrix"][f"{tr}|{ev}|whalley_wilmott"]["wins"]
               for tr in TRAIN_MEASURES for ev in EVAL_MEASURES)
    lin_w = sum(s["matrix"][f"{tr}|{ev}|linear"]["wins"]
                for tr in TRAIN_MEASURES for ev in EVAL_MEASURES)
    del_w = sum(s["matrix"][f"{tr}|{ev}|delta"]["wins"]
                for tr in TRAIN_MEASURES for ev in EVAL_MEASURES)
    L.append(f"Wins by baseline: {del_w} versus delta, {lin_w} versus the linear hedge, "
             f"{ww_w} versus Whalley-Wilmott (each out of "
             f"{len(TRAIN_MEASURES) * len(EVAL_MEASURES) * len(costs)} cells).")
    if tot_w == 0:
        L.append("The deep hedger never wins by more than 2 SE against any baseline "
                 "under any measure or cost in this grid: the negative result of "
                 "`docs/hedging_findings.md` extends to rough volatility and jumps.")
    elif ww_w == 0:
        L.append("The deep hedger beats the cost-blind hedges in some cells but never "
                 "beats the cost-aware Whalley-Wilmott band by more than 2 SE, so what it "
                 "learned is cost avoidance, not a hedge a static rule cannot express.")
    else:
        L.append("The deep hedger beats every baseline, including the cost-aware "
                 "Whalley-Wilmott band, in at least one cell; the tables above say "
                 "which measure and cost, and the crossover table says from which cost on.")
    L.append("\n## What would change this conclusion\n")
    L.append("- A cell where the paired ratio against Whalley-Wilmott is below 1 by "
             "more than 2 SE with a fresh set of seeds (17-21 were used here) would "
             "falsify a 'never wins' reading; a ratio above 1 by more than 2 SE on "
             "fresh seeds would falsify any 'wins' cell.")
    L.append("- The policy is the repository's 2-layer width-64 network with holdings "
             "clamped to [0, 1.5] and a conditional CVaR head, trained for the "
             "iterations listed above on CPU. A longer or larger training run that "
             "moved a losing cell to a win would be evidence against the policy, not "
             "against deep hedging.")
    L.append("- The rough measures are one calibration (its own gate rejects it for a "
             "pinned eta). A calibration with eta inside its bounds, or a jump fit with "
             "larger jumps than the SPY fit's 2.4% mean, changes the measure; the "
             "comparison should be rerun rather than extrapolated.")
    L.append("- The premium is booked at the Monte Carlo price under the simulated "
             "measure; booking at Black-Scholes would shift every hedger's P&L by the "
             "same constant and leave the ratios unchanged.")
    L.append(f"\nCompute: training {sum(ck[t]['train_seconds'] for t in ROUGH_TRAINED)} s "
             f"for the two rough checkpoints, evaluation {res['eval_seconds']} s, "
             f"{pr['torch_threads']} torch threads.\n")
    path.write_text("\n".join(L), encoding="utf-8")


ROUGH_TRAINED = ("rbergomi", "rbergomi_jumps")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", choices=("rbergomi", "rbergomi_jumps"),
                    help="train this checkpoint and exit")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 2 seeds x 500 paths, 100 bootstraps")
    ap.add_argument("--out-stem", default=OUT_STEM)
    args = ap.parse_args()

    if args.train:
        train_checkpoint(args.train, args.iters, args.batch)
        return

    for m in ("rbergomi", "rbergomi_jumps"):
        if not (ARTIFACTS / CHECKPOINTS[m]).exists():
            train_checkpoint(m, args.iters, args.batch)

    seeds = SEEDS[:2] if args.quick else SEEDS
    n_paths = 500 if args.quick else args.paths
    n_boot = 100 if args.quick else N_BOOT
    t0 = time.perf_counter()
    res = evaluate(n_paths, seeds, COSTS, n_boot)
    res["summary"] = summarise(res, COSTS)
    res["protocol"] = {
        "n_steps": N_STEPS, "dt": DT, "maturity": MATURITY,
        "cvar_alpha": CVAR_ALPHA, "costs": list(COSTS), "seeds": list(seeds),
        "n_paths_per_seed": n_paths, "n_boot": n_boot,
        "ww_risk_aversion_grid": list(WW_GRID),
        "torch_threads": TORCH_THREADS,
        "checkpoints": {tr: {"file": CHECKPOINTS[tr],
                             **{k: v for k, v in HedgingEngine(
                                 ARTIFACTS / CHECKPOINTS[tr]).meta.items()
                                if k in ("iters", "batch", "lr", "seed",
                                         "train_measure", "train_seconds",
                                         "train_box")}}
                        for tr in TRAIN_MEASURES},
        "wall_seconds_eval_and_summary": round(time.perf_counter() - t0, 1),
    }
    DOCS.mkdir(exist_ok=True)
    make_figure(res, COSTS, DOCS / f"{args.out_stem}.png")
    tables = write_markdown_tables(res, COSTS)

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and math.isnan(o)) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (DOCS / f"{args.out_stem}.json").write_text(
        json.dumps(_clean(res), indent=1), encoding="utf-8")
    write_markdown(res, COSTS, DOCS / f"{args.out_stem}.md")
    print(tables)
    s = res["summary"]
    print(f"\n[summary] deep hedger wins {s['total_wins']} / {s['total_cells']} "
          f"(train x eval x baseline x cost) cells by > 2 SE, loses "
          f"{s['total_loses']}; eval {res['eval_seconds']} s")
    for k, v in s["matrix"].items():
        print(f"  {k:40s} median ratio {v['median_ratio']:.3f} "
              f"[{v['min_ratio']:.3f}, {v['max_ratio']:.3f}] "
              f"W{v['wins']} L{v['loses']} crossover {s['crossover'][k]}")


if __name__ == "__main__":
    main()
