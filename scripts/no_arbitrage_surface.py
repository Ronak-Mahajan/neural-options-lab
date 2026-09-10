"""Arbitrage audit of the served 0DTE price surrogate, and a no-arbitrage IV surface.

What this script does
---------------------
1. AUDIT.  The served 0DTE ensemble (artifacts/model_0dte.pt, the model
   PricingEngine routes every maturity <= 12/252 to) outputs European call prices.
   backend.quant.iv_surface.TeacherSurface inverts them to Black-Scholes implied
   vols and total variance w(k, T) = sigma_imp^2 T, differentiably, and
   arbitrage_audit() evaluates on a dense (k, T) lattice for several (sigma, r):
     * the Durrleman butterfly function g(k) >= 0 and the calendar condition
       dw/dT >= 0 by autograd through the surrogate,
     * the price-space conditions directly (dC/dK in [-e^{-rT}, 0], d2C/dK2 >= 0,
       a 1%-wide butterfly, calendar monotonicity at fixed k and at fixed K, and
       the intrinsic / spot bounds) as a cross-check.
   Everything is reported on the full box AND on the vega-RESOLVED region, because
   in most of the box the surrogate's ~1.5 bp price noise inverts to a fake smile.

2. SURFACE.  Trains the Ackerer-Tagasovska-Vatter style surface
       w = sigma^2 T * softplus(MLP(k, T, sigma, r) + c0)
   on the teacher's prices (Huber on the vega-normalised residual) with autograd
   butterfly / calendar / Lee-slope penalties on a resampled batch every step, and
   saves artifacts/iv_surface_0dte.pt with the ranges, dynamics, penalties and fit
   metrics in its meta.

3. MEASURE.  Re-audits the constrained surface on the same lattice; IV RMSE vs the
   teacher (vol points, resolved region), price error in bps of strike, min g and
   min dw/dT with locations; and an independent rough Bergomi Monte Carlo check on
   a few smiles (common random numbers across strikes, several seeds, standard
   errors across seeds) that arbitrates between teacher and student where they
   disagree.

Outputs: docs/no_arbitrage_surface.{png,json}; the markdown is written by hand
from the JSON.  Requires matplotlib (requirements-dev.txt).

    python -m scripts.no_arbitrage_surface            # full run (~12 min CPU)
    python -m scripts.no_arbitrage_surface --quick    # smoke run to a temp dir
    python -m scripts.no_arbitrage_surface --skip-train   # reuse the saved surface
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant import iv_surface as ivs  # noqa: E402
from backend.quant.calibrate import implied_vol, bs_vega  # noqa: E402
from backend.quant.engine import PricingEngine  # noqa: E402
from backend.quant.rough_vol import rough_bergomi_mc  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
TRADING_DAYS = ivs.TRADING_DAYS
N_STEPS = 50                  # project protocol for the rough Bergomi engine

# figure palette (project dark theme)
BG, PANEL, BLUE, AMBER, MINT, WHITE, GRID, GREY = (
    "#000000", "#1c1c1e", "#0A84FF", "#FF9F0A", "#30D158", "#FFFFFF", "#3a3a3c",
    "#8E8E93")

#: slice shown in the figure and checked against Monte Carlo
FIG_SIGMA, FIG_RATE = 0.10, 0.04
MC_SIGMAS = (0.10, 0.20)
MC_T_DAYS = (1, 5, 12)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── Monte Carlo reference ──────────────────────────────────────────────────
def mc_reference(teacher: ivs.TeacherSurface, student: ivs.IVSurface, *,
                 sigmas=MC_SIGMAS, T_days=MC_T_DAYS, rate: float = FIG_RATE,
                 k_axis: np.ndarray, n_paths: int, n_seeds: int, seed: int,
                 vega_floor: float = ivs.VEGA_FLOOR) -> list[dict]:
    """Rough Bergomi MC smiles under the teacher's calibrated dynamics.

    One path set per (sigma, T) shared by every strike (common random numbers), re-run
    on `n_seeds` independent seeds; the IV standard error is the across-seed spread /
    sqrt(n_seeds).  Teacher and student IVs are compared on the strikes where the MC
    IV is defined on every seed and BS vega at the MC IV is above the floor.
    """
    dyn = teacher.meta
    eta, rho, H = float(dyn["eta"]), float(dyn["rho"]), float(dyn["H"])
    out = []
    nk = k_axis.size
    for s in sigmas:
        for d in T_days:
            T = d / TRADING_DAYS
            strikes = np.exp(k_axis + rate * T)            # K/S with S = 1
            t0 = time.perf_counter()
            iv_reps = np.full((n_seeds, nk), np.nan)
            price_reps = np.zeros((n_seeds, nk))
            for i in range(n_seeds):
                p = rough_bergomi_mc(
                    spot=torch.ones(nk), strike=torch.tensor(strikes, dtype=torch.float32),
                    maturity=torch.full((nk,), T), xi=torch.full((nk,), s * s),
                    eta=torch.full((nk,), eta), rho=torch.full((nk,), rho),
                    rate=torch.full((nk,), rate), n_paths=n_paths, n_steps=N_STEPS,
                    H=H, seed=seed + 1000 * i + 17 * d + int(s * 100)).numpy()
                price_reps[i] = p / strikes                 # per unit strike
                for j in range(nk):
                    iv = implied_vol(float(p[j]), 1.0, float(strikes[j]), T, rate)
                    iv_reps[i, j] = np.nan if iv is None else iv
            ok = np.isfinite(iv_reps).all(axis=0)
            iv_fill = np.where(np.isfinite(iv_reps), iv_reps, 0.0)
            iv_mc = np.where(ok, iv_fill.mean(axis=0), np.nan)
            se_mc = np.where(ok, iv_fill.std(axis=0, ddof=1) / math.sqrt(n_seeds), np.nan)
            price_mc = price_reps.mean(axis=0)
            price_se = price_reps.std(axis=0, ddof=1) / math.sqrt(n_seeds)
            vega_mc = np.array([bs_vega(1.0, float(K), T, float(v), rate) / float(K)
                                if np.isfinite(v) else 0.0 for K, v in zip(strikes, iv_mc)])
            kt = torch.from_numpy(k_axis); Tt = torch.full((nk,), T, dtype=torch.float64)
            st = torch.full((nk,), s, dtype=torch.float64); rt = torch.full((nk,), rate, dtype=torch.float64)
            iv_t, info = teacher.implied_vol(kt, Tt, st, rt, differentiable=False)
            iv_t = iv_t.numpy(); p_t = info["price"].detach().numpy()
            iv_s = student.iv(k_axis, T, s, rate); p_s = student.price(k_axis, T, s, rate)
            # comparison set: MC resolved (defined on all seeds, vega above floor, SE < 0.5 vp)
            comp = ok & (vega_mc >= vega_floor) & (se_mc < 0.005)
            comp_t = comp & np.isfinite(iv_t)
            def _rmse(a, b, m):
                return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")
            def _chi2(a, m):
                return float(np.mean(((a[m] - iv_mc[m]) / se_mc[m]) ** 2)) if m.any() else float("nan")
            def _med_z(a, m):
                return float(np.median(np.abs(a[m] - iv_mc[m]) / se_mc[m])) if m.any() else float("nan")
            row = {
                "sigma": float(s), "T_days": float(d), "rate": float(rate),
                "n_paths": int(n_paths), "n_seeds": int(n_seeds), "n_strikes": int(nk),
                "n_compared": int(comp.sum()), "n_compared_teacher": int(comp_t.sum()),
                "elapsed_s": time.perf_counter() - t0,
                "teacher_iv_rmse_volpts": 100.0 * _rmse(iv_t, iv_mc, comp_t),
                "student_iv_rmse_volpts": 100.0 * _rmse(iv_s, iv_mc, comp),
                "student_iv_rmse_volpts_same_set": 100.0 * _rmse(iv_s, iv_mc, comp_t),
                "teacher_chi2_per_point": _chi2(iv_t, comp_t),
                "student_chi2_per_point": _chi2(iv_s, comp),
                "teacher_median_abs_z": _med_z(iv_t, comp_t),
                "student_median_abs_z": _med_z(iv_s, comp),
                "teacher_max_abs_volpts": float(100.0 * np.max(np.abs(iv_t[comp_t] - iv_mc[comp_t]))) if comp_t.any() else float("nan"),
                "student_max_abs_volpts": float(100.0 * np.max(np.abs(iv_s[comp] - iv_mc[comp]))) if comp.any() else float("nan"),
                "teacher_price_rmse_bps": 1e4 * _rmse(p_t, price_mc, np.ones(nk, bool)),
                "student_price_rmse_bps": 1e4 * _rmse(p_s, price_mc, np.ones(nk, bool)),
                "teacher_price_max_abs_bps": float(1e4 * np.max(np.abs(p_t - price_mc))),
                "student_price_max_abs_bps": float(1e4 * np.max(np.abs(p_s - price_mc))),
                "mc_price_se_max_bps": float(1e4 * price_se.max()),
                "mc_iv_se_median_volpts_compared": float(100.0 * np.median(se_mc[comp])) if comp.any() else float("nan"),
                "teacher_undefined_strikes": int((~np.isfinite(iv_t)).sum()),
                "k": k_axis.tolist(), "iv_mc": iv_mc.tolist(), "se_mc": se_mc.tolist(),
                "iv_teacher": iv_t.tolist(), "iv_student": iv_s.tolist(),
                "teacher_resolved": (info["vega"].numpy() >= vega_floor).tolist(),
                "compared": comp.tolist(),
            }
            _log(f"  MC sigma={s:.2f} T={d:>2}d: {row['n_compared']} strikes compared, "
                 f"teacher RMSE {row['teacher_iv_rmse_volpts']:.3f} vp "
                 f"(median |z| {row['teacher_median_abs_z']:.1f}), student "
                 f"{row['student_iv_rmse_volpts']:.3f} vp (median |z| "
                 f"{row['student_median_abs_z']:.1f}); median MC SE "
                 f"{row['mc_iv_se_median_volpts_compared']:.3f} vp; "
                 f"teacher price RMSE {row['teacher_price_rmse_bps']:.2f} bps, student "
                 f"{row['student_price_rmse_bps']:.2f} bps  ({row['elapsed_s']:.0f}s)")
            out.append(row)
    return out


# ── dense-grid comparison of student vs teacher ───────────────────────────
def compare_on_grid(teacher_rep: dict, student_rep: dict, vega_floor: float) -> dict:
    A, B = teacher_rep["arrays"], student_rep["arrays"]
    assert np.allclose(A["k"], B["k"]) and np.allclose(A["T"], B["T"])
    defined = (A["defined"] > 0.5) & (A["capped"] < 0.5)
    resolved = defined & (A["vega"] >= vega_floor)
    well = defined & (A["vega"] >= 2.5 * vega_floor)
    iv_t = np.sqrt(np.where(defined, A["w"], np.nan) / A["T"])
    iv_s = np.sqrt(B["w"] / B["T"])
    div = 100.0 * (iv_s - iv_t)
    dp = 1e4 * (B["price"] - A["price"])
    below = ~defined
    out = {
        "n_points": int(A["k"].size), "n_resolved": int(resolved.sum()),
        "iv_rmse_volpts_resolved": float(np.sqrt(np.mean(div[resolved] ** 2))),
        "iv_mae_volpts_resolved": float(np.mean(np.abs(div[resolved]))),
        "iv_p95_abs_volpts_resolved": float(np.quantile(np.abs(div[resolved]), 0.95)),
        "iv_max_abs_volpts_resolved": float(np.max(np.abs(div[resolved]))),
        "iv_rmse_volpts_well_resolved": float(np.sqrt(np.mean(div[well] ** 2))),
        "price_rmse_bps_all": float(np.sqrt(np.mean(dp ** 2))),
        "price_rmse_bps_defined": float(np.sqrt(np.mean(dp[defined] ** 2))),
        "price_rmse_bps_resolved": float(np.sqrt(np.mean(dp[resolved] ** 2))),
        "price_mae_bps_all": float(np.mean(np.abs(dp))),
        "price_p95_abs_bps_all": float(np.quantile(np.abs(dp), 0.95)),
        "price_max_abs_bps_all": float(np.max(np.abs(dp))),
        "price_rmse_bps_where_teacher_below_intrinsic": float(np.sqrt(np.mean(dp[below] ** 2))) if below.any() else 0.0,
    }
    i = int(np.argmax(np.abs(dp)))
    out["price_max_abs_at"] = ivs._loc(A["T"][i], A["k"][i], A["sigma"][i], A["rate"][i])
    j = int(np.flatnonzero(resolved)[np.argmax(np.abs(div[resolved]))])
    out["iv_max_abs_at"] = ivs._loc(A["T"][j], A["k"][j], A["sigma"][j], A["rate"][j])
    # per-sigma IV RMSE on the resolved region
    out["iv_rmse_volpts_resolved_by_sigma"] = {
        f"{s:.2f}": float(np.sqrt(np.mean(div[resolved & (A["sigma"] == s)] ** 2)))
        for s in np.unique(A["sigma"])}
    out["iv_rmse_volpts_resolved_by_T"] = {}
    Td = A["T"] * TRADING_DAYS
    for name, msk in (("1-2d", Td < 2), ("2-4d", (Td >= 2) & (Td < 4)),
                      ("4-8d", (Td >= 4) & (Td < 8)), ("8-12d", Td >= 8)):
        out["iv_rmse_volpts_resolved_by_T"][name] = float(np.sqrt(np.mean(div[resolved & msk] ** 2)))
    return out


def _strip(d):
    """Drop the raw arrays from an audit report for the JSON."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items()
                if k not in ("grids", "arrays", "price_arrays")}
    return d


# ── figure ─────────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=WHITE, labelsize=12, which="both")
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)


def _slice(rep: dict, sigma: float, rate: float) -> dict:
    for g in rep["grids"]:
        if abs(g["sigma"] - sigma) < 1e-9 and abs(g["rate"] - rate) < 1e-9:
            return g
    raise KeyError((sigma, rate))


def make_figure(teacher_rep: dict, student_rep: dict, mc_rows: list[dict],
                summary: dict, out_png: Path, *, sigma: float = FIG_SIGMA,
                rate: float = FIG_RATE, vega_floor: float = ivs.VEGA_FLOOR) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D

    k = np.asarray(teacher_rep["protocol_axes"]["k"])
    Td = np.asarray(teacher_rep["protocol_axes"]["T"]) * TRADING_DAYS
    gt = _slice(teacher_rep, sigma, rate)
    gs_ = _slice(student_rep, sigma, rate)

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gsp = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.80, bottom=0.075,
                           hspace=0.5, wspace=0.2, height_ratios=[1.1, 1.0])
    cmap = LinearSegmentedColormap.from_list(
        "g", [(0.0, AMBER), (0.5, PANEL), (0.75, BLUE), (1.0, WHITE)])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0)
    extent = (k.min(), k.max(), Td.min(), Td.max())

    def heat(ax, blk, title, is_teacher):
        _style(ax)
        g = np.array(blk["g"], dtype=float)
        defined = np.array(blk["defined"]) > 0.5
        vega = np.array(blk["vega"], dtype=float)
        gplot = np.where(defined, np.clip(g, -1.0, 2.0), np.nan)
        im = ax.imshow(gplot, origin="lower", aspect="auto", extent=extent, cmap=cmap,
                       norm=norm, interpolation="nearest")
        if is_teacher and (~defined).any():
            ax.contourf(k, Td, (~defined).astype(float), levels=[0.5, 1.5], colors=[GREY],
                        alpha=0.85)
        viol = defined & (g < 0)
        if viol.any():
            ax.contour(k, Td, viol.astype(float), levels=[0.5], colors=[MINT], linewidths=1.4)
        ax.contour(k, Td, (vega >= vega_floor).astype(float), levels=[0.5], colors=[WHITE],
                   linewidths=1.2, linestyles="dotted")
        n_def = int(defined.sum()); n_v = int(viol.sum())
        res = defined & (vega >= vega_floor)
        n_vr = int((viol & res).sum())
        gmin = np.nanmin(np.where(defined, g, np.nan))
        ax.set_title(f"{title}\ng < 0: {n_v}/{n_def} IV-defined points ({100 * n_v / max(n_def, 1):.1f}%), "
                     f"{n_vr} resolved;  min g = {gmin:+.2f}",
                     color=WHITE, fontsize=11, pad=7)
        ax.set_xlabel("k = ln(K/F)", color=WHITE, fontsize=12.5)
        ax.set_ylabel("T (trading days)", color=WHITE, fontsize=12.5)
        return im

    ax1 = fig.add_subplot(gsp[0, 0])
    ax2 = fig.add_subplot(gsp[0, 1])
    im = heat(ax1, gt, f"served price surrogate: Durrleman g(k, T), sigma = {sigma:.2f}, r = {rate:.2f}", True)
    heat(ax2, gs_, "constrained IV surface, same slice", False)
    # third column: colourbar + legend + calendar summary
    axc = fig.add_subplot(gsp[0, 2])
    axc.axis("off")
    pos = axc.get_position()
    cax = fig.add_axes([pos.x0 + 0.02, pos.y0 + 0.08, 0.016, pos.height - 0.16])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("g, clipped to [-1, 2]", color=WHITE, fontsize=11)
    cb.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE, labelsize=11)
    cb.outline.set_edgecolor(GRID)
    handles = [Line2D([0], [0], color=MINT, ls="-", lw=2.2),
               Line2D([0], [0], color=WHITE, ls=":", lw=2.2),
               Line2D([0], [0], color=GREY, ls="-", lw=6)]
    labels = ["g = 0 contour: g < 0 inside is a\nnegative risk-neutral density",
              f"BS vega = {vega_floor:.2f} per unit strike;\nIV resolved inside the dots",
              "no implied vol: served price\nbelow intrinsic value"]
    axc.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.36, 0.99), fontsize=10.5,
               frameon=False, labelcolor=WHITE, handlelength=2.0, labelspacing=1.0)
    t_cal = summary["teacher"]["iv_space"]["calendar_all_defined"]
    s_cal = summary["student"]["iv_space"]["calendar_all_defined"]
    t_calr = summary["teacher"]["iv_space"]["calendar_resolved"]
    axc.text(0.36, 0.30,
             "calendar condition dw/dT >= 0\n"
             f"surrogate: {100 * t_cal['violation_fraction']:.1f}% of IV-defined\n"
             f"points violate ({100 * t_calr['violation_fraction']:.2f}% of\n"
             f"resolved); min dw/dT = {t_cal['worst_value']:+.3g}\n"
             f"constrained: {100 * s_cal['violation_fraction']:.2f}%;\n"
             f"min dw/dT = {s_cal['worst_value']:+.2e}",
             color=WHITE, fontsize=10.5, va="top", ha="left", linespacing=1.4,
             transform=axc.transAxes)

    # smiles
    rows = {(r["sigma"], r["T_days"]): r for r in mc_rows}
    for j, d in enumerate(MC_T_DAYS):
        ax = fig.add_subplot(gsp[1, j])
        _style(ax)
        row = rows.get((sigma, float(d)))
        T = d / TRADING_DAYS
        half = 6.0 * sigma * math.sqrt(T)
        lo, hi = max(k.min(), -half), min(k.max(), half)
        if row is not None:
            kk = np.array(row["k"]); ivt = np.array(row["iv_teacher"], dtype=float)
            ivs_ = np.array(row["iv_student"], dtype=float)
            ivm = np.array(row["iv_mc"], dtype=float); sem = np.array(row["se_mc"], dtype=float)
            res = np.array(row["teacher_resolved"], dtype=bool)
            comp = np.array(row["compared"], dtype=bool)
            band = (kk >= lo) & (kk <= hi)
            ax.plot(kk[band], 100 * np.where(res, ivt, np.nan)[band], color=BLUE, lw=2.4,
                    label="price surrogate, IV resolved")
            ax.plot(kk[band], 100 * np.where(~res, ivt, np.nan)[band], color=BLUE, lw=1.4,
                    ls=":", alpha=0.85, label="price surrogate, vega below floor")
            ax.plot(kk[band], 100 * ivs_[band], color=MINT, lw=2.2, label="constrained surface")
            m = band & comp
            ax.errorbar(kk[m], 100 * ivm[m], yerr=100 * sem[m], fmt="D", color=AMBER, ms=4.5,
                        ecolor=AMBER, elinewidth=1.0, capsize=2,
                        label=f"rough Bergomi MC, {row['n_seeds']} x {row['n_paths'] // 1000}k paths, +-1 SE")
            m2 = band & ~comp & np.isfinite(ivm)
            if m2.any():
                ax.errorbar(kk[m2], 100 * ivm[m2], yerr=100 * sem[m2], fmt="D", mfc="none",
                            mec=AMBER, ms=4.5, ecolor=AMBER, elinewidth=0.7, capsize=0, alpha=0.6,
                            label="MC, not compared (vega below floor or SE > 0.5 vp)")
            vals = np.concatenate([100 * ivs_[band], 100 * ivm[m]]) if m.any() else 100 * ivs_[band]
            ylo, yhi = np.nanmin(vals), np.nanmax(vals)
            pad = 0.25 * (yhi - ylo + 1e-6)
            ax.set_ylim(ylo - pad, yhi + (2.6 if j == 0 else 1.2) * pad)
            day_word = "trading days" if d > 1 else "trading day"
            ax.set_title(f"T = {d} {day_word}, sigma = {sigma:.2f}, r = {rate:.2f}\n"
                         f"IV RMSE vs MC (n = {row['n_compared']}): surrogate "
                         f"{row['teacher_iv_rmse_volpts']:.2f}, constrained "
                         f"{row['student_iv_rmse_volpts']:.2f} vol pts",
                         color=WHITE, fontsize=11, pad=7)
            if j == 0:
                ax.legend(loc="upper left", fontsize=9.5, frameon=False, labelcolor=WHITE)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("k = ln(K/F),  window |k| <= 6 sigma sqrt(T)", color=WHITE, fontsize=12)
        if j == 0:
            ax.set_ylabel("implied vol (%)", color=WHITE, fontsize=12.5)

    t_but = summary["teacher"]["iv_space"]["butterfly_all_defined"]
    s_but = summary["student"]["iv_space"]["butterfly_all_defined"]
    cov = summary["teacher"]["coverage"]
    cmp_ = summary["comparison"]
    dyn = summary["dynamics"]
    fig.suptitle("0DTE surface: static-arbitrage audit of the served price surrogate vs a "
                 "constrained implied-volatility surface", color=WHITE, fontsize=19, y=0.985)
    sub = (f"{summary['protocol']['n_points']:,}-point grid over the trained box, "
           f"{len(summary['protocol']['sigmas'])} sigma x {len(summary['protocol']['rates'])} r slices.  "
           f"Butterfly g < 0: {100 * t_but['violation_fraction']:.1f}% of IV-defined points before, "
           f"{100 * s_but['violation_fraction']:.2f}% after.  Calendar dw/dT < 0: "
           f"{100 * t_cal['violation_fraction']:.1f}% before, {100 * s_cal['violation_fraction']:.2f}% after.\n"
           f"Served price below intrinsic on {100 * cov['below_intrinsic_fraction']:.1f}% of the box "
           f"(worst {cov['below_intrinsic_worst_bps']:.1f} bps of strike), never after.  "
           f"Constrained vs teacher: IV RMSE {cmp_['iv_rmse_volpts_resolved']:.2f} vol points (resolved region), "
           f"price RMSE {cmp_['price_rmse_bps_all']:.2f} bps.\n"
           f"Teacher dynamics: rough Bergomi H = {dyn['H']:.4f}, eta = {dyn['eta']:.3f}, rho = {dyn['rho']:.3f} "
           f"(artifacts/model_0dte.pt).  Grey = no implied vol; dotted white = vega floor; mint = g = 0.")
    fig.text(0.5, 0.947, sub, color=GREY, fontsize=11, ha="center", va="top", linespacing=1.45)
    fig.savefig(out_png, dpi=100, facecolor=BG)
    plt.close(fig)


# ── driver ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke run to a temp dir")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse artifacts/iv_surface_0dte.pt instead of training")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the PNG from the saved JSON and artifacts (no MC, no training)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--mc-paths", type=int, default=400_000)
    ap.add_argument("--mc-seeds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    t_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    out_dir = args.out_dir
    artifact = ivs.SURFACE_CHECKPOINT
    if args.quick:
        out_dir = out_dir or Path(tempfile.gettempdir()) / "no_arbitrage_surface_quick"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / "iv_surface_0dte_quick.pt"
        k_axis, T_axis = ivs.default_k_axis(41), ivs.default_T_axis(1.0)
        sigmas, rates = (0.05, 0.10, 0.20), (0.04,)
        steps, n_train, mc_paths, mc_seeds = min(args.steps, 300), 65_536, 50_000, 2
    else:
        out_dir = out_dir or DOCS
        k_axis, T_axis = ivs.default_k_axis(), ivs.default_T_axis()
        sigmas, rates = (0.05, 0.10, 0.20, 0.40, 0.80), (0.0, 0.05, 0.10)
        steps, n_train, mc_paths, mc_seeds = args.steps, 262_144, args.mc_paths, args.mc_seeds
    if FIG_RATE not in rates:
        rates = tuple(rates) + (FIG_RATE,)
    if FIG_SIGMA not in sigmas:
        sigmas = tuple(sigmas) + (FIG_SIGMA,)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    engine = PricingEngine()
    teacher = ivs.TeacherSurface(engine)

    if args.figure_only:
        out_json = out_dir / "no_arbitrage_surface.json"
        summary = json.loads(out_json.read_text(encoding="utf-8"))
        if "mc_smiles" not in summary:
            raise SystemExit(f"{out_json} has no per-strike MC smiles; run without --figure-only")
        pr = summary["protocol"]
        k_axis = np.linspace(pr["k_min"], pr["k_max"], pr["n_k"])
        T_axis = np.linspace(pr["T_days_min"], pr["T_days_max"], pr["n_T"]) / TRADING_DAYS
        student = ivs.IVSurface.load(artifact)
        fs = summary["figure_slice"]
        kw = dict(sigmas=(fs["sigma"],), rates=(fs["rate"],), k_axis=k_axis, T_axis=T_axis,
                  return_grids=True)
        teacher_rep = ivs.arbitrage_audit(teacher, price_space=False, **kw)
        teacher_rep["protocol_axes"] = {"k": k_axis.tolist(), "T": T_axis.tolist()}
        student_rep = ivs.arbitrage_audit(student, **kw)
        out_png = out_dir / "no_arbitrage_surface.png"
        make_figure(teacher_rep, student_rep, summary["mc_smiles"], summary, out_png,
                    sigma=fs["sigma"], rate=fs["rate"])
        _log(f"    wrote {out_png} (figure only, {time.perf_counter() - t_start:.0f}s)")
        return
    _log(f"teacher: {teacher.meta.get('n_members')} members, dynamics H={teacher.meta['H']} "
         f"eta={teacher.meta['eta']} rho={teacher.meta['rho']}, kernel={teacher.meta.get('kernel')}")

    # 1. audit the served surrogate
    _log(f"[1] auditing the price surrogate on {len(sigmas)} x {len(rates)} slices x "
         f"{T_axis.size} T x {k_axis.size} k ...")
    t0 = time.perf_counter()
    teacher_rep = ivs.arbitrage_audit(teacher, sigmas=sigmas, rates=rates, k_axis=k_axis,
                                      T_axis=T_axis, return_grids=True)
    timings["audit_teacher_s"] = time.perf_counter() - t0
    teacher_rep["protocol_axes"] = {"k": k_axis.tolist(), "T": T_axis.tolist()}
    cov, ivsp, ps = teacher_rep["coverage"], teacher_rep["iv_space"], teacher_rep["price_space"]
    _log(f"    IV defined on {100 * cov['iv_defined_fraction']:.2f}% of the box, resolved on "
         f"{100 * cov['resolved_fraction']:.2f}%; below intrinsic on "
         f"{100 * cov['below_intrinsic_fraction']:.2f}% (worst {cov['below_intrinsic_worst_bps']:.1f} bps)")
    for name in ("butterfly_all_defined", "butterfly_resolved", "calendar_all_defined", "calendar_resolved"):
        r = ivsp[name]
        _log(f"    {name:<24} {100 * r['violation_fraction']:6.3f}% of {r['n_points']:>6}  "
             f"worst {r['worst_value']:+.4g} at {r['worst_at']}")
    for name in ("convexity_d2C_dK2", "butterfly_1pct_bps", "slope_dC_dK_ge_-discount",
                 "calendar_fixed_k", "calendar_fixed_strike", "price_below_intrinsic_bps"):
        r = ps[name]
        _log(f"    price-space {name:<26} {100 * r['violation_fraction']:6.3f}%  worst {r['worst_value']:+.4g}")
    _log(f"    sign agreement IV-space vs price-space: butterfly "
         f"{ps['sign_agreement_butterfly_vs_convexity']:.4f}, calendar "
         f"{ps['sign_agreement_calendar_w_vs_price']:.4f}  ({timings['audit_teacher_s']:.0f}s)")

    # 2. the constrained surface
    if args.skip_train and artifact.exists():
        _log(f"[2] loading {artifact}")
        student = ivs.IVSurface.load(artifact)
        train_info = {"history": student.meta.get("training", {}).get("history", []),
                      "validation": student.meta.get("fit_metrics", {}).get("validation", {}),
                      "wall_s": student.meta.get("training", {}).get("wall_s", float("nan"))}
    else:
        _log(f"[2] training the constrained surface ({steps} steps, {n_train} teacher samples) ...")
        student, train_info = ivs.train_iv_surface(teacher, n_train=n_train, steps=steps,
                                                   seed=args.seed, threads=args.threads,
                                                   log=_log)
    timings["train_s"] = float(train_info["wall_s"])
    v = train_info["validation"]
    _log(f"    held-out: IV RMSE {v['iv_rmse_volpts_resolved']:.3f} vp (resolved, n={v['n_resolved']}), "
         f"price RMSE {v['price_rmse_bps']:.2f} bps, p95 {v['price_p95_abs_bps']:.2f} bps")

    # 3. audit the student on the same grid, compare
    _log("[3] auditing the constrained surface on the same grid ...")
    t0 = time.perf_counter()
    student_rep = ivs.arbitrage_audit(student, sigmas=sigmas, rates=rates, k_axis=k_axis,
                                      T_axis=T_axis, return_grids=True)
    timings["audit_student_s"] = time.perf_counter() - t0
    for name in ("butterfly_all_defined", "calendar_all_defined"):
        r = student_rep["iv_space"][name]
        _log(f"    {name:<24} {100 * r['violation_fraction']:6.3f}% of {r['n_points']:>6}  "
             f"worst {r['worst_value']:+.4g} at {r['worst_at']}")
    comparison = compare_on_grid(teacher_rep, student_rep, ivs.VEGA_FLOOR)
    _log(f"    student vs teacher on the grid: IV RMSE {comparison['iv_rmse_volpts_resolved']:.3f} vp "
         f"(resolved), price RMSE {comparison['price_rmse_bps_all']:.2f} bps (all), "
         f"{comparison['price_rmse_bps_defined']:.2f} bps (IV-defined), max "
         f"{comparison['price_max_abs_bps_all']:.1f} bps at {comparison['price_max_abs_at']}")

    # 4. Monte Carlo arbiter
    _log(f"[4] rough Bergomi MC check: {mc_seeds} seeds x {mc_paths} paths per smile ...")
    t0 = time.perf_counter()
    mc_rows = mc_reference(teacher, student, k_axis=k_axis, n_paths=mc_paths,
                           n_seeds=mc_seeds, seed=args.seed)
    timings["mc_s"] = time.perf_counter() - t0

    # persist the surface with its audit numbers in the meta
    student.meta["audit"] = {
        "grid": teacher_rep["protocol"],
        "student": _strip({k: student_rep[k] for k in ("coverage", "iv_space")}),
        "teacher": _strip({k: teacher_rep[k] for k in ("coverage", "iv_space")}),
        "comparison": comparison,
        "mc_check": [{k: v for k, v in r.items() if not isinstance(v, list)} for r in mc_rows],
    }
    student.save(artifact)
    _log(f"    saved {artifact}")

    timings["total_s"] = time.perf_counter() - t_start
    summary = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {**teacher_rep["protocol"], "torch_threads": args.threads,
                     "training": {k: v for k, v in student.meta["training"].items() if k != "history"},
                     "penalties": student.meta["penalties"], "loss": student.meta["loss"],
                     "architecture": student.meta["architecture"],
                     "mc": {"n_paths": mc_paths, "n_seeds": mc_seeds, "n_steps": N_STEPS,
                            "sigmas": list(MC_SIGMAS), "T_days": list(MC_T_DAYS), "rate": FIG_RATE}},
        "dynamics": student.meta["dynamics"],
        "teacher": _strip({k: teacher_rep[k] for k in ("coverage", "iv_space", "price_space")}),
        "student": _strip({k: student_rep[k] for k in ("coverage", "iv_space")}),
        "comparison": comparison,
        "fit_metrics": student.meta["fit_metrics"],
        "training_history": train_info["history"],
        "mc_check": [{k: v for k, v in r.items() if not isinstance(v, list)} for r in mc_rows],
        "figure_slice": {"sigma": FIG_SIGMA, "rate": FIG_RATE},
        "timings_s": timings,
        # per-strike MC smiles (needed by --figure-only); last so the key numbers read first
        "mc_smiles": mc_rows,
    }
    out_json = out_dir / "no_arbitrage_surface.json"
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _log(f"    wrote {out_json}")

    out_png = out_dir / "no_arbitrage_surface.png"
    make_figure(teacher_rep, student_rep, mc_rows, summary, out_png)
    _log(f"    wrote {out_png}")
    _log(f"done in {timings['total_s']:.0f}s  (audit teacher {timings['audit_teacher_s']:.0f}s, "
         f"train {timings['train_s']:.0f}s, audit student {timings['audit_student_s']:.0f}s, "
         f"MC {timings['mc_s']:.0f}s)")


if __name__ == "__main__":
    main()
