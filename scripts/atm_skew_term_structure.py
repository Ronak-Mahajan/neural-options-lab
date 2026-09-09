"""ATM implied-vol skew term structure: rough Bergomi (model) against SPY and BTC (market).

The claim under test
--------------------
Under rough Bergomi the at-the-money skew

    psi(T) = d sigma_imp / dk  at  k = ln(K / F) = 0

behaves like C * T^(H - 1/2) for short maturities (Bayer, Friz & Gatheral 2016;
Fukasawa 2011/2017): a straight line of slope H - 1/2 on a log-log plot of
|psi| against T. A classical (H = 1/2) stochastic-volatility model has a skew
that is FLAT in T at the short end, so the slope is the fingerprint.

What this script measures
-------------------------
Model side. `backend.quant.rough_vol.rough_bergomi_mc` prices a five-point
strike stencil k in {-2h, -h, 0, +h, +2h} on ONE path set per maturity
(common random numbers), the prices are inverted to Black-Scholes implied vols
with `backend.quant.calibrate.implied_vol` (spot and rate as the engine uses
them, so fwd_pv == spot), and psi(T) is the central difference
(iv(+h) - iv(-h)) / 2h. The +-2h points give a second estimate at twice the
step, so |psi_h - psi_2h| / 3 is a measured bound on the O(h^2) truncation
error of the central difference. The Monte Carlo error on psi is measured, not
propagated: the whole stencil is re-priced on `--reps` independent seeds and
the standard error is the spread of psi across seeds. (Propagating the
per-strike standard errors as if the +-h prices were independent ignores the
common-random-numbers correlation and overstates the error; that "naive" value
is recorded alongside as an upper bound.) Parameters come from
artifacts/rough_calibration.json (SPY) and artifacts/rough_calibration_btc.json
(BTC); nothing is hard-coded.

Market side (offline, reproducible). For each committed SPY capture under
data/surfaces/equity/ chosen from regular trading hours, and for each expiry,
a local quadratic in k is fitted by weighted least squares to the quoted mid
implied vols with |k| <= 2 * atm_iv * sqrt(tau), weights 1 / half_spread_iv^2,
and its slope at k = 0 is the empirical psi(tau) with the regression standard
error. BTC uses the committed Deribit snapshot(s) through
backend.quant.surface.build_surface (clean OTM quotes, mid IV, half the IV
bid-ask as the weight).

Fits. log|psi| = a + b log T by weighted least squares (weights 1/SE^2 in log
space) over T <= 45 trading days for the model, and over every expiry for the
market. The reported standard error on b is the least-squares one inflated by
sqrt(chi^2/dof) when chi^2/dof > 1, so an under-dispersed error model cannot
make the exponent look better determined than the scatter says it is.

    python -m scripts.atm_skew_term_structure                # full run (~4 min CPU)
    python -m scripts.atm_skew_term_structure --quick        # smoke run
    python -m scripts.atm_skew_term_structure --skip-btc

Outputs: docs/atm_skew_term_structure.{png,json,md}. The markdown is written
by hand from the JSON; this script writes the PNG and JSON only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.calibrate import bs_vega, implied_vol  # noqa: E402
from backend.quant.calibrate_map import quotes_from_capture  # noqa: E402
from backend.quant.rough_vol import rough_bergomi_mc  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DATA = ROOT / "data"
TRADING_DAYS = 252.0
DEFAULT_T_DAYS = (1, 2, 3, 5, 8, 12, 20, 30, 45, 63, 90, 126)
SHORT_FIT_MAX_DAYS = 45.0
#: strike stencil in multiples of the step h; skew_from_stencil relies on this
#: layout (index 1 and 3 are -h and +h, 0 and 4 are -2h and +2h, 2 is ATM).
STENCIL = (-2.0, -1.0, 0.0, 1.0, 2.0)
N_STEPS = 50                    # project protocol for the rough Bergomi engine


# ── parameters ─────────────────────────────────────────────────────────────
def load_params(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {k: float(d[k]) for k in ("eta", "rho", "H", "xi")}
    spot = d.get("spot")
    if spot is None:
        spot = d.get("index_price")
    out["spot"] = float(spot)
    out["rate"] = float(d.get("rate", 0.0) or 0.0)
    out["as_of"] = d.get("as_of")
    out["accepted"] = d.get("accepted")
    out["reject_reasons"] = list(d.get("reject_reasons", []))
    out["source"] = str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    return out


def leading_order_coefficient(eta: float, rho: float, H: float) -> float:
    """First-order (in vol-of-vol) prefactor C in psi(T) ~ C T^(H - 1/2).

    From the Bergomi-Guyon first-order expansion applied to the rough Bergomi
    forward-variance dynamics d xi_t(u)/xi_t(u) = eta sqrt(2H) (u-t)^(H-1/2) dW_t:
    the spot/variance covariance functional is
    C^{x xi} = rho eta sqrt(2H) sigma xi T^(H+3/2) / ((H+1/2)(H+3/2)) and the
    ATM skew is C^{x xi} / (2 sigma^3 T^2). At H = 1/2 this collapses to the
    textbook lognormal-vol limit rho eta / 4 (vol-of-vol eta/2 on sigma, skew
    rho nu / 2). It is a small-eta*T^H statement; the MC below says how far
    from it the calibrated parameters sit.
    """
    return rho * eta * math.sqrt(2.0 * H) / (2.0 * (H + 0.5) * (H + 1.5))


# ── model side ─────────────────────────────────────────────────────────────
def moneyness_step(xi: float, T: float, h_scale: float = 0.25,
                   h_floor: float = 0.002) -> float:
    """Central-difference step in log-moneyness: a fixed fraction of the ATM
    vol scale sqrt(xi) sqrt(T), floored so a one-day stencil is not so tight
    that float32 prices cannot resolve it."""
    return max(h_scale * math.sqrt(xi) * math.sqrt(T), h_floor)


def price_stencil(params: dict, T: float, h: float, n_paths: int, seed: int,
                  n_steps: int = N_STEPS):
    """Call prices at k = STENCIL * h on ONE rough Bergomi path set (CRN)."""
    spot, rate = params["spot"], params["rate"]
    F = spot * math.exp(rate * T)
    ks = np.asarray(STENCIL, dtype=float) * h
    strikes = F * np.exp(ks)
    n = len(ks)

    def full(v: float) -> torch.Tensor:
        return torch.full((n,), float(v), dtype=torch.float32)

    prices, se = rough_bergomi_mc(
        full(spot), torch.tensor(strikes, dtype=torch.float32), full(T),
        full(params["xi"]), full(params["eta"]), full(params["rho"]),
        full(rate), n_paths=n_paths, n_steps=n_steps, H=params["H"],
        seed=seed, return_std_error=True)
    return ks, strikes, prices.double().numpy(), se.double().numpy()


def skew_from_stencil(ks: np.ndarray, strikes: np.ndarray, prices: np.ndarray,
                      spot: float, T: float, rate: float,
                      se_prices: np.ndarray | None = None) -> dict:
    """Invert a five-point stencil of call prices to implied vols and take the
    central-difference ATM skew at steps h and 2h.

    `spot` plays the role of fwd_pv in `implied_vol`: the engine drifts the
    spot at `rate` with no dividend, so Black-Scholes on (spot, rate) is the
    matching inversion. Returns NaN for any point implied_vol refuses.
    """
    ks = np.asarray(ks, dtype=float)
    h = float(ks[3])
    if not (abs(ks[1] + h) < 1e-12 and abs(ks[0] + 2 * h) < 1e-12
            and abs(ks[4] - 2 * h) < 1e-12 and abs(ks[2]) < 1e-12):
        raise ValueError("stencil must be (-2h, -h, 0, +h, +2h)")
    ivs = np.full(len(ks), np.nan)
    for i, (K, p) in enumerate(zip(strikes, prices)):
        v = implied_vol(float(p), spot, float(K), T, rate)
        if v is not None:
            ivs[i] = v
    out = {
        "h": h,
        "ivs": ivs.tolist(),
        "atm_iv": float(ivs[2]),
        "psi": float((ivs[3] - ivs[1]) / (2.0 * h)),
        "psi_2h": float((ivs[4] - ivs[0]) / (4.0 * h)),
    }
    if se_prices is not None:
        vegas = np.array([bs_vega(spot, float(K), T, float(v), rate)
                          if np.isfinite(v) else np.nan
                          for K, v in zip(strikes, ivs)])
        se_iv = np.asarray(se_prices, dtype=float) / vegas
        out["se_naive"] = float(math.sqrt(se_iv[3] ** 2 + se_iv[1] ** 2)
                                / (2.0 * h))
    return out


def model_skew_curve(params: dict, T_days=DEFAULT_T_DAYS, n_paths: int = 400_000,
                     n_reps: int = 8, seed: int = 20260908,
                     n_steps: int = N_STEPS, h_scale: float = 0.25,
                     h_floor: float = 0.002, log=None) -> list[dict]:
    """psi(T) on the maturity grid, with the across-seed standard error."""
    rows = []
    for d in T_days:
        T = float(d) / TRADING_DAYS
        h = moneyness_step(params["xi"], T, h_scale, h_floor)
        t0 = time.perf_counter()
        psis, psis2, atms, naive = [], [], [], []
        for r in range(n_reps):
            ks, K, p, se = price_stencil(params, T, h, n_paths,
                                         seed + 10_007 * r + int(d), n_steps)
            s = skew_from_stencil(ks, K, p, params["spot"], T, params["rate"], se)
            psis.append(s["psi"])
            psis2.append(s["psi_2h"])
            atms.append(s["atm_iv"])
            naive.append(s["se_naive"])
        psis = np.asarray(psis)
        psis2 = np.asarray(psis2)
        ok = np.isfinite(psis) & np.isfinite(psis2)
        n_ok = int(ok.sum())

        def mean_se(v: np.ndarray) -> tuple[float, float]:
            if n_ok == 0:
                return float("nan"), float("nan")
            m = float(np.mean(v[ok]))
            s = (float(np.std(v[ok], ddof=1) / math.sqrt(n_ok))
                 if n_ok > 1 else float("nan"))
            return m, s

        psi, se = mean_se(psis)
        psi_2h, se_2h = mean_se(psis2)
        # Richardson: the h and 2h central differences share the O(h^2) error
        # structure, so (4 psi_h - psi_2h) / 3 cancels it (per seed, then
        # averaged, so its SE is measured the same way).
        psi_rich, se_rich = mean_se((4.0 * psis - psis2) / 3.0)
        row = {
            "T_days": float(d), "T": T, "h": h,
            "atm_iv": float(np.nanmean(atms)),
            "psi": psi, "se": se,
            "se_naive": float(math.sqrt(np.nanmean(np.square(naive))) /
                              math.sqrt(max(n_ok, 1))),
            "psi_2h": psi_2h, "se_2h": se_2h,
            "psi_richardson": psi_rich, "se_richardson": se_rich,
            "truncation": float(abs(psi - psi_2h) / 3.0),
            "psi_reps": [float(x) for x in psis],
            "psi_2h_reps": [float(x) for x in psis2],
            "n_reps_ok": n_ok, "n_paths_per_rep": int(n_paths),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        if log:
            log(f"    T={d:>4g}d  h={h:.4f}  atm={row['atm_iv']:.4f}  "
                f"psi={psi:+.4f} +- {se:.4f} (naive {row['se_naive']:.4f}, "
                f"trunc {row['truncation']:.4f})  {row['seconds']:.1f}s")
    return rows


# ── fits ───────────────────────────────────────────────────────────────────
def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    r = y - X @ beta
    return beta, cov, float(np.sum(w * r ** 2)), r


def fit_power_law(T, psi, se, mask=None) -> dict:
    """log|psi| = a + b log T by WLS with weights (|psi| / se)^2.

    A log-magnitude fit only means something for a skew of one sign, so rows
    whose sign disagrees with the majority are excluded and counted
    (`n_wrong_sign`); a term structure that changes sign is not a power law.
    """
    T = np.asarray(T, dtype=float)
    psi = np.asarray(psi, dtype=float)
    se = np.asarray(se, dtype=float)
    m = (np.isfinite(T) & np.isfinite(psi) & np.isfinite(se)
         & (psi != 0.0) & (se > 0.0))
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    sign = -1.0 if (psi[m] < 0).sum() >= (psi[m] > 0).sum() else 1.0
    wrong = m & (np.sign(psi) != sign)
    m &= ~wrong
    x = np.log(T[m])
    y = np.log(np.abs(psi[m]))
    w = (np.abs(psi[m]) / se[m]) ** 2
    n = int(m.sum())
    if n < 3:
        return {"n": n, "b": float("nan"), "se_b": float("nan"),
                "n_wrong_sign": int(wrong.sum()), "sign": sign}
    X = np.column_stack([np.ones_like(x), x])
    beta, cov, chi2, _ = wls(X, y, w)
    dof = n - 2
    scale = max(1.0, chi2 / dof)
    return {
        "a": float(beta[0]), "b": float(beta[1]), "C": float(math.exp(beta[0])),
        "se_b_ls": float(math.sqrt(cov[1, 1])),
        "se_b": float(math.sqrt(cov[1, 1] * scale)),
        "se_a": float(math.sqrt(cov[0, 0] * scale)),
        "chi2": chi2, "dof": dof, "chi2_over_dof": chi2 / dof, "n": n,
        "n_wrong_sign": int(wrong.sum()), "sign": sign,
        "T_min": float(T[m].min()), "T_max": float(T[m].max()),
    }


#: (T_lo, T_hi) in trading days for the local-exponent windows
LOCAL_WINDOWS = ((1, 5), (2, 8), (3, 12), (5, 20), (8, 30), (12, 45),
                 (20, 63), (30, 126))


def local_exponents(model: list[dict], key: str = "psi",
                    se_key: str = "se") -> list[dict]:
    """Power-law slope over sliding maturity windows: does the local exponent
    drift with T (finite vol-of-vol corrections) or sit at H - 1/2?"""
    T = np.array([r["T"] for r in model])
    Td = np.array([r["T_days"] for r in model])
    psi = np.array([r[key] for r in model])
    se = np.array([r[se_key] for r in model])
    out = []
    for lo, hi in LOCAL_WINDOWS:
        f = fit_power_law(T, psi, se, (Td >= lo) & (Td <= hi))
        if f["n"] >= 3:
            out.append({"T_lo_days": lo, "T_hi_days": hi, "n": f["n"],
                        "b": f["b"], "se_b": f["se_b"]})
    return out


# ── market side ────────────────────────────────────────────────────────────
def local_quadratic_skew(k, iv, half_spread=None, band: float = 0.05,
                         min_quotes: int = 6, hs_floor: float = 1e-3) -> dict | None:
    """Slope at k = 0 of a WLS quadratic iv = a + b k + c k^2 over |k| <= band.

    Weights are 1 / half_spread^2 (floored at hs_floor vol) when a half spread
    is given, else uniform. The standard error of b is the least-squares one
    scaled by the residual variance, so it reflects the actual scatter of the
    quotes around the quadratic rather than the nominal spread.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    sel = np.isfinite(k) & np.isfinite(iv) & (np.abs(k) <= band)
    n = int(sel.sum())
    if n < min_quotes:
        return None
    k, iv = k[sel], iv[sel]
    if k.min() >= 0.0 or k.max() <= 0.0:
        return None                      # need quotes on both sides of ATM
    if half_spread is None:
        w = np.ones(n)
    else:
        hs = np.asarray(half_spread, dtype=float)[sel]
        good = np.isfinite(hs) & (hs > 0.0)
        if good.any():
            hs = np.where(good, hs, np.median(hs[good]))
            w = 1.0 / np.maximum(hs, hs_floor) ** 2
        else:
            w = np.ones(n)
    X = np.column_stack([np.ones(n), k, k ** 2])
    beta, cov, chi2, r = wls(X, iv, w)
    dof = n - 3
    s2 = chi2 / dof if dof > 0 else float("nan")
    se = np.sqrt(np.diag(cov) * s2)
    return {
        "atm_iv": float(beta[0]), "psi": float(beta[1]), "se": float(se[1]),
        "curvature": float(beta[2]), "n": n, "band": float(band),
        "k_min": float(k.min()), "k_max": float(k.max()),
        "rms_resid_volpts": float(np.sqrt(np.mean(r ** 2)) * 100.0),
    }


def market_skew_from_quotes(quotes, rate: float, band_mult: float = 2.0,
                            min_quotes: int = 6) -> list[dict]:
    """One row per expiry from calibrate.Quote objects (SPY captures).

    k is measured against the forward F = fwd_pv e^{r tau}, not against fwd_pv:
    at tau = 0.04 and r = 3.7% the difference is 0.0015, comparable to the
    one-day stencil step, so it is not negligible for a slope at k = 0.
    """
    by: dict[str, list] = {}
    for q in quotes:
        by.setdefault(q.expiry, []).append(q)
    rows = []
    for expiry, qs in sorted(by.items()):
        tau = float(np.median([q.tau for q in qs]))
        k = np.array([math.log(q.strike / q.fwd_pv) - rate * q.tau for q in qs])
        iv = np.array([q.iv for q in qs], dtype=float)
        hs = np.array([getattr(q, "half_spread_iv", float("nan")) for q in qs],
                      dtype=float)
        atm_iv0 = float(iv[np.argmin(np.abs(k))])
        band = band_mult * atm_iv0 * math.sqrt(tau)
        fit = local_quadratic_skew(k, iv, hs, band, min_quotes)
        if fit is None:
            continue
        rows.append({"expiry": expiry, "tau": tau,
                     "T_days_equiv": tau * TRADING_DAYS,
                     "n_quotes_expiry": len(qs), **fit})
    return rows


def market_skew_from_capture(path, band_mult: float = 2.0,
                             min_quotes: int = 6) -> tuple[list[dict], dict]:
    path = Path(path)
    quotes, rate, meta = quotes_from_capture(path)
    rows = market_skew_from_quotes(quotes, rate, band_mult, min_quotes)
    info = {
        "capture": path.name,
        "pricing_time": meta.get("pricing_time") or meta.get("as_of"),
        "spot": meta.get("spot"), "stale": meta.get("stale"),
        "quote_source": meta.get("quote_source"), "rate": rate,
        "n_quotes": len(quotes),
    }
    return rows, info


def trading_hour_captures(pattern: str, n_target: int = 8) -> list[Path]:
    """Committed SPY captures priced on a weekday between 09:35 and 15:55
    New York, not flagged stale, thinned to about n_target evenly in time."""
    keep = []
    for p in sorted(glob.glob(pattern)):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        ts = d.get("pricing_time")
        if not ts or d.get("stale"):
            continue
        t = datetime.fromisoformat(ts)
        minutes = t.hour * 60 + t.minute
        if t.weekday() >= 5 or not (9 * 60 + 35 <= minutes <= 15 * 60 + 55):
            continue
        keep.append(Path(p))
    if len(keep) <= n_target:
        return keep
    idx = np.unique(np.round(np.linspace(0, len(keep) - 1, n_target)).astype(int))
    return [keep[i] for i in idx]


# ── BTC (Deribit) ──────────────────────────────────────────────────────────
def btc_load_surface(path):
    from backend.quant.deribit import Snapshot, load_snapshot
    from backend.quant.surface import build_surface
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            snap = Snapshot.from_dict(json.load(fh))
    else:
        snap = load_snapshot(p)
    return build_surface(snap), snap


def btc_market_rows(path, band_mult: float = 2.0,
                    min_quotes: int = 6) -> tuple[list[dict], dict]:
    """One row per expiry from a Deribit snapshot: clean OTM quotes, mid IV,
    half the IV bid-ask as the weight. Deribit's rate is zero and the forward
    is per expiry, so k = ln(K / F) directly."""
    surf, snap = btc_load_surface(path)
    by: dict[int, list] = {}
    for q in surf.otm():
        by.setdefault(q.expiry_ms, []).append(q)
    rows = []
    for expiry_ms, qs in sorted(by.items()):
        tau = float(qs[0].tenor)
        k = np.array([math.log(q.strike / q.forward) for q in qs])
        iv = np.array([q.iv_mid for q in qs], dtype=float)
        hs = np.array([0.5 * q.iv_spread for q in qs], dtype=float)
        atm_iv0 = float(iv[np.argmin(np.abs(k))])
        band = band_mult * atm_iv0 * math.sqrt(tau)
        fit = local_quadratic_skew(k, iv, hs, band, min_quotes)
        if fit is None:
            continue
        rows.append({"expiry": qs[0].expiry_iso, "tau": tau,
                     "T_days_equiv": tau * TRADING_DAYS,
                     "n_quotes_expiry": len(qs), **fit})
    info = {"capture": Path(path).name, "pricing_time": snap.captured_at_iso,
            "spot": snap.index_price, "n_clean_otm": len(surf.otm())}
    return rows, info


# ── figure ─────────────────────────────────────────────────────────────────
BG, PANEL, BLUE, AMBER, WHITE, GRID, GREY = ("#000000", "#1c1c1e", "#0A84FF",
                                             "#FF9F0A", "#FFFFFF", "#3a3a3c",
                                             "#8E8E93")


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=WHITE, labelsize=13, which="both")
    ax.grid(True, which="major", color=GRID, lw=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5)


def _panel(ax, blk: dict, title: str):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    _style(ax)
    model = blk["model"]
    T = np.array([r["T"] for r in model])
    psi = np.abs([r["psi"] for r in model])
    se = np.array([r["se"] for r in model])
    H = blk["params"]["H"]
    ref = H - 0.5
    fits = blk["fits"]

    # The band is drawn only where psi - SE stays positive (log axis); a
    # point whose SE reaches zero gets a down-arrow instead of a fake floor.
    lower = psi - se
    band_ok = lower > 0.0
    ax.fill_between(T, np.where(band_ok, lower, psi), psi + se, color=BLUE,
                    alpha=0.3, lw=0, label="model +-1 SE (across seeds)")
    if (~band_ok).any():
        ax.errorbar(T[~band_ok], psi[~band_ok],
                    yerr=[0.5 * psi[~band_ok], se[~band_ok]], lolims=True,
                    fmt="none", ecolor=BLUE, elinewidth=1.2, capsize=3)
    ax.plot(T, psi, color=BLUE, lw=2.4, marker="o", ms=6,
            label=f"rough Bergomi, H = {H:.3f} (calibrated)")

    fm = fits["model_short"]
    if np.isfinite(fm.get("b", float("nan"))):
        Tf = np.geomspace(fm["T_min"], fm["T_max"], 50)
        ax.plot(Tf, fm["C"] * Tf ** fm["b"], "--", color="#66B2FF", lw=1.8,
                label=f"model fit, T <= 45d: slope {fm['b']:+.3f} +- {fm['se_b']:.3f}")
        Tc = math.sqrt(fm["T_min"] * fm["T_max"])
        yc = fm["C"] * Tc ** fm["b"]
        ax.plot(Tf, yc * (Tf / Tc) ** ref, ":", color=WHITE, lw=2.0,
                label=f"reference slope H - 1/2 = {ref:+.3f}")
    C_lo = abs(blk["leading_order_C"])
    Tl = np.geomspace(T.min(), T.max(), 50)
    ax.plot(Tl, C_lo * Tl ** ref, "-.", color=GREY, lw=1.3,
            label=f"first-order theory |C| T^(H-1/2), |C| = {C_lo:.3f}")

    rows = blk["market_rows"]
    y_all = [psi.min(), psi.max()]
    if rows:
        rng = np.random.default_rng(0)
        Tm = np.array([r["tau"] for r in rows])
        pm_signed = np.array([r["psi"] for r in rows])
        pm = np.abs(pm_signed)
        sm = np.array([r["se"] for r in rows])
        jit = np.exp(rng.uniform(-0.03, 0.03, size=len(Tm)))
        fk = fits["market_pooled"]
        # filled markers share the sign the T <= 45d fit was made on (the
        # majority sign there); hollow ones have the opposite sign and are
        # excluded from that fit, since log|psi| across a sign change is
        # not a power law
        fit_sign = fk.get("sign", -1.0)
        same = np.sign(pm_signed) == fit_sign
        sign_word = "negative" if fit_sign < 0 else "positive"
        ax.errorbar(Tm[same] * jit[same], pm[same], yerr=sm[same], fmt="D",
                    color=AMBER, ms=5.5, alpha=0.9, ecolor=AMBER,
                    elinewidth=0.9, capsize=0,
                    label=f"market, {sign_word} skew (n = {int(same.sum())})")
        if (~same).any():
            ax.errorbar(Tm[~same] * jit[~same], pm[~same], yerr=sm[~same],
                        fmt="D", mfc="none", mec=AMBER, ms=6.5, ecolor=AMBER,
                        elinewidth=0.9, capsize=0,
                        label=f"market, opposite sign, not in fit "
                              f"(n = {int((~same).sum())})")
        y_all += [pm.min(), pm.max()]
        if np.isfinite(fk.get("b", float("nan"))):
            Tf = np.geomspace(fk["T_min"], fk["T_max"], 50)
            ax.plot(Tf, fk["C"] * Tf ** fk["b"], "--", color=AMBER, lw=1.8,
                    label=f"market fit, T <= 45d, {sign_word} rows: slope "
                          f"{fk['b']:+.3f} +- {fk['se_b']:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # reserve a band under the data for the legend so it never covers a curve
    ax.set_ylim(min(y_all) / 3.5, max(y_all) * 1.3)
    ax.set_xlabel("T (years, log scale)", color=WHITE, fontsize=15)
    ax.set_ylabel("|psi(T)| = |d sigma_imp / dk| at k = 0", color=WHITE,
                  fontsize=15)
    ax.set_title(title, color=WHITE, fontsize=15, pad=46)
    sec = ax.secondary_xaxis("top", functions=(lambda t: t * TRADING_DAYS,
                                               lambda d: d / TRADING_DAYS))
    sec.tick_params(colors=WHITE, labelsize=12, which="both")
    days = [1, 2, 5, 10, 20, 50, 126]
    sec.xaxis.set_major_locator(FixedLocator(days))
    sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    sec.xaxis.set_minor_formatter(NullFormatter())
    sec.set_xlabel("trading days (T x 252)", color=WHITE, fontsize=12)
    sec.spines["top"].set_color(GRID)
    # explicit y ticks: a log axis spanning less than a decade otherwise
    # shows no labelled tick at all
    ylo, yhi = ax.get_ylim()
    cands = [c * 10.0 ** e for e in range(-3, 2) for c in (1, 2, 3, 5, 7)]
    yt = [c for c in cands if ylo <= c <= yhi]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    leg = ax.legend(loc="lower left", fontsize=11.5, facecolor=BG,
                    edgecolor=GRID, labelcolor=WHITE, framealpha=0.9)
    leg.get_frame().set_linewidth(0.8)


def make_figure(blocks: list[tuple[str, dict]], out_png: Path,
                subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(blocks)
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(1, n, width_ratios=([1.55, 1.0] if n == 2 else [1.0]),
                          left=0.065, right=0.985, top=0.79, bottom=0.09,
                          wspace=0.17)
    for i, (title, blk) in enumerate(blocks):
        _panel(fig.add_subplot(gs[0, i]), blk, title)
    fig.suptitle("ATM implied-vol skew term structure: rough Bergomi vs market",
                 color=WHITE, fontsize=20, y=0.985)
    fig.text(0.5, 0.955, subtitle, color=GREY, fontsize=12, ha="center",
             va="top", linespacing=1.4)
    fig.savefig(out_png, dpi=100, facecolor=BG)
    plt.close(fig)


# ── driver ─────────────────────────────────────────────────────────────────
def summarise_market(rows_by_capture: list[tuple[dict, list[dict]]]) -> dict:
    """Pooled power-law fit over every (capture, expiry) row, plus the
    per-capture exponents so the spread across captures is visible."""
    all_rows = [dict(r, capture=info["capture"], pricing_time=info["pricing_time"])
                for info, rows in rows_by_capture for r in rows]
    short = [r["T_days_equiv"] <= SHORT_FIT_MAX_DAYS for r in all_rows]
    pooled = fit_power_law([r["tau"] for r in all_rows],
                           [r["psi"] for r in all_rows],
                           [r["se"] for r in all_rows], short)
    pooled_all = fit_power_law([r["tau"] for r in all_rows],
                               [r["psi"] for r in all_rows],
                               [r["se"] for r in all_rows])
    per_capture = []
    for info, rows in rows_by_capture:
        f = fit_power_law([r["tau"] for r in rows], [r["psi"] for r in rows],
                          [r["se"] for r in rows],
                          [r["T_days_equiv"] <= SHORT_FIT_MAX_DAYS for r in rows])
        per_capture.append({"capture": info["capture"],
                            "pricing_time": info["pricing_time"],
                            "b": f.get("b"), "se_b": f.get("se_b"),
                            "n": f.get("n")})
    bs = np.array([c["b"] for c in per_capture if c["b"] is not None
                   and np.isfinite(c["b"])])
    spread = {
        "n_captures": int(len(bs)),
        "mean_b": float(bs.mean()) if len(bs) else float("nan"),
        "sd_b": float(bs.std(ddof=1)) if len(bs) > 1 else float("nan"),
        "min_b": float(bs.min()) if len(bs) else float("nan"),
        "max_b": float(bs.max()) if len(bs) else float("nan"),
    }
    return {"rows": all_rows, "pooled": pooled, "pooled_all": pooled_all,
            "per_capture": per_capture, "spread": spread, "n_rows": len(all_rows)}


def run_block(name: str, params: dict, T_days, n_paths: int, n_reps: int,
              seed: int, n_steps: int, market: list[tuple[dict, list[dict]]],
              log) -> dict:
    log(f"[{name}] model: eta={params['eta']:.4f} rho={params['rho']:.4f} "
        f"H={params['H']:.4f} sqrt(xi)={math.sqrt(params['xi']):.4f} "
        f"rate={params['rate']:.4f} spot={params['spot']:.2f} "
        f"({params['source']}, accepted={params['accepted']})")
    t0 = time.perf_counter()
    model = model_skew_curve(params, T_days, n_paths, n_reps, seed, n_steps,
                             log=log)
    model_seconds = time.perf_counter() - t0
    T = [r["T"] for r in model]
    psi = [r["psi"] for r in model]
    se = [r["se"] for r in model]
    short = [r["T_days"] <= SHORT_FIT_MAX_DAYS for r in model]
    fits = {
        "model_short": fit_power_law(T, psi, se, short),
        "model_all": fit_power_law(T, psi, se),
        "model_short_with_truncation": fit_power_law(
            T, psi, [math.hypot(r["se"], r["truncation"]) for r in model], short),
        "model_short_richardson": fit_power_law(
            T, [r["psi_richardson"] for r in model],
            [r["se_richardson"] for r in model], short),
        "model_local": local_exponents(model),
    }
    mk = summarise_market(market)
    fits["market_pooled"] = mk["pooled"]          # same window as the model
    fits["market_all"] = mk["pooled_all"]
    ref = params["H"] - 0.5
    fm = fits["model_short"]
    fits["model_minus_reference_sigmas"] = (
        (fm["b"] - ref) / fm["se_b"] if np.isfinite(fm.get("b", np.nan)) else None)
    fk = fits["market_pooled"]
    fits["market_minus_reference_sigmas"] = (
        (fk["b"] - ref) / fk["se_b"] if np.isfinite(fk.get("b", np.nan)) else None)
    fits["model_minus_market_sigmas"] = (
        (fm["b"] - fk["b"]) / math.hypot(fm["se_b"], fk["se_b"])
        if np.isfinite(fm.get("b", np.nan)) and np.isfinite(fk.get("b", np.nan))
        else None)
    log(f"[{name}] model fit T<=45d: b = {fm.get('b', float('nan')):+.4f} "
        f"+- {fm.get('se_b', float('nan')):.4f} (chi2/dof "
        f"{fm.get('chi2_over_dof', float('nan')):.2f}); Richardson "
        f"{fits['model_short_richardson'].get('b', float('nan')):+.4f}; "
        f"H - 1/2 = {ref:+.4f}")
    log(f"[{name}] local exponents: " + "  ".join(
        f"[{w['T_lo_days']}-{w['T_hi_days']}d] {w['b']:+.3f}+-{w['se_b']:.3f}"
        for w in fits["model_local"]))
    log(f"[{name}] market T<=45d: b = {fk.get('b', float('nan')):+.4f} "
        f"+- {fk.get('se_b', float('nan')):.4f} over {fk.get('n')} rows "
        f"({fk.get('n_wrong_sign')} wrong-sign excluded); per-capture mean "
        f"{mk['spread']['mean_b']:+.4f} sd {mk['spread']['sd_b']:.4f}; all "
        f"expiries b = {fits['market_all'].get('b', float('nan')):+.4f} "
        f"+- {fits['market_all'].get('se_b', float('nan')):.4f}")
    return {
        "params": params, "reference_slope": ref,
        "leading_order_C": leading_order_coefficient(
            params["eta"], params["rho"], params["H"]),
        "eta_T_H_at_1d": params["eta"] * (1.0 / TRADING_DAYS) ** params["H"],
        "eta_T_H_at_45d": params["eta"] * (45.0 / TRADING_DAYS) ** params["H"],
        "model": model, "model_seconds": model_seconds,
        "market_rows": mk["rows"], "market_per_capture": mk["per_capture"],
        "market_spread": mk["spread"],
        "captures": [info for info, _ in market],
        "fits": fits,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", type=int, default=400_000,
                   help="MC paths per seed per maturity")
    p.add_argument("--reps", type=int, default=8,
                   help="independent seeds per maturity (the SE comes from these)")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--captures", nargs="*", default=None,
                   help="SPY capture files; default: ~8 trading-hour captures")
    p.add_argument("--n-captures", type=int, default=8)
    p.add_argument("--btc-snapshots", nargs="*", default=None)
    p.add_argument("--skip-btc", action="store_true")
    p.add_argument("--eta-scan", nargs="*", type=float, default=(0.5, 1.5),
                   help="re-run the SPY model at these vol-of-vol values (same "
                        "H, rho, xi) to test whether the slope moves toward "
                        "H - 1/2 as eta T^H shrinks; pass none to skip")
    p.add_argument("--out-dir", type=Path, default=DOCS)
    p.add_argument("--quick", action="store_true",
                   help="smoke run: 50k paths x 2 seeds, 6 maturities, 3 captures")
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    t_all = time.perf_counter()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    T_days = DEFAULT_T_DAYS
    n_paths, n_reps, n_caps = args.paths, args.reps, args.n_captures
    if args.quick:
        T_days = (1, 3, 8, 20, 45, 126)
        n_paths, n_reps, n_caps = 50_000, 2, 3

    # ── SPY ──────────────────────────────────────────────────────────────
    spy_params = load_params(ARTIFACTS / "rough_calibration.json")
    caps = ([Path(c) for c in args.captures] if args.captures else
            trading_hour_captures(str(DATA / "surfaces" / "equity" / "spy_*.json.gz"),
                                  n_caps))
    log(f"[SPY] {len(caps)} captures: " + ", ".join(c.name for c in caps))
    t0 = time.perf_counter()
    spy_market = []
    for c in caps:
        rows, info = market_skew_from_capture(c)
        spy_market.append((info, rows))
        log(f"    {info['pricing_time']}  spot {info['spot']:.2f}  "
            f"{len(rows)} expiries: " + "  ".join(
                f"{r['expiry'][5:]}:{r['psi']:+.3f}+-{r['se']:.3f}(n{r['n']})"
                for r in rows))
    spy_market_seconds = time.perf_counter() - t0
    spy = run_block("SPY", spy_params, T_days, n_paths, n_reps, args.seed,
                    args.steps, spy_market, log)
    spy["market_seconds"] = spy_market_seconds

    # ── eta scan: is the departure from H - 1/2 a finite-vol-of-vol effect? ─
    eta_scan = []
    scan_T = tuple(d for d in T_days if d <= SHORT_FIT_MAX_DAYS)
    for eta in (args.eta_scan or ()):
        q = dict(spy_params, eta=float(eta), source=spy_params["source"] +
                 f" with eta overridden to {eta}")
        log(f"[SPY eta={eta}] eta T^H at 1d = "
            f"{eta * (1 / TRADING_DAYS) ** q['H']:.3f}")
        t0 = time.perf_counter()
        rows = model_skew_curve(q, scan_T, max(n_paths // 2, 25_000), n_reps,
                                args.seed + 7, args.steps, log=log)
        f = fit_power_law([r["T"] for r in rows], [r["psi"] for r in rows],
                          [r["se"] for r in rows])
        fr = fit_power_law([r["T"] for r in rows],
                           [r["psi_richardson"] for r in rows],
                           [r["se_richardson"] for r in rows])
        C_lo = leading_order_coefficient(eta, q["rho"], q["H"])
        eta_scan.append({"eta": float(eta),
                         "eta_T_H_at_1d": eta * (1 / TRADING_DAYS) ** q["H"],
                         "b": f["b"], "se_b": f["se_b"],
                         "b_richardson": fr["b"], "se_b_richardson": fr["se_b"],
                         "C_fit": f.get("C"), "C_leading_order": abs(C_lo),
                         "model": rows, "local": local_exponents(rows),
                         "seconds": time.perf_counter() - t0})
        log(f"[SPY eta={eta}] slope {f['b']:+.4f} +- {f['se_b']:.4f} "
            f"(Richardson {fr['b']:+.4f}); C_fit {f.get('C', float('nan')):.3f} "
            f"vs first-order {abs(C_lo):.3f}; H - 1/2 = {spy['reference_slope']:+.4f}")
    spy["eta_scan"] = eta_scan

    # ── BTC (optional, cheap) ────────────────────────────────────────────
    btc = None
    btc_note = "skipped by --skip-btc"
    if not args.skip_btc:
        try:
            t0 = time.perf_counter()
            btc_params = load_params(ARTIFACTS / "rough_calibration_btc.json")
            snaps = ([Path(s) for s in args.btc_snapshots] if args.btc_snapshots
                     else sorted(ARTIFACTS.glob("deribit_snapshot_btc_*.json"))
                     + sorted(DATA.glob("surfaces/deribit/btc_20260820T1717*.json.gz"))
                     + sorted(DATA.glob("surfaces/deribit/btc_20260821T1500*.json.gz")))
            if args.quick:
                snaps = snaps[:1]
            btc_market = []
            for s in snaps:
                rows, info = btc_market_rows(s)
                btc_market.append((info, rows))
                log(f"[BTC] {info['pricing_time']}  index {info['spot']:.0f}  "
                    f"{len(rows)} expiries: " + "  ".join(
                        f"{r['tau']*365:.1f}d:{r['psi']:+.3f}+-{r['se']:.3f}(n{r['n']})"
                        for r in rows))
            btc = run_block("BTC", btc_params, T_days, n_paths, n_reps,
                            args.seed + 1, args.steps, btc_market, log)
            btc["market_seconds"] = time.perf_counter() - t0 - btc["model_seconds"]
            btc_note = "included"
        except Exception as exc:          # noqa: BLE001 - report, do not hide
            btc = None
            btc_note = f"failed: {type(exc).__name__}: {exc}"
            log(f"[BTC] {btc_note}")

    # ── outputs ──────────────────────────────────────────────────────────
    total = time.perf_counter() - t_all
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "atm_skew_term_structure" + ("_quick" if args.quick else "")
    payload = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "T_days": list(T_days), "n_paths_per_seed": n_paths,
            "n_seeds": n_reps, "n_steps": args.steps, "seed": args.seed,
            "stencil": list(STENCIL), "h_rule": "max(0.25 sqrt(xi) sqrt(T), 0.002)",
            "short_fit_max_days": SHORT_FIT_MAX_DAYS,
            "market_band": "|k| <= 2 atm_iv sqrt(tau); WLS 1/half_spread^2; k vs forward",
            "torch_threads": torch.get_num_threads(),
        },
        "spy": spy, "btc": btc, "btc_note": btc_note,
        "seconds_total": total,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")
    render_from_payload(payload, out_dir / f"{stem}.png")
    log(f"wrote {out_dir / (stem + '.json')} and {out_dir / (stem + '.png')} "
        f"in {total:.0f}s total")
    return 0


def render_from_payload(payload: dict, out_png: Path) -> None:
    """Draw the figure from a saved JSON (also used by --figure-only)."""
    spy, btc = payload["spy"], payload.get("btc")
    proto = payload["protocol"]
    spy_day = (spy["params"].get("as_of") or "")[:10]
    blocks = [(f"SPY: model vs listed expiries\n(parameters calibrated "
               f"{spy_day}; {len(spy['captures'])} trading-hour captures)", spy)]
    if btc is not None:
        btc_day = (btc["params"].get("as_of") or "")[:10]
        blocks.append((f"BTC (Deribit): model vs listed expiries\n(REJECTED "
                       f"fit of {btc_day}; {len(btc['captures'])} snapshots)",
                       btc))
    fs = spy["fits"]
    sub = (f"SPY model slope {fs['model_short']['b']:+.3f} +- "
           f"{fs['model_short']['se_b']:.3f} (T <= 45d) vs H - 1/2 = "
           f"{spy['reference_slope']:+.3f}; market {fs['market_pooled']['b']:+.3f} +- "
           f"{fs['market_pooled']['se_b']:.3f}.\n{proto['n_seeds']} seeds x "
           f"{proto['n_paths_per_seed']:,} paths per maturity, "
           f"n_steps = {proto['n_steps']}. Market: one marker per expiry per "
           f"capture, jittered 3% in T, bars = fit SE.")
    make_figure(blocks, out_png, sub)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        _out = DOCS / "atm_skew_term_structure"
        render_from_payload(json.loads(_out.with_suffix(".json").read_text(
            encoding="utf-8")), _out.with_suffix(".png"))
        print(f"redrew {_out.with_suffix('.png')}")
        sys.exit(0)
    sys.exit(main())
