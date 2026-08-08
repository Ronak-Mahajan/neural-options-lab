"""Live market calibration of the rough Bergomi 0DTE engine.

Fits (eta, rho, H) — plus the forward-variance nuisance parameter xi — to the
short-dated (<= max_dte) SPY option smile, using the same PyTorch Monte Carlo
teacher (rough_vol.rough_bergomi_mc) that generates the 0DTE surrogate's
training data.

Objective
---------
    min_theta  mean_i  Huber( (P_model_i(theta) - P_mid_i) / vega_i ; delta )

* Vega-weighting: to first order dP = vega * d(sigma_iv), so vega-normalised
  price errors ARE implied-vol errors. This keeps every strike on the smile
  equally informative (raw price MSE is dominated by ATM options and fits the
  skew poorly) without inverting Black-Scholes on a *noisy MC price* at every
  optimizer step. The same vega floor (VEGA_FLOOR) is used by the objective and
  by the reported metric, so the number in the log is the number being
  minimised.
* Huber (delta = 2 vol points, Huber 1964): stale or crossed quotes get linear,
  not quadratic, influence.
* Common random numbers (Glasserman 2004, ch. 4.2): a frozen seed makes the MC
  objective a deterministic, near-smooth function of the parameters — the
  optimizer never chases resampling noise. It does NOT make the objective
  noise-free: the CRN surface has its own MC-induced ripples, which is what the
  reported noise floor measures (see below).
* Joint fit over all retained expiries: the Hurst index H is identified by the
  *term structure* of the skew (ATM skew ~ rho*eta*tau^{H-1/2}), not by any
  single smile. A run that retains one expiry cannot identify H at all, and
  says so in the log.

Optimisation and the MC noise floor
-----------------------------------
Stage 1  differential evolution (Storn & Price 1997), global, bounded, at
         `search_paths` paths.
Stage 2  Powell polish (Powell 1964) at `search_paths`.
Stage 3  Powell RE-polish at `polish_paths` (default: `final_paths`).

Stage 3 exists because H is not identifiable at 8k paths: the calibrator used
to build one Calibrator with n_paths=search_paths, and both the global stage and
the polish therefore optimised the *cheap* objective — `final_paths` only ever
entered the printed report. An audit profiling the real loss in H (with eta
re-fitted at each H) over six CRN seeds found the argmin wandering across the
entire test grid, 0.030 to 0.200, with co-fitted eta running to its upper bound;
at 64,000 paths the same profile recovers the truth. So the fit is now polished
at the precision it is reported at, and every run prints the objective's own
Monte Carlo standard deviation — measured by re-evaluating the fitted point
under `noise_reps` independent path sets — so a reader can see whether a loss
improvement is signal or resampling.

Everything is quoted off the market's own forward
-------------------------------------------------
The map from puts to calls is exact only under the true forward:

    C - P = e^{-r tau} (F - K)

Using fwd = spot * e^{r tau} assumes a zero dividend yield. SPY pays 1.1-1.3%,
and because the parity map is applied ONLY to puts the resulting error is
one-sided: a pure skew distortion, and rho is precisely the parameter that
absorbs skew. So F is backed out of the market instead, at the strike whose
call/put pair is tightest (the Cboe VIX methodology):

    F = K* + e^{r tau} (C_mid(K*) - P_mid(K*))

and the discounted forward, fwd_pv = F e^{-r tau}, replaces `spot` everywhere:
in the OTM split, in the parity map, in bs_call/bs_vega/implied_vol (which are
then Black-76 in disguise — Black 1976), and as the initial value of the Monte
Carlo, so that model and market share one forward by construction. Expiries
that span an ex-dividend date are skipped outright: SPY is American-exercise and
the early-exercise boundary bites hardest just before an ex-dividend date, where
the European parity map is not merely biased but wrong.

What the market snapshot actually is
------------------------------------
Two modes, decided per run and recorded in the JSON:

live    >= MIN_QUOTES two-sided books survive the spread/liquidity filters. Mids
        are (bid+ask)/2 and tau is measured from now. Rows without a two-sided
        book are REJECTED as unquoted; they do not silently fall back to a stale
        print stamped with the current time.
closed  no live book. Mids are the previous session's last trades, and tau is
        measured from the *median print time of that session*, not from now —
        measuring tau from now while pricing off yesterday's prints understates
        every maturity by up to a full session, which shortens the lever arm
        (log tau2/tau1) that identifies H and inflates the fitted xi. Prints
        from any earlier session, and prints with no timestamp, are dropped.
        Such runs are written for inspection and never adopted (see the gate).

Other data hygiene: OTM options only, bid > $0.02, relative spread <= 40%,
volume or open-interest floor, Yahoo IVs discarded in favour of our own
bisection inversion, and every drop reason counted and printed.

The quality gate
----------------
`accepted` is the single flag every downstream consumer keys on. A calibration
is adopted only when all of the following hold:

    1. the honest implied-vol RMSE is below MAX_RMSE_VOLPTS;
    2. no free parameter — eta, rho, H *or xi* — sits within PIN_FRAC of a
       bound (a bound hit signals misspecification or an ill-conditioned fit);
    3. the quotes came from a live book, not from last-session prints;
    4. every quote was priceable, i.e. no model price fell outside the no-arb
       range (see the metric note below).

The gate is enforced HERE, at the point of use: `main --retrain` refuses to
regenerate the dataset or retrain the surrogate on a rejected fit and exits 2.
It previously only lived inside dataset_0dte.load_calibrated_dynamics(), which
the --retrain path does not call — so drift_monitor.run_recalibration(), which
shells out to `calibrate --retrain` whenever drift is detected, would adopt a
rejected calibration automatically. The rejected fit is still written to disk,
with `reject_reasons`, for inspection.

The metric does not get to drop its worst quotes
------------------------------------------------
implied_vol() returns None when a model price falls outside the no-arb range,
and those quotes are exactly the worst fits. Filtering them out of the RMSE
(and then reporting that RMSE next to n_quotes = ALL of them) understates the
error precisely where the fit is failing. Unpriceable quotes are now scored by
the same vega-linearised error the objective uses, (P_model - P_mid)/vega,
which is the first-order continuation of the implied-vol error past the no-arb
boundary; `n_scored`, `n_unpriceable` and the old drop-the-worst number
(`iv_rmse_volpts_priceable_only`) are all recorded, and any unpriceable quote
fails the gate.

Day-count convention — READ THIS BEFORE COMPARING TO THE SURROGATE
------------------------------------------------------------------
This module measures tau in ACT/365 *calendar* time. dataset_0dte.py trains the
surrogate on k/252 *trading-day* maturities with k >= 1, i.e. its shortest
trained maturity is 1/252 yr = 34.76 h of ACT/365 time, while one calendar day
is 24 h: the two conventions disagree by 365/252 = 1.45x at the short end. The
calibration therefore drops every quote below MIN_TAU_HOURS so that no fitted
maturity lies outside the surrogate's trained domain. The conventions are still
not reconciled — a 1 DTE quote enters this fit as tau = 24/8760 yr and would
enter the surrogate as 1/252 yr — and that reconciliation is a separate change.

Usage (from the repo root):
    python -m backend.quant.calibrate                    # calibrate SPY live
    python -m backend.quant.calibrate --ticker QQQ --max-dte 2
    python -m backend.quant.calibrate --retrain          # + regenerate the
                                                         # 0DTE dataset and
                                                         # retrain the
                                                         # surrogate ensemble
                                                         # (exit 2 if the fit
                                                         # fails the gate)

Writes artifacts/rough_calibration.json, which dataset_0dte.py picks up
automatically on the next dataset build.

References
----------
Bayer, Friz & Gatheral (2016), "Pricing under rough volatility",
    Quantitative Finance 16(6), 887-904.
Black (1976), "The pricing of commodity contracts", J. Financial Economics 3.
Cboe (2019), "Cboe Volatility Index Methodology" — the forward is taken from
    the strike with the smallest |C - P|, F = K + e^{rT}(C - P).
Glasserman (2004), "Monte Carlo Methods in Financial Engineering", Springer.
Huber (1964), "Robust estimation of a location parameter",
    Ann. Math. Statist. 35(1), 73-101.
Powell (1964), "An efficient method for finding the minimum of a function of
    several variables without calculating derivatives", Computer J. 7(2).
Storn & Price (1997), "Differential evolution", J. Global Optimization 11.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from .market_data import MarketDataError
from .dataset_0dte import KERNEL_ID
from .rough_vol import rough_bergomi_mc

# Windows consoles often default to cp1252, which cannot encode the rules
# and Greek letters in the log; force UTF-8 rather than crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
CAL_FILE = ARTIFACTS / "rough_calibration.json"
BOUNDS = {"eta": (0.5, 4.0), "rho": (-1.0, 0.0), "H": (0.01, 0.5),
          "xi": (0.05 ** 2, 1.2 ** 2)}
HUBER_DELTA = 0.02          # 2 vol points
VEGA_FLOOR = 1e-4           # objective AND metric share this floor
CRN_SEED = 1234
REPORT_SEED = CRN_SEED + 9973   # out-of-sample path set for the final report
NY = ZoneInfo("America/New_York")

# ── gate thresholds ───────────────────────────────────────────────────
MAX_RMSE_VOLPTS = 3.0
# Bound-pinning tolerance as a fraction of each bound's own width, so the test
# means the same thing for a correlation (width 1.0) as for a variance
# (width 1.4375). An absolute 1e-3, as used before, is 0.03% of the eta range
# but 40% of the lower xi bound.
#: How close to a bound counts as pinned, as a fraction of that parameter's
#: range. This was 1e-3, i.e. 0.1% — so tight it could essentially never fire.
#: Caught empirically: the first live Deribit BTC calibration returned
#: eta = 3.936 against an upper bound of 4.0 — 98.2% of the way across the
#: range, pinned in any practical sense — and the gate ACCEPTED it. A bound hit
#: is the signal that the model cannot reach the market without an extreme
#: parameter, which is exactly what the gate exists to catch, so the tolerance
#: has to be wide enough to notice. At 2% that fit is correctly rejected while
#: rho (74.4% of range), H (36.4%) and xi (5.7%) still pass.
PIN_FRAC = 0.02

# ── quote filters ─────────────────────────────────────────────────────
MIN_QUOTES = 8
MIN_BID = 0.02              # a 1-cent bid is not a price
MAX_REL_SPREAD = 0.40
MIN_VOLUME = 10             # live book: volume OR open-interest floor
MIN_OPEN_INTEREST = 100
MIN_LAST = 0.05             # closed market: stricter, no spread to check
MIN_LAST_VOLUME = 50
MONEYNESS_BAND = (0.90, 1.10)
# 1/252 yr — the surrogate's shortest trained maturity — expressed in ACT/365
# hours is 365*24/252 = 34.76 h; round up so the floor is unambiguous.
SURROGATE_MIN_TAU_YEARS = 1.0 / 252.0
MIN_TAU_HOURS = 34.8
# The parity-implied forward of a <= 3 DTE index option sits within a few tens
# of basis points of spot; 2% is a sanity band on a broken pair, not a fit.
FWD_SANITY_BAND = 0.02
# A closed-market snapshot is only meaningful if the prints are roughly
# synchronous; beyond this the "surface" is a collage of different minutes.
MAX_SNAPSHOT_SPREAD_MIN = 45.0

# ── terminal aesthetics ────────────────────────────────────────────────
CY, MG, VI, GN, RD, DIM, BOLD, RS = ("\x1b[38;5;51m", "\x1b[38;5;205m",
                                     "\x1b[38;5;141m", "\x1b[38;5;84m",
                                     "\x1b[38;5;203m", "\x1b[2m",
                                     "\x1b[1m", "\x1b[0m")


def rule(title: str = "") -> None:
    pad = f"═══ {BOLD}{title}{RS} " if title else ""
    print(f"{DIM}{pad}{'═' * max(8, 74 - len(title))}{RS}")


# ── Black-76 helpers (calls only — puts are parity-mapped) ─────────────
# `fwd_pv` is the present value of the forward, F * e^{-r tau}. Substituting it
# for the spot turns the Black-Scholes formulas below into Black (1976):
#   C = e^{-r tau} [F N(d1) - K N(d2)] = fwd_pv N(d1) - K e^{-r tau} N(d2),
#   d1 = [log(F/K) + sigma^2 tau / 2] / (sigma sqrt(tau)).
# For a non-dividend-paying underlying fwd_pv == spot and nothing changes.
def bs_call(fwd_pv: float, strike: float, tau: float, sigma: float,
            rate: float) -> float:
    sd = max(sigma, 1e-8) * math.sqrt(tau)
    d1 = (math.log(fwd_pv / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    d2 = d1 - sd
    return fwd_pv * norm.cdf(d1) - strike * math.exp(-rate * tau) * norm.cdf(d2)


def bs_vega(fwd_pv: float, strike: float, tau: float, sigma: float,
            rate: float) -> float:
    sd = max(sigma, 1e-8) * math.sqrt(tau)
    d1 = (math.log(fwd_pv / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    return fwd_pv * norm.pdf(d1) * math.sqrt(tau)


def implied_vol(price: float, fwd_pv: float, strike: float, tau: float,
                rate: float, lo: float = 1e-3,
                hi: float = 5.0) -> float | None:
    """Bisection BS inversion; None if the quote is out of no-arb range.

    The no-arb range for a call on a forward F is
    (max(fwd_pv - K e^{-r tau}, 0), fwd_pv). Returning None is deliberate — a
    price outside it has no implied vol — but a None must never be silently
    dropped from an error metric; see iv_fit_report().
    """
    if price <= max(fwd_pv - strike * math.exp(-rate * tau), 0.0) + 1e-10:
        return None
    if price >= fwd_pv:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_call(fwd_pv, strike, tau, mid, rate) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── market data ingestion ──────────────────────────────────────────────
@dataclass(frozen=True)
class Quote:
    tau: float          # year fraction (ACT/365) from the PRICING time
    strike: float
    mid_call: float     # call-equivalent mid (puts parity-mapped)
    iv: float           # our own BS inversion of the mid, off fwd_pv
    vega: float
    kind: str           # original instrument: "C" or "P"
    expiry: str
    fwd_pv: float       # F * e^{-r tau} for this expiry (see Forward)


@dataclass(frozen=True)
class RawRow:
    """One row of a Yahoo option chain, before any pricing decision.

    Kept separate from Quote because tau is not knowable row-by-row: it depends
    on the pricing time, which depends on whether the whole chain is live or a
    last-session snapshot, which depends on all the rows.
    """
    expiry: str
    strike: float
    kind: str
    bid: float | None
    ask: float | None
    last: float | None
    volume: float
    open_interest: float
    last_trade: datetime | None     # NY-localised time of the last print


@dataclass(frozen=True)
class Forward:
    f: float                    # forward used for this expiry
    source: str                 # "parity" | "spot_carry_*"
    strike: float | None        # strike of the parity pair, if any
    implied_carry: float        # r - log(F/S)/tau: the dividend/borrow the
    #                             market is actually pricing, annualised
    pair_spread: float | None   # combined relative spread of that pair

    def pv(self, rate: float, tau: float) -> float:
        return self.f * math.exp(-rate * tau)


@dataclass
class MarketSnapshot:
    """Everything the fit needs, plus everything the reader needs to judge it.

    Replaces the old 3-tuple return annotation (which was a lie: the function
    returned four values, the fourth being the load-bearing staleness flag).
    """
    spot: float
    rate: float
    rate_source: str
    quotes: list[Quote]
    stale: bool                      # prices are last-session prints
    pricing_time: datetime           # the instant every tau is measured from
    run_time: datetime
    session_date: date | None
    snapshot_spread_min: float | None
    forwards: dict[str, Forward] = field(default_factory=dict)
    expiries: list[str] = field(default_factory=list)
    skipped_exdiv: list[str] = field(default_factory=list)
    exdiv_check: str = "unavailable"
    drops: dict[str, int] = field(default_factory=dict)

    @property
    def quote_source(self) -> str:
        return "last_trade_market_closed" if self.stale else "live_mid"


def _safe_float(value: Any) -> float | None:
    """Parse a Yahoo chain cell; None — never 0.0 — when it is not a number.

    Mapping a MISSING ask to 0.0 (the previous behaviour) makes `ask >= bid`
    false and pushes the row into the last-trade branch: an *unquoted* contract
    silently priced off a stale print instead of being rejected.
    """
    if value is None:
        return None
    try:
        if value != value:                      # NaN / NaT
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_ny(value: Any) -> datetime | None:
    """Coerce a yfinance timestamp (pandas Timestamp, datetime, or epoch
    seconds) to a New-York-localised datetime; None if it is not one."""
    if value is None:
        return None
    try:
        if value != value:                      # NaT / NaN
            return None
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                return None
            dt: Any = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            dt = value.to_pydatetime() if hasattr(value, "to_pydatetime") \
                else value
        if not isinstance(dt, datetime):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)   # yfinance stamps UTC
        return dt.astimezone(NY)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def book_mid(row: RawRow) -> tuple[float | None, float | None, str]:
    """Live two-sided book mid. Returns (mid, relative spread, reject reason)."""
    if row.bid is None or row.ask is None:
        return None, None, "unquoted"           # a side of the book is missing
    if row.bid <= MIN_BID or row.ask < row.bid:
        return None, None, "no_book"
    mid = 0.5 * (row.bid + row.ask)
    spread = (row.ask - row.bid) / mid
    if spread > MAX_REL_SPREAD:
        return None, spread, "wide_spread"
    if row.volume < MIN_VOLUME and row.open_interest < MIN_OPEN_INTEREST:
        return None, spread, "illiquid"
    return mid, spread, ""


def last_trade_mid(row: RawRow) -> tuple[float | None, str]:
    """Closed-market fallback: the last print, which must carry a timestamp.

    An undated print is unusable, not merely suspect: tau has to be measured
    from the moment the price was struck.
    """
    if row.last is None or row.last <= MIN_LAST:
        return None, "no_print"
    if row.volume < MIN_LAST_VOLUME:
        return None, "thin_print"
    if row.last_trade is None:
        return None, "undated_print"
    return row.last, ""


def _session_snapshot(rows: Sequence[RawRow]
                      ) -> tuple[date | None, datetime | None,
                                 float | None, int]:
    """(session date, snapshot time, print spread in minutes, rows dropped).

    The session is the latest NY calendar date any usable print carries — read
    off the data rather than off a holiday calendar we do not have. The
    snapshot time is the median print time within that session: a joint fit
    across expiries needs ONE pricing instant, and the median is the robust
    choice when a few strikes last traded early in the day.
    """
    stamps = [r.last_trade for r in rows
              if last_trade_mid(r)[0] is not None and r.last_trade is not None]
    if not stamps:
        return None, None, None, 0
    session = max(s.date() for s in stamps)
    same = sorted(s.timestamp() for s in stamps if s.date() == session)
    dropped = len(stamps) - len(same)
    snap = datetime.fromtimestamp(statistics.median(same), tz=timezone.utc)
    return (session, snap.astimezone(NY),
            (same[-1] - same[0]) / 60.0, dropped)


def implied_forward(calls: dict[float, tuple[float, float]],
                    puts: dict[float, tuple[float, float]],
                    *, spot: float, rate: float, tau: float) -> Forward:
    """Back the forward out of the market: F = K* + e^{r tau}(C - P).

    K* is the strike whose call/put pair is tightest (Cboe VIX methodology),
    tie-broken towards the money. Put-call parity is model-free, so this F
    carries the dividend, the borrow and the market's own funding assumption
    without any of them being specified — which is the point, since
    spot * e^{r tau} silently assumes a zero dividend yield on a ticker paying
    over 1%, and the resulting error lands entirely on the put wing (the only
    side that gets parity-mapped), i.e. on the skew, i.e. on rho.

    `calls`/`puts` map strike -> (mid, relative spread).
    """
    carry_f = spot * math.exp(rate * tau)
    best: tuple[tuple[float, float], float, float, float] | None = None
    for k in sorted(set(calls) & set(puts)):
        c_mid, c_spread = calls[k]
        p_mid, p_spread = puts[k]
        score = (c_spread + p_spread, abs(k - spot))
        if best is None or score < best[0]:
            best = (score, k, c_mid, p_mid)
    if best is None:
        return Forward(carry_f, "spot_carry_no_pair", None, 0.0, None)
    (spread_score, _), k_star, c_mid, p_mid = best
    f = k_star + math.exp(rate * tau) * (c_mid - p_mid)
    if not math.isfinite(f) or abs(f / spot - 1.0) > FWD_SANITY_BAND:
        return Forward(carry_f, "spot_carry_implausible", k_star, 0.0,
                       spread_score)
    carry = rate - math.log(f / spot) / tau
    return Forward(f, "parity", k_star, carry, spread_score)


def _ex_dividend_dates(tk: Any) -> tuple[list[date], str]:
    """Ex-dividend dates yfinance knows about, and whether it knew any.

    Returns ([], "unavailable") rather than ([], "ok") when the lookup fails,
    so "no ex-dividend date in the window" is never confused with "we could not
    check". An expiry spanning an ex-dividend date is skipped: the underlying
    is American-exercise and early exercise is optimal exactly there, so the
    European parity map is not merely biased but wrong.
    """
    found: list[date] = []
    ok = False
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            ok = True
            raw = cal.get("Ex-Dividend Date")
            for item in (raw if isinstance(raw, (list, tuple)) else [raw]):
                if isinstance(item, date) and not isinstance(item, datetime):
                    found.append(item)
                else:
                    dt = _to_ny(item)
                    if dt is not None:
                        found.append(dt.date())
    except Exception:                                    # noqa: BLE001
        pass
    if not found:
        try:
            dt = _to_ny((tk.info or {}).get("exDividendDate"))
            if dt is not None:
                found.append(dt.date())
                ok = True
        except Exception:                                # noqa: BLE001
            pass
    return sorted(set(found)), ("ok" if ok else "unavailable")


def _raw_rows(df: Any, kind: str, expiry: str) -> list[RawRow]:
    rows: list[RawRow] = []
    for row in df.itertuples():
        strike = _safe_float(getattr(row, "strike", None))
        if strike is None or strike <= 0:
            continue
        rows.append(RawRow(
            expiry=expiry, strike=strike, kind=kind,
            bid=_safe_float(getattr(row, "bid", None)),
            ask=_safe_float(getattr(row, "ask", None)),
            last=_safe_float(getattr(row, "lastPrice", None)),
            volume=_safe_float(getattr(row, "volume", None)) or 0.0,
            open_interest=_safe_float(getattr(row, "openInterest", None)) or 0.0,
            last_trade=_to_ny(getattr(row, "lastTradeDate", None)),
        ))
    return rows


def build_expiry_quotes(rows: Sequence[RawRow], *, tau: float, rate: float,
                        spot: float, expiry: str, live: bool,
                        session: date | None,
                        drops: dict[str, int]) -> tuple[list[Quote], Forward]:
    """Turn one expiry's raw rows into call-equivalent quotes.

    Mid selection runs over ALL strikes first (the forward extraction needs a
    call AND a put at the same strike, so it cannot be done after the OTM
    split), then the forward fixes the split point and the parity map.
    """
    calls: dict[float, tuple[float, float]] = {}
    puts: dict[float, tuple[float, float]] = {}
    usable: list[tuple[RawRow, float]] = []
    for row in rows:
        lo, hi = MONEYNESS_BAND
        if not lo <= row.strike / spot <= hi:
            drops["moneyness_band"] = drops.get("moneyness_band", 0) + 1
            continue
        if live:
            mid, spread, why = book_mid(row)
            if mid is None:
                drops[why] = drops.get(why, 0) + 1
                continue
        else:
            mid, why = last_trade_mid(row)
            if mid is None:
                drops[why] = drops.get(why, 0) + 1
                continue
            if session is not None and row.last_trade is not None \
                    and row.last_trade.date() != session:
                drops["stale_session"] = drops.get("stale_session", 0) + 1
                continue
            # No book to measure, so the forward pair is chosen on moneyness.
            spread = 0.0
        (calls if row.kind == "C" else puts)[row.strike] = (mid, spread or 0.0)
        usable.append((row, mid))

    fwd = implied_forward(calls, puts, spot=spot, rate=rate, tau=tau)
    fwd_pv = fwd.pv(rate, tau)
    disc = math.exp(-rate * tau)

    quotes: list[Quote] = []
    for row, mid in usable:
        # OTM only — the liquid half of each smile wing, split at the market's
        # own forward rather than at spot * e^{r tau}.
        if (row.kind == "C" and row.strike <= fwd.f) or \
                (row.kind == "P" and row.strike >= fwd.f):
            drops["itm"] = drops.get("itm", 0) + 1
            continue
        # parity-map puts onto the call surface: C = P + e^{-r tau}(F - K)
        mid_call = mid if row.kind == "C" else mid + fwd_pv - row.strike * disc
        iv = implied_vol(mid_call, fwd_pv, row.strike, tau, rate)
        if iv is None:
            drops["no_arb"] = drops.get("no_arb", 0) + 1
            continue
        quotes.append(Quote(
            tau=tau, strike=row.strike, mid_call=mid_call, iv=iv,
            vega=bs_vega(fwd_pv, row.strike, tau, iv, rate),
            kind=row.kind, expiry=expiry, fwd_pv=fwd_pv))
    return quotes, fwd


def fetch_calibration_set(ticker: str, max_dte: int,
                          min_tau_hours: float = MIN_TAU_HOURS
                          ) -> MarketSnapshot:
    """Fetch, validate and normalise the calibration surface.

    Every yfinance payload is validated the way market_data.fetch_market_params
    validates its own (try/except, isfinite, positivity) and raises
    MarketDataError rather than letting a NaN spot propagate into a JSON of
    NaNs.
    """
    import yfinance as yf

    from .market_data import _fetch_risk_free

    try:
        tk = yf.Ticker(ticker)
        spot = _safe_float(tk.fast_info["last_price"])
    except Exception as exc:                             # noqa: BLE001
        raise MarketDataError(f"failed to fetch {ticker} spot: {exc}") from exc
    if spot is None or spot <= 0:
        raise MarketDataError(f"{ticker} returned a non-positive or "
                              f"non-finite spot ({spot!r})")
    rate, rate_src = _fetch_risk_free(yf)
    now = datetime.now(tz=NY)

    try:
        expiries = list(tk.options)
    except Exception as exc:                             # noqa: BLE001
        raise MarketDataError(f"failed to fetch {ticker} expiries: "
                              f"{exc}") from exc

    exdiv, exdiv_check = _ex_dividend_dates(tk)

    # Pass 1: pull the raw chains for every candidate expiry. tau is NOT known
    # yet — it depends on the pricing time, which depends on all the rows.
    raw: dict[str, list[RawRow]] = {}
    exp_dts: dict[str, datetime] = {}
    for expiry in expiries:
        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d") \
                .replace(hour=16, minute=0, tzinfo=NY)
        except ValueError:
            continue
        if exp_dt <= now or (exp_dt.date() - now.date()).days > max_dte:
            continue
        try:
            chain = tk.option_chain(expiry)
        except Exception as exc:                         # noqa: BLE001
            raise MarketDataError(f"failed to fetch the {ticker} {expiry} "
                                  f"chain: {exc}") from exc
        rows = _raw_rows(chain.calls, "C", expiry) \
            + _raw_rows(chain.puts, "P", expiry)
        if rows:
            raw[expiry] = rows
            exp_dts[expiry] = exp_dt

    all_rows = [r for rows in raw.values() for r in rows]
    live_books = sum(1 for r in all_rows if book_mid(r)[0] is not None)
    live = live_books >= MIN_QUOTES
    session, snap_time, spread_min, dropped_sessions = \
        (None, None, None, 0) if live else _session_snapshot(all_rows)
    pricing_time = now if live else snap_time

    print(f"  spot {BOLD}${spot:,.2f}{RS}  ·  r {rate:.2%} ({rate_src})  ·  "
          f"{len(all_rows)} raw rows in {len(raw)} expiries")
    if not live:
        if pricing_time is None:
            raise MarketDataError(
                "no live book and no timestamped last trades: nothing to "
                "calibrate against")
        print(f"  {RD}market closed: {live_books} live books (< {MIN_QUOTES}); "
              f"pricing off the {session} session's last trades, "
              f"tau measured from the median print at "
              f"{pricing_time:%Y-%m-%d %H:%M %Z}{RS}")
        print(f"  {RD}this run will NOT be adopted downstream — a "
              f"market-closed fit is indicative only{RS}")
        if dropped_sessions:
            print(f"  {DIM}{dropped_sessions} print(s) from an earlier "
                  f"session dropped{RS}")
        if spread_min is not None and spread_min > MAX_SNAPSHOT_SPREAD_MIN:
            print(f"  {RD}prints span {spread_min:.0f} min — the snapshot is "
                  f"not synchronous{RS}")

    drops: dict[str, int] = {}
    if dropped_sessions:
        drops["stale_session_chainwide"] = dropped_sessions
    quotes: list[Quote] = []
    forwards: dict[str, Forward] = {}
    kept: list[str] = []
    skipped_exdiv: list[str] = []
    for expiry, rows in raw.items():
        exp_dt = exp_dts[expiry]
        tau = (exp_dt - pricing_time).total_seconds() / (365.0 * 24 * 3600)
        if tau * 365 * 24 < min_tau_hours:
            drops["below_tau_floor"] = drops.get("below_tau_floor", 0) + 1
            continue
        if any(pricing_time.date() < d <= exp_dt.date() for d in exdiv):
            skipped_exdiv.append(expiry)
            continue
        qs, fwd = build_expiry_quotes(rows, tau=tau, rate=rate, spot=spot,
                                      expiry=expiry, live=live,
                                      session=session, drops=drops)
        if not qs:
            continue
        quotes.extend(qs)
        forwards[expiry] = fwd
        kept.append(expiry)

    for expiry in kept:
        fwd = forwards[expiry]
        tau_h = (exp_dts[expiry] - pricing_time).total_seconds() / 3600
        col = "" if fwd.source == "parity" else RD
        print(f"  {DIM}{expiry}{RS}  tau {tau_h:6.2f}h  "
              f"F {col}{fwd.f:,.2f}{RS} ({fwd.source}"
              f"{f' @K={fwd.strike:.0f}' if fwd.strike else ''})  "
              f"implied carry {fwd.implied_carry:+.2%}  "
              f"{sum(q.expiry == expiry for q in quotes)} quotes")
    if skipped_exdiv:
        print(f"  {RD}skipped (ex-dividend inside the window, American "
              f"early exercise): {', '.join(skipped_exdiv)}{RS}")
    if exdiv_check != "ok":
        print(f"  {RD}ex-dividend calendar unavailable — expiries were NOT "
              f"checked for an ex-dividend date{RS}")
    if drops:
        print(f"  {DIM}rows dropped: "
              f"{', '.join(f'{k} {v}' for k, v in sorted(drops.items()))}{RS}")
    print(f"  {DIM}tau floor {min_tau_hours:.1f}h (= the surrogate's shortest "
          f"trained maturity, 1/252 yr, in ACT/365 hours); note this module "
          f"measures ACT/365 calendar time while the surrogate trains on "
          f"k/252 trading days — 365/252 = 1.45x apart at the short end{RS}")

    return MarketSnapshot(
        spot=spot, rate=rate, rate_source=rate_src, quotes=quotes,
        stale=not live, pricing_time=pricing_time, run_time=now,
        session_date=session, snapshot_spread_min=spread_min,
        forwards=forwards, expiries=kept, skipped_exdiv=skipped_exdiv,
        exdiv_check=exdiv_check, drops=drops)


# ── calibration engine ─────────────────────────────────────────────────
class Calibrator:
    """The CRN Monte Carlo objective.

    `n_paths` and `seed` are plain attributes and are meant to be reassigned:
    the search runs cheap, the polish runs at the precision the fit is reported
    at, and the noise floor is measured by re-seeding.
    """

    #: The Monte Carlo teacher runs wherever tensors are built. Every tensor
    #: below used to be created without a device, i.e. on the CPU, so a fit
    #: never touched the GPU: a 439-quote objective evaluation measured 15.82 s
    #: on CPU against roughly 0.07 s of equivalent GPU work, and one live SPY
    #: calibration took 2,788 s.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, rate: float, quotes: Sequence[Quote], n_paths: int,
                 seed: int = CRN_SEED) -> None:
        self.rate = rate
        self.n_paths = n_paths
        self.seed = seed
        self.n_evals = 0
        # group quotes by expiry so each MC call shares one maturity, and store
        # the quotes in that grouped order so model_prices() and self.quotes
        # cannot fall out of alignment
        self.groups: dict[str, list[Quote]] = {}
        for q in quotes:
            self.groups.setdefault(q.expiry, []).append(q)
        self.quotes: list[Quote] = [q for qs in self.groups.values()
                                    for q in qs]
        self.mids = np.array([q.mid_call for q in self.quotes])
        self.vegas = np.maximum(
            np.array([q.vega for q in self.quotes]), VEGA_FLOOR)

    def model_prices(self, eta: float, rho: float, H: float, xi: float,
                     n_paths: int | None = None,
                     seed: int | None = None) -> np.ndarray:
        out = []
        for qs in self.groups.values():
            b = len(qs)
            # The MC starts at the PV of the market's forward, not at spot, so
            # model and market agree on E[S_T] by construction.
            dev = self.device
            prices = rough_bergomi_mc(
                torch.full((b,), qs[0].fwd_pv, device=dev),
                torch.tensor([q.strike for q in qs], dtype=torch.float32,
                             device=dev),
                torch.full((b,), qs[0].tau, device=dev),
                torch.full((b,), xi, device=dev),
                torch.full((b,), eta, device=dev),
                torch.full((b,), rho, device=dev),
                torch.full((b,), self.rate, device=dev),
                n_paths=n_paths or self.n_paths, n_steps=50, H=H,
                seed=self.seed if seed is None else seed)
            out.append(prices.cpu().numpy())
        return np.concatenate(out)

    def loss(self, theta: np.ndarray) -> float:
        eta, rho, H, xi = map(float, theta)
        self.n_evals += 1
        model = self.model_prices(eta, rho, H, xi)
        e = (model - self.mids) / self.vegas          # ≈ IV error
        d = HUBER_DELTA
        hub = np.where(np.abs(e) <= d, 0.5 * e ** 2,
                       d * (np.abs(e) - 0.5 * d))
        return float(hub.mean()) * 1e4                # scaled for optimizer

    def objective_noise(self, theta: np.ndarray, n_reps: int = 4
                        ) -> tuple[float, float, list[float]]:
        """Re-evaluate the objective under independent path sets.

        CRN freezes the noise, it does not remove it: the frozen surface sits a
        random distance from the true one. The spread across seeds is the floor
        below which any loss difference — between two parameter vectors, or
        between two stages — is not evidence of anything.
        """
        keep = self.seed
        vals: list[float] = []
        try:
            for r in range(n_reps):
                self.seed = CRN_SEED + 7919 * (r + 1)
                vals.append(self.loss(theta))
        finally:
            self.seed = keep
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        return float(np.mean(vals)), sd, vals


def iv_fit_report(quotes: Sequence[Quote], model: np.ndarray,
                  rate: float) -> dict:
    """Per-quote implied-vol error, with nothing dropped.

    implied_vol() returns None when a model price lands outside the no-arb
    range — and those are exactly the worst-fitting quotes, so filtering them
    out understates the error precisely where the fit is failing. Such quotes
    are scored instead by the vega-linearised error (P_model - P_mid)/vega,
    the first-order continuation of the implied-vol error past the no-arb
    boundary and the same quantity the objective minimises (same VEGA_FLOOR).
    `priceable_only` reproduces the old drop-the-worst number for comparison.
    """
    errs: list[float] = []
    model_ivs: list[float | None] = []
    unpriceable: list[int] = []
    for i, (q, p) in enumerate(zip(quotes, model)):
        iv_m = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, rate)
        model_ivs.append(iv_m)
        if iv_m is None:
            unpriceable.append(i)
            errs.append((float(p) - q.mid_call)
                        / max(q.vega, VEGA_FLOOR) * 100)
        else:
            errs.append((iv_m - q.iv) * 100)
    arr = np.array(errs, dtype=float)
    keep = np.array([i not in set(unpriceable) for i in range(len(errs))])
    priceable = arr[keep] if keep.any() else np.array([])
    return {
        "errors_volpts": errs,
        "model_ivs": model_ivs,
        "unpriceable_idx": unpriceable,
        "rmse_volpts": float(np.sqrt((arr ** 2).mean())) if len(arr) else
        float("nan"),
        "rmse_volpts_priceable_only": (float(np.sqrt((priceable ** 2).mean()))
                                       if len(priceable) else float("nan")),
        "n_scored": int(len(arr)),
        "n_unpriceable": int(len(unpriceable)),
    }


def quality_gate(*, rmse: float, eta: float, rho: float, H: float, xi: float,
                 stale: bool, n_unpriceable: int) -> tuple[bool, list[str]]:
    """The one gate. Returns (accepted, reasons it was not).

    Downstream adoption — dataset regeneration, surrogate retraining,
    dataset_0dte.load_calibrated_dynamics — keys on this and nothing else.
    """
    reasons: list[str] = []
    if not math.isfinite(rmse):
        reasons.append("implied-vol RMSE is not finite")
    elif rmse >= MAX_RMSE_VOLPTS:
        reasons.append(f"implied-vol RMSE {rmse:.3f} vp >= "
                       f"{MAX_RMSE_VOLPTS:.1f} vp")
    # xi is a VARIANCE whose bound spans 5% to 120% vol, so its range in
    # variance space is 0.0025 to 1.44 — enormous and wildly non-uniform. A
    # fixed fraction of that range is meaningless at the low end: 2% of it is
    # 0.029, which would flag every fit with an ATM vol below ~17.7% as pinned,
    # including perfectly ordinary equity-index calibrations. Pinning is
    # therefore judged in VOL space for xi, which is the scale the bound was
    # actually chosen in, and in native units for the others.
    checks = (("eta", eta, BOUNDS["eta"]),
              ("rho", rho, BOUNDS["rho"]),
              ("H", H, BOUNDS["H"]),
              ("sqrt(xi)", math.sqrt(xi) if xi > 0 else float("nan"),
               (math.sqrt(BOUNDS["xi"][0]), math.sqrt(BOUNDS["xi"][1]))))
    for name, value, (lo, hi) in checks:
        tol = PIN_FRAC * (hi - lo)
        if not math.isfinite(value):
            reasons.append(f"{name} is not finite")
        elif not lo + tol < value < hi - tol:
            reasons.append(f"{name} = {value:.4f} is pinned at its bound "
                           f"[{lo:.4g}, {hi:.4g}]")
    if stale:
        reasons.append("quotes are last-session prints (market closed), not a "
                       "live book")
    if n_unpriceable:
        reasons.append(f"{n_unpriceable} quote(s) unpriceable: the model price "
                       f"is outside the no-arb range")
    return (not reasons), reasons


def calibrate(ticker: str, max_dte: int, search_paths: int, final_paths: int,
              seed: int = 7, polish_paths: int | None = None,
              noise_reps: int = 4,
              min_tau_hours: float = MIN_TAU_HOURS) -> dict:
    rule(f"ROUGH BERGOMI LIVE CALIBRATION · {ticker}")
    polish_paths = final_paths if polish_paths is None else polish_paths
    snap = fetch_calibration_set(ticker, max_dte, min_tau_hours)
    quotes = snap.quotes
    if len(quotes) < MIN_QUOTES:
        raise SystemExit(f"{RD}only {len(quotes)} clean quotes survived "
                         f"filtering — market closed or chain illiquid; "
                         f"try --max-dte 5{RS}")
    n_c = sum(q.kind == "C" for q in quotes)
    taus = sorted({round(q.tau * 365 * 24, 2) for q in quotes})
    print(f"  {BOLD}{len(quotes)}{RS} clean quotes "
          f"({n_c} OTM calls / {len(quotes) - n_c} OTM puts) at "
          f"tau {taus} hours")
    if len(snap.expiries) < 2:
        print(f"  {RD}only {len(snap.expiries)} expiry retained: H is "
              f"identified by the TERM STRUCTURE of the skew, so it is not "
              f"identified by this surface at all{RS}")

    cal = Calibrator(snap.rate, quotes, n_paths=search_paths)
    bounds = [BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]

    rule("STAGE 1 · DIFFERENTIAL EVOLUTION (global search)")
    print(f"  {DIM}{'gen':>4} {'eta':>7} {'rho':>7} {'H':>7} "
          f"{'sqrt(xi)':>9} {'loss':>10}{RS}")
    gen = [0]
    t0 = time.perf_counter()

    def cb(xk, convergence=None):
        gen[0] += 1
        print(f"  {gen[0]:>4} {CY}{xk[0]:>7.3f}{RS} {MG}{xk[1]:>7.3f}{RS} "
              f"{VI}{xk[2]:>7.3f}{RS} {math.sqrt(xk[3]):>8.1%} "
              f"{cal.loss(xk):>10.4f}", flush=True)

    de = differential_evolution(
        cal.loss, bounds, seed=seed, maxiter=12, popsize=6, tol=1e-3,
        mutation=(0.4, 0.9), recombination=0.8, polish=False, callback=cb,
        init="sobol", updating="deferred")

    rule(f"STAGE 2 · POWELL POLISH · {search_paths:,} paths")
    res = minimize(cal.loss, de.x, method="Powell", bounds=bounds,
                   options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    print(f"  {CY}eta={res.x[0]:.3f}{RS}  {MG}rho={res.x[1]:.3f}{RS}  "
          f"{VI}H={res.x[2]:.3f}{RS}  sqrt(xi)={math.sqrt(res.x[3]):.1%}  "
          f"loss {res.fun:.4f}")

    # Stage 3. Stages 1-2 optimise the CHEAP objective; H in particular is not
    # identifiable there, so the fit is re-polished at the precision it is
    # reported and gated at.
    rule(f"STAGE 3 · POWELL RE-POLISH · {polish_paths:,} paths")
    cal.n_paths = polish_paths
    t_eval = time.perf_counter()
    loss_before = cal.loss(res.x)
    t_eval = time.perf_counter() - t_eval
    print(f"  {DIM}{t_eval:.2f}s per objective evaluation at "
          f"{polish_paths:,} paths ({len(quotes)} quotes, "
          f"{len(cal.groups)} expiry group(s)){RS}")
    res3 = minimize(cal.loss, res.x, method="Powell", bounds=bounds,
                    options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    eta, rho, H, xi = map(float, res3.x)
    elapsed = time.perf_counter() - t0
    print(f"  converged: {CY}eta={eta:.3f}{RS}  {MG}rho={rho:.3f}{RS}  "
          f"{VI}H={H:.3f}{RS}  sqrt(xi)={math.sqrt(xi):.1%}  "
          f"({cal.n_evals} MC objective evals · {elapsed:.0f}s)")
    print(f"  loss {loss_before:.4f} -> {res3.fun:.4f} at {polish_paths:,} "
          f"paths")

    noise_mean, noise_sd, _ = cal.objective_noise(res3.x, n_reps=noise_reps)
    print(f"  {BOLD}objective noise floor{RS}: {noise_mean:.4f} ± "
          f"{noise_sd:.4f} (1 s.d. over {noise_reps} independent path sets at "
          f"{polish_paths:,} paths) — loss differences below this are "
          f"resampling, not fit")

    rule("FIT QUALITY · SMILE (high-precision repricing)")
    # Repriced under a path set the optimizer never saw: after stage 3 the
    # fitted point sits in a minimum of the CRN_SEED surface, so reporting on
    # that same surface would report the optimizer's own noise realisation.
    model = cal.model_prices(eta, rho, H, xi, n_paths=final_paths,
                             seed=REPORT_SEED)
    rep = iv_fit_report(cal.quotes, model, snap.rate)
    print(f"  {DIM}{'expiry':>11} {'K':>8} {'type':>4} {'mkt mid':>9} "
          f"{'model':>9} {'mkt IV':>7} {'mdl IV':>7} {'err':>8}{RS}")
    for i, (q, p) in enumerate(zip(cal.quotes, model)):
        iv_m = rep["model_ivs"][i]
        err = rep["errors_volpts"][i]
        col = GN if abs(err) < 1.5 else (RD if abs(err) > 4 else "")
        tag = f"{RD}*{RS}" if iv_m is None else " "
        print(f"  {q.expiry:>11} {q.strike:>8.0f} {q.kind:>4} "
              f"{q.mid_call:>9.2f} {p:>9.2f} {q.iv:>6.1%} "
              f"{'  n/a ' if iv_m is None else f'{iv_m:>6.1%}'} "
              f"{col}{err:>+7.2f}vp{RS}{tag}")
    rmse = rep["rmse_volpts"]
    print(f"\n  {BOLD}implied-vol RMSE across the smile: "
          f"{(GN if rmse < 2 else '')}{rmse:.3f} vol points{RS} "
          f"({rep['n_scored']}/{len(cal.quotes)} quotes scored, "
          f"{final_paths:,} paths)")
    if rep["n_unpriceable"]:
        print(f"  {RD}{rep['n_unpriceable']} quote(s) marked * are outside the "
              f"no-arb range and have no implied vol; they are scored by the "
              f"vega-linearised error instead of being dropped "
              f"(dropping them would report "
              f"{rep['rmse_volpts_priceable_only']:.3f} vp){RS}")

    accepted, reasons = quality_gate(
        rmse=rmse, eta=eta, rho=rho, H=H, xi=xi, stale=snap.stale,
        n_unpriceable=rep["n_unpriceable"])

    result = {
        "ticker": ticker,
        # as_of is the time the PRICES are from, not the time the script ran.
        "as_of": snap.pricing_time.isoformat(),
        "run_at": snap.run_time.isoformat(),
        "session_date": snap.session_date.isoformat()
        if snap.session_date else None,
        "snapshot_spread_min": (round(snap.snapshot_spread_min, 2)
                                if snap.snapshot_spread_min is not None
                                else None),
        "spot": snap.spot, "rate": snap.rate, "rate_source": snap.rate_source,
        "eta": round(eta, 4), "rho": round(rho, 4), "H": round(H, 4),
        "xi": round(xi, 6), "sqrt_xi": round(math.sqrt(xi), 4),
        "n_quotes": len(cal.quotes), "n_scored": rep["n_scored"],
        "n_unpriceable": rep["n_unpriceable"],
        "expiries": snap.expiries,
        "forwards": {e: {"F": round(f.f, 4), "source": f.source,
                         "strike": f.strike,
                         "implied_carry": round(f.implied_carry, 6)}
                     for e, f in snap.forwards.items()},
        "skipped_exdiv": snap.skipped_exdiv,
        "ex_dividend_check": snap.exdiv_check,
        "iv_rmse_volpts": round(rmse, 3),
        "iv_rmse_volpts_priceable_only":
            round(rep["rmse_volpts_priceable_only"], 3)
            if math.isfinite(rep["rmse_volpts_priceable_only"]) else None,
        "objective": "vega-weighted price Huber (CRN)",
        "objective_value": round(float(res3.fun), 6),
        "objective_mc_sd": round(noise_sd, 6),
        "objective_noise_reps": noise_reps,
        "search_paths": search_paths, "polish_paths": polish_paths,
        "final_paths": final_paths,
        "day_count": "ACT/365 calendar (surrogate trains on k/252 trading "
                     "days — see the module docstring)",
        "min_tau_hours": min_tau_hours,
        "quote_source": snap.quote_source,
        # Which simulated driver this fit is valid FOR. A calibration is only
        # meaningful for the kernel it was fitted under, and this project has
        # already shipped one that was not: the previous rough_calibration.json
        # was fitted against the Type-I fBm covariance, i.e. against a process
        # that is not rough Bergomi. dataset_0dte.load_calibrated_dynamics()
        # refuses any fit whose kernel id does not match the current driver, so
        # a stale-kernel calibration can never be silently re-adopted.
        "kernel": KERNEL_ID,
        "accepted": accepted,
        "reject_reasons": reasons,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    CAL_FILE.write_text(json.dumps(result, indent=2))
    if accepted:
        print(f"  {GN}accepted by the quality gate{RS}")
    else:
        print(f"  {RD}REJECTED by the quality gate:{RS}")
        for why in reasons:
            print(f"    {RD}·{RS} {why}")
    print(f"  {DIM}saved -> {CAL_FILE}{RS}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--max-dte", type=int, default=17,
                   help="calendar days to expiry to retain. The default used "
                        "to be 3, which after the tau floor retained a SINGLE "
                        "expiry — and H is identified by the TERM STRUCTURE of "
                        "the skew, so a one-expiry surface cannot identify it "
                        "at all. 17 days is the surrogate's own maturity "
                        "ceiling (12/252 yr in ACT/365 terms).")
    p.add_argument("--search-paths", type=int, default=8_000)
    p.add_argument("--polish-paths", type=int, default=None,
                   help="paths for the stage-3 re-polish (default: "
                        "--final-paths). H is not identifiable at the search "
                        "path count.")
    p.add_argument("--final-paths", type=int, default=64_000)
    p.add_argument("--min-tau-hours", type=float, default=MIN_TAU_HOURS,
                   help="drop quotes shorter than this; the default is the "
                        "surrogate's shortest trained maturity (1/252 yr) "
                        "expressed in ACT/365 hours")
    p.add_argument("--noise-reps", type=int, default=4,
                   help="independent path sets used to measure the "
                        "objective's own Monte Carlo noise floor")
    p.add_argument("--retrain", action="store_true",
                   help="after calibrating, regenerate the 0DTE dataset and "
                        "retrain the surrogate ensemble on the new dynamics "
                        "(refused, exit 2, if the fit fails the quality gate)")
    args = p.parse_args()

    result = calibrate(args.ticker, args.max_dte, args.search_paths,
                       args.final_paths, polish_paths=args.polish_paths,
                       noise_reps=args.noise_reps,
                       min_tau_hours=args.min_tau_hours)

    if not args.retrain:
        return

    # The gate, enforced where the adoption actually happens. Everything below
    # this line rewrites the served model's dynamics.
    if not result["accepted"]:
        rule("REGIME SYNC · REFUSED")
        print(f"  {RD}this calibration failed the quality gate; the 0DTE "
              f"dataset and the surrogate ensemble were NOT regenerated.{RS}")
        for why in result["reject_reasons"]:
            print(f"    {RD}·{RS} {why}")
        print(f"  {DIM}the fit is still on disk at {CAL_FILE} for inspection; "
              f"dataset_0dte.load_calibrated_dynamics() will keep returning "
              f"the historical defaults.{RS}")
        raise SystemExit(2)

    rule("REGIME SYNC · dataset regeneration + surrogate retrain")
    from .dataset_0dte import generate_0dte_dataset
    generate_0dte_dataset(eta=result["eta"], rho=result["rho"],
                          H=result["H"])
    import subprocess
    subprocess.run([sys.executable, "-m", "backend.quant.train_0dte",
                    "--ensemble", "5", "--epochs", "500"], check=True,
                   cwd=Path(__file__).resolve().parents[2])
    print(f"{GN}0DTE surrogate resynced to calibrated dynamics — "
          f"restart the API server to serve it.{RS}")


if __name__ == "__main__":
    main()
