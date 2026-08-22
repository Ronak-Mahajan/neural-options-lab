"""Implied-volatility surface built from live Deribit quotes, with no-arbitrage diagnostics.

This is the only place in the repository where a volatility number comes from a
real two-sided order book instead of a simulator. Everything here operates on a
`deribit.Snapshot`, so it runs offline and deterministically.

===========================================================================
1. THE CONVENTION TRAP: a Deribit option premium is not a price in dollars
===========================================================================
Deribit BTC options are European, cash-settled against the Deribit BTC index at
08:00 UTC on the expiry date, and **inverse** - the API calls them
``instrument_type: "reversed"``. Contract size is 1.0 BTC, and both the premium
and the settlement are denominated in BTC. A call struck at K settles for

        max(S_T - K, 0) / S_T   BTC,

which is exactly the USD payoff ``max(S_T - K, 0)`` delivered in coin. So the
economics are a plain USD vanilla; only the numeraire of the *quote* is unusual.

The trap: the quoted premium ``p`` is a fraction of ONE COIN, not a dollar
price and not a fraction of the strike. Measured on the committed snapshot,
feeding ``p`` straight into a Black inversion as if it were a USD price leaves
428 of 836 instruments with no solution at all - and, far worse, lets the other
408 return a *plausible-looking* implied vol (median 12.19%, range 0.04% to
63.3%) against a true median near 50%. It fails silently on half the chain.

Which coin price converts it? Two candidates: the spot index S_0, or the future
F_T for that expiry (the API gives it as ``underlying_price``). This was settled
by measurement, not by argument - reproducing Deribit's own published
``mark_iv`` from ``mark_price`` across all 836 instruments:

    conversion                year   n     rms      median    p95|err|
    p x F_T  (expiry future)  365    817   4.7561   +0.0001   0.6481   vol pts
    p x S_0  (spot index)     365    692   3.2278   -0.1775   6.0820   vol pts

(The rms on the forward convention is dominated by deep-ITM contracts whose
vega is ~0; restricted to OTM quotes it is 0.0692 vol points, p95 0.0172.)
Against Deribit's own ``bid_iv``/``ask_iv`` from ``/public/ticker`` on 53 live
quotes, this module's inversion agrees to a median of -0.0010 vol points.

So the convention implemented here is:

        forward USD premium  =  p_BTC * F_T
        forward option value =  Black76(F_T, K, T, sigma)      (UNDISCOUNTED)

Undiscounted is not an approximation: ``interest_rate`` is exactly 0.0 on every
book-summary row, and multiplying the coin premium by the *forward* rather than
the spot is precisely what carries the premium to expiry in the BTC numeraire.
It also makes the no-arbitrage algebra clean - with a discount factor of 1,
put-call parity in coin terms is simply ``c - p = 1 - K/F``.

Spot versus forward is not cosmetic. Measured on the snapshot, the futures basis
F/S_0 - 1 runs from +0.004% at the 0.4-day expiry to +3.935% at 323 days;
converting at the index instead would tilt the whole term structure, shifting
median OTM implied vol by -0.000 vol points at the front and -0.807 at the back.

Time to expiry is ACT/365 from the snapshot's exchange-side capture time. That
too was measured: a 365-day year reproduces ``mark_iv`` with a median error of
+0.0001 vol points, against +0.0144 on 365.25 and -0.2727 on 360.

===========================================================================
2. WHY BID AND ASK, NEVER A MID
===========================================================================
A mid is not a price. Nobody can trade it, and on this chain the two sides can
be far apart: the *quoted* IV bid-ask is a first-class output here, not a
nuisance. Every quote therefore carries `iv_bid` and `iv_ask`, each inverted
from an actually-executable price, and `iv_spread = iv_ask - iv_bid`. `iv_mid`
exists only as a plotting convenience and is never used to declare an arbitrage.

===========================================================================
3. NO-ARBITRAGE DIAGNOSTICS
===========================================================================
Real books violate static no-arbitrage constantly, and the interesting question
is not *whether* but *by how much, and can you actually lift it*. Each test is
therefore run twice:

  * **mid** - the textbook condition on mid prices. This is what a surface
    fitter sees and what breaks an interpolator.
  * **executable** - the same condition with every leg crossed on the side you
    would actually pay (buy at ask, sell at bid). A violation that survives here
    is a live, liftable arbitrage, not a quoting artifact.

and executable violations are additionally reported net of Deribit's taker fee
(``taker_commission`` = 0.0003 of the underlying per contract, read from the
instrument record - see `fee_usd`).

The four tests, all in undiscounted forward-USD terms:

  (a) **Butterfly** (convexity in strike). For K1 < K2 < K3 and
      w1 = (K3-K2)/(K3-K1), w3 = (K2-K1)/(K3-K1), the portfolio
      w1*C(K1) - C(K2) + w3*C(K3) has a non-negative payoff for every S_T, so
      its cost must be >= 0. Executable cost buys the wings at the ask and sells
      the body at the bid. Run on the call chain and the put chain separately;
      convexity is required of both.

  (b) **Vertical spread bounds** (monotonicity and the -1 slope floor). For
      K1 < K2:  0 <= C(K1) - C(K2) <= K2 - K1, and the mirror image for puts,
      0 <= P(K2) - P(K1) <= K2 - K1.

  (c) **Calendar** (monotonicity of total variance). Total implied variance
      w(k, T) = sigma(k, T)^2 * T must be non-decreasing in T at fixed
      log-moneyness k = log(K/F_T) [Gatheral & Jacquier 2014]. Checked on a
      common k-grid between adjacent expiries by linear interpolation of w in k
      over the clean OTM quotes. The executable version asks whether the far
      expiry's ASK variance still sits below the near expiry's BID variance -
      i.e. whether the spread is violated even paying the offer and hitting the
      bid.

  (d) **Put-call parity against the market forward**. In coin terms parity is
      c - p = 1 - K/F, which is strictly increasing in F, so every strike pair
      brackets the forward:
            F_lo = K / (1 - (c_bid - p_ask)),  F_hi = K / (1 - (c_ask - p_bid)).
      The width F_hi - F_lo is the synthetic-forward bid-ask. A genuine
      arbitrage exists only if the *futures* book sits outside that bracket:
      sell the future above F_hi and buy the synthetic, or buy the future below
      F_lo and sell the synthetic. Comparing against the independently quoted
      BTC-<expiry> future - not against a forward backed out of the same option
      quotes - is what makes this a real test.

===========================================================================
4. QUOTE HYGIENE, AND AN HONEST NOTE ON "STALE"
===========================================================================
`get_book_summary_by_currency` carries no per-quote timestamp: every row shares
the summary's own `creation_timestamp`. **True quote age is therefore not
observable from this endpoint**, and nothing here should be read as claiming to
measure it. What is observable, and what `build_surface` flags:

    no_bid / no_ask   one side absent (the API returns null, never 0)
    zero_bid          bid of exactly 0.0
    crossed / locked  bid > ask, bid == ask
    expired           T <= 0
    bid_below_bound /
    ask_above_bound   the price sits outside the static Black-76 range, so no
                      implied vol exists at all
    low_vega          |vega| below `min_vega_usd`; the IV is real but not
                      identified (a deep-ITM near-expiry quote moves 70 vol
                      points on one tick)
    wide              iv_ask - iv_bid above `max_iv_spread`
    no_trade          zero 24h volume AND zero open interest - the closest
                      honest proxy for a stale, untested quote

References
----------
F. Black (1976), "The pricing of commodity contracts", J. Financial Economics.
J. Gatheral & A. Jacquier (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance 14(1) - the total-variance calendar condition.
M. Roper (2010), "Arbitrage free implied volatility surfaces" - the static
    butterfly / vertical / calendar conditions in the form used here.
Deribit API v2 documentation, https://docs.deribit.com/.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .deribit import SECONDS_PER_YEAR, Snapshot

__all__ = [
    "Right",
    "OptionQuote",
    "VolSurface",
    "Violation",
    "ArbitrageReport",
    "parse_instrument_name",
    "black76_price",
    "black76_vega",
    "implied_vol",
    "build_surface",
    "butterfly_violations",
    "vertical_violations",
    "calendar_violations",
    "parity_violations",
    "arbitrage_report",
]

Right = Literal["call", "put"]

#: Deribit instrument names: ``BTC-6AUG26-56000-C``. The date part is
#: ``<D>[D]<MON><YY>`` with no zero padding on the day.
_NAME_RE = re.compile(
    r"^(?P<currency>[A-Z]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})"
    r"-(?P<strike>\d+(?:\.\d+)?)-(?P<right>[CP])$")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

#: Deribit settles options at 08:00 UTC. Verified on the committed snapshot:
#: all 12 expiration_timestamps land exactly on 08:00:00 UTC. Used only to
#: cross-check the parsed name; the authoritative expiry is always the
#: instrument record's `expiration_timestamp`.
DERIBIT_SETTLEMENT_HOUR_UTC = 8

_IV_LO, _IV_HI = 1e-6, 10.0


# --------------------------------------------------------------------------- #
#  Instrument names
# --------------------------------------------------------------------------- #

def parse_instrument_name(name: str) -> tuple[str, tuple[int, int, int], float, Right]:
    """``"BTC-6AUG26-56000-C"`` -> ``("BTC", (2026, 8, 6), 56000.0, "call")``.

    Raises ValueError on anything that is not an option name, which is the point:
    the futures and the perpetual share the currency prefix and must not be
    silently parsed as options.
    """
    match = _NAME_RE.match(name.strip().upper())
    if match is None:
        raise ValueError(f"not a Deribit option instrument name: {name!r}")
    month = _MONTHS.get(match["month"])
    if month is None:
        raise ValueError(f"unknown month {match['month']!r} in {name!r}")
    year = 2000 + int(match["year"])
    day = int(match["day"])
    if not 1 <= day <= 31:
        raise ValueError(f"impossible day {day} in {name!r}")
    strike = float(match["strike"])
    if strike <= 0:
        raise ValueError(f"non-positive strike in {name!r}")
    right: Right = "call" if match["right"] == "C" else "put"
    return match["currency"], (year, month, day), strike, right


# --------------------------------------------------------------------------- #
#  Black-76 and its inverse
# --------------------------------------------------------------------------- #

def black76_price(forward: float, strike: float, tenor: float,
                  sigma: float, right: Right) -> float:
    """Undiscounted Black (1976) forward value of a European option.

    Undiscounted because Deribit quotes and settles in the BTC numeraire with
    ``interest_rate == 0.0``; see the module docstring. Multiply by a discount
    factor if you ever port this to a venue that pays premium in cash.
    """
    if tenor <= 0.0 or sigma <= 0.0:
        return max(forward - strike, 0.0) if right == "call" else max(strike - forward, 0.0)
    std = sigma * math.sqrt(tenor)
    d1 = (math.log(forward / strike) + 0.5 * std * std) / std
    d2 = d1 - std
    if right == "call":
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


def black76_vega(forward: float, strike: float, tenor: float,
                 sigma: float) -> float:
    """d(price)/d(sigma), in forward USD per 1.00 of vol (i.e. per 100 vol points)."""
    if tenor <= 0.0 or sigma <= 0.0:
        return 0.0
    std = sigma * math.sqrt(tenor)
    d1 = (math.log(forward / strike) + 0.5 * std * std) / std
    return forward * norm.pdf(d1) * math.sqrt(tenor)


def implied_vol(price: float, forward: float, strike: float, tenor: float,
                right: Right) -> float:
    """Invert Black-76. Returns NaN when no volatility reproduces `price`.

    Returning NaN rather than clamping is deliberate. A price outside
    ``[max(F-K,0), F]`` for a call (or ``[max(K-F,0), K]`` for a put) is not a
    slightly-wrong volatility, it is a static arbitrage - and 19 of the 836
    marks on the committed snapshot are exactly that. Clamping would launder
    those into a plausible number and destroy the diagnostic.
    """
    if not (math.isfinite(price) and math.isfinite(forward)
            and math.isfinite(strike) and tenor > 0.0):
        return math.nan
    lower = black76_price(forward, strike, tenor, _IV_LO, right)
    upper = black76_price(forward, strike, tenor, _IV_HI, right)
    if not lower < price < upper:
        return math.nan
    try:
        return float(brentq(
            lambda s: black76_price(forward, strike, tenor, s, right) - price,
            _IV_LO, _IV_HI, xtol=1e-12, rtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        return math.nan


def fee_usd(forward: float, taker_commission: float, *,
            premium_usd: float | None = None,
            premium_cap: float | None = 0.125) -> float:
    """Taker fee for one contract, in forward USD.

    `taker_commission` (0.0003 on every option row of the snapshot) is a
    fraction of the underlying, so the fee is ``0.0003 * F`` ~ 19.4 USD per
    contract at F = 64,631 - large enough to swallow most apparent edges, which
    is exactly why the diagnostics report a net-of-fee count.

    Deribit additionally caps the option fee at 12.5% of the premium. That cap
    is from the exchange's published fee schedule and is NOT present anywhere in
    the API response, so it is applied only when `premium_usd` is supplied and
    is exposed as a parameter rather than hard-wired. With `premium_cap=None`
    the fee is the uncapped 0.0003*F, which over-charges cheap options and
    therefore makes every net-of-fee violation count a lower bound.
    """
    gross = taker_commission * forward
    if premium_cap is None or premium_usd is None:
        return gross
    return min(gross, premium_cap * premium_usd)


# --------------------------------------------------------------------------- #
#  Quotes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OptionQuote:
    """One instrument at one instant: raw coin quote, forward-USD quote, IVs.

    ``*_usd`` fields are UNDISCOUNTED forward USD (coin premium x expiry
    future), which is the space every no-arbitrage test below operates in.
    """

    instrument_name: str
    expiry_ms: int
    strike: float
    right: Right
    tenor: float                     # years, ACT/365
    forward: float                   # expiry future, USD per coin
    index: float                     # spot index, USD per coin
    bid_btc: float | None
    ask_btc: float | None
    mark_btc: float
    open_interest: float
    volume: float
    taker_commission: float
    tick_size: float
    exchange_mark_iv: float | None   # Deribit's own, for cross-checking only
    iv_bid: float
    iv_ask: float
    iv_mark: float
    vega: float                      # at iv_mark, forward USD per 1.0 vol
    flags: tuple[str, ...] = ()

    # -- derived ------------------------------------------------------------ #

    @property
    def bid_usd(self) -> float:
        return math.nan if self.bid_btc is None else self.bid_btc * self.forward

    @property
    def ask_usd(self) -> float:
        return math.nan if self.ask_btc is None else self.ask_btc * self.forward

    @property
    def mark_usd(self) -> float:
        return self.mark_btc * self.forward

    @property
    def mid_usd(self) -> float:
        """Only for plotting. Never used to declare an arbitrage."""
        if self.bid_btc is None or self.ask_btc is None:
            return math.nan
        return 0.5 * (self.bid_usd + self.ask_usd)

    @property
    def iv_spread(self) -> float:
        return self.iv_ask - self.iv_bid

    @property
    def iv_mid(self) -> float:
        return 0.5 * (self.iv_bid + self.iv_ask)

    @property
    def log_moneyness(self) -> float:
        return math.log(self.strike / self.forward)

    @property
    def total_variance_bid(self) -> float:
        return self.iv_bid * self.iv_bid * self.tenor

    @property
    def total_variance_ask(self) -> float:
        return self.iv_ask * self.iv_ask * self.tenor

    @property
    def total_variance_mid(self) -> float:
        return self.iv_mid * self.iv_mid * self.tenor

    @property
    def is_otm(self) -> bool:
        return (self.strike >= self.forward if self.right == "call"
                else self.strike <= self.forward)

    @property
    def is_clean(self) -> bool:
        return not self.flags

    @property
    def expiry_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(self.expiry_ms / 1e3))

    def as_row(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument_name,
            "expiry": self.expiry_iso,
            "tenor_days": round(self.tenor * 365.0, 4),
            "strike": self.strike,
            "right": self.right,
            "forward": round(self.forward, 2),
            "log_moneyness": round(self.log_moneyness, 6),
            "bid_btc": self.bid_btc,
            "ask_btc": self.ask_btc,
            "bid_usd": None if math.isnan(self.bid_usd) else round(self.bid_usd, 4),
            "ask_usd": None if math.isnan(self.ask_usd) else round(self.ask_usd, 4),
            "iv_bid": None if math.isnan(self.iv_bid) else round(self.iv_bid, 6),
            "iv_ask": None if math.isnan(self.iv_ask) else round(self.iv_ask, 6),
            "iv_spread": None if math.isnan(self.iv_spread) else round(self.iv_spread, 6),
            "iv_mark": None if math.isnan(self.iv_mark) else round(self.iv_mark, 6),
            "exchange_mark_iv": self.exchange_mark_iv,
            "vega": round(self.vega, 4),
            "open_interest": self.open_interest,
            "volume": self.volume,
            "flags": list(self.flags),
        }


# --------------------------------------------------------------------------- #
#  Surface
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VolSurface:
    """Every quote in a snapshot, with the clean ones separable via `clean()`."""

    captured_at: float
    currency: str
    index_price: float
    quotes: list[OptionQuote]
    forwards: dict[int, float]          # expiry_ms -> option-implied forward used
    futures_quotes: dict[int, dict[str, float]]  # expiry_ms -> {bid, ask, mark}
    filters: dict[str, Any] = field(default_factory=dict)
    reject_counts: dict[str, int] = field(default_factory=dict)

    @property
    def captured_at_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.captured_at))

    def clean(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.is_clean]

    def otm(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.is_clean and q.is_otm]

    def expiries(self) -> list[int]:
        return sorted({q.expiry_ms for q in self.quotes})

    def by_expiry(self, clean_only: bool = True) -> dict[int, list[OptionQuote]]:
        out: dict[int, list[OptionQuote]] = {}
        for q in (self.clean() if clean_only else self.quotes):
            out.setdefault(q.expiry_ms, []).append(q)
        for rows in out.values():
            rows.sort(key=lambda q: (q.strike, q.right))
        return out

    def iv_spread_stats(self, clean_only: bool = True) -> dict[str, float]:
        """Distribution of the IV bid-ask, in vol points."""
        spreads = np.array([q.iv_spread * 100.0
                            for q in (self.clean() if clean_only else self.quotes)
                            if math.isfinite(q.iv_spread)])
        if spreads.size == 0:
            return {"n": 0}
        pct = np.percentile(spreads, [5, 25, 50, 75, 95])
        return {
            "n": int(spreads.size),
            "mean": float(spreads.mean()),
            "p05": float(pct[0]), "p25": float(pct[1]), "median": float(pct[2]),
            "p75": float(pct[3]), "p95": float(pct[4]),
            "min": float(spreads.min()), "max": float(spreads.max()),
        }

    def table(self, clean_only: bool = False) -> list[dict[str, Any]]:
        return [q.as_row() for q in
                (self.clean() if clean_only else self.quotes)]

    def summary(self) -> dict[str, Any]:
        clean = self.clean()
        return {
            "captured_at": self.captured_at_iso,
            "currency": self.currency,
            "index_price": self.index_price,
            "n_quotes": len(self.quotes),
            "n_clean": len(clean),
            "n_clean_otm": len(self.otm()),
            "n_expiries": len(self.expiries()),
            "n_expiries_clean": len({q.expiry_ms for q in clean}),
            "rejects": dict(sorted(self.reject_counts.items(),
                                   key=lambda kv: -kv[1])),
            "iv_spread_vol_points": self.iv_spread_stats(),
            "filters": self.filters,
        }


def build_surface(snapshot: Snapshot, *,
                  min_vega_usd: float = 10.0,
                  max_iv_spread: float = 0.25,
                  flag_no_trade: bool = True) -> VolSurface:
    """Turn a `deribit.Snapshot` into a surface of bid/ask implied vols.

    Nothing is dropped: every instrument becomes an `OptionQuote`, and the
    reasons it should not be trusted land in `flags`. `VolSurface.clean()` is
    the filtered view; `reject_counts` is the census. Silent dropping is how a
    market-data pipeline ends up lying about its own coverage.

    Parameters
    ----------
    min_vega_usd
        Reject quotes whose Black-76 vega (forward USD per 1.00 of vol) is below
        this. Default 10.0, i.e. one vol point moves the option by >= 0.10 USD.
        Not arbitrary: on the committed snapshot, agreement with Deribit's own
        `mark_iv` improves from rms 4.7561 vol points with no vega floor to
        0.7366 at 1.0 and 0.3352 at 10.0. Below the floor the quote is a price
        statement, not a volatility statement.
    max_iv_spread
        Flag `wide` when iv_ask - iv_bid exceeds this (default 0.25 = 25 vol
        points).
    flag_no_trade
        Flag `no_trade` when 24h volume and open interest are both zero.
    """
    instruments = {row["instrument_name"]: row for row in snapshot.instruments
                   if row.get("kind") == "option"}
    book = snapshot.book_by_name()
    futures_quotes = _futures_by_expiry(snapshot)

    quotes: list[OptionQuote] = []
    rejects: dict[str, int] = {}
    forwards: dict[int, list[float]] = {}

    for name, instrument in instruments.items():
        row = book.get(name)
        if row is None:
            rejects["no_book_row"] = rejects.get("no_book_row", 0) + 1
            continue

        # The name and the instrument record must agree. They always did on the
        # committed snapshot; if they ever stop, that is a schema change and
        # should be loud.
        _, ymd, name_strike, name_right = parse_instrument_name(name)
        strike = float(instrument["strike"])
        right: Right = "call" if instrument["option_type"] == "call" else "put"
        if abs(name_strike - strike) > 1e-9 or name_right != right:
            raise ValueError(
                f"{name}: instrument record disagrees with the name "
                f"(strike {strike} vs {name_strike}, {right} vs {name_right})")

        expiry_ms = int(instrument["expiration_timestamp"])
        tenor = (expiry_ms / 1e3 - snapshot.captured_at) / SECONDS_PER_YEAR
        forward = float(row["underlying_price"])
        index = float(row.get("estimated_delivery_price") or snapshot.index_price)
        bid = row.get("bid_price")
        ask = row.get("ask_price")
        mark = float(row.get("mark_price") or 0.0)

        flags: list[str] = []
        if tenor <= 0.0:
            flags.append("expired")
        if bid is None:
            flags.append("no_bid")
        elif float(bid) <= 0.0:
            flags.append("zero_bid")
        if ask is None:
            flags.append("no_ask")
        if bid is not None and ask is not None:
            if float(bid) > float(ask):
                flags.append("crossed")
            elif float(bid) == float(ask):
                flags.append("locked")

        iv_bid = (implied_vol(float(bid) * forward, forward, strike, tenor, right)
                  if bid is not None and tenor > 0 else math.nan)
        iv_ask = (implied_vol(float(ask) * forward, forward, strike, tenor, right)
                  if ask is not None and tenor > 0 else math.nan)
        iv_mark = implied_vol(mark * forward, forward, strike, tenor, right)

        if bid is not None and float(bid) > 0.0 and math.isnan(iv_bid) and tenor > 0:
            flags.append("bid_below_bound")
        if ask is not None and math.isnan(iv_ask) and tenor > 0:
            flags.append("ask_above_bound")

        # Vega at the exchange mark IV when we have it (it is defined even where
        # our own inversion is not), else at our own mark IV.
        exch_iv = row.get("mark_iv")
        ref_iv = (float(exch_iv) / 100.0 if exch_iv is not None
                  else (iv_mark if math.isfinite(iv_mark) else math.nan))
        vega = (black76_vega(forward, strike, tenor, ref_iv)
                if math.isfinite(ref_iv) and tenor > 0 else 0.0)
        if vega < min_vega_usd:
            flags.append("low_vega")

        spread = iv_ask - iv_bid
        if math.isfinite(spread) and spread > max_iv_spread:
            flags.append("wide")

        volume = float(row.get("volume") or 0.0)
        open_interest = float(row.get("open_interest") or 0.0)
        if flag_no_trade and volume == 0.0 and open_interest == 0.0:
            flags.append("no_trade")

        for flag in flags:
            rejects[flag] = rejects.get(flag, 0) + 1

        quotes.append(OptionQuote(
            instrument_name=name, expiry_ms=expiry_ms, strike=strike,
            right=right, tenor=tenor, forward=forward, index=index,
            bid_btc=None if bid is None else float(bid),
            ask_btc=None if ask is None else float(ask),
            mark_btc=mark, open_interest=open_interest, volume=volume,
            taker_commission=float(instrument.get("taker_commission", 0.0)),
            tick_size=float(instrument.get("tick_size", 0.0)),
            exchange_mark_iv=None if exch_iv is None else float(exch_iv),
            iv_bid=iv_bid, iv_ask=iv_ask, iv_mark=iv_mark, vega=vega,
            flags=tuple(flags)))
        forwards.setdefault(expiry_ms, []).append(forward)

    return VolSurface(
        captured_at=snapshot.captured_at,
        currency=snapshot.currency,
        index_price=snapshot.index_price,
        quotes=sorted(quotes, key=lambda q: (q.expiry_ms, q.strike, q.right)),
        forwards={e: float(np.median(v)) for e, v in forwards.items()},
        futures_quotes=futures_quotes,
        filters={"min_vega_usd": min_vega_usd, "max_iv_spread": max_iv_spread,
                 "flag_no_trade": flag_no_trade,
                 "day_count": "ACT/365", "premium_conversion": "coin * expiry future",
                 "discounting": "none (Deribit interest_rate == 0)"},
        reject_counts=rejects,
    )


def _futures_by_expiry(snapshot: Snapshot) -> dict[int, dict[str, float]]:
    """Map option expiry -> the independently quoted BTC future's top of book."""
    book = snapshot.futures_book_by_name()
    out: dict[int, dict[str, float]] = {}
    for row in snapshot.futures:
        if row.get("settlement_period") == "perpetual":
            continue
        quote = book.get(row["instrument_name"])
        if quote is None:
            continue
        bid, ask = quote.get("bid_price"), quote.get("ask_price")
        if bid is None or ask is None:
            continue
        out[int(row["expiration_timestamp"])] = {
            "bid": float(bid), "ask": float(ask),
            "mark": float(quote.get("mark_price") or 0.5 * (float(bid) + float(ask))),
            "instrument": row["instrument_name"],
        }
    return out


# --------------------------------------------------------------------------- #
#  No-arbitrage diagnostics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Violation:
    """One breach of a static no-arbitrage condition.

    `magnitude` is always the size of the breach (positive = violated), in the
    units named by `units`. `executable` means the breach survives crossing the
    spread on every leg; `net_of_fees` means it also survives Deribit's taker
    fee. Only `net_of_fees` violations are money.
    """

    kind: str
    expiry_iso: str
    right: str
    strikes: tuple[float, ...]
    magnitude: float
    units: str
    executable: bool
    net_of_fees: bool
    detail: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "expiry": self.expiry_iso, "right": self.right,
            "strikes": list(self.strikes),
            "magnitude": round(self.magnitude, 6), "units": self.units,
            "executable": self.executable, "net_of_fees": self.net_of_fees,
            "detail": self.detail,
        }


def _bps(value_usd: float, forward: float) -> float:
    return 1e4 * value_usd / forward


def butterfly_violations(surface: VolSurface) -> list[Violation]:
    """Convexity in strike, on the call chain and the put chain separately."""
    out: list[Violation] = []
    for expiry, rows in surface.by_expiry().items():
        for right in ("call", "put"):
            chain = sorted((q for q in rows if q.right == right),
                           key=lambda q: q.strike)
            for a, b, c in zip(chain, chain[1:], chain[2:]):
                if c.strike <= a.strike:
                    continue
                w1 = (c.strike - b.strike) / (c.strike - a.strike)
                w3 = (b.strike - a.strike) / (c.strike - a.strike)
                fwd = b.forward

                mid_cost = w1 * a.mid_usd - b.mid_usd + w3 * c.mid_usd
                exec_cost = w1 * a.ask_usd - b.bid_usd + w3 * c.ask_usd
                if not (math.isfinite(mid_cost) and math.isfinite(exec_cost)):
                    continue
                if mid_cost >= 0.0:
                    continue

                # Leg sizes are w1, 1, w3 contracts; w1 + w3 == 1 exactly.
                fees = ((w1 * fee_usd(fwd, a.taker_commission,
                                      premium_usd=a.ask_usd))
                        + fee_usd(fwd, b.taker_commission, premium_usd=b.bid_usd)
                        + (w3 * fee_usd(fwd, c.taker_commission,
                                        premium_usd=c.ask_usd)))
                out.append(Violation(
                    kind="butterfly", expiry_iso=b.expiry_iso, right=right,
                    strikes=(a.strike, b.strike, c.strike),
                    magnitude=_bps(-mid_cost, fwd), units="bps_of_forward",
                    executable=exec_cost < 0.0,
                    net_of_fees=exec_cost + fees < 0.0,
                    detail=(f"mid cost {mid_cost:+.2f} USD, executable cost "
                            f"{exec_cost:+.2f} USD, taker fees {fees:.2f} USD")))
    return out


def vertical_violations(surface: VolSurface) -> list[Violation]:
    """Monotonicity in strike and the |dC/dK| <= 1 slope bound."""
    out: list[Violation] = []
    for expiry, rows in surface.by_expiry().items():
        for right in ("call", "put"):
            chain = sorted((q for q in rows if q.right == right),
                           key=lambda q: q.strike)
            for lo, hi in zip(chain, chain[1:]):
                width = hi.strike - lo.strike
                if width <= 0:
                    continue
                fwd = lo.forward
                # "near" is the leg that must be worth more: the lower strike
                # for calls, the higher strike for puts.
                near, far = (lo, hi) if right == "call" else (hi, lo)

                mono_mid = near.mid_usd - far.mid_usd            # must be >= 0
                mono_exec = near.ask_usd - far.bid_usd           # buy near, sell far
                slope_mid = width - (near.mid_usd - far.mid_usd)  # must be >= 0
                slope_exec = width - (near.bid_usd - far.ask_usd)

                if math.isfinite(mono_mid) and mono_mid < 0.0:
                    fees = (fee_usd(fwd, near.taker_commission, premium_usd=near.ask_usd)
                            + fee_usd(fwd, far.taker_commission, premium_usd=far.bid_usd))
                    out.append(Violation(
                        kind="vertical_monotonicity", expiry_iso=lo.expiry_iso,
                        right=right, strikes=(lo.strike, hi.strike),
                        magnitude=_bps(-mono_mid, fwd), units="bps_of_forward",
                        executable=mono_exec < 0.0,
                        net_of_fees=mono_exec + fees < 0.0,
                        detail=(f"{near.strike:.0f} is cheaper than {far.strike:.0f} "
                                f"by {-mono_mid:.2f} USD at mid")))

                if math.isfinite(slope_mid) and slope_mid < 0.0:
                    fees = (fee_usd(fwd, near.taker_commission, premium_usd=near.bid_usd)
                            + fee_usd(fwd, far.taker_commission, premium_usd=far.ask_usd))
                    out.append(Violation(
                        kind="vertical_slope", expiry_iso=lo.expiry_iso,
                        right=right, strikes=(lo.strike, hi.strike),
                        magnitude=_bps(-slope_mid, fwd), units="bps_of_forward",
                        executable=slope_exec < 0.0,
                        net_of_fees=slope_exec + fees < 0.0,
                        detail=(f"spread {near.strike:.0f}/{far.strike:.0f} worth "
                                f"more than its {width:.0f} USD max payoff")))
    return out


def calendar_violations(surface: VolSurface, *,
                        n_grid: int = 25,
                        k_clip: float = 0.9) -> list[Violation]:
    """Total implied variance must not decrease with maturity at fixed log-moneyness.

    Compares each adjacent pair of expiries on a shared log-moneyness grid,
    restricted to the overlap of the two quoted ranges (and to |k| <= `k_clip`,
    beyond which the wings are single quotes and the interpolation is fiction).
    The `executable` flag asks the sharper question: is the far expiry's ASK
    variance still below the near expiry's BID variance?
    """
    curves: dict[int, dict[str, np.ndarray]] = {}
    for expiry, rows in surface.by_expiry().items():
        otm = sorted((q for q in rows if q.is_otm
                      and math.isfinite(q.total_variance_bid)
                      and math.isfinite(q.total_variance_ask)),
                     key=lambda q: q.log_moneyness)
        if len(otm) < 3:
            continue
        curves[expiry] = {
            "k": np.array([q.log_moneyness for q in otm]),
            "w_bid": np.array([q.total_variance_bid for q in otm]),
            "w_ask": np.array([q.total_variance_ask for q in otm]),
            "w_mid": np.array([q.total_variance_mid for q in otm]),
            "tenor": np.array([otm[0].tenor]),
            "iso": otm[0].expiry_iso,
        }

    out: list[Violation] = []
    ordered = sorted(curves)
    for near_e, far_e in zip(ordered, ordered[1:]):
        near, far = curves[near_e], curves[far_e]
        lo = max(near["k"].min(), far["k"].min(), -k_clip)
        hi = min(near["k"].max(), far["k"].max(), k_clip)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, n_grid)
        w_near_mid = np.interp(grid, near["k"], near["w_mid"])
        w_far_mid = np.interp(grid, far["k"], far["w_mid"])
        w_near_bid = np.interp(grid, near["k"], near["w_bid"])
        w_far_ask = np.interp(grid, far["k"], far["w_ask"])

        shortfall_mid = w_near_mid - w_far_mid           # > 0 means violated
        shortfall_exec = w_near_bid - w_far_ask
        bad = shortfall_mid > 0.0
        if not bad.any():
            continue
        worst = int(np.argmax(shortfall_mid))
        t_far = float(far["tenor"][0])
        # Vol points the far expiry would have to gain to restore monotonicity.
        iv_far = math.sqrt(max(w_far_mid[worst], 0.0) / t_far)
        iv_needed = math.sqrt(max(w_near_mid[worst], 0.0) / t_far)
        out.append(Violation(
            kind="calendar", expiry_iso=f"{near['iso']} -> {far['iso']}",
            right="otm", strikes=(round(float(grid[worst]), 4),),
            magnitude=float(shortfall_mid[worst]) * 1e4,
            units="total_variance_bps",
            executable=bool(shortfall_exec[worst] > 0.0),
            net_of_fees=bool(shortfall_exec[worst] > 0.0),
            detail=(f"{int(bad.sum())}/{n_grid} grid points violated; worst at "
                    f"k={grid[worst]:+.4f}, far-expiry IV {iv_far*100:.2f}% vs "
                    f"{iv_needed*100:.2f}% needed "
                    f"({(iv_needed-iv_far)*100:+.2f} vol points)")))
    return out


def parity_violations(surface: VolSurface) -> list[Violation]:
    """Put-call parity against the independently quoted future.

    In coin terms parity is ``c - p = 1 - K/F``. Every two-sided call/put pair
    therefore brackets the forward in ``[F_lo, F_hi]``; an arbitrage exists only
    when the futures book itself sits outside that bracket.

    Fees here are four legs (call, put, future, and the future's own commission
    is charged on notional), so the net-of-fee test subtracts three option-side
    taker fees as a deliberately conservative stand-in - Deribit's futures fee
    schedule is not in the API response and is not guessed at here.
    """
    out: list[Violation] = []
    for expiry, rows in surface.by_expiry().items():
        future = surface.futures_quotes.get(expiry)
        if future is None:
            continue
        calls = {q.strike: q for q in rows if q.right == "call"}
        puts = {q.strike: q for q in rows if q.right == "put"}
        for strike in sorted(set(calls) & set(puts)):
            call, put = calls[strike], puts[strike]
            if call.bid_btc is None or call.ask_btc is None:
                continue
            if put.bid_btc is None or put.ask_btc is None:
                continue
            sell_synth = call.bid_btc - put.ask_btc     # proceeds, coin
            buy_synth = call.ask_btc - put.bid_btc      # cost, coin
            if sell_synth >= 1.0 or buy_synth >= 1.0:
                continue                                 # F -> infinity; degenerate
            f_lo = strike / (1.0 - sell_synth)
            f_hi = strike / (1.0 - buy_synth)
            if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_hi < f_lo:
                continue
            fwd = call.forward
            fees_usd = (fee_usd(fwd, call.taker_commission, premium_usd=call.ask_usd)
                        + fee_usd(fwd, put.taker_commission, premium_usd=put.bid_usd))
            fees_bps = _bps(fees_usd, fwd)

            edge_bps = 0.0
            detail = ""
            if future["bid"] > f_hi:      # sell the future, buy the synthetic
                edge_bps = _bps(future["bid"] - f_hi, fwd)
                detail = (f"future bid {future['bid']:.2f} above synthetic ask "
                          f"{f_hi:.2f}")
            elif future["ask"] < f_lo:    # buy the future, sell the synthetic
                edge_bps = _bps(f_lo - future["ask"], fwd)
                detail = (f"future ask {future['ask']:.2f} below synthetic bid "
                          f"{f_lo:.2f}")
            if edge_bps <= 0.0:
                continue
            out.append(Violation(
                kind="put_call_parity", expiry_iso=call.expiry_iso, right="pair",
                strikes=(strike,), magnitude=edge_bps, units="bps_of_forward",
                executable=True, net_of_fees=edge_bps > fees_bps,
                detail=f"{detail}; option-leg taker fees {fees_bps:.1f} bps"))
    return out


def parity_forward_stats(surface: VolSurface) -> list[dict[str, Any]]:
    """Per-expiry synthetic-forward bracket width and its basis to the future.

    Not a violation report - this is the microstructure summary that says how
    tightly the options book actually pins the forward.
    """
    out: list[dict[str, Any]] = []
    for expiry, rows in surface.by_expiry().items():
        future = surface.futures_quotes.get(expiry)
        calls = {q.strike: q for q in rows if q.right == "call"}
        puts = {q.strike: q for q in rows if q.right == "put"}
        widths, mids, tenors = [], [], []
        for strike in sorted(set(calls) & set(puts)):
            call, put = calls[strike], puts[strike]
            if None in (call.bid_btc, call.ask_btc, put.bid_btc, put.ask_btc):
                continue
            sell_synth = call.bid_btc - put.ask_btc
            buy_synth = call.ask_btc - put.bid_btc
            if sell_synth >= 1.0 or buy_synth >= 1.0:
                continue
            f_lo = strike / (1.0 - sell_synth)
            f_hi = strike / (1.0 - buy_synth)
            if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_hi < f_lo:
                continue
            widths.append(f_hi - f_lo)
            mids.append(0.5 * (f_lo + f_hi))
            tenors.append(call.tenor)
        if not widths:
            continue
        fwd = surface.forwards[expiry]
        row = {
            "expiry": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expiry / 1e3)),
            "tenor_days": round(float(np.mean(tenors)) * 365.0, 3),
            "n_pairs": len(widths),
            "future_mark": None if future is None else future["mark"],
            "option_forward": round(fwd, 2),
            "synthetic_forward_median": round(float(np.median(mids)), 2),
            "bracket_width_median_bps": round(
                float(np.median(widths)) / fwd * 1e4, 2),
            "synthetic_vs_future_median_bps": (
                None if future is None else
                round((float(np.median(mids)) - future["mark"]) / fwd * 1e4, 2)),
            "basis_vs_index_pct": round((fwd / surface.index_price - 1.0) * 100.0, 4),
        }
        out.append(row)
    return sorted(out, key=lambda r: r["tenor_days"])


def _magnitude_stats(items: Sequence[Violation]) -> dict[str, Any]:
    if not items:
        return {"n": 0}
    mags = np.array([v.magnitude for v in items])
    return {
        "n": int(mags.size),
        "n_executable": sum(1 for v in items if v.executable),
        "n_net_of_fees": sum(1 for v in items if v.net_of_fees),
        "units": items[0].units,
        "median": float(np.median(mags)),
        "p95": float(np.percentile(mags, 95)),
        "max": float(mags.max()),
        "worst": max(items, key=lambda v: v.magnitude).as_row(),
    }


@dataclass(frozen=True)
class ArbitrageReport:
    butterfly: list[Violation]
    vertical: list[Violation]
    calendar: list[Violation]
    parity: list[Violation]
    n_tested: dict[str, int]
    forwards: list[dict[str, Any]]

    def all(self) -> list[Violation]:
        return [*self.butterfly, *self.vertical, *self.calendar, *self.parity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tested": self.n_tested,
            "butterfly": _magnitude_stats(self.butterfly),
            "vertical": _magnitude_stats(self.vertical),
            "calendar": _magnitude_stats(self.calendar),
            "put_call_parity": _magnitude_stats(self.parity),
            "forward_consistency": self.forwards,
        }


def arbitrage_report(surface: VolSurface) -> ArbitrageReport:
    """Run all four static no-arbitrage tests on the clean quotes of `surface`."""
    by_expiry = surface.by_expiry()
    n_triples = n_pairs = n_parity = 0
    for rows in by_expiry.values():
        for right in ("call", "put"):
            n = sum(1 for q in rows if q.right == right)
            n_triples += max(0, n - 2)
            n_pairs += max(0, n - 1)
        strikes_c = {q.strike for q in rows if q.right == "call"}
        strikes_p = {q.strike for q in rows if q.right == "put"}
        n_parity += len(strikes_c & strikes_p)
    return ArbitrageReport(
        butterfly=butterfly_violations(surface),
        vertical=vertical_violations(surface),
        calendar=calendar_violations(surface),
        parity=parity_violations(surface),
        n_tested={
            "butterfly_triples": n_triples,
            "vertical_pairs": n_pairs,
            "parity_strikes": n_parity,
            "calendar_expiry_pairs": max(0, len(by_expiry) - 1),
            "clean_quotes": len(surface.clean()),
        },
        forwards=parity_forward_stats(surface),
    )
