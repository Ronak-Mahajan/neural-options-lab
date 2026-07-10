"""Real-world market data adapter (yfinance).

Fetches, for a given equity ticker:
    spot   — last traded price
    sigma  — 1-year realized volatility (std of daily log returns x sqrt(252))
    rate   — risk-free proxy: 13-week T-bill yield (^IRX), falling back to
             the 10-year Treasury (^TNX); both are quoted in percent on Yahoo

Values are clamped into the surrogate's trained domain and the response says
so explicitly (`clamped`) — a model should never silently extrapolate.
Results are cached for 5 minutes to be polite to Yahoo and keep the UI snappy.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .dataset import PARAM_RANGES

_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_SECONDS = 300.0


class MarketDataError(RuntimeError):
    pass


def _history_close(ticker_obj, period: str):
    hist = ticker_obj.history(period=period, auto_adjust=True)
    if hist is None or hist.empty:
        raise MarketDataError("no price history returned")
    return hist["Close"]


def _fetch_risk_free(yf) -> tuple[float, str]:
    for symbol in ("^IRX", "^TNX"):
        try:
            close = _history_close(yf.Ticker(symbol), "5d")
            quote = float(close.iloc[-1])
            if math.isfinite(quote) and quote > 0:
                return quote / 100.0, symbol
        except Exception:
            continue
    raise MarketDataError("could not fetch a Treasury yield (^IRX/^TNX)")


def fetch_market_params(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10 or not all(
            ch.isalnum() or ch in ".-^" for ch in ticker):
        raise MarketDataError(f"invalid ticker {ticker!r}")

    now = time.time()
    cached = _CACHE.get(ticker)
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MarketDataError("yfinance is not installed") from exc

    try:
        tk = yf.Ticker(ticker)
        closes = _history_close(tk, "1y")
        spot = float(closes.iloc[-1])
        try:  # prefer the live quote when available
            live = float(tk.fast_info["last_price"])
            if math.isfinite(live) and live > 0:
                spot = live
        except Exception:
            pass
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"failed to fetch {ticker}: {exc}") from exc

    log_ret = np.diff(np.log(closes.to_numpy(dtype=float)))
    log_ret = log_ret[np.isfinite(log_ret)]
    if len(log_ret) < 60:
        raise MarketDataError(f"not enough history for {ticker} "
                              f"({len(log_ret)} daily returns)")
    sigma_raw = float(np.std(log_ret, ddof=1) * math.sqrt(252.0))

    rate_raw, rate_source = _fetch_risk_free(yf)

    sig_lo, sig_hi = PARAM_RANGES["sigma"]
    r_lo, r_hi = PARAM_RANGES["rate"]
    sigma = min(max(sigma_raw, sig_lo), sig_hi)
    rate = min(max(rate_raw, r_lo), r_hi)

    result = {
        "ticker": ticker,
        "spot": round(spot, 4),
        "sigma": round(sigma, 6),
        "sigma_raw": round(sigma_raw, 6),
        "rate": round(rate, 6),
        "rate_raw": round(rate_raw, 6),
        "rate_source": rate_source,
        "n_return_days": int(len(log_ret)),
        "clamped": bool(sigma != sigma_raw or rate != rate_raw),
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _CACHE[ticker] = (now, result)
    return result
