"""Record option surfaces on a schedule, building the dataset money can't buy.

Historical option-surface data is the expensive ingredient in every options research question this project can ask next: out-of-sample calibration tests, hedging on real paths, fill-model validation. Vendors charge thousands for it.
The free alternative is to record it FORWARD: capture the live surfaces this
project already knows how to fetch, every few minutes, and let time compound.

Two sources, both free:

    Deribit (BTC, ETH)   real-time, 24/7, public API - the primary asset
    yfinance (SPY, ...)  delayed ~15 min - fine for surface research, useless
                         for execution; captures are tagged stale when the
                         market is closed so nothing downstream mistakes
                         last-session prints for a live book

Files land under data/surfaces/ (gitignored - this grows without bound and
does not belong in history), gzipped JSON, named to sort chronologically:

    data/surfaces/deribit/btc_20260820T171500Z.json.gz
    data/surfaces/equity/spy_20260820T171500Z.json.gz

Each equity capture stores the CLEANED quotes (the same filtering the
calibrator applies: two-sided, liquid, OTM, moneyness-banded) plus the
snapshot metadata needed to re-run a fit against it later: spot, rate,
per-expiry forwards, staleness, drop counts. Deribit captures reuse
deribit.Snapshot.to_dict() verbatim - raw book summaries, instruments,
futures - because the screener and fill-model work downstream want raw books,
not pre-filtered quotes.

Run one capture (cron-friendly, exits nonzero if every source failed):

    python -m scripts.record_surfaces --once

Run the loop (survives individual failures; Ctrl-C to stop):

    python -m scripts.record_surfaces --interval-min 15

Persist across reboots with Windows Task Scheduler (run from the repo root):

    schtasks /Create /TN "record_surfaces" /SC MINUTE /MO 15 ^
      /TR "\"C:\\...\\python.exe\" -m scripts.record_surfaces --once" ^
      /ST 00:00

Reading it back:

    from scripts.record_surfaces import iter_captures
    for path, payload in iter_captures("equity"):  # or "deribit"
        ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "surfaces"


def _stamp(t: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))


def _write_gz(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)          # atomic-ish: never leave a truncated capture
    return path


def capture_deribit(currency: str = "BTC") -> Path:
    """One raw Deribit chain capture, reusing the project's own client."""
    from backend.quant.deribit import DeribitClient
    snap = DeribitClient().snapshot(currency)
    out = DATA_DIR / "deribit" / f"{currency.lower()}_{_stamp(snap.captured_at)}.json.gz"
    return _write_gz(out, snap.to_dict())


def capture_equity(ticker: str = "SPY", max_dte: int = 17) -> Path:
    """One cleaned equity surface capture via the calibrator's own fetch."""
    from backend.quant.calibrate import MIN_TAU_HOURS, fetch_calibration_set
    snap = fetch_calibration_set(ticker, max_dte, min_tau_hours=MIN_TAU_HOURS)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "ticker": ticker,
        "max_dte": max_dte,
        "captured_at_iso": _stamp(),
        "pricing_time": snap.pricing_time.isoformat(),
        "spot": snap.spot,
        "rate": snap.rate,
        "rate_source": snap.rate_source,
        "stale": snap.stale,
        "quote_source": snap.quote_source,
        "session_date": (snap.session_date.isoformat()
                         if snap.session_date else None),
        "expiries": snap.expiries,
        "forwards": {k: asdict(v) for k, v in snap.forwards.items()},
        "drops": snap.drops,
        "n_quotes": len(snap.quotes),
        "quotes": [asdict(q) for q in snap.quotes],
    }
    out = DATA_DIR / "equity" / f"{ticker.lower()}_{_stamp()}.json.gz"
    return _write_gz(out, payload)


def iter_captures(kind: str, prefix: str | None = None
                  ) -> Iterator[tuple[Path, dict]]:
    """Yield (path, payload) for stored captures, oldest first."""
    base = DATA_DIR / kind
    if not base.exists():
        return
    for path in sorted(base.glob(f"{prefix or ''}*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield path, json.load(fh)


def run_once(currencies: list[str], tickers: list[str]) -> int:
    """Capture every source once. Returns how many sources succeeded.

    A source failing must not stop the others: yfinance rate limits and
    Deribit maintenance windows are routine, and a recorder that dies on the
    first hiccup records nothing. Failures are printed with tracebacks and
    the loop moves on.
    """
    ok = 0
    for cur in currencies:
        try:
            path = capture_deribit(cur)
            print(f"  deribit {cur}: {path.name} "
                  f"({path.stat().st_size / 1024:.0f} KB)", flush=True)
            ok += 1
        except Exception:
            print(f"  deribit {cur}: FAILED", flush=True)
            traceback.print_exc()
    for tk in tickers:
        try:
            path = capture_equity(tk)
            print(f"  equity {tk}: {path.name} "
                  f"({path.stat().st_size / 1024:.0f} KB)", flush=True)
            ok += 1
        except Exception:
            print(f"  equity {tk}: FAILED", flush=True)
            traceback.print_exc()
    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true",
                   help="capture each source once and exit (for cron / Task "
                        "Scheduler); exits 1 only if EVERY source failed")
    p.add_argument("--interval-min", type=float, default=15.0,
                   help="loop interval in minutes (default 15; Deribit "
                        "captures are ~200 KB gzipped, equity ~50 KB, so a "
                        "day of 15-minute captures is roughly 25 MB)")
    p.add_argument("--currencies", default="BTC,ETH",
                   help="Deribit currencies, comma-separated ('' to disable)")
    p.add_argument("--tickers", default="SPY",
                   help="yfinance tickers, comma-separated ('' to disable)")
    args = p.parse_args()

    currencies = [c.strip().upper() for c in args.currencies.split(",")
                  if c.strip()]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.once:
        ok = run_once(currencies, tickers)
        sys.exit(0 if ok else 1)

    print(f"recording {currencies + tickers} every {args.interval_min:g} min "
          f"-> {DATA_DIR}  (Ctrl-C to stop)", flush=True)
    while True:
        t0 = time.monotonic()
        print(f"[{_stamp()}]", flush=True)
        run_once(currencies, tickers)
        # Sleep the REMAINDER of the interval so capture latency does not
        # stretch the cadence; a capture longer than the interval runs
        # back-to-back rather than piling up.
        time.sleep(max(0.0, args.interval_min * 60 - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
