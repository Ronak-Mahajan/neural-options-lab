"""Avellaneda-Stoikov market making on Deribit testnet, for one instrument.

The same model the hft-market-maker project validated in simulation (500
paired seeds: inventory risk -17%, max drawdown -14%, both significant, paid
for with 36% fewer fills), now pointed at a real matching engine with paper
money. The interesting output is not P&L -- testnet P&L is play money -- it is
the FILL MODEL: the simulator assumed fills arrive as lambda(delta) =
A*exp(-kappa*delta); every live fill this loop takes is a data point on the
real curve, logged to data/trading/fills_*.jsonl for exactly that comparison.

Model (Avellaneda & Stoikov 2008), per tick:

    reservation r = m - q * gamma * sigma^2 * (T - t)
    half-spread d = gamma * sigma^2 * (T - t) / 2 + (1/gamma) * ln(1 + gamma/kappa)
    quote bid at r - d, ask at r + d, post-only, one order per side

with m the venue mark, q signed inventory, sigma the realized vol of the mark
over a ring buffer, and T the session horizon: aversion to inventory decays to
zero at the horizon because there is no time left for the position to move.

Two modes:

    --dry-run      no credentials needed: reads the live public ticker,
                   computes and prints the quotes it WOULD place. Proves the
                   whole loop except order placement.
    (default)      places real post-only orders on the TEST exchange; needs
                   DERIBIT_TESTNET_KEY / SECRET (see testnet.py). Risk limits
                   and the kill switch from risk.py wrap every action.

    python -m backend.trading.quoter --instrument BTC-26AUG26-73000-C --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

from .risk import KillSwitch, OrderBlocked, RiskLimits, RiskManager
from .testnet import TestnetClient

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "trading"


class VolEstimator:
    """Realized vol of the mark from a ring buffer of (t, price)."""

    def __init__(self, maxlen: int = 120) -> None:
        self.buf: collections.deque[tuple[float, float]] = \
            collections.deque(maxlen=maxlen)

    def update(self, t: float, price: float) -> None:
        if price > 0:
            self.buf.append((t, price))

    def annualized(self, floor: float = 0.20) -> float:
        if len(self.buf) < 10:
            return floor
        rets, spans = [], []
        items = list(self.buf)
        for (t0, p0), (t1, p1) in zip(items, items[1:]):
            if t1 > t0 and p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
                spans.append(t1 - t0)
        if not rets or sum(spans) <= 0:
            return floor
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        per_sec = var / (sum(spans) / len(spans))
        year = 365.25 * 24 * 3600
        return max(math.sqrt(per_sec * year), floor)


def as_quotes(mark: float, inventory: float, sigma: float, tau: float,
              gamma: float, kappa: float) -> tuple[float, float]:
    """The two Avellaneda-Stoikov lines, in RELATIVE (per-mark) units.

    The textbook formulas are written in absolute price units, which makes
    gamma and kappa scale-dependent: parameters tuned for a $100 asset produce
    a 0.45 half-spread on a 0.006 BTC option — 77x the price. (Found live: the
    first dry run against the real ticker quoted 0.0001/0.4521 around a 0.0058
    mark.) Working on the price normalized by the mark makes both parameters
    dimensionless: kappa is the fill-intensity decay per unit RELATIVE spread,
    so 1/kappa ~ the relative half-spread a fill model tolerates, and gamma is
    aversion per unit relative return. tau in years remaining of session.
    """
    lean = inventory * gamma * sigma ** 2 * tau
    half = gamma * sigma ** 2 * tau / 2.0 + math.log(1 + gamma / kappa) / gamma
    return mark * (1.0 - lean - half), mark * (1.0 - lean + half)


def round_to_tick(price: float, tick: float, side: str) -> float:
    """Round INWARD (bid down, ask up) so post-only never crosses by rounding."""
    steps = price / tick
    n = math.floor(steps) if side == "buy" else math.ceil(steps)
    return max(round(n * tick, 10), tick)


def run(instrument: str, *, gamma: float, kappa: float, size: float,
        session_min: float, interval: float, dry_run: bool,
        limits: RiskLimits) -> None:
    client = TestnetClient()
    meta = client.instrument(instrument)
    tick = float(meta["tick_size"])
    min_amount = float(meta["min_trade_amount"])
    size = max(size, min_amount)
    print(f"{instrument}: tick {tick}, min amount {min_amount}, "
          f"quoting size {size}", flush=True)

    oms = None
    if not dry_run:
        from .oms import OMS
        oms = OMS(client, instrument)
    risk = RiskManager(limits)
    vol = VolEstimator()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fills_log = DATA_DIR / f"fills_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"

    t_end = time.time() + session_min * 60.0
    try:
        while time.time() < t_end:
            tk = client.ticker(instrument)
            mark = float(tk["mark_price"])          # in BTC for BTC options
            best_bid = tk.get("best_bid_price") or 0.0
            best_ask = tk.get("best_ask_price") or 0.0
            vol.update(time.time(), mark)
            sigma = vol.annualized()
            tau = max(t_end - time.time(), 0.0) / (365.25 * 24 * 3600)

            inventory = 0.0
            if oms is not None:
                for fill in oms.sync():
                    fill["t"] = time.time()
                    fill["mark"] = mark
                    fill["best_bid"], fill["best_ask"] = best_bid, best_ask
                    with fills_log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(fill) + "\n")
                    print(f"  FILL {fill['side']} {fill['amount']} @ "
                          f"{fill['price']} (mark {mark:.4f})", flush=True)
                inventory = oms.position_amount
                summary = client.account_summary("BTC")
                risk.note_equity(float(summary["equity"]))

            bid, ask = as_quotes(mark, inventory, sigma, tau, gamma, kappa)
            bid = round_to_tick(bid, tick, "buy")
            ask = round_to_tick(ask, tick, "sell")

            line = (f"mark {mark:.4f}  book {best_bid:.4f}/{best_ask:.4f}  "
                    f"sigma {sigma:.1%}  q {inventory:+.1f}  "
                    f"-> quote {bid:.4f}/{ask:.4f}")
            if dry_run:
                print(f"[dry] {line}", flush=True)
            else:
                print(line, flush=True)
                oms.cancel_all()      # replace, don't ladder
                for side, px in (("buy", bid), ("sell", ask)):
                    try:
                        risk.pre_trade(side=side, amount=size,
                                       position=inventory,
                                       n_open_orders=len(oms.open))
                        oms.place(side, size, px, label="as-quoter")
                    except OrderBlocked as why:
                        print(f"  skip {side}: {why}", flush=True)
            time.sleep(interval)
    except KillSwitch as why:
        print(f"KILL SWITCH: {why}", flush=True)
    finally:
        if oms is not None:
            oms.cancel_all()
            print("all orders cancelled; session over. Fills (if any) -> "
                  f"{fills_log}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrument", required=True,
                   help="e.g. BTC-26AUG26-73000-C (pick a liquid near-ATM "
                        "option; public/get_instruments lists them)")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="risk aversion per unit RELATIVE return; higher = "
                        "stronger inventory lean")
    p.add_argument("--kappa", type=float, default=40.0,
                   help="fill-intensity decay per unit relative spread; the "
                        "resting half-spread is ~ln(1+gamma/kappa)/gamma, so "
                        "40 with gamma=2 quotes ~2.4% around the mark")
    p.add_argument("--size", type=float, default=0.1, help="contracts per quote")
    p.add_argument("--session-min", type=float, default=30.0)
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between re-quotes")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(args.instrument, gamma=args.gamma, kappa=args.kappa, size=args.size,
        session_min=args.session_min, interval=args.interval,
        dry_run=args.dry_run, limits=RiskLimits())


if __name__ == "__main__":
    main()
