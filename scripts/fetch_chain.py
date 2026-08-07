"""Fetch a live Deribit option chain, build the surface, and report on it.

Everything downstream of this script — the surface, the arbitrage tests, the
whole test suite — reads a COMMITTED snapshot rather than the network, so the
repository stays deterministic and runnable offline. This is the one place that
touches the internet, and it is opt-in.

Usage
-----
    python -m scripts.fetch_chain                 # fetch, save, report
    python -m scripts.fetch_chain --offline       # report on the newest fixture
    python -m scripts.fetch_chain --currency ETH
    python -m scripts.fetch_chain --no-save       # fetch and report, keep nothing

No API key is required or accepted; only public read-only endpoints are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.quant.deribit import (ARTIFACTS_DIR, DeribitClient, DeribitError,
                                   latest_snapshot, save_snapshot)
from backend.quant.surface import arbitrage_report, build_surface


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--currency", default="BTC")
    p.add_argument("--offline", action="store_true",
                   help="skip the network and use the newest committed snapshot")
    p.add_argument("--no-save", action="store_true",
                   help="do not write a snapshot fixture")
    p.add_argument("--min-vega-usd", type=float, default=10.0,
                   help="drop quotes whose vega is too small to invert reliably")
    p.add_argument("--report-json", type=Path, default=None,
                   help="write the arbitrage report to this path")
    args = p.parse_args()

    try:
        if args.offline:
            snap = latest_snapshot()
            print(f"offline: {snap.currency} snapshot from {snap.captured_at_iso}")
        else:
            print(f"fetching live {args.currency} chain from Deribit ...")
            snap = DeribitClient().snapshot(currency=args.currency)
            if not args.no_save:
                path = save_snapshot(snap)
                print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
    except DeribitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("If the network is unreachable, re-run with --offline.",
              file=sys.stderr)
        return 1

    surface = build_surface(snap, min_vega_usd=args.min_vega_usd)
    print(f"\nindex {snap.index_price:,.2f} | instruments {len(snap.instruments)} "
          f"| quotes {len(surface.quotes)} | clean {len(surface.clean())} "
          f"| expiries {len(surface.expiries())}")

    stats = surface.iv_spread_stats()
    if stats:
        # iv_spread_stats already returns vol points — do not rescale.
        print(f"\nIV bid-ask spread on {stats['n']} clean quotes (vol points):")
        for k in ("p05", "p25", "median", "p75", "p95", "mean", "max"):
            if k in stats:
                print(f"  {k:>6} {stats[k]:7.2f}")

    rep = arbitrage_report(surface).to_dict()
    print(f"\nno-arbitrage tests, tested {rep['n_tested']}")
    for key in ("butterfly", "vertical", "calendar", "put_call_parity"):
        v = rep[key]
        n = v.get("n", 0)
        if not n:
            print(f"  {key:<16} none")
            continue
        print(f"  {key:<16} {n:>3} flagged on mid | "
              f"{v.get('n_executable', 0)} executable | "
              f"{v.get('n_net_of_fees', 0)} after fees | "
              f"median {v.get('median', float('nan')):.2f} "
              f"max {v.get('max', float('nan')):.2f} {v.get('units', '')}")

    print("\nforward consistency (synthetic from parity vs the listed future):")
    for f in rep["forward_consistency"]:
        print(f"  {f['expiry']}  {f['tenor_days']:>7.3f}d  "
              f"future {f['future_mark']:>10,.2f}  "
              f"synthetic-vs-future {f['synthetic_vs_future_median_bps']:>6.2f} bps  "
              f"bracket {f['bracket_width_median_bps']:>6.2f} bps")

    out = args.report_json or (ARTIFACTS_DIR / "deribit_arbitrage_report.json")
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
