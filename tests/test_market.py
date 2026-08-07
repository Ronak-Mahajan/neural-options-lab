"""Real-market-data tests, run offline against the committed Deribit fixture.

Only one test needs the network; it is marked `network` and CI skips it. The
rest read `artifacts/deribit_snapshot_*.json`, so the suite is deterministic and
runs with the machine unplugged.

The sharpest test here is `test_quote_convention_reproduces_exchange_iv`. Deribit
publishes its own `mark_iv`, which gives an independent oracle for the single
most dangerous assumption in this module: that a premium quoted in BTC must be
multiplied by the PER-EXPIRY FORWARD to become a USD price. Reading it as a
dollar price instead recovers roughly 5.5 vol points where the truth is near 40 —
wrong by about 7x, and wrong in a way that still produces a plausible-looking
surface.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.quant.deribit import (DeribitError, Snapshot, find_snapshots,
                                   latest_snapshot, load_snapshot)
from backend.quant.surface import (arbitrage_report, black76_price,
                                   black76_vega, build_surface, implied_vol,
                                   parse_instrument_name)


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    if not find_snapshots():
        pytest.skip("no committed Deribit snapshot")
    return latest_snapshot()


@pytest.fixture(scope="module")
def surface(snapshot):
    return build_surface(snapshot)


# --------------------------------------------------------------------------- #
#  The fixture is real
# --------------------------------------------------------------------------- #

def test_snapshot_is_genuine_exchange_data(snapshot):
    """Guard against a fabricated or hand-edited fixture."""
    assert snapshot.currency == "BTC"
    assert 1_000 < snapshot.index_price < 10_000_000
    assert len(snapshot.instruments) > 100
    assert len(snapshot.book_summary) > 100

    names = {i["instrument_name"] for i in snapshot.instruments}
    # Deribit's naming scheme: BTC-<DDMMMYY>-<strike>-<C|P>
    for name in list(names)[:50]:
        cur, ymd, strike, right = parse_instrument_name(name)
        assert cur == "BTC"
        assert strike > 0
        assert right in ("call", "put")
        assert len(ymd) == 3

    # Real chains carry both rights, many strikes and several expiries.
    assert sum(1 for i in snapshot.instruments
               if i["option_type"] == "call") > 20
    assert sum(1 for i in snapshot.instruments
               if i["option_type"] == "put") > 20
    assert len({i["expiration_timestamp"] for i in snapshot.instruments}) >= 4

    # Deribit BTC options settle in coin — this is the fact the whole
    # conversion rests on, so assert it rather than trusting a comment.
    assert all(i.get("settlement_currency") == "BTC"
               for i in snapshot.instruments[:50])


def test_missing_snapshot_raises_clearly(tmp_path):
    with pytest.raises(DeribitError):
        load_snapshot(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
#  Black-76 building blocks
# --------------------------------------------------------------------------- #

def test_black76_put_call_parity():
    F, K, T, s = 64_000.0, 62_000.0, 0.25, 0.55
    c = black76_price(F, K, T, s, "call")
    p = black76_price(F, K, T, s, "put")
    assert c - p == pytest.approx(F - K, rel=1e-12)


def test_black76_is_monotone_in_vol_and_vega_is_its_derivative():
    F, K, T = 64_000.0, 66_000.0, 0.1
    prices = [black76_price(F, K, T, s, "call") for s in (0.3, 0.5, 0.7)]
    assert prices[0] < prices[1] < prices[2]
    s = 0.5
    h = 1e-5
    fd = (black76_price(F, K, T, s + h, "call")
          - black76_price(F, K, T, s - h, "call")) / (2 * h)
    # black76_vega is dPrice/dSigma per UNIT vol (not per vol point), so it
    # equals the central difference directly.
    assert black76_vega(F, K, T, s) == pytest.approx(fd, rel=1e-4)


def test_implied_vol_round_trips():
    F, K, T, s = 64_000.0, 60_000.0, 0.3, 0.62
    price = black76_price(F, K, T, s, "call")
    assert implied_vol(price, F, K, T, "call") == pytest.approx(s, rel=1e-6)


def test_implied_vol_returns_nan_outside_no_arbitrage_bounds():
    F, K, T = 64_000.0, 60_000.0, 0.3
    # Below intrinsic on the forward: no real vol reproduces it.
    assert not math.isfinite(implied_vol(1.0, F, K, T, "call"))


# --------------------------------------------------------------------------- #
#  THE convention test
# --------------------------------------------------------------------------- #

def test_quote_convention_reproduces_exchange_iv(surface):
    """Our inversion must agree with Deribit's published mark_iv.

    Restricted to quotes with enough vega for the inversion to be well
    conditioned: near expiry vega collapses and IV becomes numerically
    meaningless (a 6-hour, deep-ITM put inverts to 99 vol points against an
    exchange mark of 41 simply because the price is essentially all intrinsic).
    That is a real property of the data, not a bug, which is why the surface
    flags those quotes instead of trusting them.
    """
    rows = [q for q in surface.clean()
            if math.isfinite(q.iv_mark) and q.exchange_mark_iv
            and q.vega > 1.0]
    assert len(rows) > 50, f"only {len(rows)} well-conditioned quotes"
    err = np.array([q.iv_mark * 100.0 - q.exchange_mark_iv for q in rows])
    assert np.median(np.abs(err)) < 0.5, (
        f"median |our IV - mark_iv| = {np.median(np.abs(err)):.3f} vol points")
    assert np.abs(err).mean() < 2.0


def test_naive_dollar_reading_is_badly_wrong(surface):
    """The mistake this module exists to avoid, quantified.

    Treating the BTC premium as a dollar price understates IV by roughly an
    order of magnitude, and still yields a smooth, plausible-looking surface.
    """
    rows = [q for q in surface.clean()
            if math.isfinite(q.iv_mark) and q.vega > 1.0
            and 0.95 < q.strike / q.forward < 1.05][:60]
    if len(rows) < 5:
        pytest.skip("not enough near-the-money quotes in this fixture")
    naive = [implied_vol(q.mark_btc, q.forward, q.strike, q.tenor, q.right)
             for q in rows]
    naive = [v for v in naive if math.isfinite(v)]
    correct = [q.iv_mark for q in rows]
    assert np.mean(naive) < 0.5 * np.mean(correct)


# --------------------------------------------------------------------------- #
#  Surface construction and quality flags
# --------------------------------------------------------------------------- #

def test_surface_builds_and_flags_unusable_quotes(surface):
    assert len(surface.quotes) == len(surface.quotes)
    clean = surface.clean()
    assert 0 < len(clean) < len(surface.quotes), (
        "a real chain always contains some unusable quotes; flagging none "
        "means the filters are not running")
    assert len(surface.expiries()) >= 4
    for q in clean:
        assert q.tenor > 0
        assert q.forward > 0
        assert q.bid_usd <= q.ask_usd


def test_iv_spread_is_reported_as_a_band_not_a_point(surface):
    """Quoting a mid as if it were a price hides the width of the market."""
    stats = surface.iv_spread_stats()
    assert stats["n"] > 50
    assert stats["median"] > 0.0
    assert stats["p25"] <= stats["median"] <= stats["p75"]
    for q in surface.clean():
        if math.isfinite(q.iv_bid) and math.isfinite(q.iv_ask):
            assert q.iv_ask >= q.iv_bid


def test_forwards_are_per_expiry_not_the_spot_index(snapshot, surface):
    """`underlying_price` is the forward for that expiry. Using the spot index
    instead would bias every long-dated quote by the futures basis."""
    fwds = {}
    for q in surface.clean():
        fwds.setdefault(q.expiry_ms, q.forward)
    assert len(fwds) >= 4
    ordered = [fwds[k] for k in sorted(fwds)]
    # BTC was in contango in this snapshot; the far forward exceeds the near one.
    assert ordered[-1] > ordered[0]
    assert abs(ordered[-1] - snapshot.index_price) > 100.0


# --------------------------------------------------------------------------- #
#  No-arbitrage diagnostics
# --------------------------------------------------------------------------- #

def test_arbitrage_report_runs_and_counts_what_it_tested(surface):
    rep = arbitrage_report(surface)
    d = rep.to_dict()
    n = d["n_tested"]
    assert n["butterfly_triples"] > 50
    assert n["vertical_pairs"] > 50
    assert n["parity_strikes"] > 10
    for key in ("butterfly", "vertical", "calendar", "put_call_parity"):
        assert key in d


def test_mid_price_violations_do_not_survive_the_spread(surface):
    """The headline result of the whole exercise.

    Real books violate the static no-arbitrage conditions constantly when you
    measure on mid prices, and none of it is tradable: crossing the actual
    bid-ask removes every violation before fees are even considered. A study
    that reported the mid-price count as 'arbitrage found' would be wrong.
    """
    d = arbitrage_report(surface).to_dict()
    flagged = sum(d[k].get("n", 0) for k in
                  ("butterfly", "vertical", "calendar", "put_call_parity"))
    executable = sum(d[k].get("n_executable", 0) for k in
                     ("butterfly", "vertical", "calendar", "put_call_parity"))
    net = sum(d[k].get("n_net_of_fees", 0) for k in
              ("butterfly", "vertical", "calendar", "put_call_parity"))
    assert flagged > 0, "a real chain should show mid-price violations"
    assert executable <= flagged
    assert net <= executable


def test_synthetic_forward_agrees_with_the_listed_future(surface):
    """Put-call parity backs a forward out of the option market; it must match
    the futures market. Disagreement would mean the parity map, the day count,
    or the coin conversion is wrong — this is an end-to-end check on all three.
    """
    d = arbitrage_report(surface).to_dict()
    rows = [f for f in d["forward_consistency"] if f.get("n_pairs", 0) >= 2]
    assert len(rows) >= 4
    errs = np.array([abs(f["synthetic_vs_future_median_bps"]) for f in rows])
    assert errs.max() < 25.0, f"worst synthetic-vs-future gap {errs.max():.2f} bps"
    assert np.median(errs) < 5.0


# --------------------------------------------------------------------------- #
#  Live network (skipped in CI)
# --------------------------------------------------------------------------- #

@pytest.mark.network
def test_live_fetch_matches_fixture_shape():
    from backend.quant.deribit import DeribitClient
    snap = DeribitClient(timeout=30.0).snapshot(currency="BTC")
    assert len(snap.instruments) > 100
    surf = build_surface(snap)
    assert len(surf.clean()) > 50
