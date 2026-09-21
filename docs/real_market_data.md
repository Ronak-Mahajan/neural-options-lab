# Real market data: a live Deribit option chain

This module reads a full Deribit BTC option chain: live bids and asks, exchange mark IVs,
open interest, per-expiry forwards and the listed futures. It is the project's order-book
data layer. The pricing surrogates train on simulated paths (geometric Brownian motion and
rough Bergomi), and the SPY calibrations use retail-grade yfinance option mids.

All figures below are measured on the committed snapshot
`artifacts/deribit_snapshot_btc_20260805T214815Z.json`. Reproduce with:

```bash
python -m scripts.fetch_chain --offline      # committed fixture, no network
python -m scripts.fetch_chain                # fetch a fresh chain
python -m pytest tests/test_market.py -m "not network"
```

## Data source

Full-depth US equity option data carries exchange licensing fees and professional-use
rules. Deribit is the dominant venue for crypto options, and its public v2 REST API serves
full chains (live bid/ask, mark IV, open interest, per-expiry forwards and the listed
futures) without an API key or authentication. Only read-only `/api/v2/public/*` endpoints
are used here.

## The snapshot

| | |
|---|---|
| captured | 2026-08-05T21:48:15Z |
| BTC index | 64,631.30 |
| instruments | 836 live options |
| expiries | 12 |
| tenors | 0.42 days to 323 days |
| usable after filtering | 547 |

## Quote and forward conventions

A wrong premium convention moves at-the-money implied vol from 32.88 to 4.62 vol points and
still produces a smooth surface, so the convention is verified against an independent
calculation, the `mark_iv` that Deribit publishes for every instrument.

Deribit BTC options are European, cash-settled and quoted in BTC. `quote_currency` and
`settlement_currency` are both BTC, so a premium of `0.694` is 0.694 coins. The
`underlying_price` on each row is the forward for that instrument's expiry. It differs from
the spot index: measured at the same instant, the index is 64,631.30 and the forwards run
from 64,634.53 (0.4 days) to 67,175.19 (323 days). BTC was in contango, so using spot in
place of the forward would bias every long-dated quote.

The reading that reproduces `mark_iv` is:

```
price_usd = price_btc * underlying_price          # per-expiry forward
IV        = Black-76 inversion on F = underlying_price, undiscounted
```

Deribit reports `interest_rate = 0.0` on these instruments, so the discount factor is 1
and the forward already carries the carry.

Verification on the 547 well-conditioned quotes:

| reading | recovered ATM IV | agreement with `mark_iv` |
|---|---|---|
| forward-USD | 32.88 vol points | median \|error\| 0.0028 vp, max 0.172 |
| naive: BTC premium read as dollars | 4.62 vol points | low by a factor of 7.1 |

The naive reading raises no error. It returns a smooth surface whose at-the-money implied
vol is low by a factor of 7.1.

The day-count convention is pinned the same way. Reproducing `mark_iv` across all 836
instruments gives a median error of +0.0001 vol points on ACT/365, +0.0144 on 365.25
and −0.2727 on 360.

## Quote quality

Of 836 quotes, 289 are flagged and dropped. A quote can carry several flags, so the counts
below sum to 441:

| flag | count | meaning |
|---|---|---|
| `bid_below_bound` | 199 | bid sits below the no-arbitrage floor, so there is no tradable bid |
| `no_trade` | 118 | no volume and no open interest |
| `no_bid` | 66 | one-sided market |
| `low_vega` | 56 | vega too small for the implied-vol inversion to be conditioned |
| `wide` | 2 | bid-ask wider than the acceptance threshold |

Near expiry vega collapses and implied vol stops being a well-posed quantity. The 10-hour
70,000-strike put in this snapshot inverts to 70.5 vol points against an exchange mark of
41.0, because 5,366.38 of its 5,366.57 dollar mark is intrinsic value. `low_vega` quotes
are therefore flagged and excluded, and the convention test above is restricted to quotes
where the inversion is conditioned.

## Implied-vol bid-ask spread

On the 547 clean quotes, the implied-vol bid-ask spread in vol points is:

| p05 | p25 | median | p75 | p95 | mean | max |
|---|---|---|---|---|---|---|
| 0.52 | 0.79 | 1.36 | 3.00 | 9.79 | 2.70 | 24.24 |

A calibration to mids therefore targets a precision finer than the quoted spread over most
of this surface, and in the widest 5% of quotes the spread exceeds 9.79 vol points. The
calibration objective in `calibrate.py` weights by vega and does not use the quoted spread.

## No-arbitrage diagnostics

Four static tests, all in undiscounted forward-USD terms: butterfly (convexity in
strike), vertical spread bounds, calendar (monotonicity of total variance) and
put-call parity against the market forward.

| test | tested | flagged on mid | executable | net of fees | median | max |
|---|---|---|---|---|---|---|
| butterfly | 499 triples | 41 | 0 | 0 | 0.25 bps | 106.25 bps |
| vertical | 523 pairs | 9 | 0 | 0 | 20.82 bps | 68.93 bps |
| calendar | 11 expiry pairs | 1 | 0 | 0 | 0.80 | 0.80 (total-variance bps) |
| put-call parity | 195 strikes | 0 | 0 | 0 | n/a | n/a |

None of the 51 mid-price violations is executable through the quoted bid-ask, before fees.

A violation count taken on mid prices measures the width of the market. On executable
prices this book contains no static arbitrage in any of the four tests, and the mid-price
violations come from wide quotes in illiquid strikes. The worst butterfly (106.25 bps) and
the worst vertical (68.93 bps) are both on the 2027-03-26 put wing. That 232-day expiry has
the widest put quotes among the 547 clean quotes in price terms: a median bid-ask of 50 bps
of forward and a maximum of 280.

## Forward consistency

Put-call parity backs a synthetic forward out of the option market. It must agree with the
listed BTC future for that expiry. The parity map, the ACT/365 day count and the
coin-to-dollar conversion all feed into it, so the comparison checks all three at once.

| expiry | tenor | listed future | synthetic − future | parity bracket width |
|---|---|---|---|---|
| 2026-08-06 | 0.43d | 64,634.53 | +0.88 bps | 17.97 bps |
| 2026-08-08 | 2.43d | 64,650.00 | +0.73 bps | 20.20 bps |
| 2026-08-14 | 8.43d | 64,687.38 | +0.57 bps | 45.68 bps |
| 2026-09-25 | 50.43d | 65,022.82 | +0.47 bps | 48.16 bps |
| 2026-10-30 | 85.43d | 65,288.32 | +0.06 bps | 40.81 bps |
| 2026-12-25 | 141.43d | 65,751.82 | −1.71 bps | 51.91 bps |
| 2027-03-26 | 232.43d | 66,429.00 | +0.19 bps | 66.38 bps |
| 2027-06-25 | 323.43d | 67,175.19 | −0.59 bps | 63.45 bps |

The table shows eight of the twelve expiries. Across all twelve the median absolute
disagreement is 0.8 bps and the largest is 3.01 bps (2026-08-09, inside a 22.47 bps
bracket), and every synthetic forward falls inside its own bid-ask bracket
(`artifacts/deribit_arbitrage_report.json`). The option and
futures markets agree, so the conversion chain holds end to end.

## Scope

- The module does not feed the pricing surrogate. The Asian-option model is trained on
  simulated GBM, and this chain is a separate data layer.
- The data is a single point-in-time snapshot. It carries no information about dynamics,
  and no claim about signal or predictability is made.
- The forward curve, the fee structure (0.0003 taker) and the coin-settled convention are
  specific to this venue and do not carry over to equity options.
