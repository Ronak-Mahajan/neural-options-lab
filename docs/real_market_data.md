# Real market data: a live Deribit option chain

Every other number in this project is synthetic: geometric Brownian motion, Monte Carlo paths, a WGAN fitted to daily closes, or yfinance mid-prices that the README itself flags
as retail-grade. This is the part that touches a real order book.

All figures below were measured on the committed snapshot
`artifacts/deribit_snapshot_btc_20260805T214815Z.json`. Reproduce with:

```bash
python -m scripts.fetch_chain --offline      # committed fixture, no network
python -m scripts.fetch_chain                # fetch a fresh chain
python -m pytest tests/test_market.py -m "not network"
```

## Why Deribit

Full-depth US equity option data is not genuinely free once exchange licensing and
professional-use rules apply, which is why student projects end up on yfinance mids.
Deribit is the dominant venue for crypto options and its public v2 REST API serves full chains (live bid/ask, mark IV, open interest, per-expiry forwards, and the listed futures) with **no API key, no authentication and no licence**. Only read-only
`/api/v2/public/*` endpoints are used here.

## The snapshot

| | |
|---|---|
| captured | 2026-08-05T21:48:15Z |
| BTC index | 64,631.30 |
| instruments | 836 live options |
| expiries | 12 |
| tenors | 0.42 days to 323 days |
| usable after filtering | **547** |

## The convention trap, settled by measurement

This is the single most dangerous assumption in the module, so it was verified against an
independent oracle rather than reasoned about. Deribit publishes its own `mark_iv`.

Deribit BTC options are European, cash-settled, and **quoted in BTC**: `quote_currency` and `settlement_currency` are both BTC, so a premium of `0.694` means 0.694 coins, not
694 dollars. Separately, the `underlying_price` on each row is the **forward for that
instrument's expiry**, not the spot index: measured simultaneously, index 64,631.30
against forwards of 64,634.53 (0.4 days) rising to 67,175.19 (323 days). BTC was in
contango, so using spot in place of the forward would bias every long-dated quote.

The correct reading is:

```
price_usd = price_btc * underlying_price          # per-expiry forward
IV        = Black-76 inversion on F = underlying_price, undiscounted
```

Deribit reports `interest_rate = 0.0` on these instruments, so the discount factor is 1
and the forward already carries the carry.

**Verification, on 547 well-conditioned quotes:**

| reading | recovered ATM IV | agreement with `mark_iv` |
|---|---|---|
| **forward-USD (correct)** | 32.88 vol points | median \|error\| **0.0028 vp**, max 0.172 |
| naive: BTC premium read as dollars | 4.62 vol points | **off by 7.1x** |

The naive reading does not fail loudly. It produces a smooth, plausible-looking surface
that happens to be wrong by an order of magnitude, which is exactly why this needed an oracle rather than a code review.

The day-count convention was pinned the same way. Reproducing `mark_iv` across all 836
instruments gives a median error of **+0.0001 vol points on ACT/365**, +0.0144 on 365.25,
and −0.2727 on 360.

## Quote quality: what a real chain actually looks like

289 of 836 quotes are unusable, and the reasons are worth stating because a study that
silently kept them would report noise as signal:

| flag | count | meaning |
|---|---|---|
| `bid_below_bound` | 199 | bid sits below the no-arbitrage floor; effectively no real bid |
| `no_trade` | 118 | no volume and no open interest |
| `no_bid` | 66 | one-sided market |
| `low_vega` | 56 | vega too small for the IV inversion to mean anything |
| `wide` | 2 | bid-ask wider than the acceptance threshold |

`low_vega` deserves emphasis. Near expiry, vega collapses and implied vol stops being a
well-posed quantity: a 6-hour deep-ITM put in this snapshot inverts to 99 vol points
against an exchange mark of 41, purely because its price is essentially all intrinsic.
That is a real property of the data, not a bug, so those quotes are flagged rather than trusted, and the convention test above is restricted to quotes where the inversion is
conditioned.

## The IV bid-ask is a band, not a number

Treating a mid as a price hides the width of the market. On the 547 clean quotes, the
**implied-vol bid-ask spread** in vol points:

| p05 | p25 | median | p75 | p95 | mean | max |
|---|---|---|---|---|---|---|
| 0.52 | 0.79 | **1.36** | 3.00 | 9.79 | 2.70 | 24.24 |

A model calibrated to mids is being asked to fit to sub-spread precision over most of this
surface, and to nothing at all in the tail. The calibration objective in `calibrate.py`
weights by vega and ignores the spread entirely, a known defect recorded in the audit.

## No-arbitrage diagnostics: the headline result

Four static tests, all in undiscounted forward-USD terms: **butterfly** (convexity in
strike), **vertical spread** bounds, **calendar** (monotonicity of total variance), and
**put-call parity** against the market forward.

| test | tested | flagged on mid | executable | net of fees | median | max |
|---|---|---|---|---|---|---|
| butterfly | 499 triples | 41 | **0** | **0** | 0.25 bps | 106.25 bps |
| vertical | 523 pairs | 9 | **0** | **0** | 20.82 bps | 68.93 bps |
| calendar | 11 expiry pairs | 1 | **0** | **0** | 0.80 | 0.80 (total-variance bps) |
| put-call parity | 195 strikes | 0 | 0 | 0 | - | - |

**Every apparent violation vanishes once you require crossing the actual bid-ask.** Not
one of the 51 mid-price violations is executable, before transaction costs are even
considered.

This is the result worth taking away. Measuring no-arbitrage conditions on mid prices and
reporting the count as "arbitrage found" is a standard error: it mistakes the width of the market for a mispricing. The honest statement is that this book contains no static
arbitrage, and that the mid-price violations are an artifact of quoting conventions in
illiquid strikes. Note also where they cluster: the worst butterfly (106.25 bps) and the
worst vertical (68.93 bps) are both on the 2027-03-26 put wing, a 232-day tenor where the
IV spread is widest.

## Forward consistency: an end-to-end check

Put-call parity backs a synthetic forward out of the option market. It must agree with the listed BTC future for that expiry, and because the parity map, the ACT/365 day count and the coin-to-dollar conversion all feed into it, this is a single check on all three at
once.

| expiry | tenor | listed future | synthetic − future | parity bracket width |
|---|---|---|---|---|
| 2026-08-06 | 0.43d | 64,634.53 | +0.88 bps | 17.97 bps |
| 2026-08-08 | 2.43d | 64,650.00 | +0.73 bps | 20.20 bps |
| 2026-08-14 | 8.43d | 64,687.38 | +0.57 bps | 45.68 bps |
| 2026-09-25 | 50.43d | 65,022.82 | +0.47 bps | 48.16 bps |
| 2026-10-30 | 85.43d | 65,288.32 | **+0.06 bps** | 40.81 bps |
| 2026-12-25 | 141.43d | 65,751.82 | −1.71 bps | 51.91 bps |
| 2027-03-26 | 232.43d | 66,429.00 | +0.19 bps | 66.38 bps |
| 2027-06-25 | 323.43d | 67,175.19 | −0.59 bps | 63.45 bps |

Median absolute disagreement is **under 1 basis point** across all 12 expiries, and every
synthetic forward falls inside its own bid-ask bracket. The option market and the futures
market agree, which means the conversion chain is right end to end.

## What this does not do

- It does not feed the pricing surrogate. The Asian-option model is trained on simulated
  GBM; this module is a separate, honest data layer, not a retrofit of real data into a
  synthetic pipeline.
- It is a single point-in-time snapshot, not a time series. Nothing here says anything
  about dynamics, and no claim about signal or predictability is made or implied.
- Crypto options are not equity options. The forward curve, the fee structure
  (0.0003 taker) and the coin-settled convention are all specific to this venue.
