# Deep hedging on real SPY and BTC paths

Produced by `scripts/hedge_real_paths.py` on 2026-09-11; the raw output of that run is
committed as [`hedging_real_paths.txt`](hedging_real_paths.txt), and every table below is
read from it. A run of 2026-08-20 (commit `5cdd02c`) agrees with it to the second decimal
(section 5).

The `deep` rows on this page are the GAN-measure policy, `artifacts/hedger.pt`, the
checkpoint `HedgingEngine` loads when it is given none. It was trained on the WGAN market
simulator with per-step standardisation only, a measure under which discounted spot is not
a martingale and the paths realise about 1.28× the requested volatility
([hedging_findings.md](hedging_findings.md), 1.1). It is neither of the two checkpoints
the dashboard serves (`backend/api/main.py` maps the rough regime to
`hedger_rbergomi_jumps.pt` and GBM to `hedger_gbm.pt`); section 4.1 gives its provenance.
The served policies have not been replayed on history, so every comparison between this
page and a simulated result is a comparison across two different policies as well as two
different measures.

Every other hedging number in this project is measured on simulated paths: risk-neutral
GBM ([hedging_findings.md](hedging_findings.md)), rough Bergomi with jumps
([deep_hedging_regimes.md](deep_hedging_regimes.md)), or the WGAN. This experiment replays
three hedgers, the learned CVaR policy, a Black-Scholes delta hedge and a Whalley-Wilmott
no-trade band, over rolling 30-day windows of real SPY and BTC-USD daily closes, paired on
identical windows with identical ex-ante information.

At 10 bp of proportional cost the policy has no tail advantage. Plain delta has the lower
P&L standard deviation and the better CVaR₉₅ on both assets (SPY 0.027 vs the deep
policy's 0.041; BTC 0.057 vs 0.087, in units of the entry spot). Delta's whole cost bill
at that level is about 0.3% of spot per window, which leaves little for cost-awareness to
save.

At 50 bp the cost efficiency carries over. The deep policy has the better mean P&L on SPY
(−0.65% vs delta's −1.35%, better on 69% of windows, ahead of Whalley-Wilmott as well) by
trading about half as much, the behaviour its cost-aware objective trains. On BTC it is
ahead on mean P&L by a smaller margin (−0.96% vs −1.14%, 54% of windows).

Its tails stay wider at 50 bp. CVaR₉₅ is worse than delta's on both assets (SPY 0.048 vs
0.042, BTC 0.095 vs 0.071). On these paths the policy is a cheaper hedge than delta and
carries the larger tail loss.

In simulation under rough volatility with jumps the rough-measure policy has a lower
CVaR₉₅ than the delta hedge from 50 bp of cost; the checkpoint replayed here has a higher
one on history. Those two results differ in the measure and in the policy, since the
simulated tail improvement belongs to `hedger_rbergomi_jumps.pt` and these rows are
`hedger.pt`. This page shows that the tail advantage is absent on history for this
checkpoint. It does not apportion the difference between the change of measure and the
change of policy. Real daily returns add autocorrelation and volatility clustering that
neither GBM nor the WGAN reproduces. The README's tail-risk claims for deep hedging are
simulation-scoped; this page is the evidence for what carries over to history.

## 1. Protocol

The protocol and its known weak points:

| | |
|---|---|
| Contract | Short one at-the-money 30-day call, spot normalised to 1.0 at entry, daily rebalancing: the contract the policy was trained on (`N_STEPS = 30`, `MATURITY = 30/252`). |
| Data | Daily closes from yfinance, 8 years to the run date (SPY: 2,010 closes; BTC-USD: 2,922), adjusted. |
| Windows | Every 5 trading days, 30 consecutive log returns: 384 SPY windows and 567 BTC windows. Consecutive windows share 25 of 30 days, so the effective number of independent windows is closer to n/6 (about 64 and 94), and the bootstrap standard errors printed for CVaR₉₅ are optimistic. |
| Volatility | The trailing 60-day realised volatility at entry, clipped to the policy's training box [0.08, 0.65]. Every hedger receives the same forecast, and none sees the window's own realised vol. 4 SPY windows and 76 BTC windows were clipped. |
| Premium | Booked at Black-Scholes under that ex-ante vol, the price a desk quoting off this forecast would have collected, so the P&L includes the vol-forecast error. |
| Rates | 4% for SPY, 0 for BTC. |
| Hedgers | `deep`: the GAN-measure policy, `artifacts/hedger.pt`, the checkpoint `HedgingEngine` loads when it is given none. It is not one of the two the dashboard serves (section 4.1). `delta`: Black-Scholes delta at the ex-ante vol. `whalley_wilmott`: the no-trade band around that delta with risk aversion 1.0. |
| Costs | Proportional, charged on every trade and on the final unwind, at 10 bp and 50 bp. |
| Statistic | Mean, standard deviation and CVaR₉₅ (mean of the worst 5% of losses) of terminal P&L per unit of entry spot, with a 500-draw bootstrap standard error on CVaR₉₅, plus the paired deep-minus-delta difference. |

One historical path per asset is a single draw, and a strategy can lose on a draw and
still be right. The informative quantity is the paired comparison on the same windows with
the same information. The absolute P&L levels are one realisation.

## 2. Results

P&L and CVaR₉₅ are in units of the entry spot, so 0.0271 is 2.71% of strike, or 271 bp.
"costs" is the mean cost paid per window in the same units. Best value in each column in
bold.

### SPY, 384 windows (about 64 independent), ex-ante vol median 14.2%

| cost | hedger | mean P&L | std | CVaR₉₅ | (se) | costs |
|---|---|---|---|---|---|---|
| 10 bp | deep | **−0.0008** | 0.0159 | 0.0406 | 0.0070 | **0.0014** |
| 10 bp | delta | −0.0020 | **0.0119** | **0.0271** | 0.0033 | 0.0029 |
| 10 bp | Whalley-Wilmott | −0.0037 | 0.0160 | 0.0349 | 0.0028 | **0.0014** |
| 50 bp | deep | **−0.0065** | 0.0167 | 0.0484 | 0.0076 | 0.0074 |
| 50 bp | delta | −0.0135 | **0.0130** | **0.0419** | 0.0030 | 0.0144 |
| 50 bp | Whalley-Wilmott | −0.0112 | 0.0256 | 0.0560 | 0.0027 | **0.0063** |

Paired deep minus delta: +0.0011 at 10 bp (deep better on 51% of windows); +0.0070 at
50 bp (deep better on 69% of windows).

### BTC-USD, 567 windows (about 94 independent), ex-ante vol median 45.4%

| cost | hedger | mean P&L | std | CVaR₉₅ | (se) | costs |
|---|---|---|---|---|---|---|
| 10 bp | deep | −0.0035 | 0.0362 | 0.0866 | 0.0047 | 0.0016 |
| 10 bp | delta | **−0.0007** | **0.0238** | **0.0574** | 0.0053 | 0.0027 |
| 10 bp | Whalley-Wilmott | −0.0042 | 0.0364 | 0.0868 | 0.0054 | **0.0015** |
| 50 bp | deep | **−0.0096** | 0.0370 | 0.0951 | 0.0049 | 0.0083 |
| 50 bp | delta | −0.0114 | **0.0245** | **0.0705** | 0.0053 | 0.0133 |
| 50 bp | Whalley-Wilmott | −0.0127 | 0.0529 | 0.1223 | 0.0038 | **0.0066** |

Paired deep minus delta: −0.0027 at 10 bp (deep better on 47% of windows); +0.0017 at
50 bp (deep better on 54% of windows).

## 3. Reading the tables

1. At 10 bp delta has the lowest standard deviation and the best CVaR₉₅ on both assets,
   by margins well outside the (optimistic) standard errors on SPY (0.027 ± 0.003 vs
   0.041 ± 0.007) and BTC (0.057 ± 0.005 vs 0.087 ± 0.005). The deep policy's mean P&L is
   within noise of delta's on SPY and behind it on BTC. Costs at this level are about
   0.3% of spot per window for delta, so a hedger that trades half as much saves 0.15%,
   less than the tail it gives up.
2. At 50 bp the cost saving becomes the mean. Delta's cost bill rises to 1.4% of spot on
   SPY; the deep policy pays 0.7% and ends 0.70% ahead on average in the paired
   comparison, ahead on 69% of windows. Whalley-Wilmott pays even less, but its band is
   wide enough at 50 bp that its P&L dispersion is the worst of the three.
3. The deep policy's CVaR₉₅ is worse than delta's in all four (asset, cost) cells. Under
   rough Bergomi with jumps in simulation a policy of the same architecture crosses below
   delta on CVaR₉₅ at 50 bp ([deep_hedging_regimes.md](deep_hedging_regimes.md)); on
   history this checkpoint does not, and on BTC the gap is wide (0.095 vs 0.071).
4. The BTC vol forecast is the harder one. 76 of 567 BTC windows (13%) had a trailing vol
   above the 0.65 training ceiling and every hedger ran them at 0.65, and the median
   ex-ante vol of 45% sits in the upper half of the box. The policy's BTC margins over
   delta are correspondingly thinner.

## 4. Scope and limitations

### 4.1 Replayed checkpoint

The script loads `artifacts/hedger.pt`, the checkpoint `HedgingEngine` takes when it is
given none. The dashboard serves two other checkpoints: `backend/api/main.py` maps the
rough regime to `hedger_rbergomi_jumps.pt` and the GBM regime to `hedger_gbm.pt`.
`hedger.pt` is the GAN-measure policy, trained for 8,000 iterations on the WGAN market
simulator with per-step standardisation only. That measure is neither a martingale nor
correctly scaled ([hedging_findings.md](hedging_findings.md), 1.1). The checkpoint's
metadata carries no `train_measure` and no `martingale_enforced` field;
`backend/quant/hedging.py` records the training measure, and the same bytes are committed
a second time as `artifacts/hedger_v1_unconstrained_measure.pt` (both MD5
`36a6296bab32831f77c38840d0b2067b`). The three rows labelled `deep` are a replay of that
policy and of nothing the site serves. The open replays are the two served checkpoints on
the same windows, and the rough-measure checkpoints (`hedger_rbergomi.pt`,
`hedger_rbergomi_jumps.pt`) that carry the simulated tail advantage. The script takes no
checkpoint argument, so both need one added first.

### 4.2 Window overlap

Windows overlap 25 of 30 days. Overlap does not bias the paired win rates and means. The
CVaR standard errors assume independent windows and should be read as roughly 2.5x too
small.

### 4.3 One path per asset

Eight years of SPY is one path through one regime sequence (including the 2020 and 2022
vol episodes, which dominate the SPY tail). The paired comparison is informative on that
path. The absolute levels are one realisation and carry no forecast.

### 4.4 Data

yfinance daily closes, adjusted, with no intraday information; the hedgers rebalance once
a day at the close.

## 5. Rerun

```bash
python -m scripts.hedge_real_paths                  # 10 bp of cost (default)
python -m scripts.hedge_real_paths --cost 0.005     # 50 bp
python -m scripts.hedge_real_paths --tickers SPY --years 8
```

Needs network access (yfinance) and the committed `artifacts/hedger.pt` and
`artifacts/generator.pt`. Takes a few minutes on a CPU, most of it the download. The
history window is the last 8 years to the run date, so the window count and the numbers
drift slightly with that date. The committed
[`hedging_real_paths.txt`](hedging_real_paths.txt) is the 2026-09-11 run tabulated above
(384 SPY windows; SPY 10 bp CVaR₉₅ 0.027 vs 0.041; 50 bp mean P&L −0.65% vs −1.35% on 69%
of windows). The 2026-08-20 run at commit `5cdd02c` (385 SPY windows; SPY 10 bp CVaR₉₅
0.027 vs 0.041; 50 bp mean P&L −0.67% vs −1.36% on 69% of windows) agrees with it to the
second decimal.
