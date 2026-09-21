# Closed-form approximations versus the neural surrogate

On 300 Latin-hypercube points over the trained box (S/K in [0.50, 2.00], maturity in [0.05, 2.00] years, sigma in [0.05, 0.80], r in [0.00, 0.10]), scored against 200,000-path control-variate Monte Carlo, the served ensemble has a price RMSE of 1.395 bps of strike, Levy (1992) moment matching 38.911 bps and Curran (1994) conditioning 2.347 bps. On a 36-cell grid at sigma = 0.25 against 400,000-path references the mean absolute errors are 0.72, 3.65 and 0.10 bps. Both measurements are produced by `scripts/benchmark_approximations.py` and stored point by point in `docs/approximation_benchmark.json`; this page is rendered from that file.

## Trained box, 300 Latin-hypercube points

Measured on 2026-09-20; AMD64, 16 logical CPUs, torch 2.11.0+cpu (CPU), numpy 2.5.3. Checkpoint `model.pt`, sha256 `2ca9e6ecc8979e70`. Stored under `box_lhs` in `docs/approximation_benchmark.json`.

### Protocol

- Contract: arithmetic-average Asian call, strike 100, 50 equally spaced monitoring dates.
- Points: 300 Latin-hypercube points (`scipy.stats.qmc.LatinHypercube`, seed 1992) over the box the surrogate is trained on: S/K in [0.50, 2.00], maturity in [0.05, 2.00] years, sigma in [0.05, 0.80], r in [0.00, 0.10]. No training or model-selection draw uses this seed.
- Reference: `dataset._simulate_chunk`, 200,000 paths, antithetic sampling with the geometric-Asian control variate, the estimator `backend/quant/evaluate.py` scores the surrogate against.
- Reference standard error: `_simulate_chunk` returns none, so each point is priced a second time by `price_asian_mc` (the same estimator, an independent 200,000-path draw), which reports one over antithetic pairs. RMS 0.835 bps of strike, mean 0.506 bps, max 4.211 bps (m=1.71, T=1.97, sigma=0.76, r=0.031). Over the 291 points with a nonzero standard error, the two independent prices differ by an RMS of 1.01 in units of sqrt(2) standard errors; a value near 1 means the standard error describes the reference's noise.
- Surrogate: `PricingEngine` on `artifacts/model.pt`, 5 members, price only through `price_batch` with a batch of one (the float32 serving path). Greeks are not timed here.
- Latency: median of 20 calls per point x 300 points for the fast pricers; mean of one run per point for Monte Carlo. Single process, `torch.set_num_threads(1)` to match the Dockerfile's `OMP_NUM_THREADS=1`, one warm-up call per pricer. Absolute times are specific to this machine and run; the ratios between rows are the comparable quantity.
- Timing conditions: system-wide CPU utilisation averaged 64% of 16 logical CPUs over the run, of which this benchmark is one thread. The machine ran on battery power.

### Results

Errors are (method - reference), in basis points of strike.

| method | RMSE | mean abs error | bias | p95 abs error | max abs error | wall-clock per price |
|---|---|---|---|---|---|---|
| neural surrogate (5-member ensemble) | 1.395 bps | 0.951 bps | +0.308 bps | 3.312 bps | 6.703 bps | 3.21 ms |
| Turnbull-Wakeman / Levy | 38.911 bps | 19.214 bps | +17.662 bps | 99.335 bps | 220.517 bps | 299 us |
| Curran (1994), exact threshold | 2.347 bps | 1.082 bps | -1.038 bps | 4.859 bps | 13.666 bps | 388 us |
| Curran (1994), linear threshold | 2.364 bps | 1.093 bps | -1.049 bps | 4.872 bps | 13.728 bps | 328 us |
| Monte Carlo, 200,000 paths, price only (`price_asian_mc`) | (reference) | RMS SE 0.835 bps | n/a | n/a | SE <= 4.211 bps | 529.36 ms |

Worst points: surrogate at m=1.95, T=1.63, sigma=0.73, r=0.089; Levy at m=1.71, T=1.97, sigma=0.76, r=0.031; Curran at m=1.95, T=1.63, sigma=0.73, r=0.089.

Over the trained box the surrogate's price RMSE is 1.395 bps of strike and Levy's is 38.911 bps, a ratio of 27.9 (95% interval 22.9 to 33.5, paired percentile bootstrap, 10,000 resamples of the 300 points). By sigma band the ratio is 1.8, 14.0 and 37.8 (table below), and on the 36-cell grid at sigma = 0.25 it is 5.1 in mean absolute error. Levy matches a lognormal to the first two moments of the average, and the distance between that lognormal and the true law of the average grows with sigma^2 T.

Curran's RMSE on the same points is 2.347 bps. The surrogate / Curran RMSE ratio is 0.6 (95% interval 0.5 to 0.7), so over the whole box the surrogate has the lower RMSE. The ordering depends on volatility. Curran has the lower RMSE for sigma in 0.05 to 0.30 (0.188 against the surrogate at 1.180 bps) and 0.30 to 0.55 (0.863 against the surrogate at 1.211 bps). The surrogate has the lower RMSE for sigma in 0.55 to 0.80 (1.725 against Curran at 3.969 bps). Curran is a lower bound whose gap to the true price widens with sigma^2 T; its bias over the box is -1.038 bps.

The reference's RMS standard error is 0.835 bps. With that removed in quadrature the surrogate's RMSE is 1.117 bps, Levy's 38.903 bps and Curran's 2.194 bps.

One surrogate price takes 3.21 ms at the median against 529.36 ms for one 200,000-path price-only Monte Carlo run, a ratio of 165. The two are at different accuracies. That Monte Carlo price carries an RMS standard error of 0.835 bps against the surrogate's 1.395 bps RMSE. Levy costs 299 us per price and Curran 388 us, so the surrogate is 10.7x the cost of Levy and 8.3x the cost of Curran. The reference run itself (`_simulate_chunk`, which also returns pathwise delta and vega) takes 889.77 ms per point.

### Error by volatility and by price level

RMSE in bps of strike within each band.

| sigma | points | neural surrogate (5-member ensemble) | Turnbull-Wakeman / Levy | Curran (1994), exact threshold | Levy / surrogate |
|---|---|---|---|---|---|
| 0.05 to 0.30 | 100 | 1.180 | 2.159 | 0.188 | 1.8 |
| 0.30 to 0.55 | 100 | 1.211 | 17.012 | 0.863 | 14.0 |
| 0.55 to 0.80 | 100 | 1.725 | 65.178 | 3.969 | 37.8 |

| reference price (bps of strike) | points | neural surrogate (5-member ensemble) | Turnbull-Wakeman / Levy | Curran (1994), exact threshold | Levy / surrogate |
|---|---|---|---|---|---|
| below 1 | 19 | 0.396 | 0.049 | 0.003 | 0.1 |
| 1 to 10 | 11 | 0.433 | 0.943 | 0.061 | 2.2 |
| 10 to 100 | 17 | 0.346 | 3.460 | 0.238 | 10.0 |
| 100 to 1,000 | 50 | 0.994 | 7.220 | 1.488 | 7.3 |
| 1,000 and above | 203 | 1.611 | 47.156 | 2.755 | 29.3 |

Levy has the lower RMSE where the reference price is below 1 bp (0.049 against 0.396 bps over 19 points). Where the true price is near zero Levy returns a price near zero, and the surrogate's Softplus output cannot emit zero, so its error there is a floor (`scripts/fullscale_ablation.py` documents the same floor).

### Notes

- The surrogate's signed error is positive at 225 of 300 points (bias +0.308 bps). Levy's bias is +17.662 bps and Curran's -1.038 bps.
- Curran with the exact threshold is a lower bound on the true price. Over the 291 points with a nonzero reference standard error it lies between -5.79 and +2.34 standard errors of the reference. A lower bound exceeds the reference only through reference noise, and the largest of 291 independent standard normal draws has median 2.82. Curran's first-order threshold differs from the exact solve by at most 0.3601 bps over the box.
- RMSE over this box is heavy-tailed. The five largest squared errors carry 30% of Levy's total squared error and 27% of the surrogate's, so the RMSE ratio moves with the draw; the bootstrap interval above is its spread over resamples of these 300 points.

## Grid at sigma = 0.25, 36 cells

Measured on 2026-09-20; AMD64, 16 logical CPUs, torch 2.11.0+cpu (CPU), numpy 2.5.3. Checkpoint `model.pt`, sha256 `2ca9e6ecc8979e70`. Stored under `grid` in `docs/approximation_benchmark.json`.

### Protocol

- Contract: arithmetic-average Asian call, strike 100, 50 equally spaced monitoring dates.
- Grid: moneyness S/K at 0.70, 0.82, 0.94, 1.06, 1.18, 1.30 x maturity at 0.10, 0.48, 0.86, 1.24, 1.62, 2.00 years (36 cells), sigma = 0.25, r = 0.04.
- Reference: `price_asian_mc`, 400,000 paths, antithetic sampling with the geometric-Asian control variate, seed 7. Reference standard error: mean 0.066 bps of strike, max 0.207 bps (cell m=1.30, T=2.00). Errors below that scale are not resolved by this reference.
- Surrogate: `PricingEngine` on `artifacts/model.pt`, 5 members, price only through `price_batch` with a batch of one (the float32 serving path). Greeks are not timed here.
- Latency: median of 20 calls per cell x 36 cells for the fast pricers; mean of one run per cell for the reference. Single process, `torch.set_num_threads(1)` to match the Dockerfile's `OMP_NUM_THREADS=1`, one warm-up call per pricer. Absolute times are specific to this machine and run; the ratios between rows are the comparable quantity.
- Timing conditions: system-wide CPU utilisation averaged 60% of 16 logical CPUs over the run, of which this benchmark is one thread. The machine ran on battery power.

### Results

Errors are (method - reference), in basis points of strike.

| method | mean abs error | max abs error | bias | worst cell | wall-clock per price |
|---|---|---|---|---|---|
| neural surrogate (5-member ensemble) | 0.719 bps | 2.045 bps | +0.712 bps | m=1.06, T=2.00 | 4.00 ms |
| Turnbull-Wakeman / Levy | 3.654 bps | 13.282 bps | +2.028 bps | m=1.18, T=2.00 | 312 us |
| Curran (1994), exact threshold | 0.104 bps | 0.345 bps | -0.104 bps | m=1.30, T=2.00 | 399 us |
| Curran (1994), linear threshold | 0.104 bps | 0.346 bps | -0.104 bps | m=1.30, T=2.00 | 364 us |
| Monte Carlo, 400,000 paths | (reference) | SE <= 0.207 bps | n/a | n/a | 946.06 ms |

Mean absolute error by maturity (bps of strike, averaged over the 6 moneyness points):

| maturity (y) | neural surrogate (5-member ensemble) | Turnbull-Wakeman / Levy | Curran (1994), exact threshold | Curran (1994), linear threshold |
|---|---|---|---|---|
| 0.10 | 0.915 | 0.150 | 0.003 | 0.003 |
| 0.48 | 0.522 | 1.152 | 0.024 | 0.024 |
| 0.86 | 0.208 | 2.552 | 0.058 | 0.058 |
| 1.24 | 0.638 | 4.147 | 0.118 | 0.119 |
| 1.62 | 0.893 | 6.033 | 0.182 | 0.183 |
| 2.00 | 1.140 | 7.890 | 0.239 | 0.240 |

Signed error per cell for the surrogate and for Curran (bps of strike; rows are maturity, columns moneyness):

neural surrogate (5-member ensemble)

| T \ S/K | 0.70 | 0.82 | 0.94 | 1.06 | 1.18 | 1.30 |
|---|---|---|---|---|---|---|
| 0.10 | +0.29 | +0.43 | +1.11 | +1.19 | +1.00 | +1.47 |
| 0.48 | +0.27 | +0.29 | +0.30 | +0.81 | +0.87 | +0.60 |
| 0.86 | +0.33 | +0.23 | +0.10 | +0.24 | -0.14 | +0.21 |
| 1.24 | +0.45 | +0.32 | +0.23 | +0.93 | +0.76 | +1.14 |
| 1.62 | +0.68 | +0.78 | +1.02 | +1.49 | +0.66 | +0.72 |
| 2.00 | +0.43 | +0.72 | +1.53 | +2.05 | +1.24 | +0.88 |

Curran (1994), exact threshold

| T \ S/K | 0.70 | 0.82 | 0.94 | 1.06 | 1.18 | 1.30 |
|---|---|---|---|---|---|---|
| 0.10 | +0.00 | -0.00 | +0.00 | -0.01 | -0.01 | -0.01 |
| 0.48 | -0.01 | -0.03 | -0.03 | -0.01 | -0.04 | -0.02 |
| 0.86 | -0.02 | -0.05 | -0.06 | -0.05 | -0.11 | -0.06 |
| 1.24 | -0.08 | -0.11 | -0.12 | -0.08 | -0.18 | -0.15 |
| 1.62 | -0.15 | -0.19 | -0.16 | -0.11 | -0.25 | -0.24 |
| 2.00 | -0.19 | -0.29 | -0.19 | -0.14 | -0.28 | -0.34 |

### Interpretation

On this grid the surrogate's mean absolute error is 0.72 bps of strike against 3.65 bps for Turnbull-Wakeman / Levy moment matching, a ratio of 5.1 (max 2.05 against 13.28 bps). The closed form's error is a bias that grows with maturity, from 0.15 bps at 0.10 y to 7.89 bps at 2.00 y.

Curran's conditioning approximation has a mean absolute error of 0.10 bps (max 0.34 bps) at 399 us per price against the surrogate's 4.00 ms. It is a lower bound on the true price, and it sits at most 0.3 reference standard errors above the Monte Carlo price. On price alone at this vol Curran is the more accurate pricer, by a factor of 6.9 in mean absolute error. The surrogate returns all five Greeks by automatic differentiation in one call and prices in batches, and its training recipe carries over to dynamics with no geometric-conditioning closed form, such as the rough-volatility model behind the 0DTE pricer.

### Notes

- The surrogate's signed error is positive in 35 of 36 cells (bias +0.71 bps). Over the trained box the bias is +0.31 bps on 300 points.
- Curran with the exact threshold, over the 35 cells where the reference has a nonzero standard error, lies between -3.42 and +0.33 standard errors of the Monte Carlo price. A lower bound exceeds the reference only through reference noise. Curran's first-order threshold differs from the exact solve by at most 0.0055 bps on this grid.
- At m=0.70, T=0.10 every one of the 400,000 reference paths pays zero, so the reference is 0 with zero standard error. Curran returns +0.0000 bps there. The surrogate returns +0.29 bps because its Softplus output cannot emit zero.
- Timing floor: Turnbull-Wakeman is two scalar `norm.cdf` calls plus a 50x50 exponential sum, and scipy's `norm.cdf` wrapper costs 105 us per scalar call in this run against 0.4 us for `scipy.special.ndtr`, so the two wrapper calls are about 67% of its 312 us. Curran makes two `norm.cdf` calls, one on a length-50 vector; the 399 us of the exact threshold against 364 us for the linear one is the Newton solve, the only code that differs between them. Switching the cdf primitive would speed up every closed form and change none of the accuracy columns. The timings keep `norm.cdf` because `backend/quant/benchmarks.py` and `asian_approx.py` call it.

## Reproduce

```bash
python scripts/benchmark_approximations.py --lhs 300 --seed 1992 --ref-paths 200000 --repeats 20 --bootstrap 10000
python scripts/benchmark_approximations.py --ref-paths 400000 --seed 7 --repeats 20
```
