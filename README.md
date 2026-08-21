# Neural Options Lab

**Live demo: [neural-options-lab.onrender.com](https://neural-options-lab.onrender.com)**. Free-tier hosting, so the first load after an idle spell takes about a minute while the container wakes and PyTorch loads. Every number on the dashboard is computed live by the models described below.

A neural network that prices arithmetic Asian options ~500x faster than Monte Carlo and 33x more accurately than the standard closed-form approximation, wrapped in an interactive dashboard you can run locally in two commands.

The project covers the full stack of a modern quant pricing system: the numerical methods that generate ground truth, the deep learning that learns to imitate them, a rough volatility model for same-day-expiry options, a reinforcement-style hedging agent, live market calibration, and a browser front end that ties it together. Trained model weights are included, so it runs the moment you clone it.

Built with PyTorch, FastAPI, and plain JavaScript with Plotly. No frontend build step.

## Why this is not trivial

Arithmetic Asian options have no closed-form price. The payoff depends on the average price over the option's life, so the standard way to value one is Monte Carlo simulation, which is accurate but slow. At a 100,000-path budget a single price takes roughly 100 milliseconds, and a trading desk needs thousands of prices and their risk sensitivities (the Greeks) refreshed continuously.

A neural network trained on Monte Carlo prices learns the pricing function itself. Once trained it prices the same contract in ~714 microseconds (p50) and returns all five Greeks as exact derivatives of the network through automatic differentiation, not finite differences. Price plus all Greeks together costs 8.4 ms at p50, because gamma needs a second backward pass through five ensemble members. That turns a batch job into something interactive.

The interesting part is doing this with enough numerical care that the surrogate is actually trustworthy: sub-2-basis-point pricing error, Greeks accurate enough to hedge with, and a separate model for the short-dated regime where the usual assumptions break down.

This README states measured numbers and retracts the ones that did not survive measurement. Where an earlier version overclaimed, the correction is left in place rather than quietly edited out.

## What it does

The dashboard has three sections.

**Pricing Lab.** Set the contract parameters or pull live market data for a ticker, and see the neural price next to a fresh Monte Carlo price with a confidence interval, all five Greeks, a convergence chart, a latency comparison, an error-distribution chart, a 3D price surface, and a feature-attribution breakdown of what is driving the price.

**Deep Hedging.** Simulate 3,000 paths of hedging a short option to expiry under transaction costs. A neural policy trained to minimize tail risk (conditional value at risk) is compared against the textbook Black-Scholes delta hedge on the same paths, with the full profit-and-loss distributions side by side.

**AI Risk Analyst.** Streams a plain-English risk summary built from the pricing, hedging, and attribution numbers. It uses an open-source Llama 3 model through Groq when an API key is present, and a deterministic offline summary otherwise, so the feature always works.

## Quickstart

You need Python 3.12. The trained checkpoints are in the repo, so you do not need to train anything to try it.

```bash
git clone https://github.com/Ronak-Mahajan/neural-options-lab.git
cd neural-options-lab
python -m pip install -r requirements.txt
python -m uvicorn backend.api.main:app --port 8000
```

Open http://localhost:8000. Interactive API docs are at http://localhost:8000/docs.

To run it in a container instead:

```bash
docker build -t neural-options-lab .
docker run -p 8000:8000 neural-options-lab
```

The live-news feature of the risk analyst and the live-quote features are optional. Set `GROQ_API_KEY` (free tier at console.groq.com) to enable the language model; without it the offline summary is used.

To host it somewhere public so it opens from a link instead of a local clone, see [DEPLOY.md](DEPLOY.md). The app is a single service that serves the API and the dashboard together, so any host that runs the Docker image works.

## How it works

| Piece | Approach |
|---|---|
| Monte Carlo engine | Antithetic sampling with a geometric-Asian control variate (Kemna and Vorst, 1990). Cuts the standard error by about 24x (24.0x at 5,000 paths, 24.5x at 20,000), measured as the ratio of empirical standard deviations across 300 seeded replications. An earlier version of this README claimed "about 30x"; that figure came from the reported standard error, which was computed as if antithetic pairs were independent and overstated the true error by ~45%. Both the estimator and the claim are fixed. |
| Parameterization | The network prices the unit-strike call as a function of moneyness (spot over strike). Option prices scale linearly in spot and strike, so one model covers every strike exactly. Puts come from Asian put-call parity, which is exact. |
| Architecture | A residual multilayer perceptron with SiLU activations and LayerNorm, about 134k parameters. Smooth activations matter here because the Greeks are computed by differentiating the network, and something like ReLU would give zero gamma almost everywhere. |
| Differential Machine Learning | Huge and Savine (2020). The pathwise delta and vega are computed on the same Monte Carlo paths that produce the price, for almost no extra cost, and the network is trained to match both the prices and their derivatives in a variance-normalized combined loss. This teaches the model the shape of the pricing function rather than only its level. |
| Deep ensemble | Five independently initialized networks. Averaging them lowers error, and the Greeks average cleanly through the mean. |
| Same-day-expiry (0DTE) model | Very short-dated option smiles show a power-law skew that classical models cannot reproduce. A rough Bergomi Monte Carlo engine generates training data for a separate ensemble serving maturities of 12 trading days or less. The driver is the Riemann-Liouville Volterra process (Bayer-Friz-Gatheral 2016) with Hurst index near 0.1, simulated exactly via the joint law of the driving Brownian motion and the Volterra integral. The 0DTE checkpoint has been regenerated against the corrected driver; it is currently UNCALIBRATED, because the prior calibration was fitted under the old kernel from market-closed quotes. |
| Live calibration | `calibrate.py` fits the rough volatility parameters to the SPY option smile using a vega-weighted Huber loss, global search with differential evolution, and a local polish. It can then regenerate the training set and retrain the 0DTE model on the calibrated dynamics. |
| Deep hedging | Buehler and coauthors (2019). A policy network maps the hedging state to a position and is trained to minimize the 95% conditional value at risk of the terminal loss, with transaction costs inside the objective. Benchmarked against a vol-matched Black-Scholes delta hedge **and** a cost-aware Whalley-Wilmott no-trade band on identical paths, under two measures. It loses to both out of sample; see the negative result below. |
| Market simulator for hedging | A Wasserstein GAN trained on historical SPY returns generates fat-tailed paths, mapped onto the pricing measure by enforcing the terminal variance and the martingale condition. Known limitation: the shipped generator is mode-collapsed (participation ratio 4.66 of 30 factors), so its paths are forecastable and it is not a sound measure for evaluating a hedging policy. Quantified in `docs/hedging_findings.md`. |
| Real market data | A live [Deribit](https://www.deribit.com) option chain: 836 BTC instruments across 12 expiries, fetched from public endpoints with no API key. Coin-denominated premiums are converted on the per-expiry forward, implied vol is inverted on BOTH bid and ask so the market shows as a band, and the surface is checked for butterfly, vertical, calendar and put-call-parity arbitrage. Everything downstream reads a committed snapshot, so it runs offline. See [`docs/real_market_data.md`](docs/real_market_data.md). |
| Explainability | Integrated Gradients through the ensemble against an at-the-money baseline, with the completeness check (attributions sum to the price difference) reported alongside. |

## Results

Everything below is measured, not asserted. The evaluation script prices an independent test set against high-precision (200,000-path) Monte Carlo references.

Main pricer, 5-member ensemble, 600 test points:

| Quantity | Error (RMSE) |
|---|---|
| Price | 1.4 basis points of strike |
| Delta | 7.1e-4 |
| Vega | 13e-4 |

### That 1.4 bp is not a noise floor, it is a fixable bias

An earlier version of this README explained the 1.4 bp away as the label noise floor. That
was wrong, and finding out why produced the most interesting result in the project.

On 600 held-out points, **89.3% of price errors were positive** and the mean accounted for
**47.6% of total MSE**. Bucketing by true price magnitude showed a flat additive offset
(+1.05, +1.03, +1.06, +1.20, +0.89 bps across five decades of price), and at points where
the true price is below 0.01 bps the network still predicted ~1.08 bps and never went below
0.898. The cause is the output layer: `nn.Softplus()` cannot emit zero, so it floors at
about 1 bp and lifts the whole surface.

Two fixes were tested at full scale (500,000 labels, 5,000 paths each, 400 epochs,
5-member ensembles, both arms identical except for the one change under test; see
`scripts/fullscale_ablation.py` and `artifacts/ablation.json`):

| model | RMSE | bias | % positive | bias²/MSE |
|---|---|---|---|---|
| previous `model.pt` | 1.489 bps | +0.966 | 89.3% | 42.1% |
| **conditioned head (now served)** | **1.301 bps** | **+0.404** | 78.1% | **9.7%** |
| residual over geometric Asian | 1.823 bps | +0.925 | 86.8% | 25.7% |

The conditioned head is now the served checkpoint. Promotion is gated:
`scripts/promote_model.py` re-prices a fresh 1,500-point test set against
200,000-path references on a seed used by neither training nor the ablation, and
writes `model.pt` only if the candidate beats the incumbent on **both** RMSE and
|bias|. The previous checkpoint is kept as `model_legacy_unconditioned_head.pt`.
Swapping a served model on a training-time validation number is how the old one
came to carry a +0.99 bp bias that this README described as an irreducible noise
floor.

**What the promotion actually bought, and what it cost.** Paired comparison, both
models priced on the same 1,500 points against the same references:

| | price RMSE | price bias | delta RMSE | vega RMSE |
|---|---|---|---|---|
| legacy (unconditioned head) | 1.531 | +1.032 | **7.338e-4** | **17.757e-4** |
| promoted (conditioned head) | **1.366** | **+0.468** | 7.768e-4 | 17.978e-4 |

Price RMSE improves 10.8% and the systematic bias halves. **Delta gets 5.9%
worse and vega 1.2% worse.** That is a real regression and it is stated rather
than buried: conditioning the output head helps the level and slightly hurts the
shape, which is what you would expect from changing where the magnitude lives in
a network trained on a joint price-and-derivative loss. The promotion is kept
because price accuracy is this model's primary claim, but a service that hedges
off these Greeks should weigh that differently.

The first version of the gate tested price only, so it did not see the Greeks
regression at all. It now reports delta and vega alongside; they are reported,
not blocking, and the reason is written into the script.

Two further caveats. The gain is concentrated in the systematic component: p95
absolute error is essentially unchanged (2.609 vs 2.595 bps) and the worst case
is marginally wider (10.34 vs 10.20). And the absolute RMSE is heavy-tailed
enough that it moves with the test draw: the same promoted checkpoint measures
1.301, 1.329, 1.366 and 1.488 bps on four independent Latin-hypercube draws. The
*paired* comparison above is the meaningful one, because both models see
identical points and identical references.

*Head conditioning*, which carries the output magnitude in a fixed scale with the Softplus head
initialized near unity, **halved the systematic bias**. The previous initialization started
every run at `softplus(0) = 0.693`, i.e. 6,930 bps against a mean price of 3,664 bps.

*Residual over geometric* did **not** work, and that is a real result. Since AM–GM gives
`C_arith ≥ C_geo` pathwise, learning only the residual should have shrunk what the Softplus
floor can distort. It made things 37% worse, because the residual has a 3.8x wider relative
dynamic range (p99/p50 of 12.46 versus 3.27) and that outweighs the 21x smaller output
scale. Hypothesis tested, hypothesis refuted.

### Is a neural surrogate even the right tool?

Arithmetic Asians have had fast closed-form approximations since the early 1990s, so the
honest comparison is not only against Monte Carlo. Against Levy (1992) moment matching, on
300 points versus 200,000-path references:

| method | RMSE | bias | p95 abs err | latency |
|---|---|---|---|---|
| neural ensemble | 1.329 bps | +0.548 | 2.376 | 714 µs p50 |
| Levy moment matching | 44.344 bps | +19.813 | 103.300 | 56 µs |
| Monte Carlo, 200k paths | (reference) | n/a | n/a | 357,000 µs |

**33x more accurate** than the closed form at 13x its cost, and 500x faster than Monte
Carlo, a genuine point on the speed/accuracy frontier that neither alternative occupies.

The one regime where Levy still wins is where the true price is essentially zero
(0.082 vs 0.310 bps), which is the Softplus floor seen from an independent direction. Note
that gap narrowed by more than 3x when the head was conditioned (the floor shrank from
1.050 to 0.310 bps), which is corroboration from a completely different measurement that
the bias diagnosis was right.

### Latency, honestly

Measured on this machine, best of 200 calls: a single price **plus all Greeks** is
**8.4 ms at p50** (9.3 ms p95, 9.8 ms p99). The Greeks path does a double backward pass for
gamma across five ensemble members. An earlier README claim of "roughly a millisecond for a
single price plus all Greeks" was off by 8x; ~714 µs is the price-only figure. Batch
throughput does hold up: **487,843 prices/sec** at a batch of 10,000.

Label generation runs on the GPU in float64 (`backend/quant/gpu_labels.py`): 0.85 G
path-steps/s on an RTX 5080, so the full 500,000-label dataset takes 148 s instead of roughly
80 minutes on CPU. float64 is not optional: running the same kernel at float32 with
identical seeds injects 10.03 bps of price RMSE at 5,000 paths, several times the entire
error budget, because the control variate differences two deliberately near-identical
quantities. Serving stays on CPU: at a batch of 1 the GPU is *slower* (1,813 µs vs 1,336 µs),
being kernel-launch bound.

A controlled ablation (same sample budget, training on prices only versus the differential loss) cut delta error about 3x and vega error about 4x, which is the whole point of differential machine learning: better sensitivities for hedging.

### The 0DTE model needs regenerating, and its calibration was not live

Two defects here, both found by audit and both invalidating claims this README
previously made.

**The driver was the wrong process.** `rough_vol.py` built the Type-I
(Mandelbrot–Van Ness) fractional Brownian covariance
`0.5(t_i^2H + t_j^2H − |t_i−t_j|^2H)`. Rough Bergomi is driven by the
Riemann–Liouville Volterra process `W̃_t = √(2H)∫₀ᵗ(t−s)^(H−½)dW_s`. The two agree
on the diagonal (both give `Var[W̃_t] = t^2H`, which is why the martingale property
held and nothing looked wrong) and agree nowhere else: at H = 0.1172 the maximum
off-diagonal relative difference is **4.93**, and `corr(W̃_t1, W̃_t50)` was **+0.320**
against a true **+0.054**.

There was a second half to it. `chol(C)` is not the Volterra kernel, because W̃ is a
continuous stochastic integral rather than a linear function of n coarse increments.
Factorising C alone forces `corr(Z_1, W̃_t1) = 1` by construction when the truth is
`√(2H)/(H+½) = 0.7844`, so the leverage correlation ρ was being applied to the wrong
object, over-correlating spot and vol precisely at the short end where a 0DTE skew
fit is identified. Both are now fixed with the exact joint-Gaussian scheme, verified
against quadrature to 5.4e-08 and against 400,000 draws.

**`artifacts/model_0dte.pt` has been regenerated** against the corrected driver. The
previous checkpoint is kept as `model_0dte_legacy_wrong_kernel.pt` for comparison. The
old "about 2 basis points against its rough Bergomi teacher" figure described agreement
with the *wrong* teacher and has been replaced by a measurement against the right one:

| 0DTE ensemble, 400 held-out points vs 500,000-path references | |
|---|---|
| RMSE | **1.48 bps of strike** |
| bias | **+0.13 bps** |
| p95 abs error | 2.76 bps |
| training-label noise floor (20,000 paths/label) | 2.35 bps |

The surrogate sits below its own per-label noise, which is what least-squares over
25,000 noisy labels should achieve. Validation RMSE against those noisy labels is
3.5 bps and overstates the true error by 2.4x, the same gap the main pricer shows,
and the reason this project scores against high-precision references rather than
against its own training targets.

**The calibration was not live.** This README previously said the model was
"calibrated to the live SPY smile ... about 2 volatility points across 72 quotes and
two expiries." The committed `artifacts/rough_calibration.json` reads
`as_of 2026-07-09T03:43:16-04:00` (03:43 New York, market closed) with
`quote_source: "last_trade_market_closed"`, and `accepted: true`. Those exact
parameters (H 0.1172, η 2.1384, ρ −0.7387) are what `model_0dte.pt` carries. A
market-closed fit passed the quality gate and was trained into the served model,
because the gate tested only RMSE and bound-pinning and never looked at staleness.
Related: in the market-closed branch time-to-expiry was stamped from `now` while the
prices were the previous session's last trades, which on synthetic quotes with known
truth inflated √ξ by 21% and moved H by 0.021.

### Deep hedging: a negative result

**This section previously claimed the learned hedger "reduced 95% tail loss by roughly 30%
versus delta hedging." That claim was wrong and has been retracted.** It was measured
in-sample, against a handicapped baseline, on a measure that was not a valid pricing
measure. Full write-up in [`docs/hedging_findings.md`](docs/hedging_findings.md).

Three problems, all measured:

1. **The simulated measure was not risk-neutral.** `risk_neutralize` matched per-step
   marginals but left cross-step covariance free, so paths realized 1.28x the requested
   volatility and discounted spot was not a martingale (E[S_T] exceeded e^{rT} by up to
   147 bps). The option was booked at Black–Scholes while being worth ~30% more under the
   measure actually being simulated. Now fixed: terminal variance and the martingale
   condition are both enforced exactly, and the premium is priced under the simulated
   measure.
2. **The baseline was handicapped**, hedging at the requested σ while paths realized 1.28σ.
   On the old measure, vol-matching alone closed ~83% of the claimed gap.
3. **The comparison was in-sample on a degenerate generator.** The WGAN is mode-collapsed
   (participation ratio 4.66 of 30 factors), making its paths forecastable: R² = 0.8755
   regressing future returns on realized ones, versus 0.0006 for GBM. Minimizing CVaR there
   rewards market timing, not hedging.

After fixing 1 and 2 and adding a cost-aware Whalley–Wilmott baseline, evaluated
out-of-sample on risk-neutral GBM over a 12-cell (σ, cost) grid, 15,000 paths per cell:

| policy | beats vol-matched delta | median ratio | beats Whalley–Wilmott |
|---|---|---|---|
| trained on the fixed GAN measure | 2 / 12 | 1.469 | 0 / 12 |
| **trained on GBM (in-sample!)** | **5 / 12** | **1.085** | **1 / 12** |

A ratio above 1 means worse tail loss. Even trained and evaluated on the same correct
measure, the learned policy loses to a vol-matched delta hedge in 7 of 12 cells and to
Whalley–Wilmott in 11 of 12. **As implemented, deep hedging here does not beat a properly
specified baseline.** `compare()` now reports both measures side by side with bootstrap
standard errors and defaults its headline to the out-of-sample one.

### Real market data

Everything else in this project is simulated. `backend/quant/deribit.py` and
`backend/quant/surface.py` are the exception: a live institutional option chain, and the
diagnostics you would actually run on one.

Measured on the committed snapshot (836 BTC options, 12 expiries, 0.42 to 323 days):

**The convention matters more than the code.** Deribit quotes premiums in BTC and reports a
per-expiry forward, not spot. Reading `price_usd = price_btc x forward` and inverting
Black-76 reproduces Deribit's own published `mark_iv` to a median of **0.0028 vol points**.
Reading the coin premium as a dollar price instead gives 4.62 vol points where the truth is
32.88, **wrong by 7.1x**, and wrong in a way that still produces a smooth, plausible
surface. The day count was pinned the same way: ACT/365 reproduces `mark_iv` to +0.0001 vol
points, against -0.2727 for a 360-day year.

**A real chain is mostly unusable.** 289 of 836 quotes are flagged and dropped: 199 with a
bid below the no-arbitrage floor, 118 with no volume or open interest, 66 one-sided, 56 with
vega too small for the IV inversion to mean anything. The IV bid-ask on what survives has a
median of **1.36 vol points** and a 95th percentile of 9.79.

**No static arbitrage survives the spread.** Of 499 butterfly triples, 523 vertical pairs, 11
calendar pairs and 195 parity strikes, 51 violations appear on mid prices, and **zero are
executable** once you require crossing the actual bid-ask, before fees. Reporting the
mid-price count as "arbitrage found" would have been the easy, wrong answer.

**The conversion chain checks out end to end.** A forward backed out of put-call parity
agrees with the listed BTC future to a median of under **1 basis point** across all 12
expiries, which simultaneously validates the parity map, the day count and the coin-to-dollar
conversion.

## Repository layout

```
backend/
  quant/
    monte_carlo.py    Asian Monte Carlo engine: antithetic sampling, control variate, parity
    dataset.py        Training-set generation with pathwise Greeks
    model.py          Residual MLP
    train.py          Main pricer training (differential ML, ensembles)
    engine.py         Serving engine: autograd Greeks, batched pricing, benchmarks
    rough_vol.py      Rough Bergomi Monte Carlo engine (0DTE teacher)
    dataset_0dte.py   0DTE dataset generation
    train_0dte.py     0DTE ensemble training
    calibrate.py      Live smile calibration of the rough vol parameters
    hedging.py        Deep hedging policy: training and simulation
    generative.py     WGAN market simulator with risk-neutral correction
    explain.py        Integrated Gradients
    evaluate.py       Accuracy measurement against high-precision references
    drift_monitor.py  Compares the deployed model to live quotes, triggers retrain
    market_data.py    yfinance adapter (spot, realized volatility, T-bill rate)
    llm.py            Risk-report streaming with offline fallback
  api/main.py         FastAPI endpoints and static file serving
frontend/             Dashboard (HTML, CSS, vanilla JS, Plotly)
artifacts/            Trained checkpoints and evaluation results
tests/                Test suite
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks the parts that are easy to get subtly wrong: Asian put-call parity for both the Monte Carlo engine and the neural surrogate, that the control variate reduces variance without biasing the price, and that the autograd Greeks match finite-difference perturbations.

## Retraining from scratch

The committed models let the app run immediately. To rebuild them:

```bash
# Main pricer (about an hour on CPU; add --quick for a fast smoke run)
python -m backend.quant.train --samples 500000 --paths 5000 --epochs 400 --ensemble 5
python -m backend.quant.evaluate

# Hedging policy and 0DTE model
python -m backend.quant.hedging --iters 8000
python -m backend.quant.train_0dte --ensemble 5 --epochs 500

# Optional: calibrate the 0DTE dynamics to the live market and retrain
python -m backend.quant.calibrate --retrain
```

There is also a drift monitor (`backend/quant/drift_monitor.py`) that compares the deployed 0DTE model against live quotes and kicks off recalibration and retraining if the error crosses a threshold, gated on the test suite passing. That is the automation loop that keeps the model current with the market.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/price` | Surrogate price and Greeks against a Monte Carlo price with confidence interval |
| `POST /api/convergence` | Monte Carlo estimate versus path count against the surrogate price |
| `POST /api/surface` | Batched price surface over moneyness and maturity |
| `POST /api/hedge` | Deep hedge versus delta hedge profit-and-loss distributions |
| `POST /api/explain` | Integrated Gradients attributions |
| `POST /api/risk-report` | Streamed text risk report |
| `GET /api/market/{ticker}` | Live spot, realized volatility, risk-free rate |
| `GET /api/model-info` | Architecture and measured accuracy |

The Monte Carlo benchmark switches with the pricing regime automatically: Asian under geometric Brownian motion above 12 trading days to expiry, rough Bergomi at or below.

## Honest limitations

This is a research and portfolio project, not production trading infrastructure. Market data comes from yfinance, which is retail-grade and has stale or missing quotes. The pricer takes a single flat volatility rather than a full surface. The rough volatility model uses a fixed Hurst index rather than calibrating it jointly with the smile. There is no arbitrage-free constraint baked into the learned surface. Nothing here is investment advice.

## License

MIT. See [LICENSE](LICENSE).
