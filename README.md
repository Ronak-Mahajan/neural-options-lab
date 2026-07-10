# Neural Options Lab

A neural network that prices options in about a millisecond, trained to match a Monte Carlo engine that takes a hundred times longer, wrapped in an interactive dashboard you can run locally in two commands.

The project covers the full stack of a modern quant pricing system: the numerical methods that generate ground truth, the deep learning that learns to imitate them, a rough volatility model for same-day-expiry options, a reinforcement-style hedging agent, live market calibration, and a browser front end that ties it together. Trained model weights are included, so it runs the moment you clone it.

Built with PyTorch, FastAPI, and plain JavaScript with Plotly. No frontend build step.

## Why this is not trivial

Arithmetic Asian options have no closed-form price. The payoff depends on the average price over the option's life, so the standard way to value one is Monte Carlo simulation, which is accurate but slow. At a 100,000-path budget a single price takes roughly 100 milliseconds, and a trading desk needs thousands of prices and their risk sensitivities (the Greeks) refreshed continuously.

A neural network trained on Monte Carlo prices learns the pricing function itself. Once trained it prices the same contract in about a millisecond and returns all five Greeks as exact derivatives of the network through automatic differentiation, not finite differences. That turns a batch job into something interactive.

The interesting part is doing this with enough numerical care that the surrogate is actually trustworthy: sub-2-basis-point pricing error, Greeks accurate enough to hedge with, and a separate model for the short-dated regime where the usual assumptions break down.

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

## How it works

| Piece | Approach |
|---|---|
| Monte Carlo engine | Antithetic sampling with a geometric-Asian control variate (Kemna and Vorst, 1990). Cuts the standard error by about 30x, which makes generating accurate training labels cheap. |
| Parameterization | The network prices the unit-strike call as a function of moneyness (spot over strike). Option prices scale linearly in spot and strike, so one model covers every strike exactly. Puts come from Asian put-call parity, which is exact. |
| Architecture | A residual multilayer perceptron with SiLU activations and LayerNorm, about 134k parameters. Smooth activations matter here because the Greeks are computed by differentiating the network, and something like ReLU would give zero gamma almost everywhere. |
| Differential Machine Learning | Huge and Savine (2020). The pathwise delta and vega are computed on the same Monte Carlo paths that produce the price, for almost no extra cost, and the network is trained to match both the prices and their derivatives in a variance-normalized combined loss. This teaches the model the shape of the pricing function, not just its level. |
| Deep ensemble | Five independently initialized networks. Averaging them lowers error, and the Greeks average cleanly through the mean. |
| Same-day-expiry (0DTE) model | Very short-dated option smiles show a power-law skew that classical models cannot reproduce. A rough Bergomi Monte Carlo engine (fractional Brownian motion, Hurst index near 0.1) generates training data for a separate ensemble that serves maturities of 12 trading days or less. |
| Live calibration | `calibrate.py` fits the rough volatility parameters to the live SPY option smile using a vega-weighted Huber loss, global search with differential evolution, and a local polish. It can then regenerate the training set and retrain the 0DTE model on the calibrated dynamics. |
| Deep hedging | Buehler and coauthors (2019). A policy network maps the hedging state to a position and is trained to minimize the 95% conditional value at risk of the terminal loss, with transaction costs inside the objective. The benchmark is the exact Black-Scholes delta hedge on identical paths. |
| Market simulator for hedging | A Wasserstein GAN trained on historical SPY returns generates fat-tailed paths, corrected onto the pricing measure by moment matching so the hedging policy learns hedging rather than the generator's drift. |
| Explainability | Integrated Gradients through the ensemble against an at-the-money baseline, with the completeness check (attributions sum to the price difference) reported alongside. |

## Results

Everything below is measured, not asserted. The evaluation script prices an independent test set against high-precision (200,000-path) Monte Carlo references.

Main pricer, 5-member ensemble, 600 test points:

| Quantity | Error (RMSE) |
|---|---|
| Price | 1.4 basis points of strike |
| Delta | 7.1e-4 |
| Vega | 13e-4 |

For context, the label noise floor of the Monte Carlo references is itself around 1 basis point, so the pricer is close to the best it could be given its training data. Inference is roughly a millisecond for a single price plus all Greeks, and hundreds of thousands of prices per second in batch.

A controlled ablation (same sample budget, training on prices only versus the differential loss) cut delta error about 3x and vega error about 4x, which is the whole point of differential machine learning: better sensitivities for hedging.

The 0DTE ensemble prices to about 2 basis points against its rough Bergomi teacher. Calibrated to the live SPY smile it reached a fit of about 2 volatility points across 72 quotes and two expiries. The deep hedging policy reduced 95% tail loss by roughly 30% versus delta hedging at lower transaction cost.

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
