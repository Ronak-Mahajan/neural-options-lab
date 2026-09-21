# Deep hedging: a negative result

**Summary.** Three controls decide whether a deep hedging comparison means anything: the
simulated measure has to be a valid pricing measure, the baseline has to see the same
realized volatility as the policy, and the evaluation has to be out of sample. With all
three in place, **the deep hedger does not beat a properly specified baseline**, and the
headline is a negative result.

Everything below was measured by running the code in this repository. The committed record
of the run is `artifacts/hedging_final.json` (the 12-cell CVaR ratios against delta and
Whalley-Wilmott for all three policies; the table in section 2 is read from it) and
`artifacts/hedging_honest.json` (24 rows: the per-cell CVaR₉₅ of deep, delta, naive and
Whalley-Wilmott on the GBM and GAN measures, with bootstrap standard errors). The
measurement behind both is `HedgingEngine.compare()` in `backend/quant/hedging.py`, which regenerates any cell of
the grid from a checkpoint, a (σ, cost) pair and a seed. The follow-up experiments are
fully scripted: `scripts/deep_hedging_regimes.py` produces
[deep_hedging_regimes.md](deep_hedging_regimes.md) and `scripts/hedge_real_paths.py`
produces [hedging_real_paths.md](hedging_real_paths.md).

---

## 1. What the comparison has to control

### 1.1 The simulated measure has to be a pricing measure

Standardizing the generated log-returns cross-sectionally **per time step** pins each
step's marginal mean and standard deviation exactly, but a pricing measure is a property of
the joint law. Under that treatment cross-step covariance is unconstrained, and this
generator has a lot of it.

| σ requested | realized terminal vol | ratio | E[S_T] − e^{rT} | MC call vs BS call |
|---|---|---|---|---|
| 0.08 | 0.1029 | 1.286 | +2.50 bps | +24.6% |
| 0.20 | 0.2571 | 1.285 | +15.57 bps | +28.7% |
| 0.30 | 0.3846 | 1.282 | +34.62 bps | +30.4% |
| 0.45 | 0.5729 | 1.273 | +75.49 bps | +32.2% |
| 0.65 | 0.8164 | 1.256 | +147.13 bps | +33.6% |

Discounted spot is then not a martingale, and an option booked at the Black-Scholes
premium is worth about 30% more under the measure actually simulated. Both hedgers are
short a mispriced option, which is what drives their large negative mean P&L.

**The control.** `risk_neutralize` (a) rescales so terminal log-variance equals σ²T exactly,
and (b) applies a deterministic per-step shift so E[S_i] = e^{r·t_i} at every step.
Subtracting a constant from log S_i leaves its variance untouched, so the two corrections
do not conflict. After the fix:

| σ | realized vol ratio | E[S_T] − e^{rT} | MC call vs BS call |
|---|---|---|---|
| 0.08 | 1.0000 | −0.00 bps | −0.74% |
| 0.20 | 1.0000 | 0.00 bps | −0.81% |
| 0.65 | 1.0000 | −0.00 bps | −0.59% |

The residual −0.8% is genuine, not error: the generator is fat-tailed, so an ATM call under
a variance-matched non-Gaussian measure is worth slightly less than Black-Scholes. The book
now prices the premium by Monte Carlo **under the measure being simulated**, so it is
self-consistent either way.

*Limitation:* only the terminal variance is pinned. Intermediate variances
Var[log S_i], i < N, are not separately constrained; doing so would require fixing the full covariance structure and would destroy the dependence the generator exists to provide.

### 1.2 The baseline has to see the same realized volatility

A delta hedge run at the caller's σ against paths that realize 1.28 times that volatility
is hedging at the wrong vol, which is a known way to lose money in the tail. On the
unconstrained measure, vol-matching the baseline closes about 83% of the apparent gap by
itself.

Once the measure is controlled the handicap disappears on its own: realized vol equals the
requested σ, so the vol-matched and naive delta hedges agree to within noise (0.01104 vs
0.01100 at σ=0.15, cost=0.001). This one is a symptom of 1.1.

### 1.3 The evaluation has to be out of sample

Training and evaluating the policy on the same WGAN measure does not test it. That measure
is mode-collapsed: participation ratio **4.66 of 30** factors, top principal component
carrying **41.2%** of variance (an i.i.d. Gaussian reference gives 29.96 and 3.6%). The
paths are therefore forecastable: regressing the remaining log return on the realized ones
gives **R² = 0.8755 on GAN paths versus 0.0006 on GBM**. Spot predicts the future, so a
spot-conditioned holding is a directional bet and minimizing CVaR under that measure
rewards market timing rather than hedging.

This is a property of the shipped generator, which is why the headline result below is
evaluated on risk-neutral GBM instead. Retraining the WGAN to be non-degenerate is a
separate project.

---

## 2. The result

Baselines: **vol-matched delta** and **Whalley-Wilmott** (1997) no-trade band, whose
risk-aversion parameter is tuned in-sample *for the baseline*, deliberately generous, because the point is to find the strongest honest baseline the learned policy must beat.
The deep policy sees the transaction-cost parameter in its state, so comparing it against a
cost-blind delta hedge is not a fair fight.

12-cell grid, σ ∈ {0.15, 0.20, 0.30, 0.45} × cost ∈ {0.001, 0.005, 0.010}, 15,000 paths per
cell (5 seeds × 3,000), CVaR₉₅ with bootstrap standard errors.

**Evaluated on risk-neutral GBM (out-of-sample):**

| policy | beats vol-matched delta | median ratio | beats Whalley-Wilmott | median ratio |
|---|---|---|---|---|
| trained on the unconstrained measure | 1 / 12 | 1.443 | 0 / 12 | 1.561 |
| trained on the controlled GAN measure | 2 / 12 | 1.469 | 0 / 12 | 1.600 |
| **trained on GBM (in-sample)** | **5 / 12** | **1.085** | **1 / 12** | 1.153 |

A ratio above 1 means the deep hedger's tail loss is *worse*. Even the policy trained and evaluated on the same correct measure (the most favourable setup available) loses to a vol-matched delta hedge in 7 of 12 cells and to Whalley-Wilmott in 11 of 12.

**Conclusion: as implemented, the learned hedger does not beat a properly specified
baseline.** A comparison that skips any of the three controls in section 1 reverses that
conclusion, which is why they are stated first.

## 3. What would be needed to revisit this

The negative result is about *this* policy, not about deep hedging in general. Plausible
reasons it underperforms, none of which have been tested here: only 6,000 training
iterations; a 2-hidden-layer, width-64 policy; holdings hard-clamped to [0, 1.5]; a crude
conditional CVaR head; and daily rebalancing that leaves little room for a learned policy to
beat a band rule. Any claim from this line of work reports out-of-sample GBM numbers against a
Whalley-Wilmott baseline with error bars, which `compare()` does by default.
