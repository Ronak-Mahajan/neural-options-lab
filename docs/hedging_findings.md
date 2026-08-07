# Deep hedging: a negative result

**Summary.** This repository previously claimed the learned CVaR hedger "reduced 95% tail
loss by roughly 30% versus delta hedging." That claim was wrong. It was measured in-sample
against a handicapped baseline on a measure that was not a valid pricing measure. After
fixing all three problems, **the deep hedger does not beat a properly specified baseline**,
and the honest headline is a negative result.

Everything below was measured by running the code in this repository. Reproduce with
`python _hedgeresult.py` and `python _hedgefinal.py`.

---

## 1. What was wrong

### 1.1 The simulated measure was not a pricing measure

`risk_neutralize` standardized the generated log-returns cross-sectionally **per time
step**, which pins each step's marginal mean and standard deviation exactly. But a pricing
measure is a property of the joint law. Cross-step covariance was unconstrained, and the
generator has a lot of it.

| σ requested | realized terminal vol | ratio | E[S_T] − e^{rT} | MC call vs BS call |
|---|---|---|---|---|
| 0.08 | 0.1029 | 1.286 | +2.50 bps | +24.6% |
| 0.20 | 0.2571 | 1.285 | +15.57 bps | +28.7% |
| 0.30 | 0.3846 | 1.282 | +34.62 bps | +30.4% |
| 0.45 | 0.5729 | 1.273 | +75.49 bps | +32.2% |
| 0.65 | 0.8164 | 1.256 | +147.13 bps | +33.6% |

Discounted spot was not a martingale, and the option was booked at the Black–Scholes
premium while being worth ~30% more under the measure actually simulated. Both hedgers
were short a mispriced option, which is why both showed large negative mean P&L.

**Fixed.** `risk_neutralize` now (a) rescales so terminal log-variance equals σ²T exactly,
and (b) applies a deterministic per-step shift so E[S_i] = e^{r·t_i} at every step.
Subtracting a constant from log S_i leaves its variance untouched, so the two corrections
do not conflict. After the fix:

| σ | realized vol ratio | E[S_T] − e^{rT} | MC call vs BS call |
|---|---|---|---|
| 0.08 | 1.0000 | −0.00 bps | −0.74% |
| 0.20 | 1.0000 | 0.00 bps | −0.81% |
| 0.65 | 1.0000 | −0.00 bps | −0.59% |

The residual −0.8% is genuine, not error: the generator is fat-tailed, so an ATM call under
a variance-matched non-Gaussian measure is worth slightly less than Black–Scholes. The book
now prices the premium by Monte Carlo **under the measure being simulated**, so it is
self-consistent either way.

*Honest limitation:* only the terminal variance is pinned. Intermediate variances
Var[log S_i], i < N, are not separately constrained — doing so would require fixing the
full covariance structure and would destroy the dependence the generator exists to provide.

### 1.2 The baseline was handicapped

The delta hedge used the caller's σ while the paths realized 1.28× that volatility.
Hedging at the wrong vol is a known way to lose money in the tail. On the *old* measure,
vol-matching the baseline closed ~83% of the reported gap by itself.

After fixing the measure this handicap disappears on its own — realized vol now equals the
requested σ, so the vol-matched and naive delta hedges agree to within noise (0.01104 vs
0.01100 at σ=0.15, cost=0.001). The bug was a symptom of 1.1.

### 1.3 The comparison was in-sample, and the generator is degenerate

The policy was trained and evaluated on the same WGAN measure. That measure is
mode-collapsed — participation ratio **4.66 of 30** factors, top principal component
carrying **41.2%** of variance (an i.i.d. Gaussian reference gives 29.96 and 3.6%). The
paths are therefore forecastable: regressing the remaining log return on the realized ones
gives **R² = 0.8755 on GAN paths versus 0.0006 on GBM**. Spot predicts the future, so a
spot-conditioned holding is a directional bet and minimizing CVaR under that measure
rewards market timing rather than hedging.

This is a property of the shipped generator. It is documented, not fixed — retraining the
WGAN to be non-degenerate is a separate project.

---

## 2. The honest result

Baselines: **vol-matched delta** and **Whalley–Wilmott** (1997) no-trade band, whose
risk-aversion parameter is tuned in-sample *for the baseline* — deliberately generous,
because the point is to find the strongest honest baseline the learned policy must beat.
The deep policy sees the transaction-cost parameter in its state, so comparing it against a
cost-blind delta hedge is not a fair fight.

12-cell grid, σ ∈ {0.15, 0.20, 0.30, 0.45} × cost ∈ {0.001, 0.005, 0.010}, 15,000 paths per
cell (5 seeds × 3,000), CVaR₉₅ with bootstrap standard errors.

**Evaluated on risk-neutral GBM (out-of-sample):**

| policy | beats vol-matched delta | median ratio | beats Whalley–Wilmott | median ratio |
|---|---|---|---|---|
| legacy (trained on broken measure) | 1 / 12 | 1.443 | 0 / 12 | 1.561 |
| retrained on the fixed GAN measure | 2 / 12 | 1.469 | 0 / 12 | 1.600 |
| **retrained on GBM (in-sample)** | **5 / 12** | **1.085** | **1 / 12** | 1.153 |

A ratio above 1 means the deep hedger's tail loss is *worse*. Even the policy trained and
evaluated on the same correct measure — the most favourable setup available — loses to a
vol-matched delta hedge in 7 of 12 cells and to Whalley–Wilmott in 11 of 12.

**Conclusion: as implemented, the learned hedger does not beat a properly specified
baseline.** The earlier 30% figure came entirely from evaluating in-sample on a degenerate,
non-martingale measure against a handicapped benchmark.

## 3. What would be needed to revisit this

The negative result is about *this* policy, not about deep hedging in general. Plausible
reasons it underperforms, none of which have been tested here: only 6,000 training
iterations; a 2-hidden-layer, width-64 policy; holdings hard-clamped to [0, 1.5]; a crude
conditional CVaR head; and daily rebalancing that leaves little room for a learned policy to
beat a band rule. Any future claim should report out-of-sample GBM numbers against a
Whalley–Wilmott baseline with error bars, which `compare()` now does by default.
