# Joint (H, η) refit of rough Bergomi against the smiles and the ATM skew term structure

**Summary.** The served SPY calibration (η 3.94 at its bound, ρ −0.54, H 0.26) fits single-day
smiles well but its at-the-money skew steepens into expiry with exponent −0.33 (Monte Carlo,
1–45 trading days), 2.5 market standard errors away from the market's −0.249 ± 0.033. Adding the
skew term structure to the calibration objective moves the optimum to **H 0.31, ρ −0.50,
√ξ 12.5 %** (η still at the 4.0 bound) and puts the Monte Carlo exponent at **−0.264 ± 0.007**,
within half a market standard error, for a cost of **0.45 vol points** of smile RMSE
(1.31 → 1.76, pricing-map RMSE, mean of three captures). An interior-η solution (η 3.39,
H 0.30) reaches the same exponent (−0.263 ± 0.004) for about 0.15 vol points more. The
recommended parameters are written to `artifacts/rough_calibration_skewjoint.json` as an
analysis artifact; the served calibration and the 0DTE surrogate trained on it are unchanged.

Everything below was produced by `scripts/joint_skew_refit.py` (four staged runs, 680 s of
wall-clock on 16 CPU threads, no GPU; every stage checkpoints to JSON and resumes). Numbers are
in `docs/joint_skew_refit.json`; the figure is `docs/joint_skew_refit.png`; tests are
`tests/test_joint_skew_refit.py` (10 tests, 8 s).

![Joint refit](joint_skew_refit.png)

---

## 1. Question

`docs/atm_skew_term_structure.md` established two things: the SPY market's ATM skew
`ψ(T) = dσ_imp/dk` at `k = 0` steepens toward expiry with exponent `b = −0.249 ± 0.033`,
consistent with the rough-volatility prediction `H − ½ = −0.239` at the calibrated `H = 0.261`;
and the calibrated model itself does not reproduce that law, because its vol-of-vol sits at the
`η = 4` bound where `η T^H` is order one even at one day, giving `b = −0.321 ± 0.007`. The smile
calibration never sees the term structure explicitly: it minimises a vega-weighted Huber loss over
quotes, and the cheapest way to buy short-dated curvature is more `η`. This document asks whether a
joint fit of `(H, η)` (with `ρ` and `ξ` free) exists that matches the smiles acceptably *and* the
market's skew term structure, and what it costs.

## 2. Method

- **Smile objective**: `MapCalibrator.loss` from `backend/quant/calibrate_map.py` (vega-weighted
  Huber in implied-vol space, evaluated by the neural pricing map `artifacts/pricing_map.pt`), on
  three trading-hour captures: `spy_20260820T171756Z`, `spy_20260821T150017Z`,
  `spy_20260821T194519Z` (585–646 quotes each, seven expiries from 2.2 to 10.4 trading-day
  equivalents).
- **Skew objective**: for each capture and expiry, the market ATM skew `ψ_mkt(τ)` and its standard
  error from `market_skew_from_quotes` (local quadratic in `k` near the money), and the model's
  `ψ(τ)` from the pricing map by a central difference in log-moneyness (step `h ∝ σ_ATM √τ`, floor
  0.002, inside the map's box). The joint objective is
  `J(θ) = smile loss + λ · (1/n_exp) Σ_expiries ((ψ_model − ψ_mkt)/SE_mkt)²`.
- **Licensing the map**: the map-based `ψ` was checked against `model_skew_curve` (rough Bergomi
  Monte Carlo, 4 × 200k paths) on the capture expiries across the `(H, η)` grid. Relative RMS
  agreement is 1.9 % at `η = 1.45` and above, and breaks down at `η = 0.5` (24 %, the map is not
  trained to resolve the skew there). Cells with `η < 1.45` are shown hatched in the profile and
  excluded from conclusions.
- **Profile**: 12 × 12 grid over `H ∈ [0.05, 0.45]`, `η ∈ [0.5, 4.0]`; at each cell `(ρ, ξ)` are
  optimised on the smile objective (Powell, warm-started across the grid), recording smile RMSE,
  model exponent over the capture expiries and skew χ² per expiry.
- **Joint fits**: a full four-parameter optimisation (differential evolution + Powell, as
  `MapCalibrator`) for `λ ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}`, per capture and averaged; the
  Pareto front of smile RMSE against `|b_model − b_mkt|` picks the knee.
- **Uncertainty**: 24 bootstrap refits (quotes resampled by expiry) at the recommended `λ`.
- **Monte Carlo validation**: the served, pure-smile, joint and interior-η parameters on the
  standard maturity ladder (1–126 trading days, 4 × 200k paths, common random numbers across the
  strike stencil) for the exponent over `T ≤ 45` days, and a 400,000-path repricing of one capture's
  585 quotes for the smile RMSE with the true engine.

## 3. Results

### 3.1 Profile

Smile RMSE is flattest along a ridge from `(H 0.2, η 4)` down toward `(H 0.35, η 2)`; the market
exponent contour `b = −0.249` runs almost vertically at `H ≈ 0.30–0.32` for every `η ≥ 1.5`. The
two do not cross at the smile optimum, which is why the pure-smile calibration cannot reproduce the
term structure: the model's exponent on the capture expiries is `−0.41` there.

### 3.2 Pareto front (mean over the three captures, pricing map)

| λ | smile RMSE (vol pts) | model exponent b (capture expiries) | skew χ² / expiry | η | ρ | H | √ξ |
|---|---|---|---|---|---|---|---|
| 0 (pure smile) | 1.312 | −0.414 | 996 | 3.90 | −0.551 | 0.200 | 0.119 |
| 0.01 | 1.572 | −0.343 | 14.8 | 4.00 | −0.452 | 0.257 | 0.123 |
| 0.03 | 1.682 | −0.295 | 6.7 | 4.00 | −0.481 | 0.283 | 0.124 |
| **0.1 (knee)** | **1.755** | **−0.268** | **4.8** | 4.00 | −0.502 | 0.299 | 0.125 |
| 0.3 | 1.82 | −0.26 | 3.9 | interior on one capture | | | |
| 1 – 10 | 2.1 – 2.5 | −0.25 | 3.3 – 3.8 | | | | |

Served calibration on the same captures: smile RMSE 1.745 ± 0.113, exponent −0.324 ± 0.005,
χ² 266 ± 157 per expiry. The knee at `λ = 0.1` sits one market standard error from the market
exponent at less than half the served calibration's skew χ², and costs 0.44 vol points against the
pure-smile optimum.

### 3.3 Recommended parameters (λ = 0.1, capture `spy_20260821T150017Z`)

| parameter | joint refit | bootstrap (24 resamples) | served | pure smile |
|---|---|---|---|---|
| η | 4.000 (bound) | 3.99 ± 0.02, interior in 4 % of resamples | 3.935 | 4.000 |
| ρ | −0.504 | −0.503 ± 0.019 | −0.536 | −0.539 |
| H | 0.310 | 0.308 ± 0.011 | 0.261 | 0.212 |
| √ξ | 0.1253 | 0.1254 ± 0.0008 | 0.1130 | 0.1187 |
| smile RMSE, map (vol pts) | 1.794 | 1.80 ± 0.06 | 1.655 | 1.348 |
| smile RMSE, MC 400k paths (vol pts) | 2.12 | | 1.91 | 1.53 |
| skew χ² / expiry, map / MC | 7.9 / 8.9 | 8.4 ± 2.6 | 301 / 330 | 960 / 1058 |
| exponent, capture expiries, map / MC | −0.251 / −0.233 ± 0.008 | −0.253 ± 0.018 | −0.322 / −0.297 ± 0.007 | −0.400 / −0.355 ± 0.005 |
| **exponent, ladder 1–45 d, MC** | **−0.264 ± 0.007** | | −0.330 ± 0.005 | −0.369 ± 0.009 |

Market: exponent −0.249 ± 0.033 (56 rows, 2–10 trading days); on this capture's own expiries
−0.278 ± 0.024. The joint parameters are 0.5 SE from the pooled market exponent; the served
calibration is 2.5 SE away and the pure-smile optimum 3.6 SE.

Interior-η alternative (`λ = 0.3`, `η 3.39, ρ −0.551, H 0.295, √ξ 0.1266`): ladder exponent
−0.263 ± 0.004, χ² 8.2 per expiry, map smile RMSE 1.91 ± 0.05 (MC 2.26). It reproduces the term
structure equally well with `η` off its bound, for a further 0.15 vol points of smile fit.

### 3.4 The pricing map versus the true engine

Repricing the 585 quotes of one capture with 400,000 paths, the map's smile RMSE is below the Monte
Carlo one by 0.25–0.35 vol points for every parameter set (served 1.65 vs 1.91, joint 1.79 vs 2.12,
pure 1.35 vs 1.53, interior 1.93 vs 2.26); about 0.65 vol points of the Monte Carlo figure is its own
noise floor at this path count, so the map is neither systematically optimistic nor pessimistic
about the *ranking*, which the Monte Carlo reproduces. The map's skews agree with Monte Carlo to
1–3 % on the capture expiries for all four parameter sets; the map exponents are 0.02–0.04 steeper
than the Monte Carlo ones, a bias that is the same for every set and does not change the conclusion.

## 4. What it means for the served models

- `artifacts/rough_calibration.json` is left as it is: it is what the 0DTE surrogate
  (`artifacts/model_0dte.pt`) was trained on, and the surrogate's box, labels and ensemble all
  assume those dynamics. Adopting the joint parameters means regenerating the 0DTE labels
  (`backend/quant/dataset_0dte.py`) and retraining (`backend/quant/train_0dte.py`), a GPU job;
  this document recommends it and does not do it.
- The skew term structure is a stronger identifier of the dynamics than a single day's smile,
  and it disagrees with the served `H` by four bootstrap standard deviations (0.26 vs
  0.308 ± 0.011). A calibration protocol that includes the term-structure term at `λ ≈ 0.1` costs
  half a vol point of smile fit and removes the dependence on the `η` bound's placement for the
  exponent, though not for `η` itself.
- The three captures come from two consecutive trading days; the term-structure target is
  therefore the same regime three times, not three independent days.

## 5. Caveats

- The market skew is measured over 2–10 trading days and the model exponent over 1–45; the model's
  local exponent drifts (−0.22 over 1–5 d to −0.32 over 12–45 d for the joint parameters), so the
  window matters and is stated with every number.
- The profile and the joint fits use the pricing map, licensed against Monte Carlo only for
  `η ≥ 1.45`; nothing below that is claimed.
- `η` remains at its 4.0 bound at the knee. Whether that reflects the data or the map's box was
  tested one way (the interior-η solution costs 0.15 vol points) and not the other (a wider map
  box would require regenerating the map).
- The smile RMSE quoted from the map is 0.25–0.35 vol points below the Monte Carlo value with the
  true engine at every parameter set.

## 6. What would falsify this

- A four-parameter fit within the map's box reaching smile RMSE ≤ 1.35 vol points *and* a Monte
  Carlo ladder exponent within one market SE of −0.249 would show the trade-off is an artefact of
  the optimiser rather than of the model.
- Captures from other days whose market exponent is far from −0.25 would move the target; the
  joint parameters are tied to this regime.
- A Monte Carlo repricing at 400,000 paths showing the joint parameters' smile RMSE below the
  served calibration's would contradict the stated cost.
- A regenerated pricing map with `η` allowed above 4 whose smile optimum lands at an interior `η`
  with the market exponent would remove the bound from the story entirely.
