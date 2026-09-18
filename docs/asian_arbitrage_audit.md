# A static-arbitrage audit of the served Asian pricer, in price space

**Summary.** The served Asian pricer (`artifacts/model.pt`, the 5-member ensemble that
outputs the discretely monitored arithmetic-average call price per unit strike) is not
arbitrage-free. Audited by autograd on a 120,800-point lattice over its own trained box
(m = S/K in [0.5, 2.0], T in [0.05, 2.0] years, five vols, four rates, 50 monitoring
dates), it prices **negative gamma on 18.63%** of the box, **negative vega on 28.05%**,
1%-wide butterflies worth as little as **-2.67 bps of strike** on **17.68%**, and sits
**below the Asian floor e^{-rT}(E[A] - K)+ on 9.88%**, by up to **5.98 bps of strike**.
It never prices above spot, never below zero, and its delta is negative on 0.09% of the
box. The violations are concentrated where sigma sqrt(T) is small and the contract is
deep in the money: 94% of the negative-gamma points have m > 1.05, 40.1% of the points
with sigma sqrt(T) < 0.05 violate convexity against 5.2% of those above 0.20, and every
negative-delta point sits at sigma = 0.05. That region is where the arithmetic average is
almost deterministic and the true price is a hinge - its own floor, affine in spot, with
no curvature and no volatility sensitivity to reproduce - so a smooth least-squares fit
carrying about 1 bp of price error carries curvature and vega of either sign.

Inside the region where the contract's volatility sensitivity is actually resolved
(reference vega >= 0.02 per unit strike per unit vol, 42.56% of the box), the surface is
very nearly clean: **zero** violations of the floor, of monotonicity, of the slope bound,
of vega and of both delta bounds, and **0.060%** (31 points) of convexity and **0.016%**
(8 points) of the 1% butterfly, all of them in one corner at m >= 1.97, T >= 1.70,
sigma = 0.40, where the butterfly is worth -0.046 bps against a common-random-number
Monte Carlo value of +0.184 +- 0.001 bps. The clean conditions have room: in that region
dC/dK stays below -0.0085, the floor holds with at least 0.83 bps to spare, and vega
stays above +0.0127.

A 200,000-path x 4-seed Monte Carlo arbiter at the box corners and at every
worst-violation location confirms that the violations are the network's and not the
reference's: at the worst butterfly point the network prices -2.67 bps where the Monte
Carlo, on common random numbers, prices **+0.0255 +- 0.00002 bps**; at the worst floor
point the true price *is* the floor to 0.005 bps (Monte Carlo 10,005.126 bps, floor
10,005.131 bps) and the network returns 9,999.150 bps.

Everything here was produced by `python -m scripts.asian_arbitrage_audit` (107.8 s on 8
torch threads: 0.7 s of prices, 21.7 s of autograd, 46.9 s of Curran reference, the rest
Monte Carlo). Every number below is in `docs/asian_arbitrage_audit.json`, which this
document is written from; `tests/test_asian_audit.py` pins the conditions themselves.

---

## 1. What is audited

### 1.1 The model and the contract

`artifacts/model.pt` (sha256 `2ca9e6ec...`, 5 members, width 128, 4 blocks, 133,889
parameters, trained on 500,000 Latin-hypercube points) is the ensemble `PricingEngine`
routes every maturity above 12/252 to. It outputs the arithmetic-average Asian call price
per unit strike as a function of (m = S/K, T, sigma, r) over the trained box
`backend.quant.dataset.PARAM_RANGES`: m in [0.5, 2.0], T in [0.05, 2.0] years,
sigma in [0.05, 0.80], r in [0, 0.10], with n = 50 equally spaced monitoring dates
t_i = i T / n. Maturities at or below 12/252 are served by a different checkpoint,
`artifacts/model_0dte.pt`, whose own audit is `docs/no_arbitrage_surface.md`; the lattice
here starts at T = 0.05 so that every point is priced by the Asian ensemble.

Nothing in the training constrains the surface to be free of static arbitrage: the loss
is least squares on Monte Carlo prices and pathwise differentials, and least squares has
no opinion about the sign of a second derivative.

### 1.2 The conditions that transfer

The audit is done entirely in **price** space. The 0DTE audit
(`scripts/no_arbitrage_surface.py`) inverts prices to Black-Scholes implied vols and
evaluates Durrleman's g; neither step survives the change of contract, because the
arithmetic average of a lognormal is not lognormal. What survives is the set of
statements that follow from the payoff alone - A >= 0, (A - K)+ convex and decreasing in
K, increasing in every path - and those are checked directly:

| condition | reads |
|---|---|
| dC/dK <= 0 | a call struck higher cannot cost more |
| dC/dK >= -e^{-rT} | the strike spread cannot beat a discounted unit of cash |
| d2C/dK2 >= 0 | the risk-neutral density of A is non-negative |
| C(K(1-h)) - 2C(K) + C(K(1+h)) >= 0, h = 1% | the same statement as a tradable butterfly |
| C >= e^{-rT} (E[A] - K)+ | Jensen: the ASIAN floor |
| C <= e^{-rT} E[A], C <= S | the average cannot be worth more than its mean, or than the spot |
| C >= 0 | limited liability |
| d2C/dS2 >= 0 | convexity in spot |
| dC/dsigma >= 0 | more volatility cannot make the option cheaper |
| 0 <= dC/dS <= e^{-rT} E[A]/S | delta bounds, A being linear in S |

`E[A]` is the engine's own, not a European forward: with fixings t_i = i T / n,

    E[A] = (S / n) sum_{i=1..n} e^{r t_i}
         = S e^{r dt} (e^{rT} - 1) / (n (e^{r dt} - 1)),   dt = T / n,

the expression `engine._parity_adjustment_torch` evaluates to derive every served put and
`monte_carlo.expected_arithmetic_average` uses for parity. `asian_arbitrage_audit.expected_average`
is the vectorised form of it, pinned against both of them and against a direct sum over
the 50 fixing dates in `tests/test_asian_audit.py`, to floating-point roundoff (measured
5e-16 relative against the direct sum, 9e-16 absolute against the engine's parity term).

Derivatives come from reverse-mode autograd through `engine._call_price_torch`, the same
graph the served Greeks are taken from, in the served float32; prices come from
`engine.price_batch`, the served batch path. The two agree to 0.0 in price on all 120,800
points. Homogeneity of degree one makes d2C/dK2 = m^2 d2C/dS2 an identity, and the two
independent autograd paths satisfy it to **1.2e-5** absolute (0.17% relative where the
value exceeds 1e-3) and agree in sign at **99.9992%** of the lattice, which is the check
that the second derivatives are real and not float32 noise.

### 1.3 The conditions that do not transfer, and are therefore absent

- **Black-Scholes implied vol, and Durrleman's g.** The arithmetic average is not
  lognormal. A Black-Scholes implied vol read off an Asian price is not a parameter of
  anything in this model, and g(k) >= 0 is a statement about the density of a lognormal
  variable, not of A.
- **The European intrinsic floor max(S - K e^{-rT}, 0).** It is strictly above the Asian
  floor whenever r > 0, because averaging replaces the terminal forward with the mean of
  the forwards: at m = 1.0, T = 2.0, r = 0.04 the European floor is 768.8 bps of strike
  and the Asian floor is 387.0 bps. Applying it would manufacture violations that are not
  violations. The audit records both numbers at every arbitrated point so the gap is
  visible.
- **Calendar monotonicity dC/dT >= 0.** The 50 fixings are t_i = i T / n, so they move
  with T: a longer-dated contract averages over a different, wider window rather than
  being the same contract held longer. There is no self-financing position that turns a
  drop in price with maturity into an arbitrage here, and dC/dT carries no sign for this
  contract. (For the 0DTE European surrogate it does, and that audit reports it.)

### 1.4 Resolution: where a shape statistic means something

A shape statistic is only informative where the contract has more time value than the
surrogate has price error. Mirroring the vega floor of the 0DTE audit, every statistic
below is reported over the full box **and** over the region where the reference's own
vega - a central difference of Curran at sigma +- 1e-3, in which Curran's level bias
cancels to first order - is at least **0.02** per unit strike per unit vol, the same
`VEGA_FLOOR` `backend/quant/iv_surface.py` uses.

At that floor the ensemble's measured **1.33 bps** price RMSE (`artifacts/eval.json`,
600 held-out contracts against 200,000-path references) is worth 0.7 vol points, and its
measured vega error of 0.0013 per unit strike per unit vol (13.19 in the 1e-4 units of
the same file) is 7% of the floor itself. Below the floor, the contract's entire
sensitivity to volatility is smaller than the error bar on its price, and a sign read off
the surrogate there is a sign read off its noise. The resolved region is **51,418 points,
42.56%** of the box; the reference vega has median 0.0033 and maximum 0.330 over the
lattice, which is how concentrated the informative region is.

---

## 2. Results

Lattice: m in [0.5, 2.0] in 151 steps of 0.01, T in [0.05, 2.0] in 40 steps of 0.05,
sigma in {0.05, 0.10, 0.20, 0.40, 0.80}, r in {0, 0.02, 0.04, 0.10}: 120,800 points,
107.8 s including the Monte Carlo arbiter.

**Full box.** "Worst" is the extreme value of the quantity itself, at the location given.

| condition | points | violated | worst value | at (m, T, sigma, r) |
|---|---|---|---|---|
| dC/dK <= 0 | 120,800 | **2.23%** | dC/dK = +2.5e-4 | 0.93, 0.05, 0.05, 0.10 |
| dC/dK >= -e^{-rT} | 120,800 | **12.84%** | dC/dK = -1.0336 (bound -1) | 1.02, 0.05, 0.05, 0.00 |
| d2C/dK2 >= 0 | 120,800 | **18.63%** | -2.97 | 1.03, 0.05, 0.05, 0.00 |
| 1% butterfly >= 0 | 118,400 | **17.68%** | **-2.67 bps of strike** | 1.03, 0.05, 0.05, 0.00 |
| C >= e^{-rT}(E[A] - K)+ | 120,800 | **9.88%** | **-5.98 bps of strike** | 2.00, 2.00, 0.05, 0.04 |
| C <= e^{-rT} E[A] | 120,800 | 0.00% | slack >= 4,075 bps | 0.50, 2.00, 0.80, 0.10 |
| C <= S | 120,800 | 0.00% | slack >= 4,534 bps | 0.50, 2.00, 0.80, 0.10 |
| C >= 0 | 120,800 | 0.00% | C >= 0.139 bps | 0.50, 0.70, 0.20, 0.04 |
| gamma = d2C/dS2 >= 0 | 120,800 | **18.63%** | -2.80 | 1.03, 0.05, 0.05, 0.00 |
| vega = dC/dsigma >= 0 | 120,800 | **28.05%** | -0.0202 | 1.03, 0.05, 0.05, 0.00 |
| delta = dC/dS >= 0 | 120,800 | **0.09%** | -1.8e-4 | 0.94, 0.05, 0.05, 0.04 |
| delta <= e^{-rT} E[A]/S | 120,800 | **14.09%** | delta = 1.0330 (bound 1) | 1.02, 0.05, 0.05, 0.00 |

The butterfly is evaluated on 118,400 of the points: its wings read the network at
K(1 +- 1%), i.e. at m/(1 +- 1%), and at the two ends of the m axis one wing would leave
the trained box.

**Resolved region** (reference vega >= 0.02; 51,418 points, 50,995 for the butterfly):

| condition | violated | worst value | at (m, T, sigma, r) |
|---|---|---|---|
| d2C/dK2 >= 0 | **0.060%** (31 points) | -0.128 | 2.00, 2.00, 0.40, 0.00 |
| gamma >= 0 | **0.060%** (31 points) | -0.0320 | 2.00, 2.00, 0.40, 0.00 |
| 1% butterfly >= 0 | **0.016%** (8 points) | -0.046 bps of strike | 1.98, 2.00, 0.40, 0.00 |
| dC/dK <= 0 | 0.00% | dC/dK <= -0.0085 | - |
| dC/dK >= -e^{-rT} | 0.00% | slack >= +0.0070 | - |
| C >= e^{-rT}(E[A] - K)+ | 0.00% | slack >= +0.83 bps | - |
| C >= 0 | 0.00% | C >= 1.94 bps | - |
| vega >= 0 | 0.00% | vega >= +0.0127 | - |
| delta >= 0 | 0.00% | delta >= +0.0100 | - |
| delta <= e^{-rT} E[A]/S | 0.00% | slack >= +0.0034 | - |

All 31 convexity violations and all 8 butterfly violations sit in one corner:
m in [1.97, 2.00], T in [1.70, 2.00], sigma = 0.40, r in {0, 0.02, 0.04} - the top of the
moneyness range at the top of the maturity range, where the resolved region reaches
furthest into the money. They are listed point by point in the JSON
(`conditions.*.resolved.violations`).

---

## 3. Where the violations sit

**Negative gamma / negative density** (18.63% of the box). By moneyness: 27.97% of the
in-the-money points (m > 1.05) violate and they hold **94.4%** of all violations, against
3.33% of the out-of-the-money points (5.3% of violations) and 0.43% at the money
(0.1%). By volatility: 36.74% at sigma = 0.05 (39.4% of violations), 29.58% at 0.10,
20.56% at 0.20, 5.24% at 0.40, 1.06% at 0.80. By sigma sqrt(T): **40.07%** below 0.05,
31.55% from 0.05 to 0.10, 25.20% from 0.10 to 0.20, **5.16%** above 0.20. By maturity the
gradient is much flatter - 30.5% at 0.05-0.25 y, 21.5% at 0.25-0.5 y, 17.9% at 0.5-1 y,
15.9% at 1-2 y - because at sigma = 0.05 even a two-year contract has
sigma sqrt(T) = 0.07. The controlling variable is sigma sqrt(T), not T.

**Negative vega** (28.05%). Both wings are hit - 30.78% of in-the-money points (69.0% of
violations) and 28.73% of out-of-the-money points (30.5%) - while the at-the-money band
is nearly clean at 1.12%. By volatility the concentration is sharper than for gamma:
**69.69%** of the sigma = 0.05 points violate, holding half of all violations, against
35.44% at 0.10, 28.61% at 0.20, 5.66% at 0.40 and 0.88% at 0.80; 66.18% of the points
with sigma sqrt(T) < 0.05 violate.

**Below the Asian floor** (9.88%). **99.9%** of these are in the money, where the floor
is positive at all: 15.68% of in-the-money points violate, 0.11% at the money, none out
of the money. By volatility, 15.34% at sigma = 0.05 down to 0.84% at 0.80. The worst
case, -5.98 bps of strike, is at m = 2.00, T = 2.00, sigma = 0.05, r = 0.04, i.e. the
deepest, quietest, longest contract on the lattice.

**Wrong-signed strike slope** (dC/dK > 0, 2.23%) is the mirror image: **99.9%** of those
points are out of the money (7.46% of the m < 0.95 points violate, none in the money),
where the price is close to zero and its dependence on the strike is below the
surrogate's resolution. Negative delta (0.09%) is rarer and lives in exactly the same
place: every one of the 110 points is at sigma = 0.05 with sigma sqrt(T) < 0.05, 96% of
them out of the money, 81% at T < 0.25 y.

**Delta above its forward bound** (14.09%) and **dC/dK below -e^{-rT}** (12.84%) are two
readings of one defect: homogeneity gives dC/dK = (C - S dC/dS) / K, so the two bounds
coincide exactly where the price sits on its floor and separate by the contract's time
value elsewhere, which is why the rates differ by 1.25 points. Both are 99.7% in the
money.

---

## 4. Why there

At small sigma sqrt(T) the arithmetic average is almost deterministic, and the call on it
is a hinge: in the money it is worth its floor e^{-rT}(E[A] - K), out of the money it is
worth essentially nothing, and all of the curvature lives in a band around
E[A] = K whose width scales with sigma sqrt(T). The Monte Carlo arbiter measures exactly
that. At m = 2.00, T = 2.00, sigma = 0.05, r = 0.04 the 200,000-path x 4-seed price is
**10,005.126 bps** of strike against a floor of **10,005.131 bps**: the true price *is*
the floor there, to 0.005 bps, with a standard error of 0.003 bps. On that stretch the
true price surface is affine in the spot - gamma exactly zero, vega exactly zero, dC/dK
exactly -e^{-rT}. At m = 1.03, T = 0.05, sigma = 0.05, r = 0 the same measurement gives a
price of 300.000 bps against a floor of 300.000 bps, and a true 1%-wide butterfly of
+0.0255 bps - four decimal places smaller than the price it is a second difference of.

The ensemble is a smooth function fitted by least squares to noisy Monte Carlo labels
over the whole box. Its typical price error is 1.33 bps of strike, which is what
`artifacts/eval.json` reports over 600 held-out points, and it is a root-mean-square over
the box rather than a bound at any one point; but a basis point of *smooth* error on an affine stretch is a
ripple, and the second derivative of a ripple has whatever sign the ripple has. Negative
gamma, negative butterflies, delta above 1, dC/dK below -e^{-rT} and prices a few bps
below the floor are all the same artefact measured with different operators. Negative
vega is the same statement about sigma: at the worst butterfly point the contract's true
vega is 1.7e-6 by common-random-number Monte Carlo and 1.9e-6 by Curran, while the
surrogate's own vega error is 0.0013 - some seven hundred times larger - so the sign it
reports there is the sign of the fit's residual slope in sigma.

This also explains what the resolved region shows. Raising the bar to a reference vega of
0.02 - about fifteen times the surrogate's vega error - removes every vega, floor, slope and
delta violation and all but 31 of 22,510 convexity violations. What is left is not noise
about a hinge: at m = 1.98, T = 2.00, sigma = 0.40, r = 0 the contract is worth 9,831 bps
of strike with a reference vega of 0.035, comfortably resolved, and the network still
prices the 1% butterfly at -0.046 bps against a common-random-number Monte Carlo
**+0.184 +- 0.001 bps**. That corner is the top of the moneyness box at the top of the
maturity range, where the fit has neighbours on one side only.

---

## 5. The Monte Carlo arbiter

`price_asian_mc` (the project pricer: antithetic, geometric control variate, 50 steps),
200,000 paths x 4 seeds, at the twelve corners of the box and at every worst-violation
location: fifteen points, all of them in the JSON, a selection here. Prices in bps of
strike.

| point (m, T, sigma, r) | network | Curran | Monte Carlo (+- SE) | network - MC | Curran - MC |
|---|---|---|---|---|---|
| 1.00, 0.05, 0.05, 0.04 | 34.376 | 31.511 | 31.511 +- 0.0001 | +2.865 | +0.0001 |
| 1.00, 2.00, 0.05, 0.04 | 422.157 | 420.620 | 420.619 +- 0.003 | +1.538 | +0.001 |
| 1.00, 0.05, 0.80, 0.04 | 427.156 | 422.406 | 422.427 +- 0.017 | +4.728 | -0.021 |
| 1.00, 2.00, 0.80, 0.04 | 2629.300 | 2624.536 | 2633.028 +- 0.943 | -3.728 | -8.492 |
| 2.00, 2.00, 0.05, 0.04 | 9999.150 | 10005.131 | 10005.126 +- 0.003 | **-5.976** | +0.005 |
| 2.00, 0.05, 0.05, 0.04 | 9998.975 | 10000.393 | 10000.393 +- 0.0001 | -1.418 | +0.0001 |
| 2.00, 2.00, 0.80, 0.04 | 10447.435 | 10436.005 | 10455.669 +- 2.137 | -8.233 | -19.663 |
| 1.03, 0.05, 0.05, 0.00 (worst convexity) | 303.000 | 300.000 | 300.000 +- 0.00004 | +3.000 | +0.0001 |
| 1.02, 0.05, 0.05, 0.00 (worst slope) | 201.009 | 200.023 | 200.023 +- 0.00003 | +0.986 | +0.0001 |
| 1.74, 2.00, 0.80, 0.00 (worst vs reference) | 8211.582 | 8187.162 | 8206.992 +- 1.839 | +4.590 | -19.830 |

Two things follow. First, the network's violations are its own: at every worst-violation
point the Monte Carlo agrees with Curran to within 1e-4 bps, and the shape of the gap to
the network is the violation. Between m = 1.02 and m = 1.03 at T = 0.05, sigma = 0.05,
r = 0 the true price rises 99.977 bps (200.023 to 300.000, a secant delta of 1.000, the
floor's own slope) while the network rises 101.991 (201.009 to 303.000, a secant delta of
1.020): the excess slope is the delta of 1.0330 that the forward bound flags, and its
turning over is the -2.97 density. Second, the reference has a
bias of its own at high volatility - Curran is a conditioning **lower** bound, and at
sigma = 0.80, T = 2.00 the four arbitrated points put it 5.1 to 19.8 bps below the Monte
Carlo (at sigma = 0.80, T = 0.05 it is inside the Monte Carlo standard error). That is why the
largest network-vs-Curran gap on the lattice (24.42 bps at m = 1.74, T = 2.00,
sigma = 0.80, r = 0) is mostly Curran: against the Monte Carlo at that point the network
is +4.59 bps.

The butterfly itself is arbitrated on common random numbers - the three strikes priced on
one path set, so the second difference has a far smaller standard error than any leg:

| point (m, T, sigma, r) | network butterfly | Monte Carlo butterfly (+- SE) |
|---|---|---|
| 1.03, 0.05, 0.05, 0.00 (worst, full box) | -2.666 bps | +0.0255 +- 0.00002 bps |
| 1.98, 2.00, 0.40, 0.00 (worst, resolved) | -0.046 bps | +0.184 +- 0.001 bps |
| 2.00, 2.00, 0.40, 0.00 (worst convexity, resolved) | -0.129 bps | +0.165 +- 0.003 bps |

And the resolution measure is arbitrated the same way, so that the vega floor is the
contract's vega and not the reference's bias: a common-random-number Monte Carlo vega
against the Curran central difference agrees to 3e-7 at the worst violation points, to
1.5e-3 at sigma = 0.80, T = 2.00 (on a vega of 0.137) and to 7.8e-3 at the largest
reference gap (on 0.248).

Against the network, the Curran reference over the whole lattice is:

| region | RMSE | MAE | bias | p95 abs | max abs | at |
|---|---|---|---|---|---|---|
| full box (120,800) | 2.98 bps | 1.70 | +1.43 | 7.08 | 24.42 | 1.74, 2.00, 0.80, 0.00 |
| resolved (51,418) | 4.29 bps | 2.75 | +2.66 | 10.03 | 24.42 | 1.74, 2.00, 0.80, 0.00 |

by volatility, 1.32 / 1.33 / 1.34 / 1.64 / 6.04 bps RMSE at sigma = 0.05 / 0.10 / 0.20 /
0.40 / 0.80 - the last of those being Curran's own bound bias as much as the network's
error, per the Monte Carlo rows above.

---

## 6. Caveats

1. **A lattice is not a proof.** 120,800 points on 20 (sigma, r) slices, uniform in m and
   T. The violation rates are rates on that lattice; a finer grid, or a slice at a vol or
   rate not audited, will find different points. Nothing here bounds the surface between
   lattice points.
2. **The reference is a lower bound.** Curran (1994) prices this contract to 0.104 bps
   mean and 0.345 bps max error against a 400,000-path Monte Carlo on the benchmark grid
   (`docs/approximation_benchmark.md`, sigma = 0.25), and to 1e-4 bps at the
   low-volatility points that matter here - but at sigma = 0.80 and two years it is 5 to
   20 bps low (at sigma = 0.80 and 0.05 years it is inside the reference's standard error),
   and every statement about the *size* of the network's error at high volatility is
   therefore made against the Monte Carlo, not against Curran. The conditions themselves
   are not measured against the reference at all: they are properties of the network's
   own surface.
3. **The resolution floor is a choice.** `VEGA_FLOOR` = 0.02 comes from the surrogate's
   own 1.33 bps price error. Halving it roughly doubles the resolved area and admits more
   of the surrogate's noise; the fractions "over the full box" do not move with it, the
   fractions "in the resolved region" do.
4. **Puts are not audited separately.** The engine derives them from the call by the exact
   parity P = C - e^{-rT}(E[A] - K), whose adjustment is linear in K and independent of
   the network, so d2P/dK2 = d2C/dK2, dP/dK = dC/dK + e^{-rT} and every violation above
   maps to the put one for one.
5. **Black-Scholes dynamics.** This checkpoint is trained on geometric Brownian motion
   labels, so E[A], Curran and the Monte Carlo arbiter all speak the same model. The
   no-arbitrage conditions are model-free; the surface being audited is not.
6. **Second derivatives of a float32 network.** The audit runs on the served float32 path.
   The homogeneity identity d2C/dK2 = m^2 gamma, computed by two independent autograd
   routes, holds to 1.2e-5 absolute and agrees in sign on 99.9992% of the lattice, which
   bounds the noise in the curvature statistics well below the violations reported.
7. **One checkpoint.** sha256 `2ca9e6ec...`, the ensemble in `artifacts/model.pt`. The
   audit records the hash, and `tests/test_asian_audit.py` fails if the JSON and the
   checkpoint stop matching.
8. **The short end belongs to a different checkpoint.** `PricingEngine` prices maturities
   at or below 12/252 with `model_0dte.pt`, a European rough-Bergomi surrogate, so a
   quote at T = 0.004 is outside this box and inside the one
   `docs/no_arbitrage_surface.md` audits - where the same low-sigma-sqrt(T) hinge is
   measured, in implied-vol space, on that model's own conditions. Maturities between
   12/252 and 0.05 fall between the two trained boxes, and `engine.in_domain` reports
   them out of domain.

---

## 7. What would falsify this

- *"The served Asian pricer admits static arbitrage."* Falsified if the price-space
  conditions held on the lattice. They do not: negative density on 18.63% of the box,
  1%-wide butterflies as low as -2.67 bps of strike, prices up to 5.98 bps below the
  Asian floor. None of these needs a reference model, an implied vol, or a choice of
  floor: they are internal to the surface.
- *"The butterflies are real and not Monte Carlo noise."* Falsified by a
  common-random-number Monte Carlo butterfly that straddles zero at the same point. At
  the worst full-box point the Monte Carlo butterfly is +0.0255 bps with a standard error
  of 0.00002 bps - the network's -2.666 is more than 100,000 standard errors away.
- *"They sit where sigma sqrt(T) is small and the payoff is a hinge, and are of the order
  of the surrogate's own price error."* Falsified by violations of comparable size in the
  resolved, high-vega region. There are 31 convexity and 8 butterfly points there, worst
  -0.128 and -0.046 bps, against -2.97 and -2.67 on the full box; every other condition is
  clean there with the margins listed in Section 2.
- *"The Asian floor is the right floor."* Falsified by a contract priced below
  e^{-rT}(E[A] - K)+ that a static portfolio cannot exploit, or by the European intrinsic
  being the binding bound. At m = 1.00, T = 2.00, r = 0.04 the European intrinsic is
  768.8 bps and the true price 420.6 bps: a European floor would report a 348 bp
  violation where there is none. `tests/test_asian_audit.py` pins the Asian floor formula
  against the engine's parity term, against `monte_carlo.expected_arithmetic_average` and
  against a direct sum over the 50 fixing dates.
- *"Calendar monotonicity is not a no-arbitrage condition for this contract."* Falsified
  by a self-financing strategy that turns dC/dT < 0 into a riskless profit when the
  averaging window moves with T. The fixings are t_i = i T / n: the T = 1 and T = 2
  contracts average different date sets, and neither dominates the other pathwise.

Reproduce: `python -m scripts.asian_arbitrage_audit` (107.8 s for the committed JSON;
`--quick` for a coarse smoke run to a temp directory, `--no-mc` to skip the Monte Carlo
arbiter, `--checkpoint` to audit a different ensemble). Tests: `tests/test_asian_audit.py`.
