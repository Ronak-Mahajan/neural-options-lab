# 0DTE checkpoint history

Working notes kept for the record. This file is the audit trail behind the short-dated
model's driver and its calibration provenance; the README states what the served
checkpoint measures today and does not link here.

## 1. The driver

The first 0DTE checkpoint was trained on paths whose volatility driver was built from the
Type-I (Mandelbrot-Van Ness) fractional Brownian covariance
`0.5(t_i^2H + t_j^2H - |t_i-t_j|^2H)`. Rough Bergomi is driven by the Riemann-Liouville
Volterra process `W~_t = sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s`. The two agree on the
diagonal (both give `Var[W~_t] = t^2H`), which is why the martingale property held and
nothing looked wrong from the outside, and they agree nowhere else: at H = 0.1172 the
maximum off-diagonal relative difference is 4.93, and `corr(W~_t1, W~_t50)` was +0.320
against a true +0.054.

There was a second half to it. `chol(C)` is not the Volterra kernel, because W~ is a
continuous stochastic integral rather than a linear function of n coarse increments.
Factorising C alone forces `corr(Z_1, W~_t1) = 1` by construction when the truth is
`sqrt(2H)/(H+1/2) = 0.7844`, so the leverage correlation rho was applied to the wrong
object, over-correlating spot and vol precisely at the short end where a 0DTE skew fit is
identified.

Both were replaced by the exact joint-Gaussian scheme now in `rough_vol.py`, verified
against quadrature to 5.4e-08 and against 400,000 draws.

`artifacts/model_0dte.pt` was regenerated against the corrected driver on 2026-08-05, and
the wrong-kernel checkpoint is kept as `model_0dte_v1_type_i_kernel.pt`. An earlier
README figure of "about 2 basis points against its rough Bergomi teacher" described
agreement with the Type-I teacher. The 2026-08-05 checkpoint measured 1.48 bps of strike
RMSE, +0.13 bps bias and 2.76 bps p95 on 400 held-out points against 500,000-path
references under the Volterra driver, below its 2.35 bps per-label noise floor. That
checkpoint was replaced twice by the calibration adoptions in section 2, and the 1.48 bps
figure was never re-measured on the checkpoints that followed, so it is not a property of
the served model.

## 2. The calibration

An earlier README said the model was "calibrated to the live SPY smile ... about 2
volatility points across 72 quotes and two expiries." That calibration was fitted at 03:43
New York from the previous session's last trades (`quote_source:
"last_trade_market_closed"`) and passed a quality gate that tested only RMSE and
bound-pinning. The gate now also rejects stale sessions, and in the market-closed branch
time-to-expiry is no longer stamped from `now` against last-session prices: on synthetic
quotes with known truth that convention inflated sqrt(xi) by 21% and moved H by 0.021.

The served checkpoint was then rebuilt through the `calibrate --retrain` path on live
fits, twice on 2026-08-20: first on the accepted 2026-08-10 fit (eta 2.688, rho -0.328,
H 0.104; 677 quotes, 8 expiries, 1.553 vol points; ensemble validation RMSE 3.8 bps,
commit `65e67c4`), then on the accepted 2026-08-20 15:47 EDT fit (eta 3.657, rho -0.628,
H 0.255; 618 quotes, 1.568 vol points; ensemble validation RMSE 3.3 bps, commit `82c54bb`).

## 3. Deep hedging

The retracted "reduced 95% tail loss by roughly 30% versus delta hedging" claim, the three
defects behind it and their measurements are written up in full in
[hedging_findings.md](hedging_findings.md).
