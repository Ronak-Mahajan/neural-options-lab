# BTC full-surface jump-stability series (Saturday 2026-08-22)

Question: is Friday's jump result (jumps beat diffusive by +1.30 vp under
MC, interior optimum lam=58/yr, mu_j=+3.1%) a stable feature of the BTC
surface or an episodic premium? One `scripts.btc_full_surface` run every
~90 minutes through the GPU's last day; the map fits are deterministic
given the snapshot, so every difference between points is the market.

Reference point (Fri 2026-08-21, PR #16): 269 quotes 2-70d, jumps win by
+1.30 vp, lam=58/yr, mu_j=+3.1%, sig_j=2.1%, reproduced qualitatively on a
fresh Friday snapshot.

| # | UTC | index | quotes | diffusive MC rmse (vp) | jump MC rmse (vp) | jumps - diffusive (vp) | lam/yr | mu_j | verdict |
|---|-----|-------|--------|------------------------|-------------------|------------------------|--------|------|---------|
| 1 | 12:52 | 77,273 | 264 | 2.354 | 2.319 | +0.036 | 0.1 (railed 0) | +9.5% (unidentified) | RETRACTED as an optimizer artifact; see the replay rows and the basin-miss note |
| 1r | 12:45 (replay) | 77,329 | 263 | 2.335 | 2.018 | +0.317 | 31.8 | +1.9% | multi-start fit recovers the basin the original fit missed; cumulant 0.087/yr |
| 2 | 14:24 | 76,989 | 264 | 2.143 | 1.744 | +0.399 | 94.6 | +3.0% | jumps HELP again; sig_j railed at floor (0.5%), MC confirms the win |
| 3 | 15:40 | 76,997 | 265 | 1.899 | 1.562 | +0.337 | 90.4 | +2.9% | premium persists; cumulant 0.080/yr, delta easing from the 14:25 +0.45-0.50 |
| 4 | 17:11 | 77,289 | 266 | 1.832 | 1.506 | +0.326 | 92.7 | +3.0% | steady: cumulant 0.087/yr, third consecutive point with mu_j at +3% |
| 5 | 18:42 | 77,259 | 266 | 1.861 | 1.430 | +0.431 | 69.9 | +3.1% | first live point on the fixed fit: fully interior optimum (sig_j 0.99%, no rails), cumulant 0.074/yr |
| 6 | 20:13 | 77,440 | 267 | 1.921 | 1.721 | +0.201 | 19.0 | +5.4% | premium easing into the evening: cumulant 0.055/yr, delta at the day's low but still above noise |

Point-1 notes: both arms fit worse than Friday (map-RMSE ~2.5-2.6 vp vs
Friday's tighter fit), with a +2.0 vp mean bias in the 17-45d bucket on
BOTH arms, the known constant-parameter term-structure limitation, more
pronounced on this surface. With lam railed at zero the remaining jump
parameters are unidentified; their printed values carry no information.

Point-2 notes: the jump basin is back (lam=94.6, mu_j=+3.0%, the same sign
and magnitude as Friday) and the MC verdict confirms it (+0.399 vp), but
point 1's jump fit had map-RMSE 2.524 vs its diffusive 2.578, nearly
equal, which raises the question of whether point 1 MISSED the jump
basin rather than the market lacking one. Twin diagnostic (two runs on the
same surface, minutes apart) queued to measure optimizer repeatability;
until it reports, point-to-point lam differences cannot be read as market
moves.

## Twin diagnostic (14:25 UTC): what is actually identified

Two runs 26 s apart agree to three digits on every parameter (lam 67.0 vs
67.1, mu_j +1.57% vs +1.56%, sig_j 3.32% vs 3.31%, MC delta +0.499 vs
+0.489): the fit is REPEATABLE given a surface. But twin A vs point 2,
one minute apart, moved lam 94.6->67.0 and sig_j floor->3.3%. The
reconciliation is a cumulant ridge: the jump VARIANCE contribution
lam*(mu_j^2+sig_j^2) is 0.0898/yr at point 2 and 0.0902/yr at twin A,
identical to within noise, while raw parameters trade off along the ridge
under tiny data perturbations. Protocol consequence, applied from here on:

- The series reports IDENTIFIED quantities per point: the MC
  jumps-vs-diffusive delta (twin noise floor ~0.01 vp) and the jump
  variance cumulant lam*(mu_j^2+sig_j^2). Raw lam/mu_j/sig_j are
  ridge-coordinates and are logged but not interpreted individually.
- Point 1's collapse was resolved by replaying the archived 12:45Z
  capture. First, a theta transplant: the afternoon jump parameters price
  the morning surface at 2.104 vp under the map, 0.44 vp BETTER than the
  morning's own fitted diffusive, so the basin existed and the fit missed
  it. Root cause in MapCalibrator.fit: Powell polished from only two
  starts, DE's answer and a diffusive-adjacent seed (lam at its lower
  bound), so when DE's global search also landed diffusive, nothing ever
  visited the jump basin. Fix: two additional Powell starts seeded in the
  interior of the jump box, one per mu sign (so the fix does not assume
  the positive-jump conclusion). The re-replay with the fixed fit
  recovers an interior optimum (lam=31.8, mu_j=+1.9%, sig_j=4.9%, no rail
  warnings) and MC confirms jumps help by +0.317 vp. Points 1-4 ran the
  old fit; points 5 on run the fixed one. Points 2-4 stand as-is (they
  found the basin regardless).
- The recovered morning fit also vindicates the cumulant as the
  identified quantity: lam=31.8 with wide sig_j and lam=92.7 with floored
  sig_j are far apart in raw coordinates, yet both give
  lam*(mu_j^2+sig_j^2) = 0.087-0.090/yr.

## Day headline (running)

The jump premium was present all Saturday and drifted, not switched: MC
delta between +0.20 and +0.50 vp at every honestly-fitted point (peak
14:25, easing to the day's low by 20:13), jump-variance cumulant
0.055-0.090/yr, mu_j positive throughout. The one apparent absence was an
optimizer basin miss, caught by replaying the surface archive, fixed with
multi-start polishing, and documented in calibrate_map.fit. Raw lam
wandered 19-95/yr across the day while the cumulant stayed within a
factor of 1.6: the ridge is real, and only the cumulant and the MC delta
deserve interpretation.

| twin | UTC | index | lam/yr | mu_j | sig_j | cumulant/yr | MC delta (vp) |
|------|-----|-------|--------|------|-------|-------------|----------------|
| pt2  | 14:24 | 76,989 | 94.6 | +3.04% | 0.50% | 0.090 | +0.399 |
| A    | 14:25 | 77,012 | 67.0 | +1.57% | 3.32% | 0.090 | +0.499 |
| B    | 14:25 | 77,015 | 67.1 | +1.56% | 3.31% | 0.090 | +0.489 |
