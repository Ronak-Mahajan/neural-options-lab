"""The paired bootstrap that decides whether two hedgers are actually apart.

Every hedger in `HedgeComparer.compare` runs on the same simulated paths, so
comparing them is a paired problem. These tests pin the three things that can
go wrong: the estimator must agree with `cvar` on the resamples it draws, the
pairing must actually cancel the common path risk, and the verdict the
dashboard and the desk note read off it must follow the interval rather than
the point estimate.
"""
import math

import numpy as np
import pytest

from backend.quant.hedging import (CVAR_ALPHA, cvar, cvar_bootstrap_se,
                                   paired_cvar_bootstrap)


def _pair(out, a, b):
    """The pair entry for a and b, in whichever order it was stored."""
    return out["pairs"].get(f"{a}|{b}") or out["pairs"][f"{b}|{a}"]


def test_replicate_cvar_is_the_same_statistic_the_page_reports():
    """A replicate is `cvar` on a resample, not some other tail estimate.

    Reproduces the first index draw from the same seeded generator and checks
    the replicate against `cvar` evaluated on it directly, so the vectorised
    partition cannot silently drift from the sorted implementation the rest
    of the module uses.
    """
    rng = np.random.default_rng(0)
    pl = rng.normal(size=400)
    out = paired_cvar_bootstrap({"a": pl}, n_boot=1, seed=7)

    # Same draw the estimator makes: one block, because 400 rows is far
    # under max_elems, so the whole (n_boot, n) matrix comes from one call.
    idx = np.random.default_rng(7).integers(0, pl.size, size=(1, pl.size))
    expected = cvar(pl[idx[0]], CVAR_ALPHA)

    # se over one replicate is nan (ddof=1), so read the replicate through a
    # two-strategy run where the difference pins the value.
    out2 = paired_cvar_bootstrap({"a": pl, "zero": np.zeros_like(pl)},
                                 n_boot=1, seed=7)
    p = _pair(out2, "a", "zero")
    # cvar of an all-zero P&L is 0, so the replicate difference IS the
    # replicate: bootstrap mean minus point difference recovers it.
    replicate = p["bias"] + p["diff"]
    assert replicate == pytest.approx(expected, rel=1e-12)
    assert out["cvar"]["a"] == pytest.approx(cvar(pl, CVAR_ALPHA), rel=1e-12)


def test_the_tail_fraction_matches_the_sorted_implementation():
    """partition-based selection takes the same paths as `cvar`'s sort."""
    rng = np.random.default_rng(3)
    for n in (37, 100, 999):
        pl = rng.normal(size=n)
        out = paired_cvar_bootstrap({"a": pl}, n_boot=8, seed=1)
        assert out["cvar"]["a"] == pytest.approx(cvar(pl, CVAR_ALPHA),
                                                 rel=1e-12)


def test_two_identical_hedgers_are_never_separated():
    """The degenerate case the unpaired formula gets wrong.

    Two hedgers with identical P&L differ by exactly zero on every path, so
    no resampling can distinguish them. hypot(se, se) is strictly positive
    and would happily report an interval; the paired difference is zero.
    """
    rng = np.random.default_rng(11)
    pl = rng.normal(size=600)
    out = paired_cvar_bootstrap({"a": pl, "b": pl.copy()}, n_boot=200, seed=2)
    p = _pair(out, "a", "b")

    assert p["diff"] == 0.0
    assert p["se"] == 0.0
    assert (p["ci_low"], p["ci_high"]) == (0.0, 0.0)
    assert p["excludes_zero"] is False
    assert p["corr"] == pytest.approx(1.0, abs=1e-12)
    assert p["hypot_se"] > 0.0          # what the unpaired bar would claim


def test_a_constant_improvement_is_always_separated():
    """b beats a by the same amount on every path: no sampling error at all.

    Shifting the whole P&L distribution shifts CVaR by the same constant, so
    the difference is deterministic however the paths are resampled. This is
    the case the unpaired test is most wrong about: each hedger's own CVaR
    still has a large tail error, so hypot stays wide while the truth is that
    the comparison is exact.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(size=800)
    b = a + 0.25                        # strictly better on every path
    out = paired_cvar_bootstrap({"a": a, "b": b}, n_boot=300, seed=3)
    p = _pair(out, "a", "b")

    assert p["diff"] == pytest.approx(0.25, rel=1e-12)   # a loses by 0.25
    assert p["se"] == pytest.approx(0.0, abs=1e-12)
    assert p["excludes_zero"] is True
    assert p["p_first_better"] == 0.0                    # a never wins
    assert p["hypot_se"] > 10 * max(p["se"], 1e-9)


def test_pairing_beats_the_unpaired_bar_when_paths_are_shared():
    """Positively correlated hedgers: the paired error must be smaller.

    This is the whole point. Two hedgers on common paths have positively
    correlated tail losses, so Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b) is
    strictly below the hypot formula's Var(a) + Var(b).
    """
    rng = np.random.default_rng(13)
    common = rng.standard_t(df=3, size=2000)
    a = common + 0.30 * rng.normal(size=2000)
    b = common + 0.30 * rng.normal(size=2000) + 0.05
    out = paired_cvar_bootstrap({"a": a, "b": b}, n_boot=400, seed=4)
    p = _pair(out, "a", "b")

    assert p["corr"] > 0.5
    assert p["se"] < p["hypot_se"]


def test_the_paired_standard_error_obeys_the_variance_identity():
    """se^2 == se_a^2 + se_b^2 - 2 rho se_a se_b, to the digit.

    The reported correlation and the two marginal errors come from the same
    replicates as the difference, so this identity has to hold exactly. It is
    the check a reader can do by hand from the numbers on the page.
    """
    rng = np.random.default_rng(17)
    common = rng.normal(size=1500)
    pls = {"a": common + 0.5 * rng.normal(size=1500),
           "b": common + 0.5 * rng.normal(size=1500),
           "c": -abs(common) + 0.2 * rng.normal(size=1500)}
    out = paired_cvar_bootstrap(pls, n_boot=400, seed=6)

    for first, second in (("a", "b"), ("a", "c"), ("b", "c")):
        p = _pair(out, first, second)
        sa, sb = out["se"][first], out["se"][second]
        implied = math.sqrt(max(sa ** 2 + sb ** 2
                                - 2.0 * p["corr"] * sa * sb, 0.0))
        assert p["se"] == pytest.approx(implied, rel=1e-9)
        assert p["hypot_se"] == pytest.approx(math.hypot(sa, sb), rel=1e-12)


def test_the_interval_and_the_win_rate_agree_about_the_sign():
    """An interval clear of zero means one hedger wins nearly every replicate."""
    rng = np.random.default_rng(19)
    a = rng.normal(size=1200)
    b = a + 0.40 + 0.05 * rng.normal(size=1200)
    out = paired_cvar_bootstrap({"a": a, "b": b}, n_boot=400, seed=8)
    p = _pair(out, "a", "b")

    assert p["excludes_zero"] is True
    assert p["ci_low"] > 0.0            # a carries the larger loss
    assert p["p_first_better"] < 0.05   # so a almost never wins
    assert p["diff"] > 0.0


def test_marginal_errors_track_the_unpaired_estimator():
    """The marginal errors are still ordinary CVaR bootstrap errors.

    Re-using one index draw for every strategy must not change what a single
    strategy's standard error means, so it should land near the independent
    estimator `cvar_bootstrap_se` computes from its own draw.
    """
    rng = np.random.default_rng(23)
    pl = rng.normal(size=3000)
    paired = paired_cvar_bootstrap({"a": pl}, n_boot=800, seed=1)["se"]["a"]
    alone = cvar_bootstrap_se(pl, n_boot=800, seed=1)
    assert paired == pytest.approx(alone, rel=0.15)


def test_it_is_reproducible_and_refuses_misaligned_input():
    rng = np.random.default_rng(29)
    pls = {"a": rng.normal(size=500), "b": rng.normal(size=500)}
    first = paired_cvar_bootstrap(pls, n_boot=120, seed=31)
    again = paired_cvar_bootstrap(pls, n_boot=120, seed=31)
    assert _pair(first, "a", "b") == _pair(again, "a", "b")
    assert first["se"] == again["se"]

    # The pairing is only meaningful if row i is the same path everywhere, so
    # a length mismatch is a bug in the caller, not something to average over.
    with pytest.raises(ValueError, match="same path order"):
        paired_cvar_bootstrap({"a": pls["a"], "b": pls["b"][:-1]}, n_boot=8)


def test_empty_and_single_strategy_inputs_do_not_raise():
    assert paired_cvar_bootstrap({}, n_boot=4)["pairs"] == {}
    out = paired_cvar_bootstrap({"a": np.array([])}, n_boot=4)
    assert math.isnan(out["cvar"]["a"]) and out["pairs"] == {}
    solo = paired_cvar_bootstrap({"a": np.arange(50.0)}, n_boot=16, seed=1)
    assert solo["pairs"] == {} and solo["se"]["a"] >= 0.0


def test_blocking_does_not_change_which_paths_enter_the_tail():
    """Memory blocking is an implementation detail of the draw, not the tail.

    Different `max_elems` consume the random stream differently, so the
    replicates differ - that is expected of a bootstrap. What must not differ
    is the estimator: the point CVaRs and the marginal errors should agree to
    within their own Monte Carlo error.
    """
    rng = np.random.default_rng(37)
    pls = {"a": rng.normal(size=1000), "b": rng.normal(size=1000)}
    big = paired_cvar_bootstrap(pls, n_boot=600, seed=2, max_elems=10 ** 7)
    small = paired_cvar_bootstrap(pls, n_boot=600, seed=2, max_elems=2000)

    assert big["cvar"] == small["cvar"]                  # point estimate exact
    for k in ("a", "b"):
        assert big["se"][k] == pytest.approx(small["se"][k], rel=0.2)


def test_compare_reports_one_bootstrap_for_the_chips_and_the_verdict():
    """The error bars on the chips and the test behind the verdict agree.

    `compare` draws one set of resamples and reads both off it, so a reader
    cannot find the individual errors saying one thing and the difference
    saying another. This also pins that every hedger on the page enters the
    pairwise table, not just the two the headline compares.
    """
    from backend.quant.hedging import HedgingEngine, N_STEPS

    class _Stub(HedgingEngine):
        def __init__(self):            # no checkpoints: paths and rules only
            pass

        def _spots(self, measure, sigma, rate, n_paths, seed):
            rng = np.random.default_rng(seed)
            spots = np.empty((n_paths, N_STEPS + 1))
            spots[:, 0] = 1.0
            spots[:, 1:] = np.exp(np.cumsum(
                rng.normal(0.0, 0.02, size=(n_paths, N_STEPS)), axis=1))
            return spots

        def _deep_fn(self, sigma, rate, cost):
            # A crude static hedge, so its P&L is genuinely different from
            # the delta hedge's rather than identical to it.
            return lambda i, tau, s, h: np.full_like(s, 0.4)

    out = _Stub().compare(0.25, 0.04, 0.005, n_paths=256, seeds=(17, 18),
                          primary="gbm", measures=("gbm",))
    paired = out["paired_bootstrap"]

    names = paired["names"]
    assert {"deep", "delta", "whalley_wilmott", "linear"} <= set(names)
    # Every unordered pair is present exactly once.
    assert len(paired["pairs"]) == len(names) * (len(names) - 1) // 2

    for name in names:
        assert out[name]["cvar95"] == pytest.approx(paired["cvar"][name],
                                                    rel=1e-12)
        assert out[name]["cvar95_se"] == pytest.approx(paired["se"][name],
                                                       rel=1e-12)

    for key, p in paired["pairs"].items():
        a, b = key.split("|")
        assert p["diff"] == pytest.approx(
            paired["cvar"][a] - paired["cvar"][b], rel=1e-12)
        # Shared paths can only help: the paired error never exceeds the
        # unpaired one by more than bootstrap noise in the correlation.
        assert p["se"] <= p["hypot_se"] * 1.05
        assert p["ci_low"] <= p["ci_high"]
        assert 0.0 <= p["p_first_better"] <= 1.0
