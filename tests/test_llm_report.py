"""The desk note must narrate the run the Hedging tab narrates.

Both writers - the rule-based narrator and the model prompt - read one
ranking. These tests pin the case that breaks a two-way comparison: the
Whalley-Wilmott band wins and the learned policy is second, so any sentence
crowning the learned policy is a claim the numbers printed beside it refute.
"""

import asyncio

import pytest

from backend.quant import llm

# A run in the dashboard's units: dollars, CVaR95 as a negative P&L, so the
# largest value is the smallest loss. The band leads, the learned policy is
# second, the delta hedge last.
BAND_WINS = dict(
    ticker="",
    nn_price=6.73,
    bs_cvar=-5.01,
    deep_cvar=-4.07,
    ww_cvar=-3.96,
    attributions={"spot": 0.0, "maturity": 2.76, "sigma": 3.17, "rate": 0.52},
    contract="1y ATM Asian call",
    deep_cost=0.40,
    delta_cost=1.19,
    dynamics_label="rough Bergomi + jumps (SPY-calibrated)",
    cost_bps=50,
)

# The same run with the learned policy in front of both baselines.
DEEP_WINS = dict(BAND_WINS, deep_cvar=-3.40, ww_cvar=-4.20)

WINNER_CLAIMS = (
    "the learned policy carries the smaller tail loss",
    "the learned policy has the smaller tail loss",
    "the learned policy carries the smallest tail loss",
    "the learned policy has the smallest tail loss",
)


def _positions(text: str, *needles: str) -> list[int]:
    low = text.lower()
    out = []
    for n in needles:
        assert n.lower() in low, f"{n!r} missing from:\n{text}"
        out.append(low.index(n.lower()))
    return out


@pytest.mark.parametrize("writer", [llm.render_risk_note, llm.build_risk_prompt])
def test_band_run_never_crowns_the_learned_policy(writer):
    """The bug this pins: `deep_cvar > bs_cvar` ignored the band entirely."""
    text = writer(**BAND_WINS)
    low = text.lower()
    for claim in WINNER_CLAIMS:
        assert claim not in low, f"{claim!r} written for a run the band won"


@pytest.mark.parametrize("writer", [llm.render_risk_note, llm.build_risk_prompt])
def test_band_run_ranks_all_three_best_first(writer):
    text = writer(**BAND_WINS)
    band, deep, delta = _positions(
        text, "whalley-wilmott band at -$3.96",
        "the learned policy at -$4.07",
        "the black-scholes delta hedge at -$5.01")
    assert band < deep < delta


def test_band_run_reports_the_top_two_as_close_without_standard_errors():
    """0.11 on a -3.96 tail statistic is not a win anyone can defend."""
    note = llm.render_risk_note(**BAND_WINS)
    assert "the gap between the cost-aware whalley-wilmott band and the " \
           "learned policy is small" in note.lower()
    assert "does not separate them" in note.lower()


def test_standard_errors_call_the_top_two_level():
    note = llm.render_risk_note(**BAND_WINS, ww_cvar_se=0.18, deep_cvar_se=0.21,
                                bs_cvar_se=0.24)
    low = note.lower()
    assert "are level at the top" in low
    assert "combined bootstrap standard errors" in low
    for claim in WINNER_CLAIMS:
        assert claim not in low


def test_standard_errors_can_also_confirm_a_winner():
    note = llm.render_risk_note(**BAND_WINS, ww_cvar_se=0.02, deep_cvar_se=0.02,
                                bs_cvar_se=0.02)
    assert ("The cost-aware Whalley-Wilmott band carries the smallest tail "
            "loss." in note)


def test_learned_policy_is_named_when_it_actually_leads():
    note = llm.render_risk_note(**DEEP_WINS)
    assert "The learned policy carries the smallest tail loss." in note


def test_paragraph_three_does_not_claim_an_advantage_over_the_band():
    note = llm.render_risk_note(**BAND_WINS)
    assert ("The learned policy's edge over the delta hedge comes from its "
            "lower turnover; at this cost level it does not separate from "
            "the Whalley-Wilmott band." in note)


def test_no_edge_is_claimed_when_the_policy_and_delta_hedge_are_level():
    close = dict(BAND_WINS, deep_cvar=-4.95, bs_cvar=-5.01, ww_cvar=-6.00)
    note = llm.render_risk_note(**close)
    assert "does not separate from the delta hedge on this run" in note
    assert "edge over the delta hedge" not in note


def test_rank_policies_orders_by_smallest_loss():
    ranked, separated, basis = llm.rank_policies(
        bs_cvar=-5.01, deep_cvar=-4.07, ww_cvar=-3.96)
    assert [name for name, _, _ in ranked] == [llm.BAND, llm.DEEP, llm.DELTA]
    assert basis == "gap"
    assert separated is False


def test_rank_policies_without_a_band():
    ranked, _, _ = llm.rank_policies(bs_cvar=-5.01, deep_cvar=-4.07)
    assert [name for name, _, _ in ranked] == [llm.DEEP, llm.DELTA]


def test_attribution_sentence_names_the_baseline_and_counts_what_it_names():
    note = llm.render_risk_note(**BAND_WINS)
    assert "minimal at-the-money baseline option" in note
    # Four drivers are supplied, four are named, and the count word says four.
    assert "Those four contributions plus the baseline add back" in note
    for driver in ("volatility", "time to expiry", "the interest rate",
                   "the spot level"):
        assert driver in note
    # The completeness identity is F(x) - F(baseline), so the note must not
    # claim the contributions reach the quoted price on their own.
    assert "contributions add back to the quoted price" not in note


def test_attribution_sentence_quotes_the_baseline_price_when_it_is_given():
    note = llm.render_risk_note(**BAND_WINS, baseline_price=0.28)
    assert "minimal at-the-money baseline option worth $0.28" in note


def test_served_fallback_is_the_ranked_note(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = llm.get_risk_report_stream(**BAND_WINS)

    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    body = asyncio.run(collect())
    assert body.rstrip(" ") == llm.render_risk_note(**BAND_WINS)
    for claim in WINNER_CLAIMS:
        assert claim not in body.lower()


# --------------------------------------------------------------------------
# The Report tab and the Hedging tab narrate ONE run, so they must agree on
# what that run separates. frontend/app.js gates its verdict on
#   |a - b| > 2 * hypot(se_a, se_b)
# and rank_policies has to apply the same bar, or one tab names a winner the
# other calls level.
# --------------------------------------------------------------------------

def _separated(a_cvar, a_se, b_cvar, b_se):
    ranked, separated, basis = llm.rank_policies(
        bs_cvar=b_cvar, deep_cvar=a_cvar,
        bs_cvar_se=b_se, deep_cvar_se=a_se)
    assert basis == "se"
    return separated


def test_separation_bar_is_two_standard_errors_of_the_difference():
    # Gap 0.30 against se 0.09 and 0.07: 2 * hypot = 0.2280, so separated;
    # summing the errors (0.16) would also pass, which is the looser bar the
    # two tabs must not differ by.
    assert _separated(-4.00, 0.09, -4.30, 0.07)
    # Gap 0.20: inside 2 * hypot (0.2280) but outside the sum (0.16). This is
    # exactly the band where the two tabs used to disagree.
    assert not _separated(-4.00, 0.09, -4.20, 0.07)
    # Gap 0.11 with the same errors is the run in the dump: level.
    assert not _separated(-3.96, 0.09, -4.07, 0.07)


def test_a_level_top_two_never_crowns_a_winner():
    note = llm.render_risk_note(**BAND_WINS, ww_cvar_se=0.09,
                                deep_cvar_se=0.07, bs_cvar_se=0.11)
    assert "are level at the top" in note
    for claim in WINNER_CLAIMS:
        assert claim not in note.lower()


def test_no_edge_is_claimed_over_a_delta_hedge_the_run_cannot_separate():
    # The band leads; the policy and the delta hedge are NOT the top two, but
    # they are 0.10 apart with errors of 0.09 and 0.11, so the run does not
    # separate them either and paragraph 3 must not call that an edge.
    close = dict(BAND_WINS, bs_cvar=-4.17, deep_cvar=-4.07, ww_cvar=-3.50)
    note = llm.render_risk_note(**close, ww_cvar_se=0.05,
                                deep_cvar_se=0.09, bs_cvar_se=0.11)
    assert "does not separate from the delta hedge" in note
    assert "edge over the delta hedge" not in note


# ---------------------------------------------------------------------------
# The paired bootstrap decides what "level" means
# ---------------------------------------------------------------------------
#
# The three hedgers run on the same paths, so the sampling error of a
# DIFFERENCE is measured by re-sampling those paths once per replicate, not by
# combining two separate error bars. When the dashboard sends that verdict the
# note must follow it, because the Hedging tab is following the same one and
# the two narrate a single run.

PAIRED_SEPARATES_BAND = {"band|deep": True, "band|delta": True,
                         "deep|delta": True}
PAIRED_LEVEL_AT_TOP = {"band|deep": False, "band|delta": True,
                       "deep|delta": True}


def test_the_paired_verdict_overrides_the_unpaired_one_at_the_top():
    """Errors that look overlapping can still be a real difference.

    The band and the policy are 11 cents apart with error bars of 9 and 7
    cents, so the unpaired bar calls it a tie. If the paired test separates
    them - which it can, because the two hedgers' losses move together - the
    note must name the winner rather than hide behind the looser test.
    """
    unpaired = llm.compose_risk_note(
        **BAND_WINS, bs_cvar_se=0.11, deep_cvar_se=0.07, ww_cvar_se=0.09)
    assert "are level at the top" in unpaired["note"]

    paired = llm.compose_risk_note(
        **BAND_WINS, bs_cvar_se=0.11, deep_cvar_se=0.07, ww_cvar_se=0.09,
        paired=PAIRED_SEPARATES_BAND)
    note = paired["note"]
    assert "are level at the top" not in note
    assert llm.BAND in note
    # And it must still be the BAND that is crowned, not the policy.
    assert f"{llm.BAND[:1].upper()}{llm.BAND[1:]} carries the smallest" in note


def test_the_paired_verdict_can_also_refuse_a_gap_the_unpaired_bar_allows():
    """Pairing is not a licence to declare winners; it can withhold one too."""
    wide = dict(BAND_WINS)
    wide["ww_cvar"] = -3.20              # a gap the unpaired test clears
    separated = llm.compose_risk_note(
        **wide, bs_cvar_se=0.05, deep_cvar_se=0.05, ww_cvar_se=0.05)
    assert "are level at the top" not in separated["note"]

    held = llm.compose_risk_note(
        **wide, bs_cvar_se=0.05, deep_cvar_se=0.05, ww_cvar_se=0.05,
        paired=PAIRED_LEVEL_AT_TOP)
    assert "are level at the top" in held["note"]
    assert "the 95% interval for the difference" in held["note"]


def test_the_paired_map_is_read_whichever_way_the_pair_is_named():
    """Key order must not decide the answer."""
    ranked, separated, basis = llm.rank_policies(
        -5.01, -4.07, -3.96, 0.11, 0.07, 0.09,
        paired={"band|deep": True})
    assert basis == "paired" and separated is True
    assert ranked[0][0] == llm.BAND


def test_the_policy_delta_pair_is_tested_in_its_own_right():
    """The band can lead while the policy still genuinely beats delta.

    The pairwise claim against the delta hedge is a separate comparison and
    gets its own paired verdict, so a conceded top spot does not silently
    suppress a result that the run does support.
    """
    # Band level with the policy at the top, but the policy clear of delta:
    # the note keeps the result it has.
    note = llm.compose_risk_note(
        **BAND_WINS, bs_cvar_se=0.11, deep_cvar_se=0.07, ww_cvar_se=0.09,
        paired=PAIRED_LEVEL_AT_TOP)["note"]
    assert "does not separate from the delta hedge" not in note
    assert "edge over the delta hedge" in note

    # Same run, but the paired test cannot separate the policy from delta
    # either: the edge sentence has to go.
    tied = llm.compose_risk_note(
        **BAND_WINS, bs_cvar_se=0.11, deep_cvar_se=0.07, ww_cvar_se=0.09,
        paired={"band|deep": False, "deep|delta": False})["note"]
    assert "does not separate from the delta hedge" in tied
    assert "edge over the delta hedge" not in tied


def test_an_absent_paired_map_leaves_the_old_behaviour_alone():
    """An older client that sends no paired map still gets an honest note."""
    for paired in (None, {}):
        out = llm.compose_risk_note(
            **BAND_WINS, bs_cvar_se=0.11, deep_cvar_se=0.07, ww_cvar_se=0.09,
            paired=paired)
        assert "are level at the top" in out["note"]
        assert "combined bootstrap" in out["note"]
