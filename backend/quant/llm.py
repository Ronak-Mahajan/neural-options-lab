"""Risk summary: a short desk note written from the numbers on the page.

Two writers produce it. With GROQ_API_KEY set, an open-weights Llama model
on Groq's OpenAI-compatible API drafts it and the response is streamed token
by token. Without a key, a deterministic rule-based narrator writes the same
facts. Either way the figures come from the page, never from the writer, and
the page says which one wrote it.

Both writers work from one ranking, built here in `rank_policies` from the
same CVaR numbers the Hedging tab renders. Every baseline on the page enters
that ranking: comparing the learned policy against the delta hedge alone is
what let the note crown a winner and then print a better number from the
Whalley-Wilmott band one sentence later, disagreeing with the Hedging tab
about a run they both narrate.

Configuration (environment):
    GROQ_API_KEY   required for the model-written version
    GROQ_MODEL     optional override; defaults to "llama-3.1-8b-instant"
                   (Groq retired the original "llama3-8b-8192" id - the
                   3.1-8B-instant model is its direct successor)
"""

import math
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import StreamingResponse

DEFAULT_MODEL = "llama-3.1-8b-instant"

# How the three hedgers are named in prose, once, so the ranking and the
# paragraphs that read off it cannot drift apart.
DEEP = "the learned policy"
DELTA = "the Black-Scholes delta hedge"
BAND = "the cost-aware Whalley-Wilmott band"

# Without bootstrap standard errors there is nothing to test a gap against,
# so a gap under this fraction of the leading tail loss is reported as close
# rather than as a win. CVaR is a 5%-tail statistic and moves by more than
# this between seeds.
RELATIVE_TIE = 0.05

_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six"}

_DRIVER_NAMES = {"spot": "the spot level",
                 "sigma": "volatility",
                 "maturity": "time to expiry",
                 "rate": "the interest rate"}


def llm_available() -> bool:
    """Whether a model will write the summary, or the rule-based narrator."""
    return bool(os.environ.get("GROQ_API_KEY"))


def _money(value: float) -> str:
    """Sign before the currency, two decimals: -$5.01, not $-5.0072."""
    return ("-$" if value < 0 else "$") + f"{abs(value):,.2f}"


def _cap(text: str) -> str:
    """Capitalise a policy name used to open a sentence."""
    return text[:1].upper() + text[1:]


def _join(parts: list[str]) -> str:
    """a, b and c."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def rank_policies(bs_cvar: float, deep_cvar: float,
                  ww_cvar: float | None = None,
                  bs_cvar_se: float | None = None,
                  deep_cvar_se: float | None = None,
                  ww_cvar_se: float | None = None,
                  paired: dict | None = None):
    """Order every hedger on the page by tail loss, smallest loss first.

    CVaR95 reaches this module as a P&L quantile, so a *less negative* number
    is the smaller loss and the ranking is by descending value.

    Returns ``(ranked, separated, basis)``: `ranked` is a list of
    ``(name, cvar, se)`` best first; `separated` says whether the top two are
    far enough apart to call a winner; `basis` is ``"paired"`` when the
    paired bootstrap on the shared paths decided it, ``"se"`` when two
    unpaired standard errors did, and ``"gap"`` when only the raw gap was
    available.
    """
    entries = [(DEEP, float(deep_cvar), deep_cvar_se),
               (DELTA, float(bs_cvar), bs_cvar_se)]
    if ww_cvar is not None:
        entries.append((BAND, float(ww_cvar), ww_cvar_se))
    ranked = sorted(entries, key=lambda e: -e[1])

    top, second = ranked[0], ranked[1]
    gap = top[1] - second[1]
    told = _paired_separated(paired, top[0], second[0])
    if told is not None:
        # The hedgers ran on the same paths, so the difference has its own
        # sampling error, measured on the shared resamples. That is the test
        # to use whenever it is available; the unpaired bar below assumes an
        # independence these three do not have.
        return ranked, told, "paired"
    if top[2] is not None and second[2] is not None:
        # Two combined standard errors of the DIFFERENCE, which for two
        # independent bootstrap estimates is hypot(se_a, se_b) - the same test
        # the Hedging tab applies to the same numbers. Summing the two errors
        # instead would be a looser bar (about 1.4 SE), and the two tabs
        # narrate one run, so they must not disagree about what it separates.
        combined = 2.0 * math.hypot(float(top[2]), float(second[2]))
        return ranked, gap > combined, "se"
    scale = abs(top[1]) or 1.0
    return ranked, gap > RELATIVE_TIE * scale, "gap"


#: Short keys for the paired-bootstrap map the dashboard sends. The pair key
#: is the two short names joined by "|" in alphabetical order, so the caller
#: does not have to know which hedger the note will rank first.
_SHORT = {DEEP: "deep", DELTA: "delta", BAND: "band"}


def _paired_separated(paired, a_name: str, b_name: str):
    """Whether a PAIRED test separates these two, or None if it cannot say.

    Every hedger runs on the same paths, so the sampling error of a
    difference is not hypot(se_a, se_b) - that formula assumes independence,
    and a path that is bad for one hedger is usually bad for all of them. The
    dashboard sends the paired verdict computed on the shared resamples; when
    it is present it is authoritative, because it is the only one of the two
    tests that is measuring the right quantity.
    """
    if not paired:
        return None
    key = "|".join(sorted((_SHORT.get(a_name, ""), _SHORT.get(b_name, ""))))
    value = paired.get(key)
    return None if value is None else bool(value)


def _pair_is_level(a_cvar: float, a_se, b_cvar: float, b_se,
                   fallback: bool, paired=None,
                   a_name: str = "", b_name: str = "") -> bool:
    """Whether two hedgers are too close for this run to separate them.

    Prefers the paired bootstrap, which tests the difference on the shared
    paths. Falls back to two unpaired combined standard errors, and then to
    the caller's ranking-derived answer when there is nothing to test with.
    """
    told = _paired_separated(paired, a_name, b_name)
    if told is not None:
        return not told
    if a_se is None or b_se is None:
        return fallback
    return abs(a_cvar - b_cvar) <= 2.0 * math.hypot(abs(float(a_se)),
                                                    abs(float(b_se)))


def _ranking_prose(ranked, separated, basis):
    """The ordering sentence and the verdict sentence, from one ranking."""
    listing = _join([f"{name} at {_money(value)}" for name, value, _ in ranked])
    order = (f"Ranked by average loss over the worst 5% of paths: {listing}.")

    top_name, second_name = ranked[0][0], ranked[1][0]
    if separated:
        verdict = f"{_cap(top_name)} carries the smallest tail loss."
    elif basis == "paired":
        verdict = (f"{_cap(top_name)} and {second_name} are level at the top: "
                   f"re-sampling the shared paths, the 95% interval for the "
                   f"difference between them still contains zero.")
    elif basis == "se":
        verdict = (f"{_cap(top_name)} and {second_name} are level at the top: "
                   f"the gap between them is inside their combined bootstrap "
                   f"standard errors.")
    else:
        verdict = (f"The gap between {top_name} and {second_name} is small, "
                   f"so this run does not separate them.")
    return order, verdict


def compose_risk_note(ticker: str, nn_price: float, bs_cvar: float,
                      deep_cvar: float, attributions: dict,
                      contract: str = "", ww_cvar: float | None = None,
                      deep_cost: float | None = None,
                      delta_cost: float | None = None,
                      dynamics_label: str = "", cost_bps: int | None = None,
                      bs_cvar_se: float | None = None,
                      deep_cvar_se: float | None = None,
                      ww_cvar_se: float | None = None,
                      baseline_price: float | None = None,
                      paired: dict | None = None) -> dict:
    """Assemble the rule-written note and the model prompt from one ranking.

    Returns ``{"note", "prompt", "ranked", "separated"}``. Both writers are
    built here so the prompt cannot assert an outcome the template denies.
    """
    subject = (f"a {contract}" if contract else "the contract on screen")
    if ticker:
        subject = subject + f" on {ticker}"

    ranked, separated, basis = rank_policies(
        bs_cvar, deep_cvar, ww_cvar, bs_cvar_se, deep_cvar_se, ww_cvar_se,
        paired=paired)
    order_line, verdict_line = _ranking_prose(ranked, separated, basis)

    market = dynamics_label or "the simulated market"
    cost_text = f"{cost_bps} basis points a trade" if cost_bps is not None \
        else "the configured transaction cost"
    costs_line = ""
    if deep_cost is not None and delta_cost is not None:
        costs_line = (f" The learned policy paid {_money(deep_cost)} a path "
                      f"in transaction costs against {_money(delta_cost)} "
                      f"for the delta hedge.")

    # Integrated Gradients is complete against a baseline: the attributions
    # sum to F(x) - F(baseline), which is what explain.py's completeness_error
    # is measured against. Naming the baseline keeps the sentence from
    # claiming an identity the bars on screen visibly fail.
    baseline_phrase = "a minimal at-the-money baseline option"
    if baseline_price is not None:
        baseline_phrase += f" worth {_money(baseline_price)}"

    drivers = sorted(attributions, key=lambda k: abs(attributions[k]),
                     reverse=True)
    # Every driver that is named is counted and every driver counted is
    # named: an earlier magnitude filter dropped the at-the-money spot term
    # from the list while the sentence still said "four".
    count_word = _COUNT_WORDS.get(len(drivers), str(len(drivers)))

    # --- where the learned policy's edge sits, if it has one ---------------
    deep_beats_delta = float(deep_cvar) > float(bs_cvar)
    top_two = {ranked[0][0], ranked[1][0]}
    # A lead the run cannot resolve is not an edge. Test this pair directly
    # rather than only when it happens to be the top two: with the band in
    # front, the policy and the delta hedge can still be a coin toss between
    # themselves, and the note must not call that an edge either.
    deep_delta_level = _pair_is_level(
        float(deep_cvar), deep_cvar_se, float(bs_cvar), bs_cvar_se,
        top_two == {DEEP, DELTA} and not separated,
        paired=paired, a_name=DEEP, b_name=DELTA)
    band_clause = ""
    if ww_cvar is not None:
        if top_two == {DEEP, BAND} and not separated:
            band_clause = ("; at this cost level it does not separate from "
                           "the Whalley-Wilmott band")
        elif float(ww_cvar) > float(deep_cvar):
            band_clause = ("; at this cost level the Whalley-Wilmott band "
                           "keeps the smaller tail loss of the two")
        else:
            band_clause = ("; at this cost level it stays ahead of the "
                           "Whalley-Wilmott band as well")

    if deep_beats_delta and deep_delta_level:
        para3 = ("The learned policy does not separate from the delta hedge "
                 "on this run. The comparison is specific to these dynamics "
                 "and this cost level, so re-run it before reading across to "
                 "another regime.")
    elif deep_beats_delta:
        if (deep_cost is not None and delta_cost is not None
                and float(deep_cost) < float(delta_cost)):
            edge = ("The learned policy's edge over the delta hedge comes "
                    "from its lower turnover")
        elif deep_cost is not None and delta_cost is not None:
            edge = ("The learned policy's edge over the delta hedge does not "
                    "come from trading less; it pays at least as much as the "
                    "delta hedge in costs")
        else:
            edge = ("The learned policy's edge over the delta hedge holds at "
                    "this cost level")
        para3 = (f"{edge}{band_clause}. It is specific to these dynamics and "
                 f"this cost level, so re-run it before reading across to "
                 f"another regime.")
    else:
        para3 = (f"The learned policy does not clear the delta hedge under "
                 f"these dynamics at this cost level; {ranked[0][0]} sets the "
                 f"mark on this run. The comparison is specific to these "
                 f"dynamics and this cost level, so re-run it before reading "
                 f"across to another regime.")

    # --- the data the model is allowed to use ------------------------------
    attribution_line = ", ".join(
        f"{_DRIVER_NAMES.get(k, k)} {attributions[k]:.4f}" for k in drivers)
    data_block = f"""
- Contract priced: {subject}, at {_money(nn_price)}.
- What the price is made of (Integrated Gradients, in dollars): the
  attributions split the quoted price MINUS {baseline_phrase}, not the quoted
  price itself - {attribution_line}. The contributions plus that baseline add
  back to the quoted price.
- Separate hedging experiment: sell one 30-day at-the-money call and hedge it
  daily on simulated paths of {market}, paying {cost_text}. {order_line}
  {verdict_line}
- What the learned policy's edge rests on: {para3}""" + (
        f"\n- Transaction costs per path: learned policy {_money(deep_cost)}, "
        f"delta hedge {_money(delta_cost)}." if deep_cost is not None else "")

    prompt = f"""
You are a quantitative risk analyst writing a three-paragraph note for a
trading desk. Use only the figures below. Do not invent numbers, do not
reference market events, news, or anything you were not given, and do not
speculate about causes.

DATA:
{data_block}

FORMAT:
Paragraph 1: the price and what drives it, from the attribution figures. The
contributions are measured from the baseline option named in the data, so say
that they add back to the quoted price less that baseline - never that they
add up to the quoted price on their own.
Paragraph 2: the hedging comparison. Rank the hedgers in exactly the order
the data ranks them and repeat the verdict in the data as it is written; do
not name a different winner, and do not call a winner where the data says the
top two are level or close. Note that the option priced in paragraph one and
the call used in the hedging test are different contracts.
Paragraph 3: one sentence on what to watch, then note this is a research
dashboard and not investment advice.

Plain text paragraphs, no markdown, no asterisks. Technical and concise.
Write amounts with the sign before the currency symbol, e.g. -$5.01.
"""

    # --- the same facts, written by rule ----------------------------------
    if drivers:
        top = drivers[0]
        others = ", ".join(f"{_DRIVER_NAMES.get(k, k)} {_money(attributions[k])}"
                           for k in drivers[1:])
        others_text = f" Then {others}." if others else ""
        if len(drivers) == 1:
            adds_back = ("That contribution plus the baseline adds back to "
                         "the quoted price")
        else:
            adds_back = (f"Those {count_word} contributions plus the baseline "
                         f"add back to the quoted price")
        para1 = (
            f"The network prices {subject} at {_money(nn_price)}. Integrated "
            f"Gradients splits the difference from {baseline_phrase}: "
            f"{_DRIVER_NAMES.get(top, top)} accounts for the largest share at "
            f"{_money(attributions[top])}.{others_text} {adds_back}, which is "
            f"the check that the attribution is complete."
        )
    else:
        para1 = f"The network prices {subject} at {_money(nn_price)}."

    para2 = (
        f"A separate experiment hedges a short 30-day at-the-money call daily "
        f"on simulated paths of {market}, paying {cost_text}. {order_line} "
        f"{verdict_line}{costs_line}"
    )

    # The panel that shows the note already carries the "Risk summary" heading.
    note = (f"{para1}\n\n{para2}\n\n{para3} This is a "
            f"research dashboard, not investment advice.")

    return {"note": note, "prompt": prompt, "ranked": ranked,
            "separated": separated}


def render_risk_note(*args, **kwargs) -> str:
    """The rule-written desk note, as served when no model key is configured."""
    return compose_risk_note(*args, **kwargs)["note"]


def build_risk_prompt(*args, **kwargs) -> str:
    """The prompt handed to the model, carrying the same ranking."""
    return compose_risk_note(*args, **kwargs)["prompt"]


def get_risk_report_stream(ticker: str, nn_price: float, bs_cvar: float,
                           deep_cvar: float, attributions: dict,
                           contract: str = "", ww_cvar: float | None = None,
                           deep_cost: float | None = None,
                           delta_cost: float | None = None,
                           dynamics_label: str = "",
                           cost_bps: int | None = None,
                           bs_cvar_se: float | None = None,
                           deep_cvar_se: float | None = None,
                           ww_cvar_se: float | None = None,
                           baseline_price: float | None = None,
                           paired: dict | None = None):
    """Streams the risk summary from Groq, or from the rule-based narrator."""
    parts = compose_risk_note(
        ticker, nn_price, bs_cvar, deep_cvar, attributions,
        contract=contract, ww_cvar=ww_cvar, deep_cost=deep_cost,
        delta_cost=delta_cost, dynamics_label=dynamics_label,
        cost_bps=cost_bps, bs_cvar_se=bs_cvar_se, deep_cvar_se=deep_cvar_se,
        ww_cvar_se=ww_cvar_se, baseline_price=baseline_price, paired=paired,
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        def fallback_stream():
            for chunk in parts["note"].split(" "):
                yield chunk + " "
        return StreamingResponse(fallback_stream(), media_type="text/plain")

    # If API key exists, stream from Groq
    from groq import Groq
    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def groq_stream():
        try:
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": parts["prompt"]}],
                model=model,
                temperature=0.4,
                stream=True,
            )
            for chunk in stream:
                if (chunk.choices and
                        chunk.choices[0].delta.content is not None):
                    yield chunk.choices[0].delta.content
        except Exception as e:
            # Never surface a raw provider error on the page: log it and serve
            # the rule-based narrative instead.
            print(f"[risk-report] LLM provider error (model={model}): {e}")
            yield "\n\n" + parts["note"]

    return StreamingResponse(groq_stream(), media_type="text/plain")
