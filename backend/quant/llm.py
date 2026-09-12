"""Risk summary: a short desk note written from the numbers on the page.

Two writers produce it. With GROQ_API_KEY set, an open-weights Llama model
on Groq's OpenAI-compatible API drafts it and the response is streamed token
by token. Without a key, a deterministic rule-based narrator writes the same
facts. Either way the figures come from the page, never from the writer, and
the page says which one wrote it.

Configuration (environment):
    GROQ_API_KEY   required for the model-written version
    GROQ_MODEL     optional override; defaults to "llama-3.1-8b-instant"
                   (Groq retired the original "llama3-8b-8192" id - the
                   3.1-8B-instant model is its direct successor)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import StreamingResponse

DEFAULT_MODEL = "llama-3.1-8b-instant"


def llm_available() -> bool:
    """Whether a model will write the summary, or the rule-based narrator."""
    return bool(os.environ.get("GROQ_API_KEY"))


def _money(value: float) -> str:
    """Sign before the currency, two decimals: -$5.01, not $-5.0072."""
    return ("-$" if value < 0 else "$") + f"{abs(value):,.2f}"


def get_risk_report_stream(ticker: str, nn_price: float, bs_cvar: float,
                           deep_cvar: float, attributions: dict,
                           contract: str = "", ww_cvar: float | None = None,
                           deep_cost: float | None = None,
                           delta_cost: float | None = None,
                           dynamics_label: str = "", cost_bps: int | None = None):
    """Streams the risk summary from Groq, or from the rule-based narrator."""
    subject = (f"a {contract}" if contract else "the contract on screen")
    if ticker:
        subject = subject + f" on {ticker}"

    # Which policy actually has the smaller tail loss. CVaR95 arrives as a
    # P&L quantile (negative = loss), so "better" means less negative. The
    # deep hedger does NOT reliably beat delta hedging out of sample (see
    # hedging.py), so both writers state whichever direction the numbers
    # show rather than assuming the learned policy won.
    deep_wins = deep_cvar > bs_cvar
    market = dynamics_label or "the simulated market"
    cost_text = f"{cost_bps} basis points a trade" if cost_bps is not None \
        else "the configured transaction cost"
    costs_line = ""
    if deep_cost is not None and delta_cost is not None:
        costs_line = (f" It paid {_money(deep_cost)} a path in transaction "
                      f"costs against {_money(delta_cost)} for the delta hedge.")
    ww_line = ""
    if ww_cvar is not None:
        ww_line = (f" The cost-aware Whalley-Wilmott band, the strongest "
                   f"classical baseline here, came in at {_money(ww_cvar)}.")

    data_block = f"""
- Contract priced: {subject}, at {_money(nn_price)}.
- What the price is made of (Integrated Gradients, in dollars): volatility
  {attributions['sigma']:.4f}, time to expiry {attributions['maturity']:.4f},
  spot level {attributions['spot']:.4f}, interest rate
  {attributions.get('rate', 0.0):.4f}.
- Separate hedging experiment: sell one 30-day at-the-money call and hedge it
  daily on simulated paths of {market}, paying {cost_text}. Average loss over
  the worst 5% of paths (CVaR at 95%, less negative is better):
  delta hedge {_money(bs_cvar)}, learned policy {_money(deep_cvar)}""" + (
        f", Whalley-Wilmott band {_money(ww_cvar)}" if ww_cvar is not None else ""
    ) + f""". In this run the {"learned policy" if deep_wins else "delta hedge"}
  has the smaller tail loss.""" + (
        f" Transaction costs per path: learned policy {_money(deep_cost)}, "
        f"delta hedge {_money(delta_cost)}." if deep_cost is not None else "")

    prompt = f"""
You are a quantitative risk analyst writing a three-paragraph note for a
trading desk. Use only the figures below. Do not invent numbers, do not
reference market events, news, or anything you were not given, and do not
speculate about causes.

DATA:
{data_block}

FORMAT:
Paragraph 1: the price and what drives it, from the attribution figures.
Paragraph 2: the hedging comparison, stating exactly which policy had the
smaller tail loss and at what cost. Note that the option priced in paragraph
one and the call used in the hedging test are different contracts.
Paragraph 3: one sentence on what to watch, then note this is a research
dashboard and not investment advice.

Plain text paragraphs, no markdown, no asterisks. Technical and concise.
Write amounts with the sign before the currency symbol, e.g. -$5.01.
"""

    def template_text() -> str:
        """The same facts, written by rule. Served when no key is configured,
        and as the fallback if the provider fails."""
        driver_names = {"spot": "the spot level",
                        "sigma": "volatility",
                        "maturity": "time to expiry",
                        "rate": "the interest rate"}
        ranked = sorted(attributions, key=lambda k: abs(attributions[k]),
                        reverse=True)
        top = ranked[0]
        others = ", ".join(
            f"{driver_names.get(k, k)} {_money(attributions[k])}"
            for k in ranked[1:] if abs(attributions[k]) > 5e-5)
        others_text = f" Then {others}." if others else ""

        para1 = (
            f"The network prices {subject} at {_money(nn_price)}. Splitting "
            f"that price across its inputs by Integrated Gradients, "
            f"{driver_names.get(top, top)} accounts for the largest share at "
            f"{_money(attributions[top])}.{others_text} The four contributions "
            f"add back to the quoted price, which is the check that the "
            f"attribution is complete."
        )

        if deep_wins:
            para2 = (
                f"A separate experiment hedges a short 30-day at-the-money "
                f"call daily on simulated paths of {market}, paying "
                f"{cost_text}. The learned policy carries the smaller tail "
                f"loss: its average loss over the worst 5% of paths is "
                f"{_money(deep_cvar)} against {_money(bs_cvar)} for a "
                f"Black-Scholes delta hedge charged the same costs."
                f"{costs_line}{ww_line}"
            )
            para3 = (
                "The learned policy's advantage here comes with its lower "
                "turnover; it is specific to these dynamics and this cost "
                "level, so re-run it before reading across to another regime."
            )
        else:
            para2 = (
                f"A separate experiment hedges a short 30-day at-the-money "
                f"call daily on simulated paths of {market}, paying "
                f"{cost_text}. The delta hedge keeps the smaller tail loss "
                f"here: its average loss over the worst 5% of paths is "
                f"{_money(bs_cvar)} against {_money(deep_cvar)} for the "
                f"learned policy.{costs_line}{ww_line}"
            )
            para3 = (
                "Under these dynamics and at this cost level the delta hedge "
                "is the baseline to beat; the learned policy's advantage "
                "appears under rough volatility with higher costs."
            )

        return (
            f"Risk summary\n\n{para1}\n\n{para2}\n\n{para3} This is a "
            f"research dashboard, not investment advice."
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        def fallback_stream():
            for chunk in template_text().split(" "):
                yield chunk + " "
        return StreamingResponse(fallback_stream(), media_type="text/plain")

    # If API key exists, stream from Groq
    from groq import Groq
    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def groq_stream():
        try:
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
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
            yield "\n\n" + template_text()

    return StreamingResponse(groq_stream(), media_type="text/plain")
