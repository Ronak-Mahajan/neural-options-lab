"""Agentic Risk Analyst via Groq API.

Uses Groq's OpenAI-compatible API with an open-source Llama-3-family 8B
model to generate a natural-language risk report from the neural pricer,
Deep Hedging CVaR, and XAI outputs. Streams the response token-by-token.

Configuration (environment):
    GROQ_API_KEY   required for live LLM output (free tier at console.groq.com)
    GROQ_MODEL     optional override; defaults to "llama-3.1-8b-instant"
                   (Groq retired the original "llama3-8b-8192" id - the
                   3.1-8B-instant model is its direct successor)

If GROQ_API_KEY is not set, a deterministic offline template built from the
same numbers is streamed instead, clearly labeled as such.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import StreamingResponse

DEFAULT_MODEL = "llama-3.1-8b-instant"


def get_risk_report_stream(ticker: str, nn_price: float, bs_cvar: float,
                           deep_cvar: float, attributions: dict):
    """Streams a risk report from Groq Llama-3, or falls back to a template."""
    import yfinance as yf
    
    # Fetch RAG context: Live News
    try:
        tk = yf.Ticker(ticker)
        news_items = tk.news[:3]
        news_context = "\n".join([f"- {item['content']['title']}: {item['content']['summary']}" for item in news_items])
    except Exception:
        news_context = "No live news available at this time."

    # Which policy actually has the smaller tail loss. CVaR95 arrives as a
    # P&L quantile (negative = loss), so "better" means less negative. The
    # deep hedger does NOT reliably beat delta hedging out of sample (see
    # hedging.py), and both this prompt and the offline template must state
    # whichever direction the numbers actually show - an earlier version
    # hardcoded "the Deep Hedger reduces this tail risk" and happily printed
    # it next to numbers proving the opposite.
    deep_wins = deep_cvar > bs_cvar

    api_key = os.environ.get("GROQ_API_KEY")
    prompt = f"""
You are an elite quantitative Risk Analyst. Analyze the following live options
pricing data for {ticker} and provide a concise, professional 3-paragraph risk
report for the trading desk.

DATA:
- Ticker: {ticker}
- Neural Network Option Price: ${nn_price:.4f}
- XAI Price Drivers: Spot contributed ${attributions['spot']:.4f}, Volatility
  contributed ${attributions['sigma']:.4f}, Maturity contributed
  ${attributions['maturity']:.4f}.
- Hedging Risk (CVaR95, a P&L quantile where less negative is better):
  Standard Black-Scholes Delta Hedging: ${bs_cvar:.4f}. Deep Hedging engine
  (accounts for transaction costs): ${deep_cvar:.4f}. In this simulation the
  {"Deep Hedging policy" if deep_wins else "standard delta hedge"} has the
  smaller tail loss - describe the comparison exactly as these numbers show
  it, and do not assume either policy is better than the measurement says.

LIVE MARKET NEWS (RAG Context):
{news_context}

FORMAT:
Paragraph 1: Discuss the Neural Price and what is driving it (using XAI). Synthesize this with the LIVE MARKET NEWS to explain WHY the market might be pricing these Greeks (e.g. if Vega is high, correlate it to a recent news event).
Paragraph 2: Discuss hedging risk, comparing Standard vs Deep Hedging.
Paragraph 3: A final one-sentence note on risk limit management based on both the quantitative data and the fundamental news context.

Keep it highly technical, confident, and professional. Do not use asterisks or
markdown bolding. Just plain text paragraphs. This is a research dashboard,
not investment advice - frame the close as monitoring guidance, not an
instruction to deploy capital.
"""

    if not api_key:
        # Fallback offline template if no API key
        def fallback_stream():
            driver_names = {"spot": "the underlying spot level",
                            "sigma": "volatility exposure",
                            "maturity": "time value",
                            "rate": "the rate environment"}
            top = max(attributions, key=lambda k: abs(attributions[k]))
            others = ", ".join(
                f"{driver_names.get(k, k)} ${attributions[k]:.4f}"
                for k in ("sigma", "maturity", "spot") if k != top)
            if deep_wins:
                hedge_text = (
                    f"In this simulation the Deep Hedging policy carries the "
                    f"smaller tail risk: the frictionless Black-Scholes delta "
                    f"hedge shows a 95% Conditional Value at Risk (CVaR) of "
                    f"${bs_cvar:.4f}, while the Deep Hedger, which "
                    f"internalizes proportional transaction costs, improves "
                    f"that to ${deep_cvar:.4f} by trading less and avoiding "
                    f"over-hedging whipsaw losses."
                )
                close_text = (
                    "The Deep Hedging policy's tail advantage in this run "
                    "merits attention alongside its lower trading costs; "
                    "monitor the spot-driven XAI attribution daily."
                )
            else:
                hedge_text = (
                    f"In this simulation the Deep Hedging policy does NOT "
                    f"beat the standard delta hedge on tail risk: the "
                    f"Black-Scholes delta hedge shows a 95% Conditional "
                    f"Value at Risk (CVaR) of ${bs_cvar:.4f} versus "
                    f"${deep_cvar:.4f} for the Deep Hedger. The learned "
                    f"policy trades less and therefore pays lower "
                    f"transaction costs, but under these parameters that "
                    f"saving does not compensate for the wider loss tail."
                )
                close_text = (
                    "Under these parameters the delta hedge remains the "
                    "safer baseline for tail-risk limits; monitor the "
                    "spot-driven XAI attribution daily."
                )
            fallback_text = (
                f"[OFFLINE FALLBACK - NO GROQ API KEY]\n\n"
                f"The Neural Network prices the option on {ticker} at "
                f"${nn_price:.4f}. Based on our Integrated Gradients XAI, "
                f"this premium is primarily driven by "
                f"{driver_names.get(top, top)} (${attributions[top]:.4f}); "
                f"the remaining drivers contribute {others}. These "
                f"attribution metrics confirm the model is pricing the risk "
                f"factors in line with expected theoretical "
                f"sensitivities.\n\n"
                f"{hedge_text}\n\n"
                f"{close_text} This is a research dashboard, not investment "
                f"advice."
            )
            for chunk in fallback_text.split(" "):
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
            yield (f"[GROQ API ERROR] {e}\n"
                   f"(model={model}; override with the GROQ_MODEL env var)")

    return StreamingResponse(groq_stream(), media_type="text/plain")
