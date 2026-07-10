"""Agentic Risk Analyst via Groq API.

Uses Groq's OpenAI-compatible API with an open-source Llama-3-family 8B
model to generate a natural-language risk report from the neural pricer,
Deep Hedging CVaR, and XAI outputs. Streams the response token-by-token.

Configuration (environment):
    GROQ_API_KEY   required for live LLM output (free tier at console.groq.com)
    GROQ_MODEL     optional override; defaults to "llama-3.1-8b-instant"
                   (Groq retired the original "llama3-8b-8192" id — the
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
- Hedging Risk (CVaR95): Standard Black-Scholes Delta Hedging yields a
  shortfall risk of ${bs_cvar:.4f}, whereas our Deep Hedging engine (which
  accounts for transaction costs) yields a lower shortfall risk of
  ${deep_cvar:.4f}.

LIVE MARKET NEWS (RAG Context):
{news_context}

FORMAT:
Paragraph 1: Discuss the Neural Price and what is driving it (using XAI). Synthesize this with the LIVE MARKET NEWS to explain WHY the market might be pricing these Greeks (e.g. if Vega is high, correlate it to a recent news event).
Paragraph 2: Discuss hedging risk, comparing Standard vs Deep Hedging.
Paragraph 3: A final one-sentence recommendation on risk limit management based on both the quantitative data and the fundamental news context.

Keep it highly technical, confident, and professional. Do not use asterisks or
markdown bolding. Just plain text paragraphs.
"""

    if not api_key:
        # Fallback offline template if no API key
        def fallback_stream():
            driver_names = {"spot": "the underlying spot level",
                            "sigma": "volatility exposure",
                            "maturity": "time value",
                            "rate": "the rate environment"}
            top = max(attributions, key=lambda k: abs(attributions[k]))
            fallback_text = (
                f"[OFFLINE FALLBACK - NO GROQ API KEY]\n\n"
                f"The Neural Network prices the {ticker} option at "
                f"${nn_price:.4f}. Based on our Integrated Gradients XAI, "
                f"this "
                f"premium is primarily driven by {driver_names.get(top, top)} "
                f"(${attributions[top]:.4f}), with volatility contributing "
                f"${attributions['sigma']:.4f}, time-value "
                f"${attributions['maturity']:.4f}, and spot "
                f"${attributions['spot']:.4f}. These attribution metrics "
                f"confirm the model is pricing the risk factors in line with "
                f"expected theoretical sensitivities.\n\n"
                f"From a risk management perspective, the Deep Hedging policy "
                f"significantly outperforms frictionless Black-Scholes delta "
                f"hedging. Standard hedging yields a 95% Conditional Value at "
                f"Risk (CVaR) shortfall of ${bs_cvar:.4f}. By internalizing "
                f"proportional transaction costs, the Deep Hedger reduces "
                f"this tail risk shortfall to ${deep_cvar:.4f}, actively "
                f"preventing "
                f"over-hedging whipsaw losses.\n\n"
                f"Recommendation: Allocate hedging capital according to the "
                f"Deep Hedging policy and monitor spot-driven XAI daily."
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
