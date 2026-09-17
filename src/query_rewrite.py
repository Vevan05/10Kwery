from __future__ import annotations

import os
import re

from router import SEGMENT_HINTS, route_companies

REWRITE_MODEL = "openai/gpt-oss-120b"

REWRITE_SYSTEM = (
    "Rewrite the user question into an explicit retrieval query for SEC "
    "10-K/10-Q search. Inject the company name when implied (e.g. 'Services "
    "segment' -> 'Apple Services segment revenue'). Expand acronyms. Keep "
    "numbers and form hints (10-K/10-Q). Reply with ONLY the rewritten query."
)

PRONOUN_RE = re.compile(
    r"\b(it|its|they|them|their|that|those|this|these|the company|the other one)\b",
    re.I,
)


def _rule_rewrite(
    query: str, history: list[dict] | None = None, tickers: list[str] | None = None
) -> str:

    out = query.strip()

    low = out.lower()
    routed = tickers or route_companies(query, use_llm=False)
    names = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "NVDA": "Nvidia",
        "TSLA": "Tesla",
    }
    if routed and not any(n.lower() in low for n in names.values()):

        if len(out.split()) <= 14:
            out = f"{names[routed[0]]} {out[0].lower() + out[1:]}" if out else out

    if history and PRONOUN_RE.search(out):
        for msg in reversed(history):
            if msg.get("tickers"):
                t = names.get(msg["tickers"][0], msg["tickers"][0])
                out = f"{out} (referring to {t})"
                break
    return out


def rewrite_query(
    query: str,
    history: list[dict] | None = None,
    tickers: list[str] | None = None,
    use_llm: bool = True,
) -> str:

    rule = _rule_rewrite(query, history, tickers)

    if not use_llm or not os.getenv("GROQ_API_KEY"):
        return rule
    if len(query.split()) > 20 and not PRONOUN_RE.search(query):
        return rule
    try:
        from groq import Groq

        ctx = ""
        if history:
            last = history[-2:]
            ctx = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in last)
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=128,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": (f"History:\n{ctx}\n" if ctx else "")
                    + f"Question: {query}",
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip().strip('"')
        return text or rule
    except Exception:
        return rule


def condense_followup(history: list[dict], new_query: str) -> str:

    return rewrite_query(new_query, history=history, use_llm=True)
