from __future__ import annotations

import os
import re


COMPANY_ALIASES: dict[str, str] = {
    "aapl": "AAPL",
    "apple": "AAPL",
    "msft": "MSFT",
    "microsoft": "MSFT",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "youtube": "GOOGL",
    "amzn": "AMZN",
    "amazon": "AMZN",
    "aws": "AMZN",
    "meta": "META",
    "facebook": "META",
    "instagram": "META",
    "reality labs": "META",
    "nvda": "NVDA",
    "nvidia": "NVDA",
    "tsla": "TSLA",
    "tesla": "TSLA",
}


SEGMENT_HINTS: dict[str, str] = {
    "services segment": "AAPL",
    "intelligent cloud": "MSFT",
    "azure": "MSFT",
    "data center": "NVDA",
    "reality labs": "META",
    "youtube": "GOOGL",
    "energy generation": "TSLA",
    "self-driving": "TSLA",
    "autonomous vehicle": "TSLA",
}

ROUTER_MODEL = "openai/gpt-oss-120b"


def detect_companies_rule(query: str) -> list[str]:

    q = query.lower()
    found: set[str] = set()
    for alias, ticker in COMPANY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            found.add(ticker)
    for phrase, ticker in SEGMENT_HINTS.items():
        if phrase in q:
            found.add(ticker)
    return sorted(found)


def detect_companies_llm(query: str, indexed: list[str]) -> list[str] | None:

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=ROUTER_MODEL,
            max_tokens=128,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Map the user question to stock tickers. Indexed: "
                        + ", ".join(indexed)
                        + ". Reply with ONLY a JSON list of tickers, e.g. "
                        '["AAPL"]. Reply [] if no indexed company is implied.'
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        import json

        text = (resp.choices[0].message.content or "").strip()
        tickers = json.loads(text[text.index("[") : text.rindex("]") + 1])
        return sorted({t.upper() for t in tickers if t.upper() in indexed})
    except Exception:
        return None


def route_companies(
    query: str,
    indexed: list[str] | None = None,
    history: list[dict] | None = None,
    use_llm: bool = True,
) -> list[str]:

    tickers = detect_companies_rule(query)
    if tickers:
        return tickers

    if history:
        for msg in reversed(history):
            if msg.get("tickers"):

                if len(query.split()) <= 12 or re.search(
                    r"\b(it|its|they|them|that|those|the company|the other)\b",
                    query,
                    re.I,
                ):
                    return list(msg["tickers"])
                break
    if use_llm and indexed:
        llm = detect_companies_llm(query, indexed)
        if llm:
            return llm
    return []
