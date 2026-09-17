from __future__ import annotations

import os
import re

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CITE_RE = re.compile(r"\[(\d+)\]")
NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*\s?%?")
TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP = frozenset(
    "a an the and or but if then else when what which who how why with from that this these those are was were been have has had will would total".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 2}


def parse_cited_spans(answer: str) -> list[dict]:

    spans = []
    for sent in SENT_SPLIT_RE.split(answer.strip()):
        cites = sorted({int(n) for n in CITE_RE.findall(sent)})
        if cites:
            spans.append({"sentence": sent.strip(), "cites": cites})
    return spans


def lexical_support(sentence: str, chunk_text: str) -> dict:

    numbers = NUMBER_RE.findall(sentence)
    nums_ok = (
        all(n.strip() in chunk_text for n in numbers if n.strip()) if numbers else True
    )
    st, ct = _tokens(sentence), _tokens(chunk_text[:4000])
    overlap = len(st & ct) / max(len(st), 1)
    score = 0.6 * overlap + (0.4 if nums_ok else 0.0)
    missing = [n for n in numbers if n.strip() not in chunk_text] if numbers else []
    return {"score": round(min(score, 1.0), 3), "nums_ok": nums_ok, "missing": missing}


def verify_citations(
    answer: str, chunks: list[dict], threshold: float = 0.25, use_llm: bool = False
) -> dict:

    spans = parse_cited_spans(answer)
    details = []
    for sp in spans:
        best, best_idx = 0.0, None
        missing: list[str] = []
        for ci in sp["cites"]:
            if 1 <= ci <= len(chunks):
                r = lexical_support(sp["sentence"], chunks[ci - 1].get("text", ""))
                if r["score"] >= best:
                    best, best_idx, missing = r["score"], ci, r["missing"]
        details.append(
            {
                "sentence": sp["sentence"][:200],
                "cites": sp["cites"],
                "best_chunk": best_idx,
                "score": best,
                "supported": bool(best_idx and best >= threshold and not missing),
                "missing_numbers": missing,
            }
        )
    llm_flags: list[dict] = []
    if use_llm and details:
        llm_flags = _llm_judge(answer, chunks)
    total = len(details)
    supported = sum(1 for d in details if d["supported"])
    return {
        "total_cited_spans": total,
        "supported_spans": supported,
        "entailment_rate": round(supported / total, 3) if total else 1.0,
        "unsupported": [d for d in details if not d["supported"]],
        "details": details,
        "llm_flags": llm_flags,
    }


def _llm_judge(answer: str, chunks: list[dict]) -> list[dict]:
    if not os.getenv("GROQ_API_KEY"):
        return []
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        ctx = "\n\n".join(
            f"[{i}] {c.get('text', '')[:800]}" for i, c in enumerate(chunks, 1)
        )
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "For each cited sentence in the answer, say SUPPORTED or "
                        "NOT SUPPORTED by the cited excerpt. Reply one per line: "
                        "'<first 6 words>... - SUPPORTED/NOT SUPPORTED'."
                    ),
                },
                {"role": "user", "content": f"Excerpts:\n{ctx}\n\nAnswer:\n{answer}"},
            ],
        )
        return [{"judgment": (resp.choices[0].message.content or '').strip()}]
    except Exception:
        return []
