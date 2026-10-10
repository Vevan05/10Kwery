import hashlib
import json
import logging
import os
import re
import sys
import time

from dotenv import load_dotenv

from config import CACHE_PATH, ENABLE_FOLLOWUP_CONDENSE
from entailment import verify_citations
from retrieve import (
    detect_form_filter,
    last_tickers,
    last_rewrite,
    refusal_reason,
    retrieve,
)

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1024
REQUEST_DELAY = float(os.getenv("GROQ_REQUEST_DELAY", "2.5"))
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# The model sometimes emits full-width-bracket citations, with or without the
# cross-reference dagger that some providers insert. Both must become [4].
FANCY_CITATION_RE = re.compile(r"【(\d+)†?[^】]*】")

SYSTEM_PROMPT = """You are a financial research assistant that answers questions using ONLY the provided SEC filing excerpts below. Follow these rules exactly:

1. Every factual claim, especially every number, must be followed by a bracketed citation like [1] or [2] pointing to the excerpt it came from. Always use plain square brackets with just the number, e.g. [1].
2. Do not use any outside knowledge, even if you are confident about the answer. Only use what is in the excerpts.
3. If the excerpts do not contain enough information to answer the question, say so explicitly instead of guessing.
4. If excerpts disagree or are ambiguous, point that out rather than picking one silently.
5. Be concise and direct. Do not restate the question."""

_client = None


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _cache_key(query: str, chunks: list[dict]) -> str:
    chunk_ids = ",".join(c.get("chunk_id", str(i)) for i, c in enumerate(chunks))
    raw = query + "|" + chunk_ids
    return hashlib.md5(raw.encode()).hexdigest()


def clear_answer_cache() -> int:
    n = 0
    if CACHE_PATH.exists():
        n = len(json.loads(CACHE_PATH.read_text()))
        CACHE_PATH.unlink()
    logger.info("Cleared %d cached answers.", n)
    return n


def get_groq_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY") or ""
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file, e.g.\n"
                '  GROQ_API_KEY="gsk_..."'
            )
        from groq import Groq
        _client = Groq(api_key=api_key)
    return _client


def _api_call_with_retry(messages: list[dict], stream: bool = False) -> str | None:
    from groq import RateLimitError

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                time.sleep(REQUEST_DELAY * attempt)
            else:
                time.sleep(REQUEST_DELAY)

            response = get_groq_client().chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                stream=stream,
            )

            if stream:
                parts = []
                for event in response:
                    try:
                        delta = event.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        parts.append(delta)
                return "".join(parts)
            else:
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("Model returned no content.")
                return content

        except RateLimitError:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %.1fs ...",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Error on attempt %d/%d: %s, waiting %.1fs ...",
                attempt, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to get Groq response after {MAX_RETRIES} attempts.")


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['ticker']} {chunk['form']} filed {chunk['filing_date']} "
            f"— {chunk['section'][:80]}"
        )
        body = chunk["text"]
        if chunk.get("chunk_type") == "table" and chunk.get("summary"):
            body = f"[Table overview: {chunk['summary']}]\n{body}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def check_citations(answer: str, n_chunks: int) -> list[int]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in cited if n < 1 or n > n_chunks)


def _build_messages(query: str, chunks: list[dict], history: list[dict] | None = None):
    context = format_context(chunks)
    hist_block = ""
    if history:
        lines = [f"{m['role']}: {m['content'][:400]}" for m in history[-4:]]
        hist_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    user_message = f"Excerpts:\n\n{context}\n\n{hist_block}Question: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_answer(
    query: str, chunks: list[dict], history: list[dict] | None = None,
    use_cache: bool = True,
) -> str:
    cache = _load_cache() if use_cache else {}
    key = _cache_key(query, chunks) if use_cache else None

    if use_cache and key in cache:
        logger.debug("Cache hit; returning cached answer.")
        return cache[key]

    messages = _build_messages(query, chunks, history)
    content = _api_call_with_retry(messages, stream=False)
    content = FANCY_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", content)

    if key:
        cache[key] = content
        _save_cache(cache)
        logger.debug("Cache saved; %d total entries.", len(cache))

    return content


def generate_answer_stream(query: str, chunks: list[dict], history=None):
    messages = _build_messages(query, chunks, history)
    key = _cache_key(query, chunks)

    cache = _load_cache()
    if key in cache:
        print("  [cache hit] streaming cached answer")
        yield cache[key]
        return

    parts = []
    try:
        stream = get_groq_client().chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            stream=True,
        )
        for event in stream:
            try:
                delta = event.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                parts.append(delta)
                yield delta
    except Exception:
        pass

    full_answer = FANCY_CITATION_RE.sub(
        lambda m: f"[{m.group(1)}]", "".join(parts)
    )
    if key and full_answer:
        cache[key] = full_answer
        _save_cache(cache)
        print(f"  [cache saved] {len(cache)} total entries")


def answer_query(
    query: str,
    session=None,
    stream: bool = False,
    verify_entailment: bool = False,
    use_llm_judge: bool = False,
    use_cache: bool = True,
    quiet: bool = False,
    **retrieve_kwargs,
) -> dict | None:
    """Answer a query from the indexed corpus.

    ``quiet`` suppresses the human-facing stdout blocks. Callers with their own
    rendering surface (the web service in src/service.py) set it and read the
    returned dict instead of parsing printed output.
    """
    logger.info("Retrieving context for: %s", query)

    history = session.history_for_retrieval() if session else None

    eff_for_log = query
    if session and len(session.turns) > 0 and ENABLE_FOLLOWUP_CONDENSE:
        try:
            from query_rewrite import condense_followup
            eff_for_log = condense_followup(session.history_for_retrieval(), query)
            if eff_for_log != query:
                logger.info("Condensed follow-up -> retrieval query: %s", eff_for_log)
        except Exception:
            pass

    form_filter = detect_form_filter(query)
    if form_filter:
        logger.info(
            "Detected form filter: %s - restricting retrieval to %s filings",
            form_filter, form_filter,
        )

    chunks = retrieve(query, history=history, **retrieve_kwargs)

    rewritten = last_rewrite()
    if rewritten and rewritten != query:
        logger.info("Rewritten retrieval query: %s", rewritten)
    if last_tickers():
        logger.info("Routed companies: %s", last_tickers())

    if not chunks:
        logger.warning("No relevant chunks found in the index.")
        return None

    logger.info("Retrieved %d candidate chunks.", len(chunks))

    refusal = refusal_reason(query, chunks)
    if refusal:
        logger.warning("Refusing to answer: %s", refusal)
        return {"refused": True, "reason": refusal, "chunks": chunks}

    if not quiet:
        print("Generating answer...")

    if stream:
        if not quiet:
            print("=" * 70)
            print("ANSWER (streaming)")
            print("=" * 70)
        parts = []
        for delta in generate_answer_stream(query, chunks, history):
            if not quiet:
                print(delta, end="", flush=True)
            parts.append(delta)
        answer = FANCY_CITATION_RE.sub(
            lambda m: f"[{m.group(1)}]", "".join(parts)
        )
    else:
        answer = generate_answer(query, chunks, history, use_cache=use_cache)
        if not quiet:
            print("=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(answer)

    bad_citations = check_citations(answer, len(chunks))
    if bad_citations:
        logger.warning(
            "Answer cites source number(s) %s that don't exist in the %d "
            "retrieved chunks - possible hallucinated citation.",
            bad_citations, len(chunks),
        )

    entail = None
    if verify_entailment:
        entail = verify_citations(answer, chunks, use_llm=use_llm_judge)
        logger.info(
            "Entailment: %d/%d spans supported (rate %.0f%%)",
            entail["supported_spans"], entail["total_cited_spans"],
            entail["entailment_rate"] * 100,
        )
        for u in entail["unsupported"]:
            logger.info("Unsupported %s: %s...", u["cites"], u["sentence"][:120])

    for i, chunk in enumerate(chunks, 1):
        logger.info(
            "SOURCE [%d] %s %s filed %s - %s",
            i, chunk["ticker"], chunk["form"], chunk["filing_date"],
            chunk["section"][:70],
        )

    if not quiet:
        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)
        for i, chunk in enumerate(chunks, 1):
            print(
                f"[{i}] {chunk['ticker']} {chunk['form']} filed {chunk['filing_date']} "
                f"— {chunk['section'][:70]}"
            )
            if chunk.get("source_url"):
                print(f"    {chunk['source_url']}")

    if session is not None:
        session.add_user(query, tickers=list(last_tickers()))
        session.add_assistant(answer)

    return {"answer": answer, "chunks": chunks, "entailment": entail}


def chat_loop(**kwargs):
    from session import ConversationSession
    session = ConversationSession()
    print("10-Kwery chat (type 'exit' to quit). Follow-ups reuse context.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ":q"}:
            break
        if not q:
            continue
        answer_query(q, session=session, **kwargs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--legacy-rerank", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    if args.clear_cache:
        clear_answer_cache()
        return

    kwargs = dict(
        stream=args.stream,
        verify_entailment=args.verify,
        use_llm_judge=args.llm_judge,
        use_cache=not args.no_cache,
        rewrite=not args.no_rewrite,
        use_router=not args.no_router,
        reranker_mode="legacy" if args.legacy_rerank else "table-aware",
    )
    if args.chat or not args.query:
        chat_loop(**kwargs)
    else:
        answer_query(args.query, **kwargs)


if __name__ == "__main__":
    main()
