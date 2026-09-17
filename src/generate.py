import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from entailment import verify_citations
from retrieve import detect_form_filter, indexed_companies, refusal_reason, retrieve

load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1024
REQUEST_DELAY = float(os.getenv("GROQ_REQUEST_DELAY", "2.5"))
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
CACHE_PATH = Path("eval/answers_cache.json")

FANCY_CITATION_RE = re.compile(r"【(\d+)†[^】]*】")

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
    print(f"Cleared {n} cached answers.")
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
            print(f"  Rate limited (attempt {attempt}/{MAX_RETRIES}), waiting {wait:.1f}s ...")
            time.sleep(wait)
        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  Error on attempt {attempt}/{MAX_RETRIES}: {e!s}, waiting {wait:.1f}s ...")
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
        print(f"  [cache hit] returning cached answer for query")
        return cache[key]

    messages = _build_messages(query, chunks, history)
    content = _api_call_with_retry(messages, stream=False)
    content = FANCY_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", content)

    if key:
        cache[key] = content
        _save_cache(cache)
        print(f"  [cache saved] {len(cache)} total entries")

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
    **retrieve_kwargs,
) -> dict | None:
    print(f"\nRetrieving context for: {query}")

    history = session.history_for_retrieval() if session else None

    eff_for_log = query
    if session and len(session.turns) > 0:
        try:
            from query_rewrite import condense_followup
            eff_for_log = condense_followup(session.history_for_retrieval(), query)
            if eff_for_log != query:
                print(f"Condensed follow-up → retrieval query: {eff_for_log}")
        except Exception:
            pass

    form_filter = detect_form_filter(query)
    if form_filter:
        print(f"Detected form filter: {form_filter} — restricting retrieval to {form_filter} filings")

    chunks = retrieve(query, history=history, **retrieve_kwargs)
    if getattr(retrieve, "last_rewrite", query) != query:
        print(f"Rewritten retrieval query: {retrieve.last_rewrite}")
    if getattr(retrieve, "last_tickers", []):
        print(f"Routed companies: {retrieve.last_tickers}")
    if not chunks:
        print("No relevant chunks found in the index.")
        return None

    print(f"Retrieved {len(chunks)} candidate chunks.")

    refusal = refusal_reason(query, chunks)
    if refusal:
        print("\n" + "=" * 70)
        print("CANNOT ANSWER FROM INDEX")
        print("=" * 70)
        print(f"This question can't be answered with the indexed filings: {refusal}.")
        print(f"Indexed companies: {', '.join(indexed_companies())}")
        return {"refused": True, "reason": refusal, "chunks": chunks}

    print("Generating answer...\n")

    if stream:
        print("=" * 70)
        print("ANSWER (streaming)")
        print("=" * 70)
        parts = []
        for delta in generate_answer_stream(query, chunks, history):
            print(delta, end="", flush=True)
            parts.append(delta)
        answer = FANCY_CITATION_RE.sub(
            lambda m: f"[{m.group(1)}]", "".join(parts)
        )
    else:
        answer = generate_answer(query, chunks, history, use_cache=use_cache)
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

    bad_citations = check_citations(answer, len(chunks))
    if bad_citations:
        print(
            f"WARNING: answer cites source number(s) {bad_citations} that don't "
            f"exist in the {len(chunks)} retrieved chunks — possible hallucinated "
            f"citation.\n"
        )

    entail = None
    if verify_entailment:
        entail = verify_citations(answer, chunks, use_llm=use_llm_judge)
        print(
            f"Entailment: {entail['supported_spans']}/{entail['total_cited_spans']} "
            f"spans supported (rate {entail['entailment_rate']:.0%})"
        )
        for u in entail["unsupported"]:
            print(f"  UNSUPPORTED {u['cites']}: {u['sentence'][:120]}...")

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
        tickers = list(getattr(retrieve, "last_tickers", []) or [])
        session.add_user(query, tickers=tickers)
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
