import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not set. Add it to your .env file, e.g.\n"
        '  GROQ_API_KEY="gsk_..."'
    )

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a financial research assistant that answers questions using ONLY the provided SEC filing excerpts below. Follow these rules exactly:

1. Every factual claim, especially every number, must be followed by a bracketed citation like [1] or [2] pointing to the excerpt it came from.
2. Do not use any outside knowledge, even if you are confident about the answer. Only use what is in the excerpts.
3. If the excerpts do not contain enough information to answer the question, say so explicitly instead of guessing.
4. If excerpts disagree or are ambiguous, point that out rather than picking one silently.
5. Be concise and direct. Do not restate the question."""

client = Groq(api_key=GROQ_API_KEY)


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['ticker']} {chunk['form']} filed {chunk['filing_date']} "
            f"— {chunk['section'][:80]}"
        )
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def check_citations(answer: str, n_chunks: int) -> list[int]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in cited if n < 1 or n > n_chunks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    user_message = f"Excerpts:\n\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned no content.")

    return content


def answer_query(query: str) -> None:
    print(f"\nRetrieving context for: {query}")
    chunks = retrieve(query)
    if not chunks:
        print("No relevant chunks found in the index.")
        return

    print(f"Retrieved {len(chunks)} chunks, generating answer...\n")
    answer = generate_answer(query, chunks)

    bad_citations = check_citations(answer, len(chunks))
    if bad_citations:
        print(
            f"WARNING: answer cites source number(s) {bad_citations} that don't "
            f"exist in the {len(chunks)} retrieved chunks — possible hallucinated "
            f"citation.\n"
        )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

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


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    answer_query(query)


if __name__ == "__main__":
    main()