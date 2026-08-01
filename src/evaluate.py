import json
from pathlib import Path

from generate import check_citations, generate_answer
from retrieve import embed_query, load_chunk_lookup, retrieve, vector_search

EVAL_SET_PATH = Path("eval/qa_pairs.json")
TOP_K = 5


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise RuntimeError(
            f"{EVAL_SET_PATH} not found — create it first (see eval/qa_pairs.json template)."
        )
    return json.loads(EVAL_SET_PATH.read_text())


def vector_only_retrieve(
    query: str, chunk_lookup: dict[str, dict], k: int = TOP_K
) -> list[dict]:
    embedding = embed_query(query)
    ids = vector_search(embedding, k)
    return [chunk_lookup[cid] for cid in ids if cid in chunk_lookup]


def first_hit_rank(
    chunks: list[dict], expected_tickers: list[str]
) -> int | None:
    if not expected_tickers:
        return None

    for rank, chunk in enumerate(chunks, start=1):
        if chunk["ticker"] in expected_tickers:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def run_evaluation() -> None:
    eval_set = load_eval_set()
    chunk_lookup = load_chunk_lookup()

    baseline_rr = []
    hybrid_rr = []

    baseline_hit1 = 0
    hybrid_hit1 = 0
    baseline_hit5 = 0
    hybrid_hit5 = 0

    citation_clean = 0
    keyword_scores = []

    print(f"Running {len(eval_set)} questions through both retrieval configs...\n")

    for i, item in enumerate(eval_set, 1):
        query = item["question"]
        expected_tickers = item["expected_tickers"]
        expected_keywords = item.get("expected_keywords", [])

        baseline_chunks = vector_only_retrieve(query, chunk_lookup, TOP_K)
        hybrid_chunks = retrieve(query)

        b_rank = first_hit_rank(baseline_chunks, expected_tickers)
        h_rank = first_hit_rank(hybrid_chunks, expected_tickers)

        baseline_rr.append(reciprocal_rank(b_rank))
        hybrid_rr.append(reciprocal_rank(h_rank))

        baseline_hit1 += b_rank == 1
        hybrid_hit1 += h_rank == 1
        baseline_hit5 += b_rank is not None
        hybrid_hit5 += h_rank is not None

        answer = generate_answer(query, hybrid_chunks)

        bad_citations = check_citations(answer, len(hybrid_chunks))
        is_clean = len(bad_citations) == 0
        citation_clean += is_clean

        kw_score = keyword_coverage(answer, expected_keywords)
        keyword_scores.append(kw_score)

        print(f"[{i}/{len(eval_set)}] {query[:65]}")
        print(
            f"    baseline rank: {b_rank or '-':<4} hybrid rank: {h_rank or '-':<4} "
            f"citations clean: {is_clean}   keyword coverage: {kw_score:.0%}"
        )

    n = len(eval_set)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"MRR — vector-only baseline:   {sum(baseline_rr) / n:.3f}")
    print(f"MRR — hybrid + rerank:        {sum(hybrid_rr) / n:.3f}")
    print(
        f"Hit@1 — baseline / hybrid:    "
        f"{baseline_hit1}/{n} ({baseline_hit1 / n:.0%})  "
        f"vs  {hybrid_hit1}/{n} ({hybrid_hit1 / n:.0%})"
    )
    print(
        f"Hit@5 — baseline / hybrid:    "
        f"{baseline_hit5}/{n} ({baseline_hit5 / n:.0%})  "
        f"vs  {hybrid_hit5}/{n} ({hybrid_hit5 / n:.0%})"
    )
    print(
        f"Citation validity rate:       "
        f"{citation_clean}/{n} ({citation_clean / n:.0%})"
    )
    print(
        f"Avg. expected-keyword coverage: "
        f"{sum(keyword_scores) / n:.0%}"
    )


if __name__ == "__main__":
    run_evaluation()