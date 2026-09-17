import argparse
import json
import time
from pathlib import Path

from entailment import verify_citations
from generate import check_citations, generate_answer, clear_answer_cache
from retrieve import (
    detect_form_filter,
    embed_query,
    load_chunk_lookup,
    retrieve,
    vector_search,
)

EVAL_SET_PATH = Path("eval/qa_pairs.json")
TOP_K = 5


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    if not path.exists():
        raise RuntimeError(
            f"{path} not found — create it first (see eval/qa_pairs.json template)."
        )
    return json.loads(path.read_text())


def vector_only_retrieve(
    query: str, chunk_lookup: dict[str, dict], k: int = TOP_K
) -> list[dict]:
    embedding = embed_query(query)
    ids = vector_search(embedding, k, form=detect_form_filter(query))
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


def run_evaluation(
    eval_file: Path = EVAL_SET_PATH,
    split: str | None = None,
    skip_generation: bool = False,
    check_entailment: bool = False,
    legacy_rerank: bool = False,
    use_cache: bool = True,
) -> None:
    eval_set = load_eval_set(eval_file)
    if split:
        eval_set = [q for q in eval_set if q.get("split", "dev") == split]
    if not eval_set:
        raise RuntimeError(f"No questions with split={split!r} in {eval_file}")
    chunk_lookup = load_chunk_lookup()

    if use_cache and Path("eval/answers_cache.json").exists():
        cache = json.loads(Path("eval/answers_cache.json").read_text())
        print(f"Cache loaded: {len(cache)} previously answered questions.")
    else:
        cache = {}

    baseline_rr = []
    hybrid_rr = []
    baseline_hit1 = 0
    hybrid_hit1 = 0
    baseline_hit5 = 0
    hybrid_hit5 = 0
    citation_clean = 0
    keyword_scores = []
    entail_rates = []
    disagreements = 0
    latencies = []
    n_cached = 0
    n_api_calls = 0

    mode = "legacy" if legacy_rerank else "table-aware"
    print(
        f"Running {len(eval_set)} questions ({eval_file.name}"
        f"{f' split={split}' if split else ''}, rerank={mode})"
        " through both retrieval configs...\n"
    )

    for i, item in enumerate(eval_set, 1):
        query = item["question"]
        expected_tickers = item["expected_tickers"]
        expected_keywords = item.get("expected_keywords", [])

        baseline_chunks = vector_only_retrieve(query, chunk_lookup, TOP_K)
        t0 = time.time()
        hybrid_chunks = retrieve(query, reranker_mode=mode)
        latencies.append(time.time() - t0)

        b_rank = first_hit_rank(baseline_chunks, expected_tickers)
        h_rank = first_hit_rank(hybrid_chunks, expected_tickers)

        if b_rank != h_rank:
            disagreements += 1

        baseline_rr.append(reciprocal_rank(b_rank))
        hybrid_rr.append(reciprocal_rank(h_rank))

        baseline_hit1 += b_rank == 1
        hybrid_hit1 += h_rank == 1
        baseline_hit5 += b_rank is not None
        hybrid_hit5 += h_rank is not None

        if skip_generation:
            is_clean, kw_score, ent = True, 1.0, 1.0
        else:
            answer = generate_answer(query, hybrid_chunks, use_cache=use_cache)
            if answer in cache.values():
                n_cached += 1
            else:
                n_api_calls += 1
            bad_citations = check_citations(answer, len(hybrid_chunks))
            is_clean = len(bad_citations) == 0
            kw_score = keyword_coverage(answer, expected_keywords)
            ent = 1.0
            if check_entailment and answer.strip():
                ent = verify_citations(answer, hybrid_chunks)["entailment_rate"]
        citation_clean += is_clean
        keyword_scores.append(kw_score)
        entail_rates.append(ent)

        def ticker_precision(chunks: list[dict]) -> float:
            if not expected_tickers or not chunks:
                return 0.0
            hits = sum(1 for c in chunks if c["ticker"] in expected_tickers)
            return hits / len(chunks)

        marker = "DIFF" if b_rank != h_rank else "    "
        cache_marker = "[CACHED]" if answer in cache.values() else ""

        print(f"[{i}/{len(eval_set)}] {query[:65]}")
        print(
            f"    {marker} baseline rank: {b_rank or '-':<4} hybrid rank: {h_rank or '-':<4} "
            f"precision b/h: {ticker_precision(baseline_chunks):.0%}/"
            f"{ticker_precision(hybrid_chunks):.0%}  "
            f"citations clean: {is_clean}   keyword coverage: {kw_score:.0%}"
            + (f"   entailment: {ent:.0%}" if check_entailment and not skip_generation else "")
            + f" {cache_marker}"
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
    print(f"Rank disagreements b/h:       {disagreements}/{n}")
    print(
        f"Citation validity rate:       "
        f"{citation_clean}/{n} ({citation_clean / n:.0%})"
    )
    print(
        f"Avg. expected-keyword coverage: "
        f"{sum(keyword_scores) / n:.0%}"
    )
    if check_entailment and not skip_generation:
        print(f"Avg. citation entailment:       {sum(entail_rates) / n:.0%}")
    print(f"Avg. hybrid retrieval latency:    {sum(latencies) / n:.2f}s")
    if not skip_generation:
        cache_size = len(json.loads(Path("eval/answers_cache.json").read_text())) if Path("eval/answers_cache.json").exists() else 0
        print(f"Groq API calls in this run:     {n_api_calls}")
        print(f"Cache hits:                     {n_cached}")
        print(f"Total cached entries:           {cache_size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default=str(EVAL_SET_PATH))
    ap.add_argument("--split", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--legacy-rerank", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    a = ap.parse_args()

    if a.clear_cache:
        clear_answer_cache()
    else:
        run_evaluation(
            eval_file=Path(a.eval_file),
            split=a.split,
            skip_generation=a.no_llm,
            check_entailment=a.verify,
            legacy_rerank=a.legacy_rerank,
            use_cache=not a.no_cache,
        )
