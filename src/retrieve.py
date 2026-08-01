import json
import pickle
import re
import sys
from collections.abc import Sequence
from pathlib import Path
import numpy as np

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer


CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
CHROMA_DIR = Path("data/chroma_db")
BM25_PATH = Path("data/bm25_index.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

K_VECTOR = 25
K_BM25 = 25
RRF_K = 60
FINAL_K = 5

TOKEN_RE = re.compile(r"[a-z0-9]+")

print(f"Loading embedding model {EMBEDDING_MODEL}...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Loading reranker model {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL)


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_chunk_lookup() -> dict[str, dict]:
    lookup = {}

    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk

    return lookup


def embed_query(query: str) -> np.ndarray:
    text = QUERY_INSTRUCTION + query

    embeddings =  embed_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(embeddings)


def vector_search(query_embedding: np.ndarray, k: int,) -> list[str]:
    chroma_client = chromadb.PersistentClient(path = str(CHROMA_DIR))

    collection = chroma_client.get_collection("filings")

    results = collection.query(query_embeddings = [query_embedding], n_results = k,)

    return results["ids"][0]


def keyword_search(query: str, k: int) -> list[str]:
    with BM25_PATH.open("rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]

    scores = bm25.get_scores(tokenise(query))
    ranked = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True,)

    return [chunk_ids[i] for i in ranked[:k]]


def reciprocal_rank_fusion(*ranked_lists: list[str], k: int = RRF_K,) -> list[str]:
    scores = {}

    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = (scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1))

    return [
        cid
        for cid, _ in sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]


def rerank(query: str, candidate_ids: list[str], chunk_lookup: dict[str, dict], top_k: int) -> list[dict]:
    candidates = [
        chunk_lookup[cid]
        for cid in candidate_ids
        if cid in chunk_lookup
    ]

    pairs = [
        (query, c["text"])
        for c in candidates
    ]

    scores = reranker.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return candidates[:top_k]


def retrieve(query: str) -> list[dict]:
    chunk_lookup = load_chunk_lookup()

    query_embedding = embed_query(query)

    vector_ids = vector_search(query_embedding, K_VECTOR)
    bm25_ids = keyword_search(query, K_BM25)
    fused_ids = reciprocal_rank_fusion(vector_ids, bm25_ids)

    top_candidates = fused_ids[: K_VECTOR + K_BM25]

    return rerank(query,top_candidates, chunk_lookup, FINAL_K)


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]

    results = retrieve(query)

    print(f"\nTop {len(results)} chunks for: {query}\n")

    for i, chunk in enumerate(results, 1):
        preview = chunk["text"][:200].replace("\n", " ")

        print(
            f"[{i}] rerank_score={chunk['rerank_score']:.3f}  "
            f"{chunk['ticker']} {chunk['form']} "
            f"{chunk['filing_date']}  "
            f"section={chunk['section'][:60]}"
        )

        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()