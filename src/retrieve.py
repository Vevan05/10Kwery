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

K_VECTOR = 75
K_BM25 = 75
K_TABLE = 15
RERANK_WINDOW_CHARS = 1200
RERANK_OVERLAP_CHARS = 200

REFUSAL_SCORE_FLOOR = 0.0

_NON_COMPANY_CAPS = frozenset(
    """what how why when where which who whose compare according item part sec edgar
    us u.s united states federal government inc corp ltd plc incorporated companies
    company investors analysts customers suppliers employees regulators services
    segment segments management discussion analysis conditions operations risks""".split()
)

_company_dir: dict[str, str] | None = None


def _get_company_directory() -> dict[str, str]:
    """Map lowercased company names / name words / tickers -> ticker."""
    global _company_dir

    if _company_dir is None:
        directory: dict[str, str] = {}

        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                chunk = json.loads(line)
                ticker = chunk.get("ticker")

                if not ticker:
                    continue

                directory[ticker.lower()] = ticker
                name = (chunk.get("company_name") or "").strip()

                for word in re.split(r"[\s,.]+", name.lower()):
                    if len(word) >= 3:
                        directory[word] = ticker

        directory["google"] = "GOOGL"
        directory["facebook"] = "META"

        _company_dir = directory

    return _company_dir


def indexed_companies() -> list[str]:
    return sorted(set(_get_company_directory().values()))


def unindexed_company_mentions(query: str) -> list[str]:
    """Possessive capitalized names ('Boeing's') not present in the corpus."""
    directory = _get_company_directory()

    mentions = []

    for match in re.finditer(r"\b([A-Z][A-Za-z&\-]*)'s\b", query):
        name = match.group(1)
        core = name.rstrip("&-").lower()

        if len(core) < 3 or core in _NON_COMPANY_CAPS or core in directory:
            continue

        mentions.append(name)

    return mentions


def refusal_reason(query: str, chunks: list[dict]) -> str | None:
    """Return a human-readable reason if the query cannot be answered from the index."""
    if not chunks:
        return "no excerpts were retrieved from the index"

    mentions = unindexed_company_mentions(query)

    if mentions:
        return (
            f"the question asks about {', '.join(mentions)}, "
            f"which {'is' if len(mentions) == 1 else 'are'} not in the indexed corpus"
        )

    best_score = max(chunk["rerank_score"] for chunk in chunks)

    if best_score < REFUSAL_SCORE_FLOOR:
        return (
            f"no indexed excerpt is relevant enough to answer this "
            f"(best relevance score {best_score:.2f})"
        )

    return None
RRF_K = 60
FINAL_K = 5
MAX_TABLE_SLOTS = 2
TABLE_GAP_TOLERANCE = 4.0

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an the and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off over
    under again further once here there all any both each few more most other some
    such no nor not only own same so than too very can will just should now is are
    was were be been being have has had having do does did doing would could i you
    he she it we they what which who whom this that these those am its his her their
    our your my me him them as of s t don didn""".split()
)

print(f"Loading embedding model {EMBEDDING_MODEL}...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Loading reranker model {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL)


def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


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


def detect_form_filter(query: str) -> str | None:
    q = query.lower()
    forms = set()

    if re.search(r"\b10\s*-\s*k\b", q) or "annual report" in q:
        forms.add("10-K")
    if re.search(r"\b10\s*-\s*q\b", q) or "quarterly report" in q:
        forms.add("10-Q")

    return forms.pop() if len(forms) == 1 else None


def vector_search(
    query_embedding: np.ndarray,
    k: int,
    form: str | None = None,
    chunk_type: str | None = None,
) -> list[str]:
    chroma_client = chromadb.PersistentClient(path = str(CHROMA_DIR))

    collection = chroma_client.get_collection("filings")

    kwargs: dict = {"query_embeddings": [query_embedding], "n_results": k}

    conditions = []

    if form:
        conditions.append({"form": {"$eq": form}})

    if chunk_type:
        conditions.append({"chunk_type": {"$eq": chunk_type}})

    if len(conditions) == 1:
        kwargs["where"] = conditions[0]
    elif conditions:
        kwargs["where"] = {"$and": conditions}

    results = collection.query(**kwargs)

    return results["ids"][0]


_bm25_cache: dict | None = None


def _load_bm25() -> dict:
    global _bm25_cache

    if _bm25_cache is None:
        with BM25_PATH.open("rb") as f:
            _bm25_cache = pickle.load(f)

    return _bm25_cache


def keyword_search(
    query: str,
    k: int,
    form: str | None = None,
    form_by_id: dict[str, str] | None = None,
    type_by_id: dict[str, str] | None = None,
    chunk_type: str | None = None,
) -> list[str]:
    data = _load_bm25()

    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]

    scores = np.asarray(bm25.get_scores(tokenise(query)), dtype=float)

    if form and form_by_id:
        mask = np.array([form_by_id.get(cid) == form for cid in chunk_ids])
        scores[~mask] = -np.inf

    if chunk_type and type_by_id:
        mask = np.array([type_by_id.get(cid) == chunk_type for cid in chunk_ids])
        scores[~mask] = -np.inf

    ranked = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True,)

    return [chunk_ids[i] for i in ranked[:k] if np.isfinite(scores[i])]


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


def rerank_windows(text: str) -> list[str]:
    if len(text) <= RERANK_WINDOW_CHARS:
        return [text]

    windows = []
    start = 0

    while start < len(text):
        end = min(start + RERANK_WINDOW_CHARS, len(text))
        windows.append(text[start:end])

        if end == len(text):
            break

        start = end - RERANK_OVERLAP_CHARS

    return windows


def rerank_label(chunk: dict) -> str:
    name = chunk.get("company_name") or chunk["ticker"]
    return f"[{name} {chunk['form']} filed {chunk['filing_date']}] "


def rerank(query: str, candidate_ids: list[str], chunk_lookup: dict[str, dict], top_k: int) -> list[dict]:
    candidates = [
        chunk_lookup[cid]
        for cid in candidate_ids
        if cid in chunk_lookup
    ]

    if not candidates:
        return []

    pairs = []
    spans = []

    for i, c in enumerate(candidates):
        label = rerank_label(c)

        if c.get("chunk_type") == "table" and c.get("summary"):
            pairs.append((query, label + c["summary"] + "\n" + c["text"][:500]))
            spans.append(i)

            continue

        for window in rerank_windows(c["text"]):
            pairs.append((query, label + window))
            spans.append(i)

    scores = reranker.predict(pairs, show_progress_bar=False)

    best_scores: dict[int, float] = {}

    for i, score in zip(spans, scores):
        score = float(score)
        if score > best_scores.get(i, float("-inf")):
            best_scores[i] = score

    for i, c in enumerate(candidates):
        c["rerank_score"] = best_scores[i]

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return select_balanced(candidates, top_k)


def select_balanced(candidates: list[dict], top_k: int) -> list[dict]:
    """Final selection that stops prose from sweeping every slot.

    MiniLM scores tables systematically below prose even when the table
    is the best evidence (tables -8..-2, prose up to +2 in practice), so
    absolute ranking locks tables out. Instead: fill greedily with a
    cap on table slots, then reserve one seat for the single best table
    whenever it is within TABLE_GAP_TOLERANCE of the best selected
    chunk — i.e. tables compete against tables, junk tables stay out.
    """
    selected: list[dict] = []
    n_tables = 0

    for c in candidates:
        if len(selected) == top_k:
            break

        is_table = c.get("chunk_type") == "table"

        if is_table and n_tables >= MAX_TABLE_SLOTS:
            continue

        selected.append(c)

        if is_table:
            n_tables += 1

    if n_tables == 0 and selected:
        best_table = next(
            (
                c
                for c in candidates
                if c.get("chunk_type") == "table"
            ),
            None,
        )

        if (
            best_table
            and best_table["rerank_score"]
            >= selected[0]["rerank_score"] - TABLE_GAP_TOLERANCE
        ):
            selected[-1] = best_table

    selected.sort(key=lambda c: c["rerank_score"], reverse=True)

    return selected


def retrieve(query: str, form: str | None = None) -> list[dict]:
    chunk_lookup = load_chunk_lookup()

    if form is None:
        form = detect_form_filter(query)

    form_by_id = (
        {cid: c["form"] for cid, c in chunk_lookup.items()} if form else None
    )

    type_by_id = {cid: c.get("chunk_type") for cid, c in chunk_lookup.items()}

    query_embedding = embed_query(query)

    vector_ids = vector_search(query_embedding, K_VECTOR, form=form)
    bm25_ids = keyword_search(query, K_BM25, form=form, form_by_id=form_by_id)
    table_ids = vector_search(
        query_embedding, K_TABLE, form=form, chunk_type="table"
    )
    table_bm25_ids = keyword_search(
        query,
        K_TABLE,
        form=form,
        form_by_id=form_by_id,
        type_by_id=type_by_id,
        chunk_type="table",
    )

    fused_ids = reciprocal_rank_fusion(
        vector_ids, bm25_ids, table_ids, table_bm25_ids
    )

    top_candidates = fused_ids[: K_VECTOR + K_BM25 + K_TABLE]

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