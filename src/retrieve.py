import copy
import json
import logging
import pickle
import re
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import BM25_PATH, CHROMA_DIR, CHUNKS_PATH

logger = logging.getLogger(__name__)

# Per-thread retrieval trace (routed tickers, rewritten query). A server handles
# several requests concurrently, so this must never be shared module state.
_tls = threading.local()

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
\
\
\

    directory = _get_company_directory()

    mentions = []

    for match in re.finditer(r"\b([A-Z][A-Za-z&\-]*)'s\b", query):
        name = match.group(1)
        core = name.rstrip("&-").lower()

        if len(core) < 3 or core in _NON_COMPANY_CAPS or core in directory:
            continue

        mentions.append(name)


    if re.search(
        r"\b(10\s*-\s*[kq]|10k|10q|filing|dividend|revenue|segment|report|disclos"
        r"|cash|flow|margin|profit|earning|income|sales|growth|guidance|outlook)\b",
        query,
        re.I,
    ):
        for match in re.finditer(r"\b([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})?)\b", query):
            for part in match.group(1).split():
                core = part.strip("&-").lower()
                if (
                    len(core) >= 3
                    and core not in _NON_COMPANY_CAPS
                    and core not in directory
                    and part not in mentions
                ):

                    if match.start() == 0 and core in {"what", "how", "compare"}:
                        continue
                    mentions.append(part)
                    break


    seen, out = set(), []
    for m in mentions:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def refusal_reason(query: str, chunks: list[dict]) -> str | None:

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


_embed_model = None
_reranker_model = None
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()


def get_embed_model():
    """Load the bi-encoder once per process.

    Guarded by a lock because SentenceTransformer's download-and-load step is
    not safe to run concurrently from two request threads.
    """
    global _embed_model

    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                logger.info("Loading embedding model %s...", EMBEDDING_MODEL)
                _embed_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("Embedding model ready.")

    return _embed_model


def get_reranker():
    global _reranker_model

    if _reranker_model is None:
        with _rerank_lock:
            if _reranker_model is None:
                logger.info("Loading reranker model %s...", RERANKER_MODEL)
                _reranker_model = CrossEncoder(RERANKER_MODEL)
                logger.info("Reranker model ready.")

    return _reranker_model


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


_chunk_lookup_cache: dict[str, dict] | None = None
_chunk_lookup_lock = threading.Lock()


def get_chunk_lookup() -> dict[str, dict]:
    """Return the chunk table, parsed once per process.

    Parsing 15k JSON lines costs seconds, so a web deployment must not repeat
    it per request. The returned dict is shared: treat entries as read-only and
    shallow-copy (see rerank) before attaching per-request scores.
    """
    global _chunk_lookup_cache

    if _chunk_lookup_cache is None:
        with _chunk_lookup_lock:
            if _chunk_lookup_cache is None:
                logger.info("Loading chunk lookup from %s...", CHUNKS_PATH)
                _chunk_lookup_cache = load_chunk_lookup()
                logger.info(
                    "Chunk lookup ready (%d chunks).", len(_chunk_lookup_cache)
                )

    return _chunk_lookup_cache


def embed_query(query: str) -> np.ndarray:
    text = QUERY_INSTRUCTION + query

    embeddings = get_embed_model().encode(
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


_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """Return a shared Chroma persistent client.

    Constructing a PersistentClient rewrites the index directory, so it is
    deliberately created once per process instead of on every query (the
    pipeline issues two vector searches per request).
    """
    global _chroma_client

    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                logger.info("Opening Chroma index at %s...", CHROMA_DIR)
                _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                logger.info("Chroma index ready.")

    return _chroma_client


def vector_search(
    query_embedding: np.ndarray,
    k: int,
    form: str | None = None,
    chunk_type: str | None = None,
    tickers: list[str] | None = None,
) -> list[str]:
    collection = get_chroma_client().get_collection("filings")

    kwargs: dict = {"query_embeddings": [query_embedding], "n_results": k}

    conditions = []

    if form:
        conditions.append({"form": {"$eq": form}})

    if chunk_type:
        conditions.append({"chunk_type": {"$eq": chunk_type}})

    if tickers:
        if len(tickers) == 1:
            conditions.append({"ticker": {"$eq": tickers[0]}})
        else:
            conditions.append({"ticker": {"$in": tickers}})

    if len(conditions) == 1:
        kwargs["where"] = conditions[0]
    elif conditions:
        kwargs["where"] = {"$and": conditions}

    results = collection.query(**kwargs)

    return results["ids"][0]


_bm25_cache: dict | None = None
_bm25_lock = threading.Lock()


def _load_bm25() -> dict:
    global _bm25_cache

    if _bm25_cache is None:
        with _bm25_lock:
            if _bm25_cache is None:
                logger.info("Loading BM25 index from %s...", BM25_PATH)
                with BM25_PATH.open("rb") as f:
                    _bm25_cache = pickle.load(f)
                logger.info("BM25 index ready.")

    return _bm25_cache


def get_bm25() -> dict:
    """Public accessor so callers can preload the keyword index.

    The pipeline reaches it through ``keyword_search``; the web warm-up needs to
    force it into memory before the first request arrives.
    """
    return _load_bm25()


def keyword_search(
    query: str,
    k: int,
    form: str | None = None,
    form_by_id: dict[str, str] | None = None,
    type_by_id: dict[str, str] | None = None,
    chunk_type: str | None = None,
    tickers: list[str] | None = None,
    ticker_by_id: dict[str, str] | None = None,
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

    if tickers and ticker_by_id:
        wanted = set(tickers)
        mask = np.array([ticker_by_id.get(cid) in wanted for cid in chunk_ids])
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


def rerank(query: str, candidate_ids: list[str], chunk_lookup: dict[str, dict], top_k: int, mode: str = "table-aware") -> list[dict]:
    # Shallow-copy so per-request scores land on request-local dicts. The
    # lookup itself is shared across request threads (see get_chunk_lookup),
    # and rerank writes rerank_score / _calibrated onto each chunk.
    candidates = [
        dict(chunk_lookup[cid])
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


            pairs.append((query, label + c["summary"] + "\n" + c["text"][:1000]))
            spans.append(i)

            continue

        for window in rerank_windows(c["text"]):
            pairs.append((query, label + window))
            spans.append(i)

    scores = get_reranker().predict(pairs, show_progress_bar=False)

    best_scores: dict[int, float] = {}

    for i, score in zip(spans, scores):
        score = float(score)
        if score > best_scores.get(i, float("-inf")):
            best_scores[i] = score

    for i, c in enumerate(candidates):
        c["rerank_score"] = best_scores[i]

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    if mode == "legacy":
        return select_balanced(candidates, top_k)
    return select_table_aware(candidates, top_k)


def select_balanced(candidates: list[dict], top_k: int) -> list[dict]:
\
\
\
\
\
\
\
\

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


    selected.sort(key=lambda c: c["rerank_score"], reverse=True)

    return selected


def select_table_aware(
    candidates: list[dict],
    top_k: int,
    max_table_slots: int = MAX_TABLE_SLOTS,
    table_z_floor: float = -1.0,
) -> list[dict]:
\
\
\
\
\
\
\
\

    if not candidates:
        return []
    import math

    tables = [c for c in candidates if c.get("chunk_type") == "table"]
    proses = [c for c in candidates if c.get("chunk_type") != "table"]

    def zmap(items: list[dict]) -> dict[int, float]:
        if not items:
            return {}
        scores = [c["rerank_score"] for c in items]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var) if var > 1e-12 else 1.0
        return {id(c): (c["rerank_score"] - mean) / std for c in items}

    zt, zp = zmap(tables), zmap(proses)
    for c in candidates:
        c["_calibrated"] = (zt if c.get("chunk_type") == "table" else zp).get(id(c), 0.0)

    ordered = sorted(candidates, key=lambda c: c["_calibrated"], reverse=True)
    selected: list[dict] = []
    n_tables = 0
    for c in ordered:
        if len(selected) == top_k:
            break
        if c.get("chunk_type") == "table" and n_tables >= max_table_slots:
            continue
        selected.append(c)
        if c.get("chunk_type") == "table":
            n_tables += 1

    if n_tables == 0 and tables and selected:
        best_table = max(tables, key=lambda c: c["_calibrated"])
        if best_table["_calibrated"] >= table_z_floor:
            selected[-1] = best_table
    for c in candidates:
        c.pop("_calibrated", None)
    selected.sort(key=lambda c: c["rerank_score"], reverse=True)
    return selected


def last_rewrite() -> str | None:
    """Retrieval query actually used by the calling thread, if changed."""
    return getattr(_tls, "last_rewrite", None)


def last_tickers() -> list[str]:
    """Tickers routed by the calling thread."""
    return getattr(_tls, "last_tickers", [])


def retrieve(
    query: str,
    form: str | None = None,
    tickers: list[str] | None = None,
    rewrite: bool = True,
    history: list[dict] | None = None,
    use_router: bool = True,
    reranker_mode: str = "table-aware",
) -> list[dict]:
\
\
\
\
\
\
\

    chunk_lookup = get_chunk_lookup()

    if form is None:


        form = detect_form_filter(query)


    routed: list[str] = []
    if tickers is not None:
        routed = list(tickers)
    elif use_router:
        try:
            from router import route_companies as _route

            routed = _route(query, indexed=indexed_companies(), history=history)
        except Exception:
            routed = []

    # Per-thread state: a server answers several requests at once, so the
    # routing trace must not live on the module or on the function object.
    _tls.last_tickers = routed


    eff_query = query
    if rewrite:
        try:
            from query_rewrite import rewrite_query as _rewrite

            eff_query = _rewrite(query, history=history, tickers=routed or None)
        except Exception:
            eff_query = query
    _tls.last_rewrite = eff_query

    form_by_id = (
        {cid: c["form"] for cid, c in chunk_lookup.items()} if form else None
    )

    type_by_id = {cid: c.get("chunk_type") for cid, c in chunk_lookup.items()}
    ticker_by_id = {cid: c.get("ticker") for cid, c in chunk_lookup.items()}
    tick_filter = routed or None

    query_embedding = embed_query(eff_query)

    vector_ids = vector_search(query_embedding, K_VECTOR, form=form, tickers=tick_filter)
    bm25_ids = keyword_search(
        eff_query, K_BM25, form=form, form_by_id=form_by_id,
        tickers=tick_filter, ticker_by_id=ticker_by_id,
    )
    table_ids = vector_search(
        query_embedding, K_TABLE, form=form, chunk_type="table", tickers=tick_filter
    )
    table_bm25_ids = keyword_search(
        eff_query,
        K_TABLE,
        form=form,
        form_by_id=form_by_id,
        type_by_id=type_by_id,
        chunk_type="table",
        tickers=tick_filter,
        ticker_by_id=ticker_by_id,
    )

    fused_ids = reciprocal_rank_fusion(
        vector_ids, bm25_ids, table_ids, table_bm25_ids
    )

    top_candidates = fused_ids[: K_VECTOR + K_BM25 + K_TABLE]

    selected = rerank(
        eff_query, top_candidates, chunk_lookup, FINAL_K, mode=reranker_mode
    )

    # Detach from the shared lookup: callers (and the answer cache) mutate and
    # serialise these, and a web worker serves several threads at once.
    return [copy.deepcopy(c) for c in selected]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--legacy-rerank", action="store_true")
    args = ap.parse_args()

    query = args.query

    results = retrieve(
        query,
        rewrite=not args.no_rewrite,
        use_router=not args.no_router,
        reranker_mode="legacy" if args.legacy_rerank else "table-aware",
    )

    print(f"\nTop {len(results)} chunks for: {query}\n")
    if last_rewrite() and last_rewrite() != query:
        print(f"Rewritten retrieval query: {last_rewrite()}")
    if last_tickers():
        print(f"Routed tickers: {last_tickers()}")

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
