"""Web-facing service layer for 10-Kwery.

Wraps the retrieval pipeline with the lifecycle a server needs:

* ``ensure_index``  - fetch the index if it is missing (Spaces start on empty disk)
* ``warmup``        - one-shot load of models, vector + keyword indexes
* ``ask``           - one structured answer, with concurrency bounded

Keeping this separate from the HTTP layer (app.py) means the CLI and the web
app run the identical code path.
"""

from __future__ import annotations

import logging
import threading
import time

from config import (
    BM25_PATH,
    CHROMA_DIR,
    CHUNKS_PATH,
    DATA_DIR,
    DATASET_TOKEN,
    INDEX_DATASET,
    MAX_CONCURRENT_REQUESTS,
)
from generate import answer_query
from retrieve import (
    get_bm25,
    get_chroma_client,
    get_chunk_lookup,
    get_embed_model,
    get_reranker,
    indexed_companies,
)
from session import ConversationSession

logger = logging.getLogger(__name__)

REQUIRED_ARTEFACTS = (CHUNKS_PATH, CHROMA_DIR, BM25_PATH)

EXAMPLE_QUESTIONS = [
    "How did Apple's R&D spending change year over year?",
    "What were Microsoft's Intelligent Cloud segment revenues?",
    "Compare Tesla's automotive gross margin in 2024 against 2023.",
    "What does Nvidia say about data centre demand?",
    "Which risk factors does Alphabet list for its advertising business?",
]


class IndexMissingError(RuntimeError):
    """Raised when the retrieval index cannot be found or fetched."""


class TenKweryService:
    """Owns retrieval state and the single Groq key shared by all visitors."""

    def __init__(self) -> None:
        self._ready = False
        self._load_error: str | None = None
        self._gate = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_REQUESTS))
        self._init_lock = threading.Lock()
        self.started_at = time.monotonic()
        self._companies: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def ensure_index(self) -> None:
        """Make the three index artefacts available, downloading if needed.

        A Spaces container starts on a wiped disk, so this is where the index
        arrives. Locally it is a no-op because the artefacts already exist.
        """
        missing = [p for p in REQUIRED_ARTEFACTS if not p.exists()]

        if not missing:
            logger.info("Index artefacts already present.")
            return

        if not INDEX_DATASET:
            raise IndexMissingError(
                "Missing index artefact(s): "
                + ", ".join(str(p) for p in missing)
                + ". Run `python src/index.py --force` locally, or set "
                "INDEX_DATASET to a Hugging Face dataset holding the index."
            )

        logger.info(
            "Index missing; downloading dataset %s into %s ...",
            INDEX_DATASET, DATA_DIR,
        )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise IndexMissingError(
                "huggingface_hub is required to download INDEX_DATASET."
            ) from e

        snapshot_download(
            repo_id=INDEX_DATASET,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            token=DATASET_TOKEN or None,
        )

        still_missing = [p for p in REQUIRED_ARTEFACTS if not p.exists()]
        if still_missing:
            raise IndexMissingError(
                "Dataset %s did not contain: %s"
                % (INDEX_DATASET, ", ".join(str(p) for p in still_missing))
            )

        logger.info("Index downloaded.")

    def warmup(self) -> bool:
        """Load every model and index up front. Idempotent and thread-safe."""
        with self._init_lock:
            if self._ready:
                return True
            if self._load_error:
                raise RuntimeError(self._load_error)

        try:
            self.ensure_index()
        except IndexMissingError as e:
            self._load_error = str(e)
            logger.error("Index unavailable: %s", e)
            raise

        logger.info("Warming up: loading embedding model...")
        get_embed_model()
        logger.info("Warming up: loading reranker...")
        get_reranker()
        logger.info("Warming up: loading chunk table...")
        get_chunk_lookup()

        # Pull the keyword index + vector collection into memory so the first
        # real request is not the request that pays for them.
        logger.info("Warming up: loading keyword index...")
        get_bm25()
        get_chroma_client()
        self._companies = indexed_companies()

        self._ready = True
        logger.info(
            "Warm-up complete in %.1fs. Companies: %s",
            time.monotonic() - self.started_at,
            ", ".join(self._companies),
        )
        return True

    def companies(self) -> list[str]:
        if not self._ready:
            try:
                self.warmup()
            except Exception:
                return []
        return list(self._companies)

    # -- querying -----------------------------------------------------------

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        *,
        use_cache: bool = True,
        verify_entailment: bool = False,
        use_router: bool = True,
        rewrite: bool = True,
    ) -> dict:
        """Answer one question. Raises if the service is not warmed up.

        ``use_router`` and ``rewrite`` mirror the CLI defaults. The rule-based
        router resolves the seven indexed tickers for free and only falls back to
        the LLM when no alias matches, so routing costs nothing for most
        questions; rewriting is what makes pronoun follow-ups ("how did that
        change?") retrievable.
        """
        if not self.ready:
            self.warmup()

        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if len(question) > 1000:
            raise ValueError("Question is too long (1000 character limit).")

        started = time.monotonic()

        # The Groq free tier is shared, so a burst of requests must not all
        # reach the API at once. Anything beyond the limiter waits its turn.
        with self._gate:
            result = answer_query(
                question,
                session=_session_from_history(history),
                use_cache=use_cache,
                verify_entailment=verify_entailment,
                use_router=use_router,
                rewrite=rewrite,
                quiet=True,
            )

        elapsed = round(time.monotonic() - started, 2)

        if not result:
            return {
                "question": question,
                "refused": True,
                "reason": "nothing relevant was retrieved from the index",
                "answer": (
                    "I could not retrieve any relevant excerpts for that "
                    "question, so I cannot answer it from the indexed filings."
                ),
                "sources": [],
                "latency_s": elapsed,
            }

        payload = {
            "question": question,
            "refused": bool(result.get("refused")),
            "answer": result.get("answer")
            or _REFUSAL_MESSAGE.format(reason=result.get("reason", "unknown")),
            "reason": result.get("reason"),
            "sources": [_source_view(i, c) for i, c in enumerate(result["chunks"], 1)],
            "latency_s": elapsed,
        }

        if result.get("entailment"):
            payload["entailment"] = result["entailment"]

        return payload


_REFUSAL_MESSAGE = (
    "I cannot answer that from the indexed filings: {reason}."
)


def _session_from_history(history: list[dict] | None) -> ConversationSession | None:
    """Rebuild a ConversationSession from plain turn dicts.

    Multi-turn context needs a real session object: answer_query resolves its
    history through ``session.history_for_retrieval()``, and passing history any
    other way would collide with its own keyword argument.
    """
    if not history:
        return None

    session = ConversationSession()
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            session.add_user(content)
        else:
            session.add_assistant(content)
    return session


def _source_view(index: int, chunk: dict) -> dict:
    """Client-facing projection of a chunk (no full document text)."""
    return {
        "n": index,
        "ticker": chunk.get("ticker"),
        "form": chunk.get("form"),
        "filing_date": chunk.get("filing_date"),
        "section": chunk.get("section"),
        "chunk_type": chunk.get("chunk_type"),
        "rerank_score": round(float(chunk.get("rerank_score", 0.0)), 3),
        "source_url": chunk.get("source_url"),
        "excerpt": (chunk.get("text") or "")[:600],
    }


_service: TenKweryService | None = None
_service_lock = threading.Lock()


def get_service() -> TenKweryService:
    """Process-wide singleton."""
    global _service

    if _service is None:
        with _service_lock:
            if _service is None:
                _service = TenKweryService()
    return _service
