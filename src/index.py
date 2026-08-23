import json
import pickle
import re
import shutil
import sys
from pathlib import Path
import numpy as np

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
CHROMA_DIR = Path("data/chroma_db")
BM25_PATH = Path("data/bm25_index.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 256

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

model = SentenceTransformer(EMBEDDING_MODEL)


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"{CHUNKS_PATH} not found. run chunk.py first")

    chunks = []

    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks

def embed_batch(texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

def build_vector_index(chunks: list[dict]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = chroma_client.get_or_create_collection(
        name="filings",
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(
        collection.get(include=[])["ids"]
        if collection.count()
        else set()
    )

    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = [
            c
            for c in chunks[i : i + EMBED_BATCH_SIZE]
            if c["chunk_id"] not in existing_ids
        ]

        if not batch:
            continue

        texts = [c.get("summary") or c["text"] for c in batch]
        embeddings = embed_batch(texts)

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings, # type: ignore
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "ticker": c["ticker"],
                    "form": c["form"],
                    "filing_date": c["filing_date"],
                    "section": c["section"],
                    "chunk_type": c["chunk_type"],
                    "accession_number": c["accession_number"],
                    "company_name": c.get("company_name") or "",
                }
                for c in batch
            ],
        )

    print(f"Vector index: {collection.count()} chunks stored in {CHROMA_DIR}")


def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_keyword_index(chunks: list[dict]) -> None:
    tokenized_corpus = [
        tokenise(c["text"] + " " + (c.get("summary") or ""))
        for c in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BM25_PATH.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [c["chunk_id"] for c in chunks],
            },
            f,
        )

    print(f"Keyword index: {len(chunks)} chunks stored in {BM25_PATH}")


def main():
    if "--force" in sys.argv:
        print("Force rebuild: wiping existing indexes...")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        BM25_PATH.unlink(missing_ok=True)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} from {CHUNKS_PATH}")

    build_vector_index(chunks)
    build_keyword_index(chunks)


if __name__ == "__main__":
    main()