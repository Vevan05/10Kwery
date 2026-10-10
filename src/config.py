"""Runtime configuration and path resolution for 10-Kwery.

Every path the pipeline touches is derived from ``DATA_DIR`` so the exact same
code runs three ways:

    locally          -> paths resolve to <repo>/data/...      (default)
    in a container   -> DATA_DIR points at the mounted index
    during ingestion -> paths point at the repo as before

Logging is configured here (instead of ``print``) because the deployment runs
this pipeline inside a web worker, where ``print`` floods the request log.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


# --------------------------------------------------------------------------
# Retrieval index. Only these three artefacts are needed to answer queries;
# the raw filings under data/filings/ are inputs to ingestion, not queries.
# --------------------------------------------------------------------------
DATA_DIR = _path_env("DATA_DIR", REPO_ROOT / "data")
CHUNKS_PATH = _path_env("CHUNKS_PATH", DATA_DIR / "chunks" / "chunks.jsonl")
CHROMA_DIR = _path_env("CHROMA_DIR", DATA_DIR / "chroma_db")
BM25_PATH = _path_env("BM25_PATH", DATA_DIR / "bm25_index.pkl")

# --------------------------------------------------------------------------
# Answer cache. Deliberately outside the index directory: it is a write-heavy,
# disposable artefact and lives on ephemeral disk in a deployment.
# --------------------------------------------------------------------------
CACHE_DIR = _path_env("CACHE_DIR", REPO_ROOT / "eval")
CACHE_PATH = _path_env("ANSWERS_CACHE_PATH", CACHE_DIR / "answers_cache.json")

# --------------------------------------------------------------------------
# Web service
# --------------------------------------------------------------------------
INDEX_DATASET = os.getenv("INDEX_DATASET", "")
DATASET_TOKEN = os.getenv("HF_TOKEN", "")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
# Comfortably above what a single visitor can produce by hand, so the limiter is
# a guard rail against scripted abuse rather than a throttle on real use. The
# real bottleneck is the bounded semaphore plus the shared free-tier Groq key.
RATE_LIMIT_RPM = float(os.getenv("RATE_LIMIT_RPM", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# The follow-up condenser costs an extra Groq call per turn. It is worth it in
# interactive CLI use but it doubles free-tier spend for a public demo, so the
# web app switches it off (set to "1" to opt back in).
ENABLE_FOLLOWUP_CONDENSE = os.getenv("ENABLE_FOLLOWUP_CONDENSE", "1") != "0"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-10s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs() -> None:
    for d in (CACHE_DIR, DATA_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.getLogger("config").warning("could not create %s: %s", d, e)


setup_logging()
ensure_dirs()
