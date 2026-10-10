"""Publish the retrieval index as a Hugging Face dataset.

A Space's 50 GB runtime disk is wiped on every restart, so the index cannot live
in the container. This script stages the three artefacts the query path needs
(chunk table, Chroma vector index, BM25 index) and uploads them to a dataset
that the Space downloads on boot.

The raw filings under ``data/filings`` are *not* uploaded: they are inputs to
ingestion and chunking, and at 185 MB they are the single largest thing in the
repo while being useless at query time.

Usage
-----
    HF_TOKEN=hf_... python deploy/prepare_dataset.py --repo-id you/10kwery-index

Add ``--dry-run`` to print the manifest and total size without uploading.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import BM25_PATH, CHROMA_DIR, CHUNKS_PATH, DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# Relative to DATA_DIR, so the dataset layout mirrors the local data/ layout.
ALLOW_PATTERNS = [
    "chunks/chunks.jsonl",
    "chroma_db/**",
    "bm25_index.pkl",
]

# Never uploaded, even if present: inputs to ingestion, not retrieval.
IGNORE_PATTERNS = [
    "filings/**",
    "answers_cache.json",
    "__pycache__/**",
]


def required_artifacts() -> list[tuple[Path, int]]:
    """Return (path, size) for every file the query path needs."""
    missing = []
    manifest = []

    for path in (CHUNKS_PATH, BM25_PATH, CHROMA_DIR):
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing index artefacts: "
            + ", ".join(missing)
            + "\nBuild them locally with:\n"
            "  python src/ingestion.py\n"
            "  python src/chunk.py\n"
            "  python src/index.py --force"
        )

    for path in (CHUNKS_PATH, BM25_PATH):
        manifest.append((path, path.stat().st_size))

    for path in sorted(CHROMA_DIR.rglob("*")):
        if path.is_file():
            manifest.append((path, path.stat().st_size))

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("INDEX_DATASET", ""),
        help="Dataset repo id, e.g. 'username/10kwery-index'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset as private (public by default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest and size without uploading anything.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error(
            "No dataset repo id. Pass --repo-id or set INDEX_DATASET."
        )

    try:
        manifest = required_artifacts()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    total = sum(size for _, size in manifest)
    logger.info("Staged %d files, %.1f MB total.", len(manifest), total / 1e6)
    for path, size in manifest:
        logger.info("  %-60s %8.2f MB", path, size / 1e6)

    if any(size > 2 * 1024**3 for _, size in manifest):
        logger.warning(
            "A single file exceeds git-LFS's 2 GB limit; the upload will fail."
        )

    if args.dry_run:
        logger.info("Dry run; nothing uploaded.")
        return 0

    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is not set; a write token is required to upload.")
        return 1

    api = HfApi(token=token)

    logger.info("Creating dataset %s (private=%s) ...", args.repo_id, args.private)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Refresh retrieval index",
    )

    logger.info("Uploaded to https://huggingface.co/datasets/%s", args.repo_id)
    logger.info(
        "Set these on the Space's Settings > Variables and secrets page:\n"
        "  INDEX_DATASET = %s\n"
        "  HF_TOKEN     = %s...",
        args.repo_id,
        token[:6] if token else "",
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s"
    )
    raise SystemExit(main())
