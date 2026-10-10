"""Create (or update) the Hugging Face Space that hosts this app.

One command does the whole mechanical part of the deployment:

    HF_TOKEN=hf_... python deploy/push_space.py

It:
  1. creates the Space with the Gradio SDK on the free CPU Basic tier
  2. stages a clean tree (the project README is replaced by the Space card)
  3. pushes it

What it deliberately does *not* do: set secrets. `GROQ_API_KEY` must never pass
through this script or through git - add it under the Space's
Settings > Variables and secrets page.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything the Space needs to run. The retrieval index is deliberately NOT in
# this list: it lives in a dataset (see prepare_dataset.py) and is downloaded on
# boot, because a Space's runtime disk is wiped on every restart.
FILES_TO_SYNC = [
    "app.py",
    "requirements.txt",
    "README.md",
    "src",
]

# The Space's README doubles as its landing page and must carry YAML
# frontmatter (title, sdk, python_version). The project README does not, so the
# Space gets the deploy/ copy instead.
SPACE_README_SOURCE = "deploy/space_README.md"
SPACE_README_TARGET = "README.md"

EXCLUDED_PATHS = ["__pycache__", "*.pyc", ".DS_Store"]


def stage_tree(destination: Path) -> list[str]:
    staged = []

    for item in FILES_TO_SYNC:
        source = REPO_ROOT / item
        target = destination / item

        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(*EXCLUDED_PATHS),
            )
        elif source.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            logging.warning("Skipping %s: not found.", item)
            continue

        staged.append(item)

    readme_source = REPO_ROOT / SPACE_README_SOURCE
    if readme_source.is_file():
        shutil.copy2(readme_source, destination / SPACE_README_TARGET)
        staged.append(SPACE_README_TARGET)
    else:
        logging.warning("%s missing; the Space will have no frontmatter.", readme_source)

    return staged


def git(directory: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-id",
        default=os.getenv("SPACE_ID", ""),
        help="Space repo id, e.g. 'username/10kwery'. Defaults to SPACE_ID.",
    )
    args = parser.parse_args()

    if not args.space_id:
        parser.error("No Space id. Pass --space-id or set SPACE_ID.")

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN is not set; a write token is required.")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        parser.error(f"pip install huggingface_hub ({e})")

    api = HfApi(token=token)

    logging.info("Creating Space %s ...", args.space_id)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="cpu-basic",
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="10kwery-space-") as tmp:
        staging = Path(tmp)
        staged = stage_tree(staging)
        logging.info("Staged: %s", ", ".join(staged))

        git(staging, "init", "--quiet")
        git(staging, "config", "user.email", "deploy@10kwery.local")
        git(staging, "config", "user.name", "10-Kwery deploy")
        git(staging, "add", "-A")
        git(
            staging,
            "-c", "commit.gpgsign=false",
            "commit", "--quiet",
            "-m", "Deploy 10-Kwery",
            "--no-verify",
        )

        remote = (
            f"https://user:{token}@huggingface.co/spaces/{args.space_id}"
        )
        logging.info("Pushing to %s ...", args.space_id)
        git(staging, "push", "--force", "--quiet", remote, "HEAD:main")

    logging.info("Deployed: https://huggingface.co/spaces/%s", args.space_id)
    logging.info(
        "Now set GROQ_API_KEY (and optionally INDEX_DATASET, HF_TOKEN) under "
        "the Space's Settings > Variables and secrets page, then restart it."
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    raise SystemExit(main())
