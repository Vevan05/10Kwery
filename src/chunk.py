import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm


FILINGS_DIR = Path("data/filings")
OUTPUT_PATH = Path("data/chunks/chunks.jsonl")

CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 20

ENCODING = tiktoken.get_encoding("cl100k_base")

ITEM_HEADER_RE = re.compile(
    r"^\s*ITEM\s+\d{1,2}[A-Z]?\.?\s*[-\u2013\u2014.:]?\s*[A-Z]",
    re.IGNORECASE
)
MAX_HEADER_LEN = 120


def count_tokens(text : str) -> int:
    return len(ENCODING.encode(text))


def is_section_header(line : str) -> bool:
    return bool(ITEM_HEADER_RE.match(line)) and len(line) <= MAX_HEADER_LEN


def clean_soup(soup : BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    for tag in soup.find_all(re.compile(r"^ix:", re.I)):
        tag.unwrap()


def table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip = True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]

        if cells:
            rows.append('|'.join(cells))

    return "\n".join(rows)


def extract_blocks(soup : BeautifulSoup) -> list[tuple[str, str]]:
    tables: list[str] = []

    for table in soup.find_all("table"):
        if table.parent is None:
            continue

        text = table_to_text(table)

        if not text.strip():
            table.decompose()
            continue

        idx = len(tables)
        tables.append(text)

        table.replace_with(NavigableString(f"\n@@TABLE_{idx}@@\n"))

    full_text = soup.getText("\n", strip = True)

    blocks: list[tuple[str, str]] = []
    for line in full_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        match = re.match(r"@@TABLE_(\d+)@@$", line)
        if match:
            blocks.append(("table", tables[int(match.group(1))]))
        else:
            blocks.append(("text", line))

    return blocks


def make_chunk(text: str, section: str, chunk_type: str, meta: dict, chunk_index: int) -> dict:
    return{
        "chunk_id": f"{meta['accession_number']}_{chunk_index}",
        "text": text,
        "section": section,
        "chunk_type": chunk_type,
        "token_count": count_tokens(text),
        "ticker": meta["ticker"],
        "cik": meta["cik"],
        "company_name": meta.get("company_name"),
        "form": meta["form"],
        "filing_date": meta["filing_date"],
        "accession_number": meta["accession_number"],
        "source_url": meta.get("source_url")  
    }

def build_chunks(blocks: list[tuple[str, str]], meta: dict) -> list[dict]:
    chunks: list[dict] = []
    current_section = "Unknown"
    buffer_lines: list[str] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer_lines, buffer_tokens
        text = "\n".join(buffer_lines).strip()

        token_count = count_tokens(text)

        if text and token_count >= MIN_CHUNK_TOKENS:
            chunks.append(make_chunk(text, current_section, "text", meta, len(chunks)))


        overlap: list[str] = []
        tokens = 0

        for line in reversed(buffer_lines):
            tokens += count_tokens(line)
            overlap.insert(0, line)

            if tokens >= CHUNK_OVERLAP_TOKENS:
                break

        buffer_lines = overlap
        buffer_tokens = tokens

    for block_type, content in blocks:
        if block_type == "table":
            flush()
            chunks.append(make_chunk(content, current_section, "table", meta, len(chunks)))
            continue

        if is_section_header(content):
            flush()
            buffer_lines, buffer_tokens = [], 0
            current_section = content
            continue

        line_tokens = count_tokens(content)
        if buffer_lines and buffer_tokens + line_tokens > CHUNK_SIZE_TOKENS:
            flush()

        buffer_lines.append(content)
        buffer_tokens += line_tokens

    flush()
    return chunks


def process_filing(html_path: Path) -> list[dict]:
    meta_path = html_path.with_suffix(".json")

    if not meta_path.exists():
        tqdm.write(f"WARNING: no metadata sidecar for {html_path.name}, skipping...")
        return []

    meta = json.loads(meta_path.read_text())
    html = html_path.read_text(encoding = "utf-8", errors = "ignore")

    soup = BeautifulSoup(html, 'lxml')
    clean_soup(soup)

    blocks = extract_blocks(soup)
    return build_chunks(blocks, meta)


def main():
    html_files = sorted(FILINGS_DIR.glob("*/*.htm"))

    if not html_files:
        raise RuntimeError(f"No .htm files found under {FILINGS_DIR}/. Run ingest.py first")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    type_counts = {"text": 0, "table": 0}

    with OUTPUT_PATH.open("w", encoding='utf-8') as out:
        for html_path in tqdm(html_files, desc = 'Filings'):
            try:
                chunks = process_filing(html_path)
            except Exception as e:
                tqdm.write(f"ERROR processing {html_path.name}: {e}")
                continue

            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")
                total_chunks += 1
                type_counts[chunk["chunk_type"]] += 1

    print(f"\nDone. {total_chunks} chunks from {len(html_files)} filings "
    f"({type_counts['text']} text, {type_counts['table']} table) "
    f"written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

        