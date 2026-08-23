import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from tqdm import tqdm


FILINGS_DIR = Path("data/filings")
OUTPUT_PATH = Path("data/chunks/chunks.jsonl")

CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 20
MIN_TABLE_TOKENS = 15

ENCODING = tiktoken.get_encoding("cl100k_base")

ITEM_HEADER_RE = re.compile(
    r"^\s*ITEM\s+\d{1,2}[A-Z]?\.?\s*[-\u2013\u2014.:]?\s*[A-Z]",
    re.IGNORECASE,
)

MAX_HEADER_LEN = 120

MAX_SUMMARY_ROWS = 8
MAX_SUMMARY_COLS = 6
MAX_SUMMARY_VALUES = 6


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def summarize_table(
    table_text: str,
    meta: dict,
    section: str,
    lead_in: str = "",
) -> str:
    """Rule-based natural-language summary so dense tables embed well.

    Semantic content first (lead-in, columns, line items); company/form
    identity reduced to a short tag so shared boilerplate does not drown
    the distinguishing tokens in the embedding space.
    """
    lines = [ln.strip() for ln in table_text.split("\n") if ln.strip()]

    if not lines:
        return ""

    company = meta.get("company_name") or meta["ticker"]

    parts = []

    if lead_in:
        parts.append(lead_in[:220].strip())

    parts.append(f"{company} {meta['form']} data table")

    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    if header_cells:
        parts.append(f"with columns {', '.join(header_cells[:MAX_SUMMARY_COLS])}")

    row_labels = []
    for line in lines[1:]:
        first_cell = line.split("|")[0].strip()

        if (
            first_cell
            and first_cell not in row_labels
            and len(row_labels) < MAX_SUMMARY_ROWS
            and not re.fullmatch(r"[\d$%,.\s()\-]*", first_cell)
        ):
            row_labels.append(first_cell)

    if row_labels:
        parts.append(f"covering line items {', '.join(row_labels)}")

    values = re.findall(r"\(?\d{1,3}(?:\.\d+)?\)?%", table_text)[:3]
    values += re.findall(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?", table_text)[:3]

    if values:
        parts.append(f"including values such as {', '.join(values)}")

    return ". ".join(parts) + "."


def is_section_header(line: str) -> bool:
    return bool(ITEM_HEADER_RE.match(line)) and len(line) <= MAX_HEADER_LEN


def clean_soup(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    for tag in soup.find_all(re.compile(r"^ix:", re.I)):
        tag.unwrap()


def table_to_text(table: Tag) -> str:
    rows = []

    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]

        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def extract_blocks(soup: BeautifulSoup) -> list[tuple[str, str]]:
    tables = []

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

    full_text = soup.get_text("\n", strip=True)

    blocks = []

    for line in full_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        match = re.match(r"@@TABLE_(\d+)@@$", line)

        if match:
            table_text = tables[int(match.group(1))]

            if count_tokens(table_text) < MIN_TABLE_TOKENS:
                blocks.append(("text", table_text.replace("\n", " | ")))
            else:
                blocks.append(("table", table_text))

        else:
            blocks.append(("text", line))

    return blocks


def make_chunk(
    text: str,
    section: str,
    chunk_type: str,
    meta: dict,
    chunk_index: int,
    summary: str = "",
) -> dict:
    return {
        "chunk_id": f"{meta['accession_number']}_{chunk_index}",
        "text": text,
        "summary": summary,
        "section": section,
        "chunk_type": chunk_type,
        "token_count": count_tokens(text),
        "ticker": meta["ticker"],
        "cik": meta["cik"],
        "company_name": meta.get("company_name"),
        "form": meta["form"],
        "filing_date": meta["filing_date"],
        "accession_number": meta["accession_number"],
        "source_url": meta.get("source_url"),
    }


def build_chunks(blocks: list[tuple[str, str]], meta: dict) -> list[dict]:
    chunks = []

    current_section = "Unknown"

    buffer_lines = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer_lines, buffer_tokens

        text = "\n".join(buffer_lines).strip()

        if text and count_tokens(text) >= MIN_CHUNK_TOKENS:
            chunks.append(
                make_chunk(
                    text,
                    current_section,
                    "text",
                    meta,
                    len(chunks),
                )
            )

        overlap = []
        tokens = 0

        for line in reversed(buffer_lines):
            tokens += count_tokens(line)
            overlap.insert(0, line)

            if tokens >= CHUNK_OVERLAP_TOKENS:
                break

        buffer_lines = overlap
        buffer_tokens = tokens

    for block_index, (block_type, content) in enumerate(blocks):
        if block_type == "table":
            flush()

            lead_in = ""

            for prev_type, prev_content in reversed(blocks[:block_index]):
                if prev_type == "text" and prev_content:
                    lead_in = prev_content
                    break

            chunks.append(
                make_chunk(
                    content,
                    current_section,
                    "table",
                    meta,
                    len(chunks),
                    summary=summarize_table(
                        content, meta, current_section, lead_in
                    ),
                )
            )

            continue

        if is_section_header(content):
            flush()

            buffer_lines = []
            buffer_tokens = 0
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
        tqdm.write(f"WARNING: no metadata sidecar for {html_path.name}, skipping")
        return []

    meta = json.loads(meta_path.read_text())

    html = html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(html, "lxml")

    clean_soup(soup)

    blocks = extract_blocks(soup)

    return build_chunks(blocks, meta)


def main():
    html_files = sorted(FILINGS_DIR.glob("*/*.htm"))

    if not html_files:
        raise RuntimeError(
            f"No .htm files found under {FILINGS_DIR}. run ingest.py first"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chunks = 0

    type_counts = {
        "text": 0,
        "table": 0,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as out:

        for html_path in tqdm(
            html_files,
            desc="Filings",
        ):
            try:
                chunks = process_filing(html_path)

            except Exception as e:
                tqdm.write(f"ERROR processing {html_path.name}: {e}")
                continue

            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")

                total_chunks += 1
                type_counts[chunk["chunk_type"]] += 1

    print(
        f"Done. {total_chunks} chunks from {len(html_files)} filings "
        f"({type_counts['text']} text, {type_counts['table']} table) "
        f"written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()