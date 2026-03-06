from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

def split_long_text(
    text: str,
    max_chars: int = 6000,
    overlap: int = 500,
) -> List[str]:
    """
    Split long text into overlapping chunks.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == n:
            break
        start = end - overlap

    return chunks



# Heuristic patterns for policy-style headings (works for CFR-style docs + many policies)
HEADING_PATTERNS = [
    re.compile(r"^\s*§\s*\d+(\.\d+)*\s+.*$"),          # § 200.403 ...
    re.compile(r"^\s*\d+\.\d+\s+.*$"),                # 200.403 ...
    re.compile(r"^\s*Subpart\s+[A-Z]\s+—\s+.*$", re.I) # Subpart E — ...
]


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: Optional[str]
    start_page: int
    end_page: int
    text: str


def load_pages_jsonl(path: Path) -> List[Dict[str, Any]]:
    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    return pages


def is_heading(line: str) -> bool:
    return any(p.match(line) for p in HEADING_PATTERNS)


def extract_section_id(heading_line: str) -> Optional[str]:
    """
    Try to pull a stable section identifier (e.g., 200.403) from a heading line.
    """
    # From "§ 200.403 Factors affecting allowability of costs."
    m = re.search(r"§\s*(\d+\.\d+)", heading_line)
    if m:
        return m.group(1)

    # From "200.403 Factors affecting allowability of costs."
    m = re.search(r"^\s*(\d+\.\d+)", heading_line)
    if m:
        return m.group(1)

    return None


def make_chunk_id(source: str, section: Optional[str], start_page: int, end_page: int, text: str) -> str:
    """
    Stable ID so chunks can be compared across runs.
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(section).encode("utf-8"))
    h.update(f"{start_page}-{end_page}".encode("utf-8"))
    # Use a slice to avoid huge hashing but keep stability
    h.update(text[:5000].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def chunk_policy_pages(pages: List[Dict[str, Any]], source_name: str) -> List[Chunk]:
    """
    Build chunks by detecting headings. Each chunk starts at a heading and
    continues until the next heading (possibly across pages).
    """
    # Flatten into (page_num, line) items while keeping page numbers
    flat_lines: List[Dict[str, Any]] = []
    for p in pages:
        if p["source"] != source_name:
            continue
        page_num = int(p["page"])
        text = (p.get("text") or "").strip()
        if not text:
            continue
        for line in text.splitlines():
            # keep even short lines; policy headings are often short
            flat_lines.append({"page": page_num, "line": line.rstrip()})

    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_start_page: Optional[int] = None
    current_section: Optional[str] = None

    def flush(end_page: int):
        nonlocal current_lines, current_start_page, current_section, chunks
        if not current_lines or current_start_page is None:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return

        # NEW: split oversized sections
        sub_texts = split_long_text(body)

        for idx, sub_text in enumerate(sub_texts):
            sub_section = current_section
            cid = make_chunk_id(
                source_name,
                sub_section,
                current_start_page,
                end_page,
                sub_text,
            )
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source=source_name,
                    section=sub_section,
                    start_page=current_start_page,
                    end_page=end_page,
                    text=sub_text,
                )
            )

            last_page_seen = None

    for item in flat_lines:
        page = item["page"]
        line = item["line"]
        last_page_seen = page

        if is_heading(line):
            # New heading begins → flush previous chunk
            if current_lines:
                flush(end_page=page)  # conservative end_page
            # Start new chunk
            current_lines = [line]
            current_start_page = page
            current_section = extract_section_id(line)
        else:
            # If we haven't hit any heading yet, start a "preamble" chunk
            if current_start_page is None:
                current_start_page = page
                current_section = None
                current_lines = []
            current_lines.append(line)

    # Flush the last chunk
    if current_lines and last_page_seen is not None:
        flush(end_page=last_page_seen)

    return chunks


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            rec = {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "section": c.section,
                "start_page": c.start_page,
                "end_page": c.end_page,
                "text": c.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    pages_path = Path("data/processed/uniform_guidance_pages.jsonl")
    chunks_out = Path("data/processed/uniform_guidance_chunks.jsonl")
    source_name = "uniform_guidance.pdf"

    pages = load_pages_jsonl(pages_path)
    chunks = chunk_policy_pages(pages, source_name=source_name)
    save_chunks_jsonl(chunks, chunks_out)

    print(f"✅ Built {len(chunks)} chunks -> {chunks_out}")

    # show a tiny preview
    if chunks:
        print("\n--- Preview of first chunk metadata ---")
        first = chunks[0]
        print(
            f"chunk_id={first.chunk_id} section={first.section} pages={first.start_page}-{first.end_page}"
        )


if __name__ == "__main__":
    main()
