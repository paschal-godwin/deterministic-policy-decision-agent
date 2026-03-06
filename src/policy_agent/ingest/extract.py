from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF


@dataclass
class PageText:
    source: str
    page: int
    text: str


def _clean_text(text: str) -> str:
    """
    Light cleaning only.
    Policies rely on numbering and structure, so we avoid aggressive cleaning.
    """
    # Normalize weird whitespace while keeping paragraph breaks
    text = text.replace("\u00a0", " ")  # non-breaking space
    # Collapse lines that are just spaces
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Remove empty lines at the top/bottom but keep internal empties (paragraphs)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def extract_pdf_pages(pdf_path: Path) -> List[PageText]:
    """
    Extract per-page text so we always know where a quote came from.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: List[PageText] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("text")  # simple, reliable text extraction
        cleaned = _clean_text(raw)

        pages.append(
            PageText(
                source=pdf_path.name,
                page=i + 1,  # humans count from 1
                text=cleaned,
            )
        )

    doc.close()
    return pages


def save_pages_jsonl(pages: List[PageText], out_path: Path) -> None:
    """
    Save as JSON Lines: one JSON object per line.
    This format is great for debugging and streaming.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            rec: Dict[str, Any] = {
                "source": p.source,
                "page": p.page,
                "text": p.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    pdf_path = Path("data/raw/uniform_guidance.pdf")
    out_path = Path("data/processed/uniform_guidance_pages.jsonl")

    pages = extract_pdf_pages(pdf_path)
    save_pages_jsonl(pages, out_path)

    # quick sanity output
    non_empty = sum(1 for p in pages if p.text.strip())
    print(f"✅ Extracted {len(pages)} pages ({non_empty} non-empty) -> {out_path}")


if __name__ == "__main__":
    main()
