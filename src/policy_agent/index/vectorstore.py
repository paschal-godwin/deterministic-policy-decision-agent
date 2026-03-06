from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from policy_agent.index.embed import embed_texts


def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_index():
    """
    Build embeddings for chunks and save:
    - vectors.npy
    - chunks_meta.json
    This is a simple “poor man’s vector store” that is Windows-friendly.
    We'll switch to FAISS/Chroma after validation if needed.
    """
    chunks_path = Path("data/processed/uniform_guidance_chunks.jsonl")
    out_dir = Path("data/index")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]

    vectors = embed_texts(texts)  # (N, D)

    np.save(out_dir / "vectors.npy", vectors)

    # Store only metadata (not full text) to keep files smaller
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "section": c["section"],
            "start_page": c["start_page"],
            "end_page": c["end_page"],
            "text": c["text"],  # keep for now; later we can store separately
        }
        for c in chunks
    ]
    (out_dir / "chunks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Built index for {len(chunks)} chunks.")
    print(f"Saved: {out_dir / 'vectors.npy'} and {out_dir / 'chunks_meta.json'}")


if __name__ == "__main__":
    build_index()
