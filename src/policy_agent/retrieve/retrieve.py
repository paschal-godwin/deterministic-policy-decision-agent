from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from policy_agent.index.embed import embed_texts

import re

RULE_TERMS = [
    r"\ballowable\b",
    r"\bunallowable\b",
    r"\bmust\b",
    r"\bmay not\b",
    r"\bshall\b",
    r"\brequired\b",
    r"\bprohibited\b",
    r"\bexcept\b",
    r"\bsubject to\b",
    r"\breasonable\b",
    r"\ballocable\b",
    r"\bnecessary\b",
    r"\bequipment\b",
    r"\bsupplies\b",
    r"\bfederal award\b",
]

_rule_regexes = [re.compile(pat, re.IGNORECASE) for pat in RULE_TERMS]

SEC_ALL_RE = re.compile(r"(?:§\s*)?(\d+\.\d+)")

def rule_signal_score(text: str) -> float:
    """
    Heuristic score: counts presence of rule-like terms.
    Returns a small bonus, not a replacement for semantic score.
    """
    if not text:
        return 0.0
    hits = 0
    for rx in _rule_regexes:
        if rx.search(text):
            hits += 1
    # scale down so it nudges rather than dominates
    return min(0.25, hits * 0.03)

def sections_in_text(text: str) -> list[str]:
    if not text:
        return []
    # unique but keep order
    seen = set()
    out = []
    for m in SEC_ALL_RE.finditer(text):
        sec = m.group(1)
        if sec not in seen:
            seen.add(sec)
            out.append(sec)
    return out

def normalize_section(section) -> str:
    if section is None:
        return ""
    s = str(section).strip()
    s = s.replace("§", "").strip()
    # keep just the leading numeric part like 200.403
    m = re.match(r"^(\d+\.\d+)", s)
    return m.group(1) if m else s

SEC_RE = re.compile(r"(?:§\s*)?(\d+\.\d+)")  # matches "200.403" or "§ 200.403"

def infer_section(meta_section, text: str) -> str:
    """
    Prefer meta_section if it exists; otherwise infer from text.
    Returns '' if nothing found.
    """
    if meta_section:
        s = str(meta_section).strip().replace("§", "").strip()
        m = re.search(r"(\d+\.\d+)", s)
        if m:
            return m.group(1)

    t = text or ""
    m = SEC_RE.search(t)
    return m.group(1) if m else ""

def section_boost(text: str) -> float:
    secs = sections_in_text(text)

    if "200.403" in secs:
        return 0.20
    if "200.404" in secs or "200.405" in secs:
        return 0.08
    return 0.0


def load_index(index_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(index_dir / "vectors.npy")
    meta = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    return vectors, meta


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_top_k(question: str, *, index_dir: Path, k: int = 5, candidate_k: int = 200):
    vectors, meta = load_index(index_dir)

    # DEBUG: confirm the section exists in corpus
    found = any("200.403" in sections_in_text(m.get("text", "")) for m in meta)
    print(f"[DEBUG] corpus_has_200.403={found}")


    # Embed the query
    q_vec = embed_texts([question])[0]

    # Compute similarity to each chunk
    scored = []
    for i, chunk_vec in enumerate(vectors):
        score = cosine_similarity(q_vec, chunk_vec)
        scored.append((score, meta[i]))

    # Sort by similarity (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    # DEBUG: where does 200.403 rank by similarity?
    rank_200403 = None
    sim_200403 = None
    for rank, (sim, m) in enumerate(scored, start=1):
        if "200.403" in sections_in_text(m.get("text", "")):
            rank_200403 = rank
            sim_200403 = sim
            break
    print(f"[DEBUG] 200.403_rank={rank_200403} sim={None if sim_200403 is None else round(sim_200403,4)}")
    # Take more candidates, then rerank with lexical rule signals
    candidates = scored[:candidate_k]

    reranked = []
    for sim, m in candidates:
        text = m.get("text", "")
        lexical_bonus = rule_signal_score(text)
        sec_bonus = section_boost(text)
        final = sim + lexical_bonus + sec_bonus
        reranked.append((final, sim, lexical_bonus, sec_bonus, m))



    # Sort by final score
    reranked.sort(key=lambda x: x[0], reverse=True)

    print("[DEBUG] top5_after_rerank:")
    for i, (final, sim, lex, sec, m) in enumerate(reranked[:5], start=1):
        print(i, m.get("section"), ("200.403" in sections_in_text(m.get("text",""))),
            round(sim,4), round(lex,4), round(sec,4), round(final,4))
    
    pos = None
    for i, (final, sim, lex, sec, m) in enumerate(reranked, start=1):
        if "200.403" in sections_in_text(m.get("text","")):
            pos = (i, final, sim, lex, sec)
            break
    print(f"[DEBUG] 200.403_pos_after_rerank={pos}")
    # Take top-k
    results = []
    for final, sim, lex, sec, m in reranked[:k]:
        results.append(
            {
                "score": float(sim),                     # original cosine similarity
                "score_type": "cosine_similarity",
                "rerank_bonus": round(lex + sec, 4),
                "final_score": round(final, 4),
                "chunk_id": m["chunk_id"],
                "section": m["section"],
                "source": m["source"],
                "start_page": m["start_page"],
                "end_page": m["end_page"],
                "text": m["text"],
            }
        )

    return results