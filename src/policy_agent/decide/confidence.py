from __future__ import annotations
from typing import List, Dict, Any, Tuple

# How much rerank is allowed to influence confidence (deterministic cap)
RERANK_BONUS_CAP = 0.20

# Thresholds for the *hybrid* score (cosine + capped rerank bonus)
RELEVANT_THRESHOLD = 0.40
MODERATE_THRESHOLD = 0.45
STRONG_THRESHOLD = 0.65


def _hybrid_evidence_score(r: Dict[str, Any]) -> float:
    """
    Hybrid evidence score for confidence:
      hybrid = cosine_score + min(rerank_bonus, cap)

    Why: cosine is grounded semantic similarity; rerank_bonus is a useful signal,
    but we cap it to avoid over-trusting heuristic boosts.
    """
    try:
        base = float(r.get("score", 0.0))  # cosine similarity
    except Exception:
        base = 0.0

    try:
        bonus = float(r.get("rerank_bonus", 0.0))
    except Exception:
        bonus = 0.0

    bonus = min(max(bonus, 0.0), RERANK_BONUS_CAP)
    return base + bonus


def compute_confidence(
    decision: str,
    retrieved: List[Dict[str, Any]],
    clarifying_questions: List[str],
    needs_more_info: bool,
) -> Tuple[float, List[str]]:
    """
    Deterministic confidence based on observable signals.

    Confidence is a proxy for reliability of the *final* decision, based on:
    - Evidence strength (hybrid score: cosine + capped rerank bonus)
    - Evidence coverage (how many chunks clear a relevance threshold)
    - Decision type (UNCERTAIN lowers confidence)
    - Missing required inputs lowers confidence (only when gated)
    """
    reasons: List[str] = []

    scores = [_hybrid_evidence_score(r) for r in retrieved]
    best_score = max(scores, default=0.0)

    relevant = [s for s in scores if s >= RELEVANT_THRESHOLD]
    coverage = len(relevant)

    # Base confidence
    conf = 0.30

    # 1) Evidence strength proxy: best hybrid score
    if best_score >= STRONG_THRESHOLD:
        conf += 0.20
        reasons.append(f"Strong evidence match (best_hybrid={best_score:.2f})")
    elif best_score >= MODERATE_THRESHOLD:
        conf += 0.10
        reasons.append(f"Moderate evidence match (best_hybrid={best_score:.2f})")
    else:
        reasons.append(f"Weak evidence match (best_hybrid={best_score:.2f})")

    # 2) Coverage: how many chunks are reasonably relevant (by hybrid score)
    if coverage >= 3:
        conf += 0.15
        reasons.append(f"Multiple relevant evidence chunks (coverage={coverage})")
    elif coverage == 2:
        conf += 0.08
        reasons.append("Some evidence coverage (2 chunks)")
    else:
        reasons.append("Low evidence coverage (<=1 chunk)")

    # 3) Decision-type adjustment
    if decision == "UNCERTAIN":
        conf -= 0.10
        reasons.append("Decision is UNCERTAIN by design")

    # 4) Clarifying questions: only reduce confidence if we're gated (missing required inputs)
    if clarifying_questions:
        if needs_more_info:
            conf -= min(0.15, 0.05 * len(clarifying_questions))
            reasons.append(f"Missing required inputs (clarifying_questions={len(clarifying_questions)})")
        else:
            reasons.append(f"Optional follow-ups present (clarifying_questions={len(clarifying_questions)})")

    # Clamp to [0, 1]
    conf = max(0.0, min(1.0, conf))
    return conf, reasons
