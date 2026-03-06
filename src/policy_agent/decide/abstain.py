from __future__ import annotations
from typing import Tuple, List

def apply_abstain_rule(
    decision: str,
    confidence: float,
    threshold: float = 0.50
) -> Tuple[str, List[str]]:
    """
    If confidence is below threshold, force UNCERTAIN.
    """
    reasons = []
    if confidence < threshold and decision != "UNCERTAIN":
        reasons.append(f"Forced abstain: confidence {confidence:.2f} < threshold {threshold:.2f}")
        return "UNCERTAIN", reasons
    return decision, reasons
