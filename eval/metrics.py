from typing import List, Dict, Any


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    correct = sum(1 for r in results if r["actual"] == r["expected"])
    return correct / len(results) if results else 0.0


def compute_abstain_rate(results: List[Dict[str, Any]]) -> float:
    abstains = sum(1 for r in results if r["actual"] == "UNCERTAIN")
    return abstains / len(results) if results else 0.0


def compute_false_allow_rate(results: List[Dict[str, Any]]) -> float:
    false_allow = sum(
        1 for r in results if r["actual"] == "ALLOW" and r["expected"] != "ALLOW"
    )
    return false_allow / len(results) if results else 0.0


def compute_false_not_allow_rate(results: List[Dict[str, Any]]) -> float:
    false_not_allow = sum(
        1 for r in results if r["actual"] == "NOT_ALLOW" and r["expected"] != "NOT_ALLOW"
    )
    return false_not_allow / len(results) if results else 0.0


def compute_avg_confidence(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(r["confidence"] for r in results) / len(results)

def compute_avg_cost(results):
    if not results:
        return 0.0
    return sum(r.get("cost", 0.0) for r in results) / len(results)

