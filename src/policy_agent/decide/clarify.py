from typing import List, Dict, Any


def system_clarifying_questions(
    retrieved: List[Dict[str, Any]],
    confidence_reasons: List[str],
) -> List[str]:
    """
    Generate clarifying questions when the system forces abstention.
    These are heuristic and explainable.
    """
    questions: List[str] = []

    # If retrieval was weak
    if any("Weak retrieval match" in r for r in confidence_reasons):
        questions.append(
            "Can you provide more details about the purpose of the expense?"
        )

    # If coverage was low
    if any("Low evidence coverage" in r for r in confidence_reasons):
        questions.append(
            "Is this expense directly tied to a specific policy, grant, or project?"
        )

    # Generic fallback
    if not questions:
        questions.append(
            "Can you provide additional context or documentation to support this request?"
        )

    return questions
