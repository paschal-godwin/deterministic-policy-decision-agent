from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from attrs import fields

from policy_agent.decide.schema import InfoNeeded
from policy_agent.retrieve.retrieve import retrieve_top_k
from policy_agent.decide.judge import decide_with_evidence
from policy_agent.store.db import init_db
from policy_agent.store.runs import save_run
from policy_agent.decide.abstain import apply_abstain_rule
from policy_agent.decide.clarify import system_clarifying_questions
from policy_agent.decide.confidence import compute_confidence
from policy_agent.decide.schema import Facts  # import


def classify_question(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["travel", "hotel", "airfare", "flight", "per diem", "conference"]):
        return "travel"
    if any(w in q for w in ["meal", "lunch", "dinner", "catering", "food", "refreshment"]):
        return "meals"
    if any(w in q for w in ["consultant", "contractor", "professional service", "legal", "audit"]):
        return "services"
    if any(w in q for w in ["laptop", "computer", "printer", "ipad", "equipment", "purchase", "buy"]):
        return "equipment"
    return "general"


INFO_PROMPTS = {
    "general": {
        "purpose": "What is the specific purpose/benefit to the Federal award?",
        "federal_award": "Which Federal award or project will this be charged to?",
        "budgeted_and_approved": "Is this included in the approved budget/procurement plan?",
        "estimated_cost": "What is the estimated total cost?",
        "intended_user": "Who will primarily use this (role, not name)?",
        "equipment_or_supply": "Is this treated as equipment or supplies under your capitalization threshold?",
    },
    "equipment": {
        "purpose": "What specific work will this item support (and how does it benefit the award)?",
        "federal_award": "Which Federal award or project will this be charged to?",
        "estimated_cost": "What is the estimated purchase cost?",
        "equipment_or_supply": "Is it treated as equipment or supplies under policy thresholds?",
        "budgeted_and_approved": "Is it included in the approved budget and procurement plan?",
        "intended_user": "Who will primarily use it (role, not name)?",
    },
    "travel": {
        "purpose": "What is the travel for, and how does it benefit the Federal award?",
        "federal_award": "Which Federal award or project will this travel be charged to?",
        "budgeted_and_approved": "Is the travel authorized/approved under the award and your travel policy?",
        "estimated_cost": "What is the estimated total travel cost (airfare + hotel + per diem, etc.)?",
        "intended_user": "Who is traveling (role, not name)?",
    },
    "meals": {
        "purpose": "What is the purpose of the meal, and how does it directly support the award?",
        "federal_award": "Which Federal award or project will this be charged to?",
        "budgeted_and_approved": "Is this meal expense allowed/approved under the award and your policy?",
        "estimated_cost": "What is the estimated total meal cost and number of attendees?",
    },
    "services": {
        "purpose": "What service is being procured and how does it benefit the award?",
        "federal_award": "Which Federal award or project will this be charged to?",
        "budgeted_and_approved": "Is this service budgeted/approved and procured per policy?",
        "estimated_cost": "What is the estimated cost and period of performance?",
        "intended_user": "Who will receive/use the service (role, not name)?",
    },
}

REQUIRED_FIELDS = {
    "general": ["purpose", "federal_award"],
    "equipment": ["purpose", "federal_award", "estimated_cost", "budgeted_and_approved"],
    "travel": ["purpose", "federal_award", "budgeted_and_approved"],
    "meals": ["purpose", "federal_award", "budgeted_and_approved"],
    "services": ["purpose", "federal_award", "budgeted_and_approved"],
}


def build_info_needed(qtype: str, missing_fields: list[str]) -> tuple[InfoNeeded, dict[str, str]]:

    # prefer category-specific prompts, fall back to general prompts
    cat = INFO_PROMPTS.get(qtype, {})
    gen = INFO_PROMPTS["general"]

    missing_map: dict[str, str] = {}
    for f in missing_fields:
        missing_map[f] = cat.get(f) or gen.get(f) or f"Please provide: {f}"

    # Build InfoNeeded with ONLY the missing fields filled (others None)
    fields = getattr(InfoNeeded, "model_fields", None) or getattr(InfoNeeded, "__fields__", {})
    info_kwargs = {k: None for k in fields.keys()}

    for f, prompt in missing_map.items():
        if f in info_kwargs:
            info_kwargs[f] = prompt

    return InfoNeeded(**info_kwargs), missing_map



def answer_question(question: str, k: int = 5, facts: Facts | None = None):
    init_db()

    # 1) Retrieve evidence
    retrieved = retrieve_top_k(
        question,
        index_dir=Path("data/index"),
        k=k,
        candidate_k=80,
    )

    # 2) LLM draft decision (may be overridden by system gates)
    decision = decide_with_evidence(question, retrieved, facts=facts)

    # 3) Determine required fields + missing fields (SYSTEM)
    qtype = classify_question(question)
    required_fields = REQUIRED_FIELDS.get(qtype, REQUIRED_FIELDS["general"])


    missing: list[str] = []
    if facts is None:
        missing = required_fields.copy()
    else:
        for field in required_fields:
            val = getattr(facts, field, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(field)

    decision.missing_fields = missing

    # 4) Gate on missing required inputs (SYSTEM)
    # If missing required inputs, force UNCERTAIN and mark needs_more_info
    if missing:
        decision.needs_more_info = True
        decision.decision = type(decision.decision)("UNCERTAIN")
    else:
        decision.needs_more_info = False
    
    # 4b) Gate: decisive decisions must be grounded with citations
    # If the model returns ALLOW/NOT_ALLOW but provides no citations, force UNCERTAIN.
    if not decision.needs_more_info and decision.decision.value in ("ALLOW", "NOT_ALLOW"):
        if not decision.citations:
            decision.decision = type(decision.decision)("UNCERTAIN")
            decision.needs_more_info = True  # treat as needs-more-info/grounding
            decision.clarifying_questions = decision.clarifying_questions or []
            decision.clarifying_questions.append(
                "Please provide more context or clarify the scenario so I can cite the relevant policy evidence."
            )

    # 5) Build dynamic info_needed + missing_info_needed (SYSTEM)
    # Do this whenever we need more info (not only when LLM said UNCERTAIN)
    if decision.needs_more_info:
        # If missing is empty but we still need more info (e.g., grounding/citations issue),
        # fall back to the required_fields for this question type.
        fields_to_ask = missing if missing else required_fields
        info_needed, missing_map = build_info_needed(qtype, fields_to_ask)
        decision.info_needed = info_needed
        decision.missing_info_needed = missing_map

        # Ensure clarifying questions exist (optional UX improvement)
        if not decision.clarifying_questions:
            decision.clarifying_questions = system_clarifying_questions(
                retrieved=retrieved,
                confidence_reasons=["Missing required inputs"],
            )
    else:
        decision.info_needed = None
        decision.missing_info_needed = {}

    # 6) SYSTEM confidence (deterministic, explainable) — computed AFTER gating
    score, reasons = compute_confidence(
        decision=decision.decision.value,
        retrieved=retrieved,
        clarifying_questions=decision.clarifying_questions,
        needs_more_info=decision.needs_more_info,
    )
    decision.confidence.score = score
    decision.confidence.reasons = reasons

    # If gated, cap confidence (keep this deterministic)
    if decision.needs_more_info:
        decision.confidence.score = min(decision.confidence.score, 0.30)
        if "Gated: missing required fields for allowability decision" not in decision.confidence.reasons:
            decision.confidence.reasons.append("Gated: missing required fields for allowability decision")

    # 7) Abstain rule (SYSTEM) — after confidence is finalized
    new_decision, abstain_reasons = apply_abstain_rule(
        decision.decision.value,
        decision.confidence.score,
        threshold=0.40,
    )

    if abstain_reasons:
        decision.confidence.reasons.extend(abstain_reasons)
        decision.decision = type(decision.decision)(new_decision)

        # If abstained, ensure user gets prompts/questions
        if decision.decision.value == "UNCERTAIN" and not decision.needs_more_info:
            # Not missing required inputs; still uncertain due to low confidence
            if not decision.clarifying_questions:
                decision.clarifying_questions = system_clarifying_questions(
                    retrieved=retrieved,
                    confidence_reasons=decision.confidence.reasons,
                )

    # 8) Safety fallback justification
    if not decision.justification.strip():
        decision.justification = (
            "UNCERTAIN: The retrieved policy evidence is relevant but not decisive for this question. "
            "More context is required to determine allowability."
        )

    # 9) Persist run
    run_id = save_run(
        question=decision.question,
        decision=decision.decision.value,
        justification=decision.justification,
        confidence=decision.confidence.score,
    )
    decision.run_id = str(run_id)


    return decision.model_dump()

