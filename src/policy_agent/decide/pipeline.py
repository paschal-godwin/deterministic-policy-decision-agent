from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from policy_agent.decide.schema import InfoNeeded, Facts
from policy_agent.retrieve.retrieve import retrieve_top_k
from policy_agent.decide.judge import decide_with_evidence
from policy_agent.store.db import init_db
from policy_agent.store.runs import save_run
from policy_agent.decide.abstain import apply_abstain_rule
from policy_agent.decide.clarify import system_clarifying_questions
from policy_agent.decide.confidence import compute_confidence


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
    cat = INFO_PROMPTS.get(qtype, {})
    gen = INFO_PROMPTS["general"]

    missing_map: dict[str, str] = {}
    for f in missing_fields:
        missing_map[f] = cat.get(f) or gen.get(f) or f"Please provide: {f}"

    info_kwargs = {k: None for k in InfoNeeded.model_fields.keys()}
    for f, prompt in missing_map.items():
        if f in info_kwargs:
            info_kwargs[f] = prompt

    return InfoNeeded(**info_kwargs), missing_map


def extract_facts_from_question(question: str) -> Facts:
    q = question.strip()
    q_lower = q.lower()

    purpose: Optional[str] = None
    federal_award: Optional[str] = None
    estimated_cost: Optional[float] = None
    budgeted_and_approved: Optional[bool] = None
    intended_user: Optional[str] = None
    equipment_or_supply: Optional[str] = None

    # Purpose extraction
    purpose_patterns = [
        r"use (?:it|this|the laptop|the equipment)?\s*for ([^.?,]+)",
        r"for ([^.?,]+?) for the award",
        r"to support ([^.?,]+)",
        r"to be used for ([^.?,]+)",
        r"needed for ([^.?,]+)",
    ]
    for pattern in purpose_patterns:
        m = re.search(pattern, q_lower, flags=re.IGNORECASE)
        if m:
            purpose = m.group(1).strip(" .,:;")
            break

    # Federal award extraction
    award_patterns = [
        r"\b(nih\s*[a-z0-9-]+)\b",
        r"\b(nsf\s*[a-z0-9-]+)\b",
        r"\b(doe\s*[a-z0-9-]+)\b",
        r"\b(award\s*[a-z0-9-]+)\b",
        r"\b(project\s*[a-z0-9-]+)\b",
    ]
    for pattern in award_patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            federal_award = m.group(1).strip()
            break

    # Estimated cost extraction
    cost_patterns = [
        r"(?:estimated at|estimated cost is|cost is|priced at|for)\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
        r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
    ]
    for pattern in cost_patterns:
        m = re.search(pattern, q_lower, flags=re.IGNORECASE)
        if m:
            estimated_cost = float(m.group(1).replace(",", ""))
            break

    # Budget / approval extraction
    if any(x in q_lower for x in ["budgeted and approved", "budgeted & approved", "approved in budget"]):
        budgeted_and_approved = True
    elif any(x in q_lower for x in ["not budgeted", "not approved", "unapproved"]):
        budgeted_and_approved = False

    # Intended user extraction (simple heuristic)
    if "research assistant" in q_lower:
        intended_user = "research assistant"
    elif "student" in q_lower:
        intended_user = "student"
    elif "analyst" in q_lower:
        intended_user = "analyst"
    elif "staff" in q_lower:
        intended_user = "staff"

    # Equipment / supply extraction
    if any(x in q_lower for x in ["laptop", "computer", "printer", "ipad", "equipment"]):
        equipment_or_supply = "equipment"
    elif any(x in q_lower for x in ["supply", "supplies", "materials"]):
        equipment_or_supply = "supply"

    return Facts(
        purpose=purpose,
        federal_award=federal_award,
        intended_user=intended_user,
        estimated_cost=estimated_cost,
        equipment_or_supply=equipment_or_supply,
        budgeted_and_approved=budgeted_and_approved,
    )


def merge_facts(extracted: Facts | None, provided: Facts | None) -> Facts | None:
    if extracted is None and provided is None:
        return None
    if extracted is None:
        return provided
    if provided is None:
        return extracted

    extracted_data = extracted.model_dump()
    provided_data = provided.model_dump()

    merged = {}
    for key in Facts.model_fields.keys():
        provided_val = provided_data.get(key)
        extracted_val = extracted_data.get(key)
        merged[key] = provided_val if provided_val not in (None, "") else extracted_val

    return Facts(**merged)


def canonicalize_question(question: str, qtype: str, facts: Facts | None) -> str:
    q_lower = question.lower()

    if qtype == "equipment":
        item = "item"
        if "laptop" in q_lower:
            item = "laptop"
        elif "printer" in q_lower:
            item = "printer"
        elif "computer" in q_lower:
            item = "computer"
        elif "ipad" in q_lower:
            item = "ipad"
        elif "equipment" in q_lower:
            item = "equipment"

        purpose_part = f" for {facts.purpose}" if facts and facts.purpose else ""
        award_part = f" under {facts.federal_award}" if facts and facts.federal_award else ""
        return f"Is the purchase of a {item}{purpose_part}{award_part} allowable under federal award cost principles?"

    return question


def answer_question(question: str, k: int = 5, facts: Facts | None = None):
    init_db()

    qtype = classify_question(question)

    extracted_facts = extract_facts_from_question(question)
    merged_facts = merge_facts(extracted_facts, facts)

    required_fields = REQUIRED_FIELDS.get(qtype, REQUIRED_FIELDS["general"])

    # Determine missing fields before judging
    missing: list[str] = []
    if merged_facts is None:
        missing = required_fields.copy()
    else:
        for field in required_fields:
            val = getattr(merged_facts, field, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(field)

    retrieval_question = canonicalize_question(question, qtype, merged_facts)

    retrieved = retrieve_top_k(
        retrieval_question,
        index_dir=Path("data/index"),
        k=k,
        candidate_k=80,
    )

    decision = decide_with_evidence(question, retrieved, facts=merged_facts)
    decision.missing_fields = missing

    # Gate on missing required inputs
    if missing:
        decision.needs_more_info = True
        decision.decision = type(decision.decision)("UNCERTAIN")
    else:
        decision.needs_more_info = False

    # Must have grounding if decisive
    if not decision.needs_more_info and decision.decision.value in ("ALLOW", "NOT_ALLOW"):
        if not decision.citations:
            decision.decision = type(decision.decision)("UNCERTAIN")
            decision.needs_more_info = True
            decision.clarifying_questions = decision.clarifying_questions or []
            decision.clarifying_questions.append(
                "Please provide more context or clarify the scenario so I can cite the relevant policy evidence."
            )

    # Ask only for truly missing fields
    if decision.needs_more_info:
        fields_to_ask = missing if missing else required_fields
        info_needed, missing_map = build_info_needed(qtype, fields_to_ask)
        decision.info_needed = info_needed
        decision.missing_info_needed = missing_map

        if not decision.clarifying_questions:
            decision.clarifying_questions = system_clarifying_questions(
                retrieved=retrieved,
                confidence_reasons=["Missing required inputs"],
            )
    else:
        decision.info_needed = None
        decision.missing_info_needed = {}

    # Deterministic confidence
    score, reasons = compute_confidence(
        decision=decision.decision.value,
        retrieved=retrieved,
        clarifying_questions=decision.clarifying_questions,
        needs_more_info=decision.needs_more_info,
    )
    decision.confidence.score = score
    decision.confidence.reasons = reasons

    if decision.needs_more_info:
        decision.confidence.score = min(decision.confidence.score, 0.30)
        if "Gated: missing required fields for allowability decision" not in decision.confidence.reasons:
            decision.confidence.reasons.append("Gated: missing required fields for allowability decision")

    new_decision, abstain_reasons = apply_abstain_rule(
        decision.decision.value,
        decision.confidence.score,
        threshold=0.40,
    )

    if abstain_reasons:
        decision.confidence.reasons.extend(abstain_reasons)
        decision.decision = type(decision.decision)(new_decision)

        if decision.decision.value == "UNCERTAIN" and not decision.needs_more_info:
            if not decision.clarifying_questions:
                decision.clarifying_questions = system_clarifying_questions(
                    retrieved=retrieved,
                    confidence_reasons=decision.confidence.reasons,
                )

    if not decision.justification.strip():
        decision.justification = (
            "UNCERTAIN: The retrieved policy evidence is relevant but not decisive for this question. "
            "More context is required to determine allowability."
        )

    run_id = save_run(
        question=decision.question,
        decision=decision.decision.value,
        justification=decision.justification,
        confidence=decision.confidence.score,
    )
    decision.run_id = str(run_id)

    return decision.model_dump()