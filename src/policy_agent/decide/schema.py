# src/policy_agent/decide/schema.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Decision(str, Enum):
    ALLOW = "ALLOW"
    NOT_ALLOW = "NOT_ALLOW"
    UNCERTAIN = "UNCERTAIN"


class Citation(BaseModel):
    source: str
    section: Optional[str] = None
    chunk_id: str
    quote: str


class InfoNeeded(BaseModel):
    purpose: Optional[str] = None
    federal_award: Optional[str] = None
    intended_user: Optional[str] = None
    estimated_cost: Optional[str] = None
    equipment_or_supply: Optional[str] = None
    budgeted_and_approved: Optional[str] = None


class Confidence(BaseModel):
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasons: List[str] = Field(default_factory=list)


class Facts(BaseModel):
    purpose: Optional[str] = None
    federal_award: Optional[str] = None
    intended_user: Optional[str] = None
    estimated_cost: Optional[float] = None
    equipment_or_supply: Optional[str] = None
    budgeted_and_approved: Optional[bool] = None


class PolicyDecisionResponse(BaseModel):
    question: str
    decision: Decision
    justification: str

    citations: List[Citation] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    clarifying_questions: List[str] = Field(default_factory=list)
    info_needed: Optional[InfoNeeded] = None
    run_id: Optional[str] = None

    needs_more_info: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    missing_info_needed: dict[str, str] = Field(default_factory=dict)

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None