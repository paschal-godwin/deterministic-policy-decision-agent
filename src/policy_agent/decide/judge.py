from __future__ import annotations

import json
import re

from dotenv import load_dotenv
from openai import OpenAI

from policy_agent.decide.schema import (
    Decision,
    Facts,
    PolicyDecisionResponse,
    Citation,
    Confidence,
)

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"


def _pick_quote(text: str, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:max_len] + "...") if len(text) > max_len else text


def decide_with_evidence(question, retrieved, facts: Facts | None = None) -> PolicyDecisionResponse:
    evidence_blocks = []
    for i, c in enumerate(retrieved, start=1):
        evidence_blocks.append(
            f"""[EVIDENCE {i}]
source: {c['source']}
section: {c.get('section')}
pages: {c['start_page']}-{c['end_page']}
chunk_id: {c['chunk_id']}
text:
{c['text']}
"""
        )

    facts_block = "None."
    if facts is not None:
        lines = []
        for k, v in facts.model_dump(exclude_none=True).items():
            lines.append(f"- {k}: {v}")
        facts_block = "\n".join(lines) if lines else "None."

    prompt = f"""
You are a compliance/policy decision assistant.

Task:
Given the QUESTION, optional FACTS PROVIDED, and the POLICY EVIDENCE, produce a decision:
- ALLOW, NOT_ALLOW, or UNCERTAIN.

Core rules:
- Use ONLY the POLICY EVIDENCE to justify policy claims. Do NOT invent rules.
- You MAY use FACTS PROVIDED as case context.
- If the evidence and facts together support a clear decision, choose ALLOW or NOT_ALLOW.
- Do NOT include a "confidence" field. The system computes confidence deterministically.
- If key case facts are missing OR the evidence is not decisive, choose UNCERTAIN and ask clarifying questions.
- If there is no explicit prohibition in the evidence, and the evidence describes criteria/conditions that the facts satisfy, you may return ALLOW with conditions.

Decision guidance:
- ALLOW: Evidence supports allowability given the provided facts/conditions.
- NOT_ALLOW: Evidence clearly prohibits it under the provided facts/conditions.
- UNCERTAIN: Evidence is relevant but not decisive, or facts required to apply the evidence are missing.

Output requirements:
Return ONLY valid JSON. No markdown. No extra text.

Schema requirements:
- question: string
- decision: one of "ALLOW", "NOT_ALLOW", "UNCERTAIN"
- justification: concise explanation tying evidence to the decision and the provided facts
- citations: LIST of OBJECTS with:
  - source (string)
  - section (string or null)
  - chunk_id (string)
  - quote (string, copied from evidence text)
- clarifying_questions: list of strings

QUESTION:
{question}

FACTS PROVIDED:
{facts_block}

POLICY EVIDENCE:
{chr(10).join(evidence_blocks)}
""".strip()

    resp = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    usage = resp.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    PRICE_INPUT = 0.00015 / 1000
    PRICE_OUTPUT = 0.0006 / 1000
    cost = (input_tokens * PRICE_INPUT) + (output_tokens * PRICE_OUTPUT)

    raw = resp.output_text.strip()

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Model did not return valid JSON.\nRaw output:\n{raw}") from e

    citations = []
    raw_citations = obj.get("citations", [])

    for cit in raw_citations[:3]:
        if isinstance(cit, str):
            citations.append(
                Citation(
                    source="unknown",
                    section=None,
                    chunk_id="unknown",
                    quote=cit[:240],
                )
            )
            continue

        if isinstance(cit, dict):
            citations.append(
                Citation(
                    source=str(cit.get("source", "unknown")),
                    section=cit.get("section"),
                    chunk_id=str(cit.get("chunk_id", "unknown")),
                    quote=str(cit.get("quote", "")),
                )
            )

    confidence = Confidence(
        score=0.0,
        reasons=["System-computed confidence placeholder"],
    )

    return PolicyDecisionResponse(
        question=obj.get("question", question),
        decision=Decision(obj.get("decision", "UNCERTAIN")),
        justification=obj.get("justification", ""),
        citations=citations,
        confidence=confidence,
        clarifying_questions=list(obj.get("clarifying_questions", [])),
        run_id=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )