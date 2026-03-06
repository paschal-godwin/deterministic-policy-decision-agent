# Deterministic Policy Decision Engine

Compliance-grade RAG with abstain-first gating and regression-tested confidence calibration.
---

## Overview

This project implements a deterministic, explainable decision support system over real-world policy documents.

Unlike naive LLM chatbots that produce fluent but unverified answers, this system:

 - Retrieves relevant policy evidence

 - Produces structured ALLOW / NOT_ALLOW / UNCERTAIN decisions

 - Cites exact supporting sections

 - Computes confidence deterministically (not via model self-reporting)

 - Enforces abstention below calibrated thresholds

 - Tracks false-ALLOW risk explicitly

 - Supports regression-tested evaluation

This is not a demo chatbot.
It is an engineered reliability-focused RAG system.

---

## Why This Exists

 - Policy and compliance documents are:

 - Long and cross-referential

 - Exception-heavy

 - Context-dependent

 - Risk-sensitive

In these domains, a confident but incorrect approval is more dangerous than abstaining.

Most RAG demos optimize for fluency.

This system optimizes for:

 - Controlled uncertainty

 - Evidence grounding

 - Risk minimization

 - Measurable system behavior
---

## System Guarantees

This system enforces:

  - ❌ No LLM-generated confidence scores

  - ✅ Deterministic confidence computation

  - ✅ Required-field gating before allowability decisions

  - ✅ Explicit abstention below threshold

  - ✅ False-ALLOW rate tracked during evaluation

  - ✅ Regression comparison between runs

  - ✅ Cost tracking per decision
---
## Architecture

  1. Section-aware PDF ingestion

  2. Embedding-based retrieval (ANN search)

  3. Rerank bonus integration

  4. Hybrid evidence scoring

  5. Required-field gating layer

  6. Deterministic confidence computation

  7. Abstain threshold enforcement

  8. Evaluation + regression comparison

---
## Deterministic Confidence Design

  - Confidence is a system property, not a model output.

  - It is computed from:

  - Hybrid evidence strength (cosine similarity + rerank bonus)

  - Evidence coverage (number of relevant chunks)

  - Missing required field penalties

  - Explicit uncertainty penalty

  - Confidence capping when gated

 LLM self-reported confidence is intentionally excluded.

If confidence falls below threshold, the system abstains.

---
## Decision Flow

  1. Retrieve top-k policy sections

  2. Draft decision using retrieved evidence

  3. Determine required fields for question type

  4. Gate if required inputs missing

  5. Compute deterministic confidence

  6. Apply abstain rule

  7. Log run for evaluation & regression tracking

---

## Output Schema (Example)

```json
{
  "question": "Is purchasing a $1,200 laptop allowable?",
  "decision": "ALLOW",
  "justification": "The cost is necessary and reasonable under §200.403...",
  "citations": [
    {
      "source": "uniform_guidance.pdf",
      "section": "200.403",
      "quote": "Costs must be necessary and reasonable..."
    }
  ],
  "confidence": {
    "score": 0.65,
    "reasons": [
      "Strong evidence match",
      "Multiple relevant chunks"
    ]
  },
  "needs_more_info": false
}
```

# 
## Evaluation Framework

Evaluation is built into the system.

Metrics tracked:

 - Accuracy

  - Abstain rate

  - False ALLOW rate

  - False NOT_ALLOW rate

  - Average confidence

  - Average cost per decision

Example regression comparison:

      accuracy            70.00%  -> 80.00%
      abstain_rate        50.00%  -> 40.00%
      false_allow_rate     0.00%  -> 0.00%
      false_not_allow_rate 0.00%  -> 0.00%
      avg_confidence       0.468  -> 0.468
      avg_cost     $0.001074  -> $0.001072

Threshold tuning is regression-tested to prevent false-ALLOW inflation

## Set Up

      git clone https://github.com/paschal-godwin/deterministic-policy-decision-agent.git
      cd policy-decision-agent

      pip install -r requirements.txt

      pip install -e .

## Running the CLI
```python -m policy_agent.cli "Is travel for a conference hotel allowable?" ```

## Running Evaluation
    python -m eval.run_eval

## Set baseline:

    python -m eval.run_eval --set-baseline

## Project Structure
      src/policy_agent/
          decide/
          retrieval/
          confidence/
          judge.py
          pipeline.py

      eval/
          cases/
          metrics.py
          run_eval.py

      docs/
          architecture.png
# Design Decisions
## Abstain-First Philosophy

In compliance domains, false approvals are more costly than abstentions.

The system prioritizes risk minimization.

## Deterministic Confidence

Confidence is derived from measurable signals — not LLM output.

## Required-Field Gating

Allowability decisions are blocked if required inputs are missing.

## Regression Comparison

All evaluation runs are stored and compared to detect performance regressions.

## Motivation

This project explores:

Reliability-focused RAG engineering

Deterministic confidence calibration

Risk-aware AI system design

Evaluation-driven iteration

Production-minded architecture for high-stakes domains

## Disclaimer

This system is for educational and architectural demonstration purposes only.
It does not constitute legal advice or replace formal compliance review.