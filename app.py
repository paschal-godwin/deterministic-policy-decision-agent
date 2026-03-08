import streamlit as st
from typing import Dict, Any, Optional

from policy_agent.decide.pipeline import answer_question, extract_facts_from_question
from policy_agent.decide.schema import Facts


FACT_FIELDS_ORDER = [
    "purpose",
    "federal_award",
    "intended_user",
    "estimated_cost",
    "equipment_or_supply",
    "budgeted_and_approved",
]


FRIENDLY_LABELS = {
    "purpose": "Purpose / relevance to award",
    "federal_award": "Federal award / project",
    "intended_user": "Intended user (role)",
    "estimated_cost": "Estimated cost",
    "equipment_or_supply": "Equipment or supply",
    "budgeted_and_approved": "Budgeted and approved?",
}


def _blank_facts_dict() -> Dict[str, Any]:
    return {
        "purpose": None,
        "federal_award": None,
        "intended_user": None,
        "estimated_cost": None,
        "equipment_or_supply": None,
        "budgeted_and_approved": None,
    }


def _make_facts(existing: Optional[Facts], updates: Dict[str, Any]) -> Facts:
    base = existing.model_dump() if existing is not None else _blank_facts_dict()
    base.update(updates)
    return Facts(**base)


def _render_output(out: Dict[str, Any]):
    st.subheader("Final Output")
    st.json(out)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    decision = out.get("decision", "")
    confidence = out.get("confidence", {}).get("score", 0.0)
    needs_more_info = out.get("needs_more_info", False)

    col1.metric("Decision", getattr(decision, "value", decision))
    col2.metric("Confidence", f"{confidence:.2f}")
    col3.metric("Needs More Info", "Yes" if needs_more_info else "No")

    st.subheader("Citations")
    cits = out.get("citations", [])
    if not cits:
        st.info("No citations returned.")
    else:
        for c in cits:
            st.markdown(
                f"- **{c.get('source','')}** | **{c.get('section','')}** | chunk `{c.get('chunk_id','')}`\n\n"
                f"  > {c.get('quote','')}"
            )


def _render_facts_editor(header: str, facts: Optional[Facts], key_prefix: str) -> Dict[str, Any]:
    st.subheader(header)
    current = facts.model_dump() if facts is not None else _blank_facts_dict()
    updates: Dict[str, Any] = {}

    updates["purpose"] = st.text_input(
        FRIENDLY_LABELS["purpose"],
        value=current.get("purpose") or "",
        key=f"{key_prefix}_purpose",
    )
    updates["federal_award"] = st.text_input(
        FRIENDLY_LABELS["federal_award"],
        value=current.get("federal_award") or "",
        key=f"{key_prefix}_federal_award",
    )
    updates["intended_user"] = st.text_input(
        FRIENDLY_LABELS["intended_user"],
        value=current.get("intended_user") or "",
        key=f"{key_prefix}_intended_user",
    )

    est_cost_current = current.get("estimated_cost")
    updates["estimated_cost"] = st.number_input(
        FRIENDLY_LABELS["estimated_cost"],
        min_value=0.0,
        step=1.0,
        value=float(est_cost_current) if est_cost_current is not None else 0.0,
        key=f"{key_prefix}_estimated_cost",
    )

    equipment_current = current.get("equipment_or_supply") or ""
    updates["equipment_or_supply"] = st.text_input(
        FRIENDLY_LABELS["equipment_or_supply"],
        value=equipment_current,
        key=f"{key_prefix}_equipment_or_supply",
    )

    budget_current = current.get("budgeted_and_approved")
    budget_options = ["Unknown", True, False]
    default_index = 0
    if budget_current is True:
        default_index = 1
    elif budget_current is False:
        default_index = 2

    budget_choice = st.selectbox(
        FRIENDLY_LABELS["budgeted_and_approved"],
        options=budget_options,
        index=default_index,
        key=f"{key_prefix}_budgeted_and_approved",
    )
    updates["budgeted_and_approved"] = None if budget_choice == "Unknown" else budget_choice

    cleaned: Dict[str, Any] = {}
    for key, val in updates.items():
        if isinstance(val, str):
            cleaned[key] = val.strip() or None
        elif key == "estimated_cost":
            cleaned[key] = None if val == 0.0 else val
        else:
            cleaned[key] = val

    return cleaned


def main():
    st.set_page_config(page_title="Policy Decision Agent", layout="centered")
    st.title("Deterministic Policy Decision Agent")
    st.caption("Abstain-first gating • Deterministic confidence • Evidence-backed decisions")

    if "facts" not in st.session_state:
        st.session_state.facts = None
    if "last_out" not in st.session_state:
        st.session_state.last_out = None
    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input(
        "Question",
        value=st.session_state.question,
        placeholder="e.g., Can I purchase a laptop for NIH R01 data analysis? Estimated cost is $1200.",
    )
    k = st.slider("Top-K evidence chunks", min_value=3, max_value=10, value=5, step=1)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Extract facts from question"):
            extracted = extract_facts_from_question(question)
            st.session_state.facts = _make_facts(st.session_state.facts, extracted.model_dump(exclude_none=True))
            st.session_state.question = question
            st.rerun()

    with col_b:
        if st.button("Reset session"):
            st.session_state.facts = None
            st.session_state.last_out = None
            st.session_state.question = ""
            st.rerun()

    updates = _render_facts_editor(
        header="Structured facts (editable before run)",
        facts=st.session_state.facts,
        key_prefix="main_facts",
    )

    if st.button("Run"):
        st.session_state.question = question
        st.session_state.facts = _make_facts(st.session_state.facts, updates)
        st.session_state.last_out = answer_question(
            question,
            k=k,
            facts=st.session_state.facts,
        )

    out = st.session_state.last_out
    if out is None:
        st.info("Enter a question, optionally extract/review facts, then click Run.")
        return

    if out.get("needs_more_info") and out.get("missing_info_needed"):
        missing_fields = out.get("missing_fields", [])
        missing_map: Dict[str, str] = out.get("missing_info_needed", {})

        st.warning("Missing required inputs. Please provide the fields below to continue.")

        st.subheader("Missing required inputs")
        for f in missing_fields:
            reason = missing_map.get(f, "")
            label = FRIENDLY_LABELS.get(f, f)
            if reason:
                st.markdown(f"- **{label}** — {reason}")
            else:
                st.markdown(f"- **{label}**")

        st.divider()

        with st.form("missing_facts_form"):
            st.subheader("Fill missing fields")
            existing = st.session_state.facts.model_dump() if st.session_state.facts is not None else _blank_facts_dict()
            missing_updates: Dict[str, Any] = {}

            ordered_missing = [f for f in FACT_FIELDS_ORDER if f in missing_fields]
            for field in ordered_missing:
                prompt = missing_map.get(field) or FRIENDLY_LABELS.get(field, field)
                current = existing.get(field)

                if field == "estimated_cost":
                    missing_updates[field] = st.number_input(
                        prompt,
                        min_value=0.0,
                        step=1.0,
                        value=float(current) if current is not None else 0.0,
                        key=f"missing_{field}",
                    )
                elif field == "budgeted_and_approved":
                    default_index = 0 if current is True else 1
                    missing_updates[field] = st.selectbox(
                        prompt,
                        options=[True, False],
                        index=default_index,
                        key=f"missing_{field}",
                    )
                else:
                    missing_updates[field] = st.text_input(
                        prompt,
                        value=current or "",
                        key=f"missing_{field}",
                    )

            submitted = st.form_submit_button("Continue (Re-run with provided facts)")

        if submitted:
            cleaned: Dict[str, Any] = {}
            for key, val in missing_updates.items():
                if isinstance(val, str):
                    cleaned[key] = val.strip() or None
                elif key == "estimated_cost":
                    cleaned[key] = None if val == 0.0 else val
                else:
                    cleaned[key] = val

            st.session_state.facts = _make_facts(st.session_state.facts, cleaned)
            st.session_state.last_out = answer_question(
                question,
                k=k,
                facts=st.session_state.facts,
            )
            st.rerun()

    _render_output(st.session_state.last_out)

    st.divider()
    st.subheader("Session Facts (current)")
    if st.session_state.facts is None:
        st.caption("No facts stored yet.")
    else:
        st.json({k: v for k, v in st.session_state.facts.model_dump().items() if v is not None})


if __name__ == "__main__":
    main()