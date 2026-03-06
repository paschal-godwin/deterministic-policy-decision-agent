import json
import streamlit as st
from typing import Dict, Any, Optional

from policy_agent.decide.pipeline import answer_question
from policy_agent.decide.schema import Facts


FACT_FIELDS_ORDER = [
    "purpose",
    "federal_award",
    "intended_user",
    "estimated_cost",
    "equipment_or_supply",
    "budgeted_and_approved",
]


def _make_facts(existing: Optional[Facts], updates: Dict[str, Any]) -> Facts:
    base = existing.model_dump() if existing is not None else {
        "purpose": None,
        "federal_award": None,
        "intended_user": None,
        "estimated_cost": None,
        "equipment_or_supply": None,
        "budgeted_and_approved": None,
    }
    base.update(updates)
    return Facts(**base)


def _render_output(out: Dict[str, Any]):
    st.subheader("Final Output")
    st.json(out)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", out.get("decision", ""))
    col2.metric("Confidence", f"{out.get('confidence', {}).get('score', 0.0):.2f}")
    col3.metric("Needs More Info", str(out.get("needs_more_info", False)))

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


def main():
    st.set_page_config(page_title="Policy Decision Agent", layout="centered")
    st.title("Deterministic Policy Decision Agent")
    st.caption("Abstain-first gating • Deterministic confidence • Evidence-backed decisions")

    if "facts" not in st.session_state:
        st.session_state.facts = None # type: Optional[Facts]
    if "last_out" not in st.session_state:
        st.session_state.last_out = None # type: Optional[Dict[str, Any]]  
    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input(
        "Question",
        value=st.session_state.question,
        placeholder="e.g., Can we pay for hotel for a conference trip?",
    )
    k = st.slider("Top-K evidence chunks", min_value=3, max_value=10, value=5, step=1)

    run = st.button("Run")

    if run:
        st.session_state.question = question
        st.session_state.last_out = answer_question(question, k=k, facts=st.session_state.facts)

    out = st.session_state.last_out
    if out is None:
        st.info("Enter a question and click Run.")
        return

    # --- Missing inputs UX: show explicit checklist + form ---
    if out.get("needs_more_info") and out.get("missing_info_needed"):
        missing_fields = out.get("missing_fields", [])
        missing_map: Dict[str, str] = out.get("missing_info_needed", {})

        st.warning("Missing required inputs. Please provide the fields below to continue.")

        # 1) Explicit checklist panel (field + why)
        st.subheader("Missing required inputs")
        for f in missing_fields:
            reason = missing_map.get(f, "")
            if reason:
                st.markdown(f"- **{f}** — {reason}")
            else:
                st.markdown(f"- **{f}**")

        st.divider()
        st.subheader("Fill missing fields")

        # Pre-fill with existing facts (if any)
        existing = st.session_state.facts.model_dump() if st.session_state.facts is not None else {}
        existing = {k: existing.get(k) for k in FACT_FIELDS_ORDER}

        # 2) Dynamic form for only missing fields
        with st.form("missing_facts_form"):
            updates: Dict[str, Any] = {}

            # Keep ordering stable
            ordered_missing = [f for f in FACT_FIELDS_ORDER if f in missing_fields]
            for field in ordered_missing:
                prompt = missing_map.get(field) or field

                if field == "estimated_cost":
                    current = existing.get(field)
                    updates[field] = st.number_input(
                        prompt,
                        min_value=0.0,
                        step=1.0,
                        value=float(current) if current is not None else 0.0,
                    )
                elif field == "budgeted_and_approved":
                    current = existing.get(field)
                    default = True if current is None else bool(current)
                    updates[field] = st.selectbox(
                        prompt,
                        options=[True, False],
                        index=0 if default is True else 1,
                    )
                else:
                    current = existing.get(field) or ""
                    updates[field] = st.text_input(prompt, value=current)

            submitted = st.form_submit_button("Continue (Re-run with provided facts)")

        if submitted:
            # Convert empty strings to None so required-field gating stays correct
            cleaned: Dict[str, Any] = {}
            for key, val in updates.items():
                if isinstance(val, str) and val.strip() == "":
                    cleaned[key] = None
                else:
                    cleaned[key] = val

            st.session_state.facts = _make_facts(st.session_state.facts, cleaned)
            st.session_state.last_out = answer_question(question, k=k, facts=st.session_state.facts)
            st.rerun()

    # Show output (either final, or still gated)
    _render_output(st.session_state.last_out)

    # Optional: show facts currently stored in session (transparency)
    st.divider()
    st.subheader("Session Facts (current)")
    if st.session_state.facts is None:
        st.caption("No facts stored yet.")
    else:
        st.json({k: v for k, v in st.session_state.facts.model_dump().items() if v is not None})

    if st.button("Reset session"):
        st.session_state.facts = None
        st.session_state.last_out = None
        st.session_state.question = ""
        st.rerun()


if __name__ == "__main__":
    main()