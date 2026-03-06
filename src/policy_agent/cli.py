import argparse
import json
from typing import Optional, Dict, Any

from policy_agent.decide.pipeline import answer_question
from policy_agent.decide.schema import Facts


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("budgeted_and_approved must be true/false")


def _facts_from_args(args: argparse.Namespace) -> Optional[Facts]:
    facts = Facts(
        purpose=args.purpose,
        federal_award=args.federal_award,
        intended_user=args.intended_user,
        estimated_cost=args.estimated_cost,
        equipment_or_supply=args.equipment_or_supply,
        budgeted_and_approved=args.budgeted_and_approved,
    )
    if all(v is None for v in facts.model_dump().values()):
        return None
    return facts


def prompt_for_missing(missing_info_needed: Dict[str, str], facts: Optional[Facts]) -> Facts:
    # Start from existing facts if provided, otherwise blank schema
    base = facts.model_dump() if facts is not None else {
        "purpose": None,
        "federal_award": None,
        "intended_user": None,
        "estimated_cost": None,
        "equipment_or_supply": None,
        "budgeted_and_approved": None,
    }

    print("\n--- Missing info required to continue ---")
    for field, prompt in missing_info_needed.items():
        msg = prompt or f"Please provide: {field}"
        raw = input(f"{msg}\n> ").strip()

        if raw == "":
            continue  # allow skip, pipeline may still gate if required remains missing

        if field == "estimated_cost":
            try:
                base[field] = float(raw)
            except ValueError:
                print("Invalid number. Skipping.")
        elif field == "budgeted_and_approved":
            try:
                base[field] = parse_bool(raw)
            except Exception:
                print("Invalid boolean (true/false). Skipping.")
        else:
            base[field] = raw

    return Facts(**base)


def main():
    parser = argparse.ArgumentParser(description="Policy Decision Agent CLI")

    parser.add_argument(
        "--question",
        type=str,
        default="Is purchasing a laptop allowable under this policy?",
        help="Question to ask the policy agent",
    )

    # Facts (optional)
    parser.add_argument("--purpose", type=str, default=None)
    parser.add_argument("--federal-award", type=str, default=None)
    parser.add_argument("--intended-user", type=str, default=None)
    parser.add_argument("--estimated-cost", type=float, default=None)
    parser.add_argument("--equipment-or-supply", type=str, default=None)
    parser.add_argument("--budgeted-and-approved", type=parse_bool, default=None)

    parser.add_argument("-k", type=int, default=5, help="Top K evidence chunks")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If the system needs more info, prompt for missing fields and rerun automatically.",
    )

    args = parser.parse_args()
    facts = _facts_from_args(args)

    # First run
    out: Dict[str, Any] = answer_question(args.question, k=args.k, facts=facts)

    # Auto-continue loop (safe max iterations to avoid infinite loops)
    # Mode A loop
    if args.interactive and out.get("needs_more_info") and out.get("missing_info_needed"):
        missing = out.get("missing_fields", [])
        missing_map = out.get("missing_info_needed", {})

        print("\n==============================")
        print("MISSING REQUIRED INPUTS")
        print("==============================")

        # 1) Show an explicit checklist
        for f in missing:
            msg = missing_map.get(f, "")
            if msg:
                print(f"- {f}: {msg}")
            else:
                print(f"- {f}")

    print("\nPlease provide the missing fields below.\n")
    

    # 2) Then prompt only for those fields
    facts2 = prompt_for_missing(missing_map, facts)
    out = answer_question(args.question, k=args.k, facts=facts2)
    print("\n==============================")
    print("FACTS PROVIDED")
    print("==============================")
    for k, v in facts2.model_dump().items():
        if v is not None:
            print(f"- {k}: {v}")
    print()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()