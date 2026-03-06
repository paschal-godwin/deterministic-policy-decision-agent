from policy_agent.store.db import init_db
from policy_agent.store.runs import save_run, get_run

if __name__ == "__main__":
    init_db()

    run_id = save_run(
        question="Is buying a laptop allowable?",
        decision="ALLOW",
        justification="Temporary test justification.",
        confidence=0.82
    )

    print("Run ID:", run_id)
    print(get_run(run_id))
