from typing import Optional
from policy_agent.store.db import get_connection

def save_run(
    *,
    question: str,
    decision: str,
    justification: str,
    confidence: float
) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO runs (question, decision, justification, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (question, decision, justification, confidence)
    )

    conn.commit()
    run_id = cursor.lastrowid
    conn.close()

    return run_id


def get_run(run_id: int) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM runs WHERE id = ?",
        (run_id,)
    )
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None
