from pathlib import Path

from policy_agent.retrieve.retrieve import retrieve_top_k


if __name__ == "__main__":
    question = "Is purchasing computer equipment an allowable expense?"

    results = retrieve_top_k(
        question,
        index_dir=Path("data/index"),
        k=5
    )

    print(f"\nQuestion: {question}\n")
    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Score: {r['score']}")
        print(f"Section: {r['section']}")
        print(f"Pages: {r['pages']}")
        print(r["text_preview"])
        print()
