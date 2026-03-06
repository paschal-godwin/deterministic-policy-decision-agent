import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from policy_agent.decide.schema import Facts


from policy_agent.decide.pipeline import answer_question
from eval.metrics import (
    compute_accuracy,
    compute_abstain_rate,
    compute_false_allow_rate,
    compute_false_not_allow_rate,
    compute_avg_confidence,
    compute_avg_cost,
)

CASES_PATH = Path("eval/cases/core.jsonl")
REPORT_PATH = Path("eval/last_report.json")
REPORTS_DIR = Path("eval/reports")
LATEST_RESULTS_PATH = REPORTS_DIR / "latest_results.json"
LATEST_SUMMARY_PATH = REPORTS_DIR / "latest_summary.json"
BASELINE_SUMMARY_PATH = REPORTS_DIR / "baseline_summary.json"


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def save_report(results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def build_summary(results):
    return {
        "total_cases": len(results),
        "accuracy": compute_accuracy(results),
        "abstain_rate": compute_abstain_rate(results),
        "false_allow_rate": compute_false_allow_rate(results),
        "false_not_allow_rate": compute_false_not_allow_rate(results),
        "avg_confidence": compute_avg_confidence(results),
        "avg_cost": compute_avg_cost(results),
    }

def run():
    cases = load_cases(CASES_PATH)

    results = []

    print(f"Running evaluation on {len(cases)} cases...\n")

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        facts_data = case.get("facts")

        facts = None
        if isinstance(facts_data, dict):
            facts = Facts(**facts_data)
        elif facts_data is None:
            facts = None
        else:
            raise ValueError(f"Invalid facts type in case: {type(facts_data)}")

        output = answer_question(question, facts=facts)

        expected = case["expected_decision"]

        start = time.time()
        latency = time.time() - start

        actual = output["decision"]
        confidence = output["confidence"]["score"]

        cost = output.get("cost", 0.0)

        results.append(
            {
                "expected": expected,
                "actual": actual,
                "confidence": confidence,
                "latency": latency,
                "cost": cost,
                
            }
        )

        print(f"[{i}] Expected={expected} | Actual={actual} | Conf={confidence:.2f}")

    print("\n==============================")
    print("EVALUATION REPORT")
    print("==============================")

    print(f"Accuracy: {compute_accuracy(results):.2%}")
    print(f"Abstain Rate: {compute_abstain_rate(results):.2%}")
    print(f"False ALLOW Rate: {compute_false_allow_rate(results):.2%}")
    print(f"False NOT_ALLOW Rate: {compute_false_not_allow_rate(results):.2%}")
    print(f"Avg Confidence: {compute_avg_confidence(results):.2f}")
    print(f"Avg Cost per Decision: ${compute_avg_cost(results):.6f}")

    print("==============================")

    summary = build_summary(results)

    save_json(LATEST_RESULTS_PATH, results)
    save_json(LATEST_SUMMARY_PATH, summary)
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-baseline", action="store_true", help="Save current summary as baseline")
    args = parser.parse_args()

    results, summary = run()

    if args.set_baseline:
        save_json(BASELINE_SUMMARY_PATH, summary)
        print(f"\n✅ Baseline saved to: {BASELINE_SUMMARY_PATH}")

if __name__ == "__main__":
    main()

