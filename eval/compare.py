import json
from pathlib import Path

REPORTS_DIR = Path("eval/reports")
BASELINE = REPORTS_DIR / "baseline_summary.json"
LATEST = REPORTS_DIR / "latest_summary.json"

KEYS = [
    "accuracy",
    "abstain_rate",
    "false_allow_rate",
    "false_not_allow_rate",
    "avg_confidence",
    "avg_cost",
]

def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pct(x):
    return f"{x*100:.2f}%"

def money(x):
    return f"${x:.6f}"

def fmt(key, val):
    if "rate" in key or key == "accuracy":
        return pct(val)
    if "cost" in key:
        return money(val)
    return f"{val:.4f}"

def compare():
    if not BASELINE.exists():
        print("❌ No baseline found. Run: python -m eval.run_eval --set-baseline")
        return
    if not LATEST.exists():
        print("❌ No latest summary found. Run: python -m eval.run_eval")
        return

    baseline = load(BASELINE)
    latest = load(LATEST)

    print("\n==============================")
    print("REGRESSION COMPARISON")
    print("==============================")
    print(f"Baseline cases: {baseline.get('total_cases')} | Latest cases: {latest.get('total_cases')}\n")

    for key in KEYS:
        bval = baseline.get(key, 0.0)
        lval = latest.get(key, 0.0)
        delta = lval - bval

        sign = "+" if delta >= 0 else "-"
        print(f"{key:20} {fmt(key,bval):>10}  ->  {fmt(key,lval):>10}   ({sign}{abs(delta):.4f})")

    # Critical compliance safety gate
    if latest.get("false_allow_rate", 0.0) > baseline.get("false_allow_rate", 0.0):
        print("\n🚨 REGRESSION ALERT: false_allow_rate increased. Investigate immediately.")
    else:
        print("\n✅ No false-ALLOW regression detected.")

if __name__ == "__main__":
    compare()