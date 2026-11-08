"""Run held-out lexical retrieval checks without API calls or account data."""
import argparse
import json
import tempfile
import time
from pathlib import Path

from learning_app.data import EVALUATION_CASES
from learning_app.learning import Learning, retrieve
from learning_app.store import Store


def evaluate():
    work = Path(__file__).resolve().parent / "data/evaluation-work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as folder:
        store = Store(Path(folder) / "evaluation.sqlite3")
        user = store.register("Evaluation", "evaluation@local.invalid", "evaluation-only-secret")
        learning = Learning(store, provider="local")
        courses = {course["slug"]: course for course in learning.courses(user["id"])}
        details = []
        latencies = []
        for slug, question, expected in EVALUATION_CASES:
            started = time.perf_counter()
            found = retrieve(learning.materials(courses[slug]["id"]), question)
            latencies.append((time.perf_counter() - started) * 1000)
            titles = [row["source"] for row in found]
            details.append({"course": slug, "question": question, "expected": expected, "returned": titles, "hit_at_1": bool(titles) and titles[0] == expected, "hit_at_3": expected in titles})
        unsupported = [("python", "How does gradient descent update weights?"), ("sql", "How do mutable Python defaults behave?"), ("machine-learning", "Which planet grows purple jellyfish?")]
        abstentions = [not retrieve(learning.materials(courses[slug]["id"]), question) for slug, question in unsupported]
        result = {"evaluation": "held-out seeded-course lexical retrieval", "cases": len(details), "hit_at_1": sum(item["hit_at_1"] for item in details) / len(details), "hit_at_3": sum(item["hit_at_3"] for item in details) / len(details), "unsupported_abstention_rate": sum(abstentions) / len(abstentions), "retrieval_mean_latency_ms": round(sum(latencies) / len(latencies), 3), "provider": "local", "external_calls": 0, "details": details}
        result["passed"] = result["hit_at_3"] >= .9 and result["unsupported_abstention_rate"] == 1
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional destination for the measured JSON report")
    args = parser.parse_args()
    report = evaluate()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)
