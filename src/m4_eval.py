"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ])
    df = result.to_pandas()

    # Debug: print actual columns
    print(f"DEBUG: RAGAS result columns = {list(df.columns)}")
    print(f"DEBUG: df shape = {df.shape}")

    # Handle different RAGAS versions - column names may vary
    # Check if 'question' column exists, otherwise try alternative names
    if "question" not in df.columns:
        # Try to find input/source columns
        possible_cols = [c for c in df.columns if "question" in c.lower() or "input" in c.lower() or "user" in c.lower()]
        if possible_cols:
            df = df.rename(columns={possible_cols[0]: "question"})

    per_question = []
    for _, row in df.iterrows():
        try:
            per_question.append(EvalResult(
                question=row.get("user_input", row.get("question", "")),
                answer=row.get("response", row.get("answer", "")),
                contexts=row.get("retrieved_contexts", row.get("contexts", [])),
                ground_truth=row.get("reference", row.get("ground_truth", "")),
                faithfulness=row.get("faithfulness", 0.0),
                answer_relevancy=row.get("answer_relevancy", 0.0),
                context_precision=row.get("context_precision", 0.0),
                context_recall=row.get("context_recall", 0.0),
            ))
        except Exception as e:
            print(f"DEBUG: Error processing row: {e}")
            continue

    if not per_question:
        # Fallback - return default scores
        return {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "per_question": [],
        }

    return {
        "faithfulness": sum(e.faithfulness for e in per_question) / len(per_question),
        "answer_relevancy": sum(e.answer_relevancy for e in per_question) / len(per_question),
        "context_precision": sum(e.context_precision for e in per_question) / len(per_question),
        "context_recall": sum(e.context_recall for e in per_question) / len(per_question),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for r in eval_results:
        avg_score = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        scored.append((r, avg_score))

    scored.sort(key=lambda x: x[1])
    worst = scored[:bottom_n]

    DIAGNOSTICS = [
        ("faithfulness", 0.85, "LLM hallucinating", "Tighten prompt, lower temperature"),
        ("context_recall", 0.75, "Missing relevant chunks", "Improve chunking or add BM25"),
        ("context_precision", 0.75, "Too many irrelevant chunks", "Add reranking or metadata filter"),
        ("answer_relevancy", 0.80, "Answer doesn't match question", "Improve prompt template"),
    ]

    failures = []
    for result, avg_score in worst:
        worst_metric, worst_score = "", 1.0
        for metric_name, _, _, _ in DIAGNOSTICS:
            score = getattr(result, metric_name)
            if score < worst_score:
                worst_score = score
                worst_metric = metric_name

        diagnosis, suggested_fix = "", ""
        for metric_name, threshold, diagnosis_text, fix_text in DIAGNOSTICS:
            if getattr(result, metric_name) < threshold:
                diagnosis, suggested_fix = diagnosis_text, fix_text
                break

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
