---
name: Nguyen Thanh Luan
type: reflection
module: eval
---

# Module 4: RAGAS Evaluation - Implementation Notes

## Implemented Functions

### evaluate_ragas()
- Builds a `Dataset` from `datasets.Dataset` via `from_dict` with question/answer/contexts/ground_truth columns
- Runs `evaluate()` with all 4 RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Converts result to pandas DataFrame and computes per-question `EvalResult` records
- Returns aggregate means plus per-question list matching the expected dict shape

### failure_analysis()
- Computes per-question average score from the 4 metrics and sorts ascending to find bottom-N
- Identifies the `worst_metric` (lowest-scoring metric) and maps it to a diagnostic category using fixed thresholds:
  - faithfulness < 0.85 → "LLM hallucinating" → "Tighten prompt, lower temperature"
  - context_recall < 0.75 → "Missing relevant chunks" → "Improve chunking or add BM25"
  - context_precision < 0.75 → "Too many irrelevant chunks" → "Add reranking or metadata filter"
  - answer_relevancy < 0.80 → "Answer doesn't match question" → "Improve prompt template"
- Returns a list of failure dictionaries ready for `save_report()`

## Notes
- All imports are inside the function (lazy loading) to avoid top-level failures if ragas is not installed
- The Diagnostic Tree thresholds are fixed constants per the task specification
- `save_report()` and `load_test_set()` were pre-implemented and left untouched
