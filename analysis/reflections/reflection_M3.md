---
name: M3 Reflection
type: reflection
module: m3_rerank
---

# Module 3: Reranking - Implementation Notes

## Implemented Components

### CrossEncoderReranker
- Uses `FlagReranker` from FlagEmbedding (Option A from hints)
- Lazy-loaded model via `_load_model()` with `use_fp16=True` for efficiency
- Default model: `BAAI/bge-reranker-v2-m3`
- `rerank()` computes query-document pairs, sorts descending by score, returns top-k `RerankResult`

### FlashrankReranker
- Lightweight alternative using `flashrank.Ranker`
- Lazy-initializes on first call
- Maps flashrank results back to `RerankResult` with original scores looked up via text match

### benchmark_reranker()
- Times `n_runs` iterations with `time.perf_counter()`
- Returns `{"avg_ms", "min_ms", "max_ms"}` in milliseconds

## Key Design Decisions
- FlagReranker chosen over CrossEncoder for simpler API (`compute_score` vs `predict`)
- Lazy loading ensures models are only downloaded/loaded when first used
- RerankResult dataclass preserves both original_score and rerank_score for analysis
