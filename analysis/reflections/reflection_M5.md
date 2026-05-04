---
name: Trịnh Kế Tiến
type: reflection
module: m5_enrichment
---

# Module 5: Enrichment Pipeline - Implementation Notes

## Implemented Functions

### summarize_chunk()
- Uses `gpt-4o-mini` via OpenAI API to generate a 2-3 sentence Vietnamese summary
- Fallback (no API key): extractive approach using first 2 sentences of the text
- Lazy import of OpenAI client inside the function

### generate_hypothesis_questions()
- Generates N questions the chunk can answer using `gpt-4o-mini`
- Parses newline-separated questions and strips numbering prefixes
- Fallback (no API key): returns empty list

### contextual_prepend()
- Uses `gpt-4o-mini` to write 1 sentence describing where the chunk sits in the document
- Prepends this context to the original text (Anthropic style)
- Fallback (no API key): returns original text unchanged

### extract_metadata()
- Uses `gpt-4o-mini` to extract `{topic, entities, category, language}` as JSON
- Fallback (no API key): returns default metadata dict with empty/unknown values

### enrich_chunks()
- Orchestrates the full pipeline over a list of chunks
- Applies methods conditionally based on the `methods` parameter
- Default methods: `["contextual", "hyqa", "metadata"]`
- Also supports `"summary"` and `"full"` as method options
- Merges auto-extracted metadata with existing chunk metadata via `{**metadata, **auto_meta}`
- Returns list of `EnrichedChunk` dataclass instances

## Design Notes
- All functions check `OPENAI_API_KEY` before attempting API calls and fall back gracefully
- OpenAI imports are lazy (inside function body) to avoid import errors when key is absent
- `enriched_text` contains the original text with context prepended; `original_text` is preserved separately
- The pipeline is designed for one-time offline enrichment to improve retrieval for all subsequent queries
