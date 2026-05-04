---
name: M2 Reflection
type: reflection
module: m2_search
---

# Module 2: Hybrid Search - Implementation Notes

## Implemented Functions

### `segment_vietnamese()`
- Uses `underthesea.word_tokenize` with `format="text"` to segment Vietnamese text
- Returns space-separated words: "nghỉ phép" stays as one token rather than splitting into characters
- Critical for BM25 which needs proper word boundaries

### `BM25Search.index()`
- Stores chunks in `self.documents`
- Tokenizes each chunk text via `segment_vietnamese()` then splits by space
- Builds `BM25Okapi` index from the tokenized corpus

### `BM25Search.search()`
- Tokenizes query the same way: `segment_vietnamese(query).split()`
- Gets BM25 scores via `get_scores()`
- Returns top-k results sorted by score descending

### `DenseSearch.index()`
- Recreates Qdrant collection with cosine similarity
- Encodes all chunk texts using `sentence_transformers` (BGE-M3)
- Uploads points with vector + payload (metadata + text)

### `DenseSearch.search()`
- Encodes query to dense vector
- Performs Qdrant nearest-neighbor search
- Returns results with score and payload

### `reciprocal_rank_fusion()`
- Merges ranked lists using RRF formula: `score(d) = Σ 1/(k + rank + 1)` with k=60
- Accumulates scores across all result lists for each unique document
- Returns top-k hybrid results sorted by fused score

## Design Decisions
- Lazy encoder initialization in `DenseSearch._get_encoder()` to avoid loading model until needed
- Used `c.get("metadata", {})` for safety when chunks lack metadata field
- RRF uses `rank + 1` to avoid division by zero (rank is 0-indexed)
