---
name: Thái Tuấn Khang
ID: 2A202600289
module: m1_chunking
---

# Module 1: Advanced Chunking - Implementation Notes

## What was implemented

Four chunking strategies were implemented in `/mnt/f/Lab18/src/m1_chunking.py`:

1. **chunk_semantic()** - Encodes sentences using sentence-transformers (all-MiniLM-L6-v2), groups consecutive sentences by cosine similarity threshold. When similarity drops below threshold, a new chunk is started.

2. **chunk_hierarchical()** - Creates parent chunks (2048 chars) by grouping paragraphs, then splits each parent into child chunks (256 chars) using a sliding window. Each child carries a `parent_id` reference linking back to its parent.

3. **chunk_structure_aware()** - Parses markdown headers (#, ##, ###) using regex, pairs each header with its following content as a single chunk. Preserves tables, code blocks, and lists intact.

4. **compare_strategies()** - Runs all four strategies across loaded documents, collects statistics (num_chunks, avg_length, min_length, max_length), prints a formatted comparison table, and returns a dict of results.

## Challenges faced

- The hierarchical chunking required careful handling of the parent-child relationship. Ensuring parent_id linkage was correctly set on every child while maintaining proper chunk indices required tracking both parents and children separately.
- For structure-aware chunking, the regex split creates alternating header/non-header sections, requiring proper state management to pair headers with their content blocks.
- The compare_strategies function needed to handle the different return types (list vs tuple) across strategies - basic/semantic/structure return lists while hierarchical returns a tuple of (parents, children).

## Key design decisions

- Used all-MiniLM-L6-v2 for semantic encoding as it's fast and produces 384-dimensional embeddings suitable for cosine similarity comparisons.
- Hierarchical chunking uses simple sliding window for child splitting rather than sentence-aware splitting, trading some granularity for simplicity and predictability.
- Structure-aware chunking treats content before any header as an "untitled" section rather than discarding it, ensuring no content is lost.
- The comparison function focuses on children for hierarchical strategy since children are what get indexed in the vector database in production use.