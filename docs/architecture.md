# Architecture

## Phase 1 data flow

Source CSV -> source-specific column mapping (future adapter) -> canonical job schema -> validation report -> processed dataset (future phase).

## Design decisions

- Keep ingestion, NLP, matching, evaluation and UI in separate modules.
- Use a canonical schema so multiple public datasets can be compared consistently.
- Preserve source data and provenance metadata; never overwrite raw data.
- Run the initial pipeline on CPU. The proposed `all-MiniLM-L6-v2` embedding model is a later-phase option, not a Phase 1 dependency.
- Treat scores as relevance signals, not hiring recommendations.

## Planned ranking pipeline

1. Lexical baseline: TF-IDF with cosine similarity.
2. Semantic retrieval: sentence-transformer embeddings.
3. Reranking: cross-encoder only if CPU latency and memory benchmarks justify it.
4. Explanation: surface matched requirements and source-text evidence.
5. Evaluation: query-level relevance judgments and ranking metrics.
