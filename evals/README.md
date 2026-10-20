# RAG Evaluation

`golden_rag.jsonl` contains retrieval cases with expected source files and key terms.

Run:

```bash
python -m evals.rag_eval --rebuild --min-hit-rate 0.8
```

Use this set as a regression gate whenever the chunker, embedding model, or retrieval
strategy changes. Add cases from real user questions instead of only demo questions.
