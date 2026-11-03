# RAG Evaluation

`golden_rag.jsonl` contains retrieval cases with expected source files and key terms.

Run:

```bash
python -m evals.rag_eval --rebuild --min-hit-rate 0.8
```

Use this set as a regression gate whenever the chunker, embedding model, or retrieval
strategy changes. Add cases from real user questions instead of only demo questions.

## Task Evaluation

`golden_tasks.jsonl` evaluates the full task layer:

```bash
python -m evals.task_eval --min-task-success 0.9 --min-routing-accuracy 0.9
```

It measures routing, Skill selection, expected facts, expected sources, latency, and
two deterministic end-to-end workflow smoke cases. Add one case for every production
bug or new business intent.
