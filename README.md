# Football Agentic RAG

A lightweight **Agentic RAG framework for automated football match
summarization and tactical insight generation**, built on structured
StatsBomb event data (not video/audio). It implements and benchmarks two
retrieval-augmented pipelines side by side against the same knowledge base,
so the comparison isolates retrieval-strategy and agent-architecture
differences rather than data differences.

## What's here

- **Naive RAG baseline** — ingest → chunk → embed → dense retrieve →
  single-pass Gemini generate. The benchmark the agentic pipeline is
  compared against.
- **6-agent Agentic RAG pipeline**, orchestrated with LangGraph:
  `Query Understanding → Retriever → Context Aggregator → Reasoning →
  Generator → Evaluator`. Retrieval is hybrid (dense cosine similarity +
  BM25 keyword search, fused via Reciprocal Rank Fusion, then optionally
  re-ranked by a cross-encoder). The Evaluator scores each response and can
  loop back to the Generator with feedback for a bounded number of retries.
- **`StatsAgent`** — a seventh agent alongside the six, routed to directly
  for counting/aggregate questions ("how many yellow cards", "how many
  shots did Messi take"). Retrieval only ever samples a top-N of chunks, so
  it structurally can't answer "count all X" reliably; `StatsAgent` instead
  filters the full, un-sampled event list deterministically — no LLM call,
  no quota cost, and every number is traceable back to real events.

Both pipelines retrieve from the same **hierarchical knowledge base** built
from StatsBomb's free open-data event feeds: event-level chunks, 5-minute
time-window chunks, possession-sequence chunks, and lineup/formation
chunks. That range is what lets you ask broad "summarize the match"
questions, entity/timeframe-scoped questions ("what did Mbappé do in the
second half"), exact counts, and formation questions against the same
match.

All generation is Gemini-backed (free tier by default); all embedding and
re-ranking runs locally via `sentence-transformers` at zero API cost.

## Repository layout

```
src/football_rag/
  ingestion/       StatsBomb open-data fetch + parsing into a typed Event model
  chunking/        event / time-window / sequence / lineup chunk builders
  embeddings/      local sentence-transformers wrapper
  retrieval/       Chroma vector store, BM25 index, cross-encoder re-ranker
  generation/      naive RAG baseline (prompts + generator)
  agents/          the 6-agent pipeline + StatsAgent, wired with LangGraph
  evaluation/      P@K/R@K, ROUGE, BLEU, BERTScore + a starter query dataset
  api/             FastAPI service exposing the agentic pipeline over HTTP
scripts/           CLI entry points + the Streamlit UI
experiment/        research-scale evaluation sweep (18 matches, ~120 queries)
tests/             import/smoke tests
```

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and add a free Gemini API key
(https://aistudio.google.com/apikey):

```
GEMINI_API_KEY="your-key-here"
```

`.env` is gitignored — never put a real key in `.env.example`, that file
*is* committed. `.env.example` also documents `GEMINI_MODEL` /
`GEMINI_MODEL_LITE` (optional overrides for which Gemini tier each agent
uses) and `HF_API_KEY` (optional — only needed if you hit anonymous
Hugging Face rate limits downloading the local embedding/re-ranker
models).

## Running it

**Naive baseline** (single Gemini call):

```bash
uv run python scripts/run_vertical_slice.py [match_id] ["query"]
```

**Full 6-agent pipeline** (CLI, prints every agent's intermediate output):

```bash
uv run python scripts/run_agentic_pipeline.py [match_id] ["query"]
```

**Browser UI** for the agentic pipeline — a competition/season/match picker
plus a stage-by-stage view of all six agents. Two processes: a FastAPI
backend, then Streamlit as a pure HTTP client of it.

```bash
uv run python scripts/run_api.py        # terminal 1 — backend, http://localhost:8000
uv run streamlit run scripts/app.py     # terminal 2 — UI, http://localhost:8501
```

Both default to `match_id` 3869685 (2022 FIFA World Cup Final, Argentina vs
France) and a "summarize the match" query if you don't pass your own.

Every run (CLI or API) saves its generated summary/evidence to
`outputs/<prefix>_<match_id>_<timestamp>.txt` and its execution log to
`logs/<prefix>_<timestamp>.log` — both directories are gitignored.

## Evaluation

A small illustrative evaluation harness ships in the repo itself:

```bash
uv run python scripts/run_evaluation.py   # naive vs. agentic: P@5/R@5, ROUGE-L, BLEU
uv run python scripts/run_ablation.py     # 5 configs, each with one component disabled
```

The `experiment/` directory holds the full research-scale version of this
harness — 18 matches, ~120 queries, resumable multi-day sweeps to stay
inside Gemini's free-tier daily quota. See `experiments.md` for what's
actually been found so far, including headline numbers such as:

| Comparison | Result |
|---|---|
| Naive vs. agentic (122 query pairs) | see `experiments.md` §2–3 for retrieval and generation quality deltas |
| Embedding model (MiniLM vs. E5-small) | MiniLM: P@5 0.467 / R@5 0.294 — edges out E5 on every metric, though narrowly |
| Ablation (75 rows, 5 configs) | isolates the contribution of hybrid retrieval, re-ranking, reasoning, and the evaluator loop |
| Human evaluation | 3 raters × 20 blinded pairs, with inter-rater agreement analysis |

## Testing

There's a small import/smoke test suite:

```bash
uv run python -m unittest discover -s tests
```

Beyond that, no automated test suite covers the Gemini-backed pipelines
end-to-end yet — see `TESTING.md` for the full manual test plan (every
entry point above, expected output, and troubleshooting for known issues
like Gemini free-tier rate limits).

## License

MIT — see `LICENSE`.

## Author

Arnab Bera — [linkedin.com/in/arnabbera-tech](https://www.linkedin.com/in/arnabbera-tech)
