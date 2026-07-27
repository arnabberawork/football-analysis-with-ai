# Football Agentic RAG

A lightweight Agentic RAG framework for automated football match
summarization and tactical insight generation, built on structured
StatsBomb event data. Implements the design in
`Research_Proposal_MSML_Arnab_Bera.pdf` — see `CLAUDE.md` for the full
architecture writeup (module-by-module, mapped to the proposal's numbered
sections).

Two pipelines, plus an evaluation harness comparing them:

- **Naive RAG baseline** — dense-only retrieve, single-pass generate. The
  benchmark the agentic pipeline is compared against.
- **6-agent Agentic RAG pipeline** — Query Understanding → Retriever →
  Context Aggregator → Reasoning → Generator → Evaluator, with hybrid
  (dense + BM25 + cross-encoder re-rank) retrieval. A `StatsAgent` sits
  alongside the six, routed to directly for counting/aggregate questions
  ("how many yellow cards", "how many shots did Messi take") that plain
  retrieval can't answer reliably — see `CLAUDE.md` for why.

Both are Gemini-backed (free tier) and retrieve from the same hierarchical
knowledge base (event-level, time-window, possession-sequence, and
lineup/formation chunks) — designed so you can ask broad summary
questions, entity/timeframe-scoped questions ("what did Mbappé do in the
second half"), exact counts, and lineup/formation questions all against the
same match.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and add a free Gemini API key
(https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key-here
```

`.env` is gitignored — never put a real key in `.env.example`, that file
*is* committed. The Gemini model tier is also configurable there
(`GEMINI_MODEL`, `GEMINI_MODEL_LITE` — see `.env.example` for what each
controls and their defaults) rather than hardcoded per file.

## Running it

**Naive baseline** (single Gemini call):

```bash
uv run python scripts/run_vertical_slice.py [match_id] ["query"]
```

**Full 6-agent pipeline** (CLI, prints every agent's output):

```bash
uv run python scripts/run_agentic_pipeline.py [match_id] ["query"]
```

**Browser UI** for the agentic pipeline — a match browser plus a
stage-by-stage view of all six agents. Two processes: a FastAPI backend,
then Streamlit as a pure HTTP client of it.

```bash
uv run python scripts/run_api.py        # terminal 1 — backend, http://localhost:8000
uv run streamlit run scripts/app.py     # terminal 2 — UI, http://localhost:8501
```

Both default to `match_id` 3869685 (2022 FIFA World Cup Final, Argentina vs
France) and a "summarize the match" query if you don't pass your own.

**Evaluation harness** — naive-vs-agentic baseline comparison (P@5/R@5,
ROUGE-L, BLEU) and an ablation study, both over a small illustrative
starter query set (see `TESTING.md` for what that dataset is and isn't):

```bash
uv run python scripts/run_evaluation.py
uv run python scripts/run_ablation.py   # quota-heavy — best after a quota reset
```

For the full research-scale version of this harness (18 matches, ~61
queries, resumable multi-day sweeps) see `experiment/README.md` for how to
run it and **`experiments.md`** for what's actually been run so far and
what it found.

Every script above (including each API call) saves its final generated
output to `outputs/<prefix>_<timestamp>.txt` for later comparison, and its
execution progress to a leveled, timestamped log under
`logs/<prefix>_<timestamp>.log` (see `TESTING.md` §11 for the split between
the two).

## Testing

No automated test suite yet — see `TESTING.md` for the full manual test
plan (every entry point above, expected output, and troubleshooting for
known issues like Gemini free-tier rate limits).

## More detail

`CLAUDE.md` has the module-by-module architecture writeup, mapped to the
research proposal's sections — useful background even if you're not using
Claude Code.
