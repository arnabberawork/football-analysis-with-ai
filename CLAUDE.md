# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Implementation of the research proposal in `Research_Proposal_MSML_Arnab_Bera.pdf`: a lightweight **Agentic RAG framework** for automated football match summarization and tactical insight generation, built on structured StatsBomb event data rather than video/audio. Treat that PDF as the design spec — module names, prompt structure, and chunking strategy below map directly to its numbered sections (referenced inline as `§x.y`).

The codebase implements two parallel pipelines, both described in the Commands/Architecture sections below:

- The **naive RAG baseline vertical slice** (ingest → event-level chunk → embed → dense retrieve → single-pass generate), §6.11.5.
- The **6-agent Agentic RAG pipeline** (Query Understanding → Retriever → Context Aggregator → Reasoning → Generator → Evaluator), §6.2.3/§6.3, orchestrated with LangGraph and using hybrid (dense + BM25) retrieval per §6.8.

`NaiveRAGGenerator` is deliberately kept unmodified as the benchmark baseline the agentic pipeline is compared against (§6.11.5) — do not fold its logic into the agentic pipeline or vice versa. Both pipelines now retrieve from the **same hierarchical knowledge base** (event + time-window + possession-sequence chunks) — the baseline/agentic comparison isolates retrieval-strategy and agent-architecture differences, not corpus differences.

## Commands

Package management is via `uv` (src-layout package `football_rag`, installed editable — see `pyproject.toml`'s `[tool.hatch.build.targets.wheel]`).

```bash
uv sync                      # install/update the environment from pyproject.toml + uv.lock
uv add <package>             # add a new dependency
uv run python <script.py>    # run any script with the project environment active
uv run python -c "..."       # one-off snippets against the football_rag package
```

Run the end-to-end naive baseline (ingests one match, embeds, retrieves, generates a summary via Gemini):

```bash
uv run python scripts/run_vertical_slice.py [match_id] ["query"]
# defaults to match_id 3869685 (2022 FIFA World Cup Final, Argentina vs France)
```

Run the same match through the full 6-agent pipeline instead (prints every agent's intermediate output — structured query, fused retrieval hits, deduplicated context, reasoning insights, response, evaluator verdict):

```bash
uv run python scripts/run_agentic_pipeline.py [match_id] ["query"]
```

Or drive the agentic pipeline from a browser UI instead of the CLI — a competition/season/match picker plus a stage-by-stage view of all six agents' output. This is two processes: a FastAPI backend, then the Streamlit UI (a pure HTTP client of it):

```bash
uv run python scripts/run_api.py        # terminal 1 — backend on http://localhost:8000
uv run streamlit run scripts/app.py     # terminal 2 — UI, talks to the backend above
```

Run the evaluation harness (§6.11) — naive-vs-agentic baseline comparison (P@5/R@5, ROUGE-L, BLEU) over a small starter query set:

```bash
uv run python scripts/run_evaluation.py
```

Or the ablation study (§6.11.6) — re-runs the same starter query set through five agentic-pipeline configurations, each with one component toggled off, to isolate its contribution. Quota-heavy (5 configs × N queries × several Gemini calls each) — best run once your free-tier daily quota has reset:

```bash
uv run python scripts/run_ablation.py
```

All of the above require `GEMINI_API_KEY` in a `.env` file at the repo root (copy `.env.example`) — get a free key at https://aistudio.google.com/apikey. For the UI, the key only needs to be set where `scripts/run_api.py` runs (the backend), not where Streamlit runs. The agentic pipeline makes several Gemini calls per run (Query Understanding, Reasoning, Generator, and optionally Evaluator), all Flash-tier models chosen to stay comfortably inside the free-tier rate limits, but it still costs more per invocation than the single-call baseline. No test suite, linter, or formatter is configured yet.

## Architecture

Pipeline stages live under `src/football_rag/`, each corresponding to a proposal section:

1. **`ingestion/statsbomb.py`** (§6.4.1, §6.5) — fetches raw JSON from the public `statsbomb/open-data` GitHub mirror (no auth required), with optional disk caching (`cache_dir`). `parse_events()` normalizes raw StatsBomb event dicts into the `Event` pydantic model (`schema.py`), filtering out administrative event types with no tactical value (`LOW_SIGNAL_EVENT_TYPES`) and deriving a human-readable `detail`/`outcome` per event type in `_describe_event()`.

2. **`chunking/`** (§6.6) — three chunk-granularity tiers, all sharing the `Event -> Chunk` shape:
   - `event_chunker.py` — `chunk_events()`, one sentence per event via a fixed template (`event_to_sentence()`), so the knowledge base stays consistent rather than free LLM phrasing. `chunk_level="event"`.
   - `time_window_chunker.py` — `chunk_time_windows()`, groups events into fixed 5-minute windows per period and renders each window's events (reusing `event_to_sentence()`) into one chunk. Gives the generator narrative substance a single event can't — e.g. a "20'-24'" chunk, not just an isolated Duel sentence. `chunk_level="time_window"`.
   - `sequence_chunker.py` — `chunk_sequences()`, groups events by the existing `Event.possession_id` into one chunk per possession (a team's uninterrupted spell in control). `chunk_level="sequence"`.
   - `__init__.py` — `build_hierarchical_chunks(events)` concatenates all three tiers; this is the one ingestion call site every script/API uses (chunk_id schemes `-ev-`/`-tw-`/`-seq-` are disjoint, so `ChunkVectorStore.ingest()`'s upsert just grows one collection additively).

3. **`embeddings/embedder.py`** (§6.7) — wraps a local `sentence-transformers` model (default `all-MiniLM-L6-v2`, chosen for zero API cost per §6.13's resource constraints). `get_default_embedder()` caches one model instance process-wide; swap models by changing `DEFAULT_MODEL_NAME`.

4. **`retrieval/vector_store.py`** (§6.7, §6.8) — `ChunkVectorStore` wraps a `PersistentClient` ChromaDB collection, chosen over raw FAISS specifically because Chroma supports metadata filtering *during* retrieval (`query(..., where={...})`), which the proposal's metadata-aware retrieval strategy depends on. Embeddings are always computed by our own `Embedder`, never Chroma's built-in embedding function, so the model choice stays explicit. `ingest()` upserts (not adds) so re-running ingestion on the same match is idempotent. `_metadata_to_chroma_dict()` strips `None` values because Chroma metadata must be flat str/int/float/bool.

5. **`generation/prompts.py` + `generation/generator.py`** (§6.10, §6.11.5) — `NaiveRAGGenerator` is the **baseline** system: dense-only retrieval, no metadata filtering, no reasoning/evaluator stage, single Gemini call. It exists to be benchmarked against, not as the final architecture. Prompts use the exact structured template from §6.10 (system role constrains the model to only use provided evidence; user turn separates query / numbered evidence chunks / task instructions) so outputs stay traceable to `chunk_id`s. Default model is `gemini-flash-latest`, read via `agents._llm.get_client()` (the same shared Gemini client helper the agentic pipeline uses; picks up `GEMINI_API_KEY` from the environment via `load_dotenv()`).

`schema.py` defines the two data models everything else depends on: `Event` (mirrors the proposal's Table 4 event schema: match/team/player IDs, minute/second/period, x/y pitch coordinates, event_type, possession_id) and `Chunk`/`ChunkMetadata` (the retrievable unit — text + metadata link back to source event(s), with a `chunk_level` field already anticipating the not-yet-built time_window/sequence tiers).

6. **`agents/`** (§6.2.3, §6.3) — the 6-agent Agentic RAG pipeline, built on top of the same ingestion/chunking/embedding/vector-store modules above (steps 1-4) rather than duplicating them:
   - `query_understanding.py` — `QueryUnderstandingAgent` (`gemini-flash-lite-latest`, structured-output extraction) turns the raw query into a `StructuredQuery` (intent, teams, players, event_types, minute range, period); `to_where_filter()` maps it to a Chroma `where` filter over the same `ChunkMetadata` fields the baseline uses.
   - `retriever.py` — `RetrieverAgent` does hybrid retrieval (§6.8): dense search via the existing `ChunkVectorStore` fused with keyword search via `retrieval/bm25_index.py` (`BM25Index`, using the `rank-bm25` dependency), combined with Reciprocal Rank Fusion since cosine distance and BM25 scores aren't on comparable scales. `BM25Index` treats an empty candidate corpus as zero hits rather than crashing (`BM25Okapi` divides by corpus size internally). Both retrieval arms are filtered by the same metadata `where`, via `_matches_where()` on the BM25 side (re-implementing the equality/`$and`/`$gte`/`$lte` subset of Chroma's `where` syntax that `to_where_filter()` can produce) — except `team_name`/`player_name`, which `_matches_where()` matches on accent-/case-insensitive substring since LLM-extracted names (e.g. "Mbappe") rarely match the stored metadata verbatim (e.g. "Kylian Mbappé"). Chroma's own `where` filtering on the dense side has no equivalent fuzzy matching, so a name-filtered query can end up dense-empty/BM25-populated — the RRF fusion handles that fine since it doesn't require both arms to contribute. The fused pool is then optionally re-ranked by `retrieval/reranker.py`'s `Reranker` (§6.8's optional refinement step — a local `sentence-transformers` `CrossEncoder`, `cross-encoder/ms-marco-MiniLM-L-6-v2` by default, cached the same way `embeddings/embedder.py::get_default_embedder()` is), so salient events (goals, cards, shots) outrank generic ones (duels, ball receipts) within the top-N, at zero added Gemini quota cost. `enable_hybrid`/`enable_reranker` constructor flags exist purely so `scripts/run_ablation.py` can toggle each independently.
   - `context_aggregator.py` — `ContextAggregatorAgent` deduplicates near-identical chunk text (`difflib.SequenceMatcher`) from the fused hits. Rule-based, no LLM call, per §6.3.3 and the resource-efficiency goal in §6.13.
   - `reasoning.py` — `ReasoningAgent` (`gemini-flash-latest`, structured output) performs the multi-step analysis naive RAG skips entirely: produces a list of `ReasoningInsight` (pattern/trend text + supporting `chunk_id`s) from the aggregated context, *before* any prose is generated.
   - `generator.py` — `GeneratorAgent` (`gemini-flash-latest` — same model as `NaiveRAGGenerator`, for a fair benchmark comparison) reuses `generation/prompts.py`'s `SYSTEM_PROMPT` plus a new `build_agentic_user_prompt()` that adds an "Analytical Insights" section from the Reasoning agent. Appends the Evaluator's feedback as a revision instruction on retry.
   - `evaluator.py` — `EvaluatorAgent` (`gemini-flash-latest`, structured output, optional per §6.3.6) scores the response's relevance/coherence/factual-consistency against the aggregated evidence and returns pass/fail + feedback.
   - `graph.py` — `build_agentic_graph()`/`run_agentic_pipeline()` wire the above into a LangGraph `StateGraph` over `state.AgentState`: linear through Reasoning → Generator, then a conditional edge from Evaluator back to Generator (with feedback) for up to `max_retries` attempts before ending. `enable_reasoning=False` skips the Reasoning node entirely (`context_aggregator` edges straight to `generator`; `generator_node` reads `state.get("reasoning_insights", [])` rather than indexing directly, so it doesn't `KeyError` when reasoning was skipped). Together with `enable_hybrid`/`enable_reranker` (on `RetrieverAgent`) and the existing `enable_evaluator`, these four flags are exactly what `scripts/run_ablation.py` flips one at a time.
   - `_llm.py` — shared Gemini plumbing (`get_client()`, `call_structured()`) so the four LLM-backed agents above (plus `NaiveRAGGenerator`) don't each reimplement client setup and structured-output parsing. `call_structured()` passes a Pydantic model (or `list[Model]`) directly as `response_schema` and returns the already-validated `response.parsed` — no hand-written JSON Schema or manual parsing needed.

**`api/`** is a third entry point over the same agentic pipeline (no pipeline logic of its own) — a FastAPI service (`main.py`) so a UI can drive the pipeline over HTTP instead of importing it in-process. Reuses the exact same functions the CLI scripts do (`load_match_events`, `chunk_events`, `ChunkVectorStore`, `fetch_competitions`/`fetch_matches`, `run_agentic_pipeline`); `schemas.py` defines `PipelineRunRequest`/`PipelineRunResponse`, reusing `StructuredQuery`/`ReasoningInsight`/`EvaluationResult` from `agents/state.py` as field types rather than redefining them. Ingestion caching moves server-side as process-lifetime module state (`get_vector_store()`/`get_or_ingest_chunks()` — the FastAPI equivalent of what Streamlit's `st.cache_resource`/`st.cache_data` used to do in-process), so re-embedding only happens once per distinct `match_id` per server run, not per request. Routes: `GET /health`, `GET /competitions`, `GET /matches`, `POST /pipeline/run`. Run via `uv run python scripts/run_api.py`.

**`scripts/app.py`** is the Streamlit UI — a competition/season/match browser (finally exercising `ingestion/statsbomb.py`'s previously-unused `fetch_competitions()`/`fetch_matches()`, now via the API above) and a stage-by-stage view of all six agents' output. It has **no direct `football_rag` pipeline imports** — every piece of data comes from `scripts/_api_client.py`, a small `requests`-based HTTP client hitting the FastAPI service (base URL from `FOOTBALL_RAG_API_URL`, default `http://localhost:8000`). `scripts/app.py` still uses `st.cache_data` for the competitions/matches lookups (cheap, per-session), but ingestion caching itself now lives entirely in the API process, not in Streamlit.

### Data flow

Shared ingestion (both pipelines):

```
StatsBomb open-data JSON (HTTP, cached to data/statsbomb_cache/)
  -> parse_events()        -> list[Event]
  -> chunk_events()        -> list[Chunk]              (event-level only)
  -> ChunkVectorStore.ingest()  -> Chroma collection at data/chroma/
```

Naive baseline (`scripts/run_vertical_slice.py`):

```
  -> ChunkVectorStore.query()   -> top-N chunks (dense cosine similarity, optional `where` metadata filter)
  -> build_user_prompt()   -> structured prompt (evidence chunks numbered by chunk_id)
  -> NaiveRAGGenerator.generate() -> gemini-flash-latest call -> grounded summary text
```

Agentic pipeline (`scripts/run_agentic_pipeline.py`, `agents.graph.run_agentic_pipeline()`):

```
  -> QueryUnderstandingAgent -> StructuredQuery + where filter (gemini-flash-lite-latest)
  -> RetrieverAgent          -> dense (Chroma) + BM25 hits fused via Reciprocal Rank Fusion
  -> ContextAggregatorAgent  -> deduplicated evidence list (rule-based, no LLM)
  -> ReasoningAgent          -> list[ReasoningInsight] (gemini-flash-latest, structured output)
  -> GeneratorAgent          -> response_text (gemini-flash-latest, build_agentic_user_prompt)
  -> EvaluatorAgent          -> pass/fail + feedback (gemini-flash-latest); on fail, loops back to
                                 GeneratorAgent with feedback for up to max_retries attempts
```

`data/` (raw cache + Chroma persistence) and `.env` are gitignored — regenerate by re-running either script, not by committing them.

### Planned next phases (not yet implemented)

Per the proposal's methodology, still to build: time-window and possession-sequence chunking tiers (§6.6); cross-encoder/LLM re-ranking as an optional retrieval refinement step (§6.8); and the full evaluation harness (P@K/R@K, ROUGE/BLEU/BERTScore, human evaluation, ablation study, §6.11) for benchmarking the agentic pipeline against `NaiveRAGGenerator`.
