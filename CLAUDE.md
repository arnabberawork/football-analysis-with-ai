# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Implementation of the research proposal in `Research_Proposal_MSML_Arnab_Bera.pdf`: a lightweight **Agentic RAG framework** for automated football match summarization and tactical insight generation, built on structured StatsBomb event data rather than video/audio. Treat that PDF as the design spec — module names, prompt structure, and chunking strategy below map directly to its numbered sections (referenced inline as `§x.y`).

The codebase implements two parallel pipelines, both described in the Commands/Architecture sections below:

- The **naive RAG baseline vertical slice** (ingest → event-level chunk → embed → dense retrieve → single-pass generate), §6.11.5.
- The **6-agent Agentic RAG pipeline** (Query Understanding → Retriever → Context Aggregator → Reasoning → Generator → Evaluator), §6.2.3/§6.3, orchestrated with LangGraph and using hybrid (dense + BM25 + cross-encoder re-rank) retrieval per §6.8. A `StatsAgent` (`agents/stats.py`) sits alongside these six — a deliberate addition beyond the proposal's original architecture, routed to directly from Query Understanding for counting/aggregate questions ("how many yellow cards") that RAG retrieval structurally cannot answer reliably (retrieval samples a top-N, it never enumerates "all X"); see the `agents/` section below.

`NaiveRAGGenerator` is deliberately kept unmodified as the benchmark baseline the agentic pipeline is compared against (§6.11.5) — do not fold its logic into the agentic pipeline or vice versa. Both pipelines now retrieve from the **same hierarchical knowledge base** (event + time-window + possession-sequence + lineup/formation chunks) — the baseline/agentic comparison isolates retrieval-strategy and agent-architecture differences, not corpus differences.

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

Every script (`run_vertical_slice.py`, `run_agentic_pipeline.py`, `run_evaluation.py`, `run_ablation.py`) and every `POST /pipeline/run` API call saves its output to `outputs/<prefix>_<match_id>_<timestamp>.txt`, via `output_writer.py` (`src/football_rag/output_writer.py`) — useful for comparing runs, and the saved file has correctly-encoded accented characters even on a Windows console whose codepage mangles them on screen. `outputs/` is gitignored, same as `data/`. Two entry points into it: `tee_to_file()` (CLI scripts — redirects `sys.stdout` to both console and file for the duration of `main()`, not safe for concurrent use) and `write_output()` (the API — writes a complete string in one call, safe under concurrent requests since it never touches global state). The API's saved text and the CLI agentic script's console output are rendered by the same shared `agents/formatting.py::format_pipeline_result()`, so the two never drift apart; the API also returns the saved path as `output_file` in its JSON response.

All of the above require `GEMINI_API_KEY` in a `.env` file at the repo root (copy `.env.example`) — get a free key at https://aistudio.google.com/apikey. For the UI, the key only needs to be set where `scripts/run_api.py` runs (the backend), not where Streamlit runs. The agentic pipeline makes several Gemini calls per run (Query Understanding, Reasoning, Generator, and optionally Evaluator), all Flash-tier models chosen to stay comfortably inside the free-tier rate limits, but it still costs more per invocation than the single-call baseline. No test suite, linter, or formatter is configured yet.

## Architecture

Pipeline stages live under `src/football_rag/`, each corresponding to a proposal section:

1. **`ingestion/statsbomb.py`** (§6.4.1, §6.5) — fetches raw JSON from the public `statsbomb/open-data` GitHub mirror (no auth required), with optional disk caching (`cache_dir`). `parse_events()` normalizes raw StatsBomb event dicts into the `Event` pydantic model (`schema.py`), filtering out administrative event types with no tactical value (`LOW_SIGNAL_EVENT_TYPES`) and deriving a human-readable `detail`/`outcome` per event type in `_describe_event()`. For `Shot` events, `detail` includes `shot.type` (Penalty/Free Kick/Corner - "Open Play" is excluded since it's the vast majority and adds no signal) alongside `technique`/`body_part` - confirmed live that omitting `shot.type` silently dropped the fact that a shot was a penalty from the chunk text entirely, breaking keyword-based retrieval for exactly the queries that ask about penalties (see TESTING.md). Same class of fix applied to two more event types found to be silently losing sub-fields: `Duel.type` ("Tackle" vs "Aerial Lost") and `Block.deflection`/`.offensive` (rendered as "deflected"/"offensive" flags) now populate `detail` too. There is no StatsBomb "Card" event type — cards are the `outcome` of a `Foul Committed` or `Bad Behaviour` event (see `agents/query_understanding.py` below, which had this exact confusion until a live test caught it).

2. **`chunking/`** (§6.6) — four chunk-granularity tiers, all sharing the `Event -> Chunk` shape:
   - `event_chunker.py` — `chunk_events()`, one sentence per event via a fixed template (`event_to_sentence()`), so the knowledge base stays consistent rather than free LLM phrasing. `chunk_level="event"`.
   - `time_window_chunker.py` — `chunk_time_windows()`, groups events into fixed 5-minute windows per period and renders each window's events into one chunk. Gives the generator narrative substance a single event can't — e.g. a "20'-24'" chunk, not just an isolated Duel sentence. `chunk_level="time_window"`.
   - `sequence_chunker.py` — `chunk_sequences()`, groups events by the existing `Event.possession_id` into one chunk per possession (a team's uninterrupted spell in control). `chunk_level="sequence"`.
   - `lineup_chunker.py` — `chunk_lineups(match_id, raw_events)`, a fourth tier beyond the proposal's original three, added because "what formation did X play" / "who started" was structurally unanswerable otherwise: `Starting XI` events are filtered out of `Event` entirely by `LOW_SIGNAL_EVENT_TYPES` (their `tactics.formation`/`tactics.lineup` payload has no tactical-narrative sentence form), so this reads them straight from the pre-filter `raw_events` list instead. One chunk per team ("Argentina started in a 4-3-3 formation: ..." plus all 11 players with position and jersey number). `chunk_level="lineup"`.
   - The time-window and sequence tiers render their events via `event_chunker.render_events()` (not raw `event_to_sentence()` joins), which caps each chunk at `MAX_EVENTS_PER_CHUNK` (12) with a "(+N more events omitted for brevity)" note past that. A busy 5-minute window or a long possession chain can otherwise contain 60+ events and balloon a single chunk past 10,000 characters — confirmed live to blow up the evidence prompt (~31,000 chars/~7,750 tokens for just 10 retrieved chunks) enough that Gemini hit `finish_reason=MAX_TOKENS` with little or no visible output. `render_events()` bounds every multi-event chunk to a few hundred–~1,700 characters regardless of how eventful the window/sequence was.
   - `__init__.py` — `build_hierarchical_chunks(events, raw_events=None)` concatenates all tiers (`raw_events` is optional; lineup chunks are only produced when it's given); this is the one ingestion call site every script/API uses (chunk_id schemes `-ev-`/`-tw-`/`-seq-`/`-lineup-` are disjoint, so `ChunkVectorStore.ingest()`'s upsert just grows one collection additively).

3. **`embeddings/embedder.py`** (§6.7) — wraps a local `sentence-transformers` model (default `all-MiniLM-L6-v2`, chosen for zero API cost per §6.13's resource constraints). `get_default_embedder()` caches one model instance process-wide; swap models by changing `DEFAULT_MODEL_NAME`.

4. **`retrieval/vector_store.py`** (§6.7, §6.8) — `ChunkVectorStore` wraps a `PersistentClient` ChromaDB collection, chosen over raw FAISS specifically because Chroma supports metadata filtering *during* retrieval (`query(..., where={...})`), which the proposal's metadata-aware retrieval strategy depends on. Embeddings are always computed by our own `Embedder`, never Chroma's built-in embedding function, so the model choice stays explicit. `ingest()` upserts (not adds) so re-running ingestion on the same match is idempotent. `_metadata_to_chroma_dict()` strips `None` values because Chroma metadata must be flat str/int/float/bool.

5. **`generation/prompts.py` + `generation/generator.py`** (§6.10, §6.11.5) — `NaiveRAGGenerator` is the **baseline** system: dense-only retrieval, no metadata filtering, no reasoning/evaluator stage, single Gemini call. It exists to be benchmarked against, not as the final architecture. Prompts use the exact structured template from §6.10 (system role constrains the model to only use provided evidence; user turn separates query / numbered evidence chunks / task instructions) so outputs stay traceable to `chunk_id`s. Default model is `gemini-flash-latest`, read via `agents._llm.get_client()` (the same shared Gemini client helper the agentic pipeline uses; picks up `GEMINI_API_KEY` from the environment via `load_dotenv()`).

`schema.py` defines the two data models everything else depends on: `Event` (mirrors the proposal's Table 4 event schema: match/team/player IDs, minute/second/period, x/y pitch coordinates, event_type, possession_id) and `Chunk`/`ChunkMetadata` (the retrievable unit — text + metadata link back to source event(s), with a `chunk_level` field distinguishing the three chunking tiers described in step 2 below).

6. **`agents/`** (§6.2.3, §6.3) — the 6-agent Agentic RAG pipeline plus `StatsAgent` (see below), built on top of the same ingestion/chunking/embedding/vector-store modules above (steps 1-4) rather than duplicating them:
   - `query_understanding.py` — `QueryUnderstandingAgent` (`gemini-flash-lite-latest`, structured-output extraction) turns the raw query into a `StructuredQuery` (intent, teams, players, event_types, outcome, minute range, period); `to_where_filter()` maps it to a Chroma `where` filter over the same `ChunkMetadata` fields the baseline uses. `intent` includes a `"statistic"` value for counting/aggregate questions ("how many yellow cards", "how many shots did Messi take") — these route to `StatsAgent` instead of retrieval (see `graph.py` below), since retrieval samples a top-N and can never reliably enumerate "all X". `outcome` captures a specific named result ("Goal", "Yellow Card", "Saved") for `StatsAgent` to filter on. The `SYSTEM_PROMPT` enumerates the actual StatsBomb `event_type` vocabulary explicitly and calls out that there is no "Card" event type (cards are the `outcome` of `Foul Committed`/`Bad Behaviour`) — added after a live test showed the model extracting a plausible-but-nonexistent `event_types=["Card"]` for a "how many yellow cards" query, silently returning zero matches for a match that has 8 real yellow cards.
   - `stats.py` — `StatsAgent`, **a deliberate addition beyond the proposal's original 6-agent design**. No LLM call: filters the full `list[Event]` (not a retrieved/sampled chunk set — the only way to get an exact, un-undercounted answer) by whatever `StructuredQuery` fields are set, groups matches by team/player, and returns a deterministic formatted count plus a capped list of matching events (minute, team, player, detail/outcome) so every number is traceable back to real events and can never be altered by generation. Zero added Gemini quota cost.
   - `retriever.py` — `RetrieverAgent` does hybrid retrieval (§6.8): dense search via the existing `ChunkVectorStore` fused with keyword search via `retrieval/bm25_index.py` (`BM25Index`, using the `rank-bm25` dependency), combined with Reciprocal Rank Fusion since cosine distance and BM25 scores aren't on comparable scales. `BM25Index` treats an empty candidate corpus as zero hits rather than crashing (`BM25Okapi` divides by corpus size internally), and `_tokenize()` stems every token (`nltk`'s `PorterStemmer`) before matching — confirmed live that without stemming, a plural query ("penalties") completely missed chunks containing the exact singular keyword ("Penalty"), since BM25 does exact-token matching with no morphological awareness on its own. Both retrieval arms are filtered by the same metadata `where`, via `_matches_where()` on the BM25 side (re-implementing the equality/`$and`/`$gte`/`$lte` subset of Chroma's `where` syntax that `to_where_filter()` can produce) — except `team_name`/`player_name`, which `_matches_where()` matches on accent-/case-insensitive substring since LLM-extracted names (e.g. "Mbappe") rarely match the stored metadata verbatim (e.g. "Kylian Mbappé"). Chroma's own `where` filtering on the dense side has no equivalent fuzzy matching, so a name-filtered query can end up dense-empty/BM25-populated — the RRF fusion handles that fine since it doesn't require both arms to contribute. The fused pool is then optionally re-ranked by `retrieval/reranker.py`'s `Reranker` (§6.8's optional refinement step — a local `sentence-transformers` `CrossEncoder`, `cross-encoder/ms-marco-MiniLM-L-6-v2` by default, cached the same way `embeddings/embedder.py::get_default_embedder()` is), so salient events (goals, cards, shots) outrank generic ones (duels, ball receipts) within the top-N, at zero added Gemini quota cost. `enable_hybrid`/`enable_reranker` constructor flags exist purely so `scripts/run_ablation.py` can toggle each independently. **Known limitation, confirmed live** (see TESTING.md): this general-purpose MS-MARCO-trained cross-encoder doesn't reliably recognize football-domain relevance — for a "penalties awarded and scored" query it demoted correctly-surfaced penalty-shot chunks below generic "Foul Won" chunks, even though the pre-rerank RRF-fused pool already ranked ~half its top 10 as the right penalty chunks. Not fixed in this pass; `enable_reranker=False` is the workaround if a specific query set is measurably hurt by it. `retrieve()` also takes `ensure_salient: bool` — when true, chunks for Goals/Own Goals/Red Cards/Yellow Cards/Substitutions are force-included (up to half of `top_n`) regardless of similarity/BM25/rerank score, with the remaining slots filled normally. Wired from `graph.py`'s `retriever_node` exactly when `intent=="match_summary"` and no metadata filter is set — i.e. a genuinely broad, unscoped "summarize the match" request. Confirmed live this is necessary, not optional: without it, the top 8 hits for the default broad query were all generic Duel/Pass chunks with zero goals; with it, real match goals (e.g. Messi's 22' penalty, Mbappé's 79'/80' goals) appear in the top 4. Forced hits carry no `rerank_score`/`fusion_score` key at all (they bypass ranking by design) — `formatting.py::format_pipeline_result()` and `scripts/app.py`'s retriever table both display `"forced"` in place of a score for these, having crashed the first time this path was exercised live.
   - `context_aggregator.py` — `ContextAggregatorAgent` deduplicates near-identical chunk text (`difflib.SequenceMatcher`) from the fused hits. Rule-based, no LLM call, per §6.3.3 and the resource-efficiency goal in §6.13.
   - `reasoning.py` — `ReasoningAgent` (`gemini-flash-latest`, structured output) performs the multi-step analysis naive RAG skips entirely: produces a list of `ReasoningInsight` (pattern/trend text + supporting `chunk_id`s) from the aggregated context, *before* any prose is generated. `reason()` normalizes `supporting_chunk_ids` back to `[]` if Gemini returns an explicit `null` for it — confirmed live that structured output treats a field with a Python default (`default_factory=list`) as nullable and can return `null` there, which bypasses the default via the SDK's own parsing path and crashed `build_agentic_user_prompt()`'s `', '.join(...)` downstream.
   - `generator.py` — `GeneratorAgent` (`gemini-flash-latest` — same model as `NaiveRAGGenerator`, for a fair benchmark comparison) reuses `generation/prompts.py`'s `SYSTEM_PROMPT` plus a new `build_agentic_user_prompt()` that adds an "Analytical Insights" section from the Reasoning agent. Appends the Evaluator's feedback as a revision instruction on retry. `max_output_tokens=4096` (same in `NaiveRAGGenerator`) — bumped from 2048 after live testing showed a large evidence prompt could exhaust that budget on internal reasoning tokens before any visible answer, returning an empty `response_text` with `finish_reason=MAX_TOKENS`; both call sites also now do `response.text or ""` defensively.
   - `evaluator.py` — `EvaluatorAgent` (`gemini-flash-latest`, structured output, optional per §6.3.6) scores the response's relevance/coherence/factual-consistency against the aggregated evidence and returns pass/fail + feedback.
   - `graph.py` — `build_agentic_graph()`/`run_agentic_pipeline()` wire the above into a LangGraph `StateGraph` over `state.AgentState` (which now also carries `events: list[Event]` — the full, un-sampled event list `StatsAgent` needs). A conditional edge after `query_understanding` routes `intent=="statistic"` straight to `stats_node` → END, bypassing retriever/aggregator/reasoning/generator/evaluator entirely — both a correctness fix (exact counts, not retrieval samples) and a quota-cost win (one Gemini call total instead of up to four). All other intents flow through the original linear path: Retriever → Context Aggregator → Reasoning → Generator, then a conditional edge from Evaluator back to Generator (with feedback) for up to `max_retries` attempts before ending. `enable_reasoning=False` skips the Reasoning node entirely (`context_aggregator` edges straight to `generator`; `generator_node` reads `state.get("reasoning_insights", [])` rather than indexing directly, so it doesn't `KeyError` when reasoning was skipped). Together with `enable_hybrid`/`enable_reranker` (on `RetrieverAgent`) and the existing `enable_evaluator`, these four flags are exactly what `scripts/run_ablation.py` flips one at a time.
   - `_llm.py` — shared Gemini plumbing (`get_client()`, `call_structured()`) so the four LLM-backed agents above (plus `NaiveRAGGenerator`) don't each reimplement client setup and structured-output parsing. `call_structured()` passes a Pydantic model (or `list[Model]`) directly as `response_schema` and returns the already-validated `response.parsed` — no hand-written JSON Schema or manual parsing needed.

**`api/`** is a third entry point over the same agentic pipeline (no pipeline logic of its own) — a FastAPI service (`main.py`) so a UI can drive the pipeline over HTTP instead of importing it in-process. Reuses the exact same functions the CLI scripts do (`load_match_events`, `build_hierarchical_chunks`, `ChunkVectorStore`, `fetch_competitions`/`fetch_matches`, `run_agentic_pipeline`); `schemas.py` defines `PipelineRunRequest`/`PipelineRunResponse`, reusing `StructuredQuery`/`ReasoningInsight`/`EvaluationResult` from `agents/state.py` as field types rather than redefining them. Ingestion caching moves server-side as process-lifetime module state (`get_vector_store()`/`get_or_ingest_chunks()` — the FastAPI equivalent of what Streamlit's `st.cache_resource`/`st.cache_data` used to do in-process), so re-embedding only happens once per distinct `match_id` per server run, not per request. Routes: `GET /health`, `GET /competitions`, `GET /matches`, `POST /pipeline/run` (also saves the run's output to `outputs/` on every call and returns the path as `output_file`, same as the CLI scripts). Run via `uv run python scripts/run_api.py`.

**`scripts/app.py`** is the Streamlit UI — a competition/season/match browser (finally exercising `ingestion/statsbomb.py`'s previously-unused `fetch_competitions()`/`fetch_matches()`, now via the API above) and a stage-by-stage view of all six agents' output. It has **no direct `football_rag` pipeline imports** — every piece of data comes from `scripts/_api_client.py`, a small `requests`-based HTTP client hitting the FastAPI service (base URL from `FOOTBALL_RAG_API_URL`, default `http://localhost:8000`). `scripts/app.py` still uses `st.cache_data` for the competitions/matches lookups (cheap, per-session), but ingestion caching itself now lives entirely in the API process, not in Streamlit.

**`evaluation/`** (§6.11) is the evaluation harness, run via `scripts/run_evaluation.py` (naive-vs-agentic baseline comparison) and `scripts/run_ablation.py` (§6.11.6 ablation study):
   - `retrieval_metrics.py` — `precision_at_k()`/`recall_at_k()`, plain functions over `(retrieved_ids, relevant_ids, k)`.
   - `generation_metrics.py` — `rouge_scores()`/`bleu_score()` always available (lightweight); `bertscore()` lazy-imports `bert_score` since its model is a ~1GB+ download that shouldn't be paid on every harness run per §6.13's resource-efficiency goal.
   - `dataset.py` — **an illustrative starter dataset, not a rigorous benchmark** (see its module docstring). Real SoccerNet integration and human-written match reports (§6.4.2, §7.1) aren't pulled in — SoccerNet needs separate dataset access this project doesn't have, and scraping published reports raises copyright concerns. Instead: `STARTER_DATASET`, ~5 hand-written queries against the WC Final with `RelevanceCriterion`s matched against chunk **text** (robust across all three chunking tiers, unlike hardcoded chunk_ids) via `resolve_relevant_chunk_ids()`, and originally-written reference summaries (from the public factual record, not copied from any article).

**`output_writer.py`** and **`agents/formatting.py`** back the output-saving described above. `format_pipeline_result()` renders a `run_agentic_pipeline()` result dict into the same numbered-stage text both `scripts/run_agentic_pipeline.py` prints and the API saves — one source of truth for the format. `output_writer.py` has two entry points depending on the caller's concurrency needs: `tee_to_file()` (stdout-redirect, CLI-only) and `write_output()` (plain write, safe for the API's concurrent requests).

### Data flow

Shared ingestion (both pipelines):

```
StatsBomb open-data JSON (HTTP, cached to data/statsbomb_cache/)
  -> fetch_raw_events() -> raw_events; parse_events(raw_events) -> list[Event]
  -> build_hierarchical_chunks(events, raw_events)  -> list[Chunk]  (event + time-window + sequence + lineup tiers)
  -> ChunkVectorStore.ingest()    -> Chroma collection at data/chroma/
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
     |
     +-- intent == "statistic" --> StatsAgent -> response_text (no LLM, exact count over all events) -> END
     |
     +-- otherwise --> RetrieverAgent -> dense (Chroma) + BM25 hits fused via Reciprocal Rank Fusion,
                                          then re-ranked by a local cross-encoder (Reranker)
                     -> ContextAggregatorAgent  -> deduplicated evidence list (rule-based, no LLM)
                     -> ReasoningAgent          -> list[ReasoningInsight] (gemini-flash-latest, structured output)
                     -> GeneratorAgent          -> response_text (gemini-flash-latest, build_agentic_user_prompt)
                     -> EvaluatorAgent          -> pass/fail + feedback (gemini-flash-latest); on fail, loops back to
                                                   GeneratorAgent with feedback for up to max_retries attempts
```

`data/` (raw cache + Chroma persistence) and `.env` are gitignored — regenerate by re-running either script, not by committing them.

### Planned next phases (not yet implemented)

Hierarchical chunking, re-ranking, and the evaluation harness (P@K/R@K, ROUGE/BLEU/BERTScore, baseline comparison, ablation study — §6.11) are now built (`chunking/`, `retrieval/reranker.py`, `evaluation/`, `scripts/run_evaluation.py`, `scripts/run_ablation.py`). Still to build, per the proposal's methodology: real SoccerNet integration and human-written match report ingestion (§6.4.2, §7.1) — `evaluation/dataset.py`'s `STARTER_DATASET` is an illustrative stand-in, not this; human evaluation (§6.11.4); and Phase 6 "Refinement" (§8.1), which depends on having real numbers from the harness above to act on.
