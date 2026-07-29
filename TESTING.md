# Testing guide: Gemini-backed RAG pipelines

Manual test plan for the four runnable entry points in this repo, all
Gemini-backed: the naive baseline (`scripts/run_vertical_slice.py`), the
6-agent pipeline CLI (`scripts/run_agentic_pipeline.py`), a FastAPI service
over the same agentic pipeline (`scripts/run_api.py`,
`src/football_rag/api/main.py`), and a Streamlit UI (`scripts/app.py`) that
talks to that API over HTTP rather than importing pipeline code itself -
plus the evaluation harness (`scripts/run_evaluation.py`,
`scripts/run_ablation.py`). An automated unit test suite (`tests/`, see §1)
now covers the pure-logic modules (schema, chunking helpers, StatsAgent,
BM25, retrieval/generation metrics, context dedup) with no network or
Gemini calls - but the four Gemini-backed entry points and the
Chroma/embedding-model-backed retrieval path have no automated coverage,
so the rest of this document is still a run-it-and-read-the-output
checklist for those.

## 0. Prerequisites

- [ ] `GEMINI_API_KEY` is set in `.env` at the repo root (a free key:
      https://aistudio.google.com/apikey). `.env` is gitignored — never put
      a real key in `.env.example`, that file is committed to git.
- [ ] `uv sync` runs clean.
- [ ] Internet access (StatsBomb open-data fetch on first run, Gemini API
      calls, and a one-time `sentence-transformers` model download).

```bash
uv sync
```

## 1. Offline sanity check (no API calls, no network)

Confirms every module still imports and the LangGraph pipeline still wires
up correctly, without spending any API quota. Good first step if anything
below fails, to isolate "broken code" from "broken API call".

```bash
uv run python -c "
from football_rag.agents.graph import build_agentic_graph
from football_rag.generation.generator import NaiveRAGGenerator
print('imports OK')
"
```

Expect: `imports OK`, no traceback.

**Automated unit tests** — the same "no network, no Gemini quota" property,
but as a real `unittest` suite under `tests/` instead of a one-off snippet:

```bash
uv run python -m unittest discover -s tests
```

Expect: `OK` and a passing count (57 tests as of this writing; a few
seconds once `rouge_score`/`sacrebleu`/`sentence-transformers` are already
import-cached, up to ~60-120s cold). Covers, by file:

| File | What it covers |
|---|---|
| `test_imports.py` | module import smoke test + `HF_API_KEY` → `HF_TOKEN`/`HF_API_TOKEN` env aliasing |
| `test_schema.py` | `Event`/`Chunk`/`ChunkMetadata` field defaults and required-field validation |
| `test_retrieval_metrics.py` | `precision_at_k`/`recall_at_k` edge cases (empty lists, `k` larger/smaller than input, no relevant ids) |
| `test_generation_metrics.py` | `rouge_scores`/`bleu_score` sanity (identical text scores high, unrelated text scores low) - `bertscore()` deliberately not covered here since it lazy-downloads a ~1GB model |
| `test_context_aggregator.py` | near-duplicate chunk dedup, including the `similarity_threshold` boundary |
| `test_stats_agent.py` | `StatsAgent` counting/filtering logic, including the penalty-shootout (period 5) exclusion for goal-counting queries and its "count stays exact, detail list is capped" behavior |
| `test_bm25_index.py` | empty-corpus handling (no `ZeroDivisionError`), stemming (plural query matches singular chunk text), `top_n` truncation |
| `test_retriever_helpers.py` | `_salience_priority` (goal/card/sub tiers, penalty-shootout period 5 excluded), `_matches_where` (equality/`$and`/`$gte`/`$lte`, accent-/case-insensitive fuzzy name matching), `_normalize`, and `RetrieverAgent._force_salient` (every real goal forced in uncapped, cards/subs stay budget-limited, no duplicates against the already-ranked pool) - built via a `vector_store=None, enable_reranker=False` `RetrieverAgent` instance so no model download is needed to exercise this pure logic |

Not yet covered by this suite (still manual, via the sections below):
`ChunkVectorStore`/`Embedder`/`Reranker` (each loads a real model or opens a
real Chroma collection), the chunking tier builders themselves
(`chunk_events`/`chunk_time_windows`/`chunk_sequences`/`chunk_lineups`),
`RetrieverAgent`'s hybrid dense+BM25 fusion and reranking end-to-end,
`query_understanding.py`'s `to_where_filter`, `output_writer.write_output`,
and every Gemini-backed agent (`QueryUnderstandingAgent`, `ReasoningAgent`,
`GeneratorAgent`, `EvaluatorAgent`, `NaiveRAGGenerator`) plus the
FastAPI/Streamlit layers.

## 2. Naive baseline (single Gemini call)

Cheapest real test — one retrieval + one `gemini-flash-latest` call.

```bash
uv run python scripts/run_vertical_slice.py
```

First run will be slow (~1-2 min): downloads the StatsBomb match JSON,
caches it to `data/statsbomb_cache/`, downloads the `all-MiniLM-L6-v2`
embedding model, embeds ~3500+ chunks (event + time-window + sequence
tiers, via `build_hierarchical_chunks()`) into `data/chroma/`. Re-running
afterwards should be much faster (cache hits + idempotent Chroma upsert).

**Expect:**
- `N events parsed`, `N chunks generated (event + time-window + sequence
  tiers)`, `vector store now holds N chunks` (chunk count should match
  after any re-run).
- A "Retrieved evidence" list of `[chunk_id] (distance) text` lines.
- A "Generated response" section: a coherent ~150-200 word match summary
  mentioning specific minutes/players, grounded in the retrieved evidence.

**Fail signs:** `RuntimeError: GEMINI_API_KEY is not set` (check `.env`),
a `429`/quota error from the Gemini call (see Troubleshooting), or an empty/
garbled response.

Also check `outputs/run_vertical_slice_<match_id>_<timestamp>.txt` — every
run saves its full printed output there too (not just the console), via
`football_rag.output_writer`. Useful for diffing two runs, and the saved
file has correctly-encoded accented characters even where the console
itself shows mojibake. Every script and every API call gets this treatment
now (see §11) — this isn't special-cased to the baseline.

## 3. Full 6-agent pipeline

More expensive — up to 4 Gemini calls per run (Query Understanding,
Reasoning, Generator, Evaluator). Run this after step 2 passes, so you're
only isolating agent-specific issues.

```bash
uv run python scripts/run_agentic_pipeline.py
```

Walk through each numbered section in the output and check it's sane:

1. **Query Understanding** — `StructuredQuery(intent=..., teams=[], ...)`
   printed with a real `where filter` (likely `None` for the default broad
   "summarize the match" query, since it names no team/player/event type).
2. **Retriever** — a list of `[chunk_id] (score) text` lines (re-ranked by
   the cross-encoder by default, so the score printed is `rerank_score`,
   falling back to `fusion_score` only if re-ranking is disabled), ranked
   highest-first. `chunk_id` prefixes tell you which tier each hit came
   from (`-ev-` event, `-tw-` time-window, `-seq-` sequence).
3. **Context Aggregator** — a chunk count that's `<=` the retriever's count
   (it only drops near-duplicates, so equal is fine too).
4. **Reasoning insights** — several `- <tactical pattern> (evidence:
   chunk_id, ...)` lines. Each should read as actual analysis (e.g.
   "France dominated second-half possession"), not a restatement of a single
   event.
5. **Generated response** — prose summary, similar shape to the baseline's
   but should read as informed by the insights from step 4.
6. **Evaluator verdict** — `passed=True/False` with three 0-1 scores and a
   `feedback` string. If `passed=False`, re-run and check the response
   changed (confirms the retry-with-feedback loop actually fired — see §5).

## 4. Exercise metadata filtering (entity + timeframe extraction)

Confirmed working live: `gemini-flash-lite-latest` cleanly fills
`StructuredQuery`'s optional `minute_min`/`minute_max`/`period` fields, and
the resulting `where` filter narrows retrieval correctly. Test with a query
that names an entity and a timeframe:

```bash
uv run python scripts/run_agentic_pipeline.py 3869685 "What did Mbappe do in the second half?"
```

**Expect:** `StructuredQuery(intent='player_focus', players=['Mbappe'],
minute_min=46, minute_max=90, period=2, ...)`, a non-`None` `$and` where
filter, and every retrieved chunk mentioning Mbappé specifically (note the
metadata stores the accented full name "Kylian Mbappé Lottin" — matching
still works because `agents/retriever.py::_matches_where` does
accent-/case-insensitive substring matching on `team_name`/`player_name`
specifically for this reason). If `where filter` stays `None` or retrieval
pulls in unrelated players, the entity extraction or the `to_where_filter`
mapping needs a look (`src/football_rag/agents/query_understanding.py`).

Also worth noting from testing: broad, unspecific queries (e.g. the default
"Summarize the match and provide key tactical insights") can still retrieve
low-signal events instead of goals/shots even with hierarchical chunking
and re-ranking (§9 below has the detailed finding and why) — both the
naive baseline and the agentic pipeline may correctly respond "Not enough
information" rather than hallucinate a summary in that case. That's the
grounding constraint in `generation/prompts.py`'s `TASK_INSTRUCTIONS`
working as intended, not a bug — entity/timeframe/topic-specific queries
like the one above retrieve much more useful evidence, as confirmed live.

## 5. Exercise the Evaluator retry loop

Hard to force deterministically (it depends on the LLM-judge's own
verdict), but you can raise the odds and confirm the mechanism works:

```bash
uv run python -c "
from pathlib import Path
from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.retrieval.vector_store import ChunkVectorStore

events = load_match_events(3869685, cache_dir=Path('data/statsbomb_cache'))
chunks = build_hierarchical_chunks(events)
store = ChunkVectorStore(persist_dir=Path('data/chroma'))
store.ingest(chunks)

result = run_agentic_pipeline('3869685', 'Summarize the match.', store, chunks, max_retries=2)
print('retry_count:', result['retry_count'])
print('final passed:', result['evaluation'].passed)
"
```

**Expect:** `retry_count` between 1 and 3. If the first evaluation fails,
`retry_count` should be `> 1` and the final response should differ from a
plain single-shot run (confirms `feedback` was actually appended to the
regeneration prompt in `agents/generator.py`).

## 6. Confirm re-ingestion is idempotent

Run either script twice in a row for the same `match_id` and check the
vector store count printed by the script doesn't grow the second time
(`ChunkVectorStore.ingest()` upserts by `chunk_id`, so re-running should not
create duplicates).

## 7. FastAPI backend

The API is what actually runs ingestion + the agentic pipeline now (§3's
`run_agentic_pipeline()` call, wrapped in HTTP). Test it standalone before
touching the UI, so a UI bug and an API bug never get confused:

```bash
uv run python scripts/run_api.py
```

Starts on http://localhost:8000. No `--reload` — uvicorn's file-watcher
reload proved unreliable in at least one sandboxed test environment
(silently kept serving stale code after an edit instead of restarting), so
if you change `src/football_rag/api/main.py` you need to stop (Ctrl+C) and
re-run `scripts/run_api.py` yourself to pick it up. In another terminal:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/competitions | head -c 300
# a JSON array of competition/season objects

curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"match_id": 3869685, "query": "Summarize the match and provide key tactical insights."}'
# a full PipelineRunResponse JSON body: structured_query, where_filter,
# retrieved_chunks, aggregated_context, reasoning_insights, response_text,
# evaluation, retry_count
```

First `/pipeline/run` call for a given `match_id` is slow (~1 min —
ingestion + embedding, same cost as §2/§3's first run); repeat the same
`curl` command and it should return in seconds (server-side
`get_or_ingest_chunks()` cache hit, per `src/football_rag/api/main.py`).

**Fail signs:** connection refused (server isn't running, or crashed on
startup — check the terminal for a traceback); `500` with a `GEMINI_API_KEY
is not set` detail (the key needs to be set where `run_api.py` runs, not
where Streamlit runs — they can be different terminals/environments); `429`
with a `RESOURCE_EXHAUSTED`/quota-exceeded detail (see Troubleshooting —
this is a clean, expected error response now, not a bug).

**Streaming variant** (`POST /pipeline/run/stream`) — same pipeline, but
returns newline-delimited JSON, one line per agent stage as it completes,
so a client can show live progress instead of waiting for one big response
(this is what `scripts/app.py`'s live status display in §8 actually calls):

```bash
curl -N -X POST http://localhost:8000/pipeline/run/stream \
  -H "Content-Type: application/json" \
  -d '{"match_id": 3869685, "query": "How many yellow cards were shown in the match?"}'
```

**Expect** (`-N` disables curl's output buffering so lines print as they
arrive, not all at once at the end) one `{"event": "stage", "stage": ...,
"label": ..., "detail": ...}` line per completed agent — for this
statistic-intent query specifically: `ingest` (only on the first call for
this match_id), `query_understanding` (`detail` should read `intent:
statistic`), then `stats_node` — followed by exactly one final
`{"event": "done", "result": {...}}` line whose `result` has the same shape
as `/pipeline/run`'s response body. Try the default broad
"Summarize the match and provide key tactical insights." query too to see
all six stages (`query_understanding`, `retriever`, `context_aggregator`,
`reasoning`, `generator`, `evaluator`) stream through in order — and note
that a `generator`/`evaluator` pair can repeat if the Evaluator's first
verdict fails and a retry fires (§5).

**Fail signs:** a line with `{"event": "error", ...}` followed by the
stream ending — this is the pipeline's own error path (Gemini quota, bad
API key, etc.), not a transport bug, and its `detail` is the same message
`/pipeline/run` would have returned as an HTTP error body; the only
difference is it can't be signalled via HTTP status once streaming has
already started (headers are already sent), so `scripts/_api_client.py::stream_pipeline()`
raises `RuntimeError(detail)` on this event instead.

## 8. Streamlit UI

Same agentic pipeline as §3-§5, now consumed purely over HTTP from the
API in §7 — `scripts/app.py` has no direct `football_rag` pipeline
imports, only `scripts/_api_client.py`. **Start the API first** (§7), then:

```bash
uv run streamlit run scripts/app.py
```

Opens at http://localhost:8501. Confirmed working via an actual browser
session (not just "the server started"):

1. Sidebar's Competition/Season/Match dropdowns should populate and default
   to FIFA World Cup / 2022 / "2022-12-18: Argentina 3-3 France" (the Final,
   match_id 3869685) — confirms `_api_client.get_competitions()`/
   `get_matches()` are reaching the API correctly.
2. Click **Run pipeline** with the default query. **Expect** an
   `st.status()` block that stays open and updates live, one line per
   agent, as each one finishes (via `/pipeline/run/stream`, §7's streaming
   variant) — `:material/check_circle: **Query Understanding** — intent:
   match_summary`, then `Retriever`, `Context Aggregator`, `Reasoning`,
   `Generator`, `Evaluator` in order, ending with the status block
   collapsing to "Pipeline complete". First click also shows an
   `Ingesting match data` line before any agent runs (~1 min server-side
   ingest+embed); confirms someone unfamiliar with the pipeline's internals
   can watch it work stage by stage instead of staring at one opaque
   spinner for up to a minute. After the status block finishes, **expect**
   two tabs: **Response** (selected by default — just the generated
   summary, plus a pass/fail badge and relevance/coherence/factual-
   consistency metrics if the Evaluator ran) and **Audit trail** (the
   structured query JSON, the retriever's scored hit table, the aggregator's
   dedup count, and the reasoning insights — everything the old single-page
   view showed at once, now one tab over). For a counting query (e.g. "How
   many yellow cards were shown?"), **expect** the live status to show only
   `Query Understanding` → `StatsAgent` (no Retriever/Aggregator/Reasoning/
   Generator/Evaluator lines), and the Audit trail tab to show a caption
   explaining the StatsAgent bypass instead of an empty/misleading retriever
   table.
3. Change only the query text (same match) and click **Run pipeline** again.
   **Expect** it to skip straight to the pipeline run with no multi-second
   ingestion delay — confirms the *API's* server-side ingestion cache is
   working (re-embedding only happens once per distinct `match_id` per API
   process, not per Streamlit session anymore). Verified pre-API-split: a
   query-only rerun completed in ~35s vs. the ~1 min+ first run; re-verify
   this still holds now that the cache lives in `run_api.py`'s process
   instead of Streamlit's.
4. Switch to a different match in the dropdowns and run again. **Expect** a
   fresh ingestion delay this time (new `match_id`, cache miss on the API
   side), then normal timing after that.
5. Try the Mbappé-style query from §4 in the UI's query box. **Expect** the
   same entity/timeframe extraction and correctly-filtered retrieval as the
   CLI test (with properly-rendered accented characters in the browser,
   unlike the mojibake seen in a Windows terminal — see Troubleshooting).
6. Stop the API process (Ctrl+C on `run_api.py`) and click **Run pipeline**
   again without restarting it. **Expect** the live status block to switch
   to its error state (red) and a friendly `st.error` naming the API URL it
   couldn't reach, not an unhandled exception/blank page — confirms
   `run_pipeline_with_live_status()`'s `except (RuntimeError, ConnectionError)`
   around the `stream_pipeline()` loop.

**Fail signs:** blank page after load (check the terminal running
`streamlit run` for a traceback); dropdowns stuck showing a loading
skeleton or an in-page error naming the API URL (API isn't running, or
crashed — check §7 first).

## 9. Hierarchical chunking + re-ranking (quota-free)

Both new chunking tiers and the cross-encoder re-ranker can be verified
without spending any Gemini quota, since none of this touches the LLM:

```bash
uv run python -c "
from pathlib import Path
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.chunking import build_hierarchical_chunks
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.agents.retriever import RetrieverAgent

events = load_match_events(3869685, cache_dir=Path('data/statsbomb_cache'))
chunks = build_hierarchical_chunks(events)
print(len(chunks), 'total chunks (was 1842 event-only before this feature)')

store = ChunkVectorStore(persist_dir=Path('data/chroma'))
store.ingest(chunks)

retriever = RetrieverAgent(store, chunks)
for hit in retriever.retrieve('What goals were scored in the match?', top_n=5):
    print(hit['metadata']['chunk_level'], round(hit['rerank_score'], 3), hit['text'][:90])
"
```

**Expect:** ~3480 total chunks (1842 event + ~200 time-window + ~1440
sequence, roughly — exact counts vary only with StatsBomb data, not code).
For a goal-related query, the top hit should be one of Messi's/Mbappe's
shot chunks with a `rerank_score` clearly higher (less negative) than the
rest — confirmed live: `0.882` for Messi's decisive shot vs. `-3.7` for the
next-best candidate, a large, confident gap.

**Important honest caveat, found during this exact test:** this fix works
well for queries with *some* topical anchor (player names, "goals", a
timeframe). It does **not** fully fix the original *broad* default query
("Summarize the match and provide key tactical insights.") in isolation —
re-run the same snippet with that query instead and the top hits are still
generic Pass/Duel chunks, just with weak, closely-clustered rerank scores
(around -9.1 to -9.8) rather than one standout. That's because an abstract
instructional query has no semantic or keyword anchor into literal
event-description text — no amount of chunking/re-ranking fixes that on
its own. Closing that gap needs either query expansion/rewriting (having
Query Understanding infer that "summarize + tactical insights" implicitly
means "surface goals/cards/momentum shifts") or a metadata heuristic that
always includes high-salience event types regardless of similarity score.
Neither is built yet — this is exactly the kind of finding Phase 6
"Refinement" (§8.1) is for, once the evaluation harness below produces
numbers to act on.

**Update: fixed in §13 below.** The metadata heuristic described as
missing above is now built — `RetrieverAgent.retrieve(..., ensure_salient=True)`,
wired automatically for exactly this kind of broad, unscoped summary query.

**A second, more serious bug was found and fixed via this same broad
query**, running it through the actual naive baseline (`scripts/run_vertical_slice.py`):
the response came back completely empty, no traceback. Root cause: the
time-window/sequence chunks initially had no size cap - a busy 5-minute
window or long possession chain can contain 60+ events, and a few such
chunks in the top-10 evidence bloated the prompt to ~31,000 characters
(~7,750 tokens). Combined with a 2048-token output cap, and this model
family spending part of that budget on internal reasoning before its
visible answer, Gemini returned `finish_reason=MAX_TOKENS` with an empty
`response.text`. **Fixed**: `chunking/event_chunker.py::render_events()`
now caps every time-window/sequence chunk at `MAX_EVENTS_PER_CHUNK` (12
events, dropped the max chunk size from 12,148 to 1,718 characters in this
match), and `max_output_tokens` was bumped 2048 → 4096 in both
`NaiveRAGGenerator` and `GeneratorAgent` as defense-in-depth. Confirmed
live afterward: a full, well-structured summary generates for the exact
same query on the exact same match with no truncation.

If you still see an empty response after this fix, check
`response.candidates[0].finish_reason` directly (see the Troubleshooting
entry below) rather than assuming it's the same root cause.

## 10. Evaluation harness

**Baseline comparison** (§6.11.5) — retrieval + generation metrics for
naive vs. agentic, side by side, over `evaluation/dataset.py`'s
`STARTER_DATASET` (5 hand-written queries against the WC Final - see that
file's docstring for what this dataset is and isn't):

```bash
uv run python scripts/run_evaluation.py
```

**Expect:** for each query, a `[naive]` and `[agentic]` line with
`P@5`/`R@5`/`rougeL`/`bleu`. Don't expect either side to score high on
every metric — the reference summaries are short and specific, while
generated responses are longer and more free-form, so ROUGE/BLEU absolute
values will look low even for a "good" response; what matters is the
naive-vs-agentic *comparison*, not the absolute numbers.

**Ablation study** (§6.11.6) — same query set through `full`, `no_hybrid`,
`no_reranker`, `no_reasoning`, `no_evaluator` configurations:

```bash
uv run python scripts/run_ablation.py
```

**Quota cost:** this is the most expensive thing in this repo to run - 5
configs × 5 queries × up to 4 Gemini calls each. If you hit a `429`, that's
the free tier's 20-requests/day/model cap (see Troubleshooting) - wait for
the daily reset or run a subset of `CONFIGS`/`STARTER_DATASET` by editing
the script.

**Fail signs:** a `KeyError` on `reasoning_insights` would mean the
`no_reasoning` config path is broken (`generator_node` in `agents/graph.py`
should read `state.get("reasoning_insights", [])`, not index directly);
metrics that don't change at all between configs likely means a config
dict typo (e.g. `no_hybrid` not actually reaching `RetrieverAgent`).

## 11. Output saving and execution logging (every script + every API call)

Updated after a live complaint that there was no way to check a
long-running `experiment/` script's progress independently while it was
running — the original `tee_to_file()` design (below, now removed) only
mirrored `print()` output verbatim, with no timestamp/level/module info
and no visibility into third-party library activity. Two concerns are now
handled separately rather than conflated into one mechanism:

- **The final, human-readable deliverable** (`run_vertical_slice.py`,
  `run_agentic_pipeline.py`, and every `POST /pipeline/run` API call) is
  saved to `outputs/<prefix>_<timestamp>.txt` via
  `football_rag.output_writer::write_output(prefix, text)` — a plain write
  of a complete string, safe under concurrent requests since it never
  touches global state. Called after `run_agentic_pipeline()` returns, with
  the text rendered by `agents/formatting.py::format_pipeline_result()` —
  the same function used by both the CLI script and the API, so the two
  never drift apart. The saved path comes back in the API's JSON response
  as `output_file`, and `scripts/app.py` shows it as a caption under
  "Match: ..." after a successful run.
- **Execution progress** (every `experiment/*.py` script, every
  `scripts/run_*.py` script, and the API process) is now real, leveled
  logging via `football_rag.logging_config::setup_logging(name)` — a
  console handler at INFO and a file handler at DEBUG writing to
  `logs/<name>_<timestamp>.log`. Any module just does
  `logging.getLogger(__name__)` and inherits this configuration
  automatically. Known-noisy third-party loggers (`httpx`,
  `sentence_transformers`, `chromadb`, etc.) are quieted to WARNING -
  confirmed live that without this, `httpx` alone logs a line for every
  single HTTP HEAD request HuggingFace Hub makes while resolving a model,
  completely burying the project's own progress messages once the root
  logger has handlers attached.

`run_evaluation.py` and `run_ablation.py` (the small starter-scale
evaluation/ablation scripts) only ever printed metrics, not a generated
deliverable - they now log via `setup_logging()` and have no `outputs/`
file at all, unlike `run_vertical_slice.py`/`run_agentic_pipeline.py` which
still produce one.

The old `tee_to_file(prefix)` (a context manager that redirected
`sys.stdout` to both the console and a file for the duration of `main()`)
has been removed from `output_writer.py` - confirmed via a full-codebase
grep that nothing called it anymore once every CLI script had a real
logger.

**Test it:**

```bash
uv run python scripts/run_vertical_slice.py
ls outputs/   # should show a new run_vertical_slice_<match_id>_<timestamp>.txt (the deliverable)
ls logs/      # should show a new run_vertical_slice_<match_id>_<timestamp>.log (the execution trace)
```

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"match_id": 3869685, "query": "What did Mbappe do in the second half?"}' \
  | python -c "import json,sys; print(json.load(sys.stdin)['output_file'])"
# prints something like outputs\api_pipeline_run_3869685_20260726_092601_423593.txt -
# confirm that file exists and its content matches the API response. The
# API process's own logs/api_<timestamp>.log should show a
# "POST /pipeline/run match=... query=..." line for the same request.
```

**Fail signs:** no new file appears in `outputs/` or `logs/` — check the
script/API process actually has write access to the working directory it
was launched from (both directories are created relative to the current
working directory, not the repo root, if you launched from somewhere
else); a `KeyError`/`AttributeError` inside `format_pipeline_result()` most
likely means `result` is missing a key it expects (e.g. called with a dict
that's already been through `PipelineRunResponse` serialization instead of
the raw `run_agentic_pipeline()` output — the API calls it *before*
constructing the response model specifically to avoid this); a flood of
third-party HTTP/model-loading noise in the console or log file means a
new noisy logger needs adding to `logging_config.py::_NOISY_LOGGERS`.

## 12. Penalty retrieval investigation (data completeness + reranker limitation)

Triggered by inspecting a specific raw StatsBomb event by ID
(`657afc07-537a-44cc-9411-cb0a713b0a63`, a `Goal Keeper`/"Penalty Conceded"
event for match 3869685) and asking why penalty context wasn't showing up
in generated summaries. Found three distinct, separately-verified issues:

**Bug 1 (fixed): the word "Penalty" was silently dropped from chunk text.**
The `Goal Keeper` event itself is correctly filtered as low-signal
(`LOW_SIGNAL_EVENT_TYPES`) - that's intentional, not the bug. Its
`related_events` link led to the actual penalty *shot* (Mbappé's 79th
minute penalty goal), which is **not** filtered - but
`ingestion/statsbomb.py::_describe_event()` only read `shot.technique`
("Normal"/"Volley"/etc.) and `shot.body_part`, never `shot.type` (the field
that actually distinguishes Penalty/Free Kick/Corner/Open Play). Confirmed
before the fix: the chunk read `"...took a shot...; Normal, Right Foot; xG
0.78; outcome: Goal."` - the word "penalty" appeared nowhere. **Fixed**:
`_describe_event()` now includes `shot.type` in `detail` (excluding "Open
Play", which is the vast majority and adds no signal). Verified live: all
11 penalty-type shots in this match (2 in-game penalties + the shootout
kicks, which StatsBomb also tags `shot.type=Penalty`) now correctly say
"Penalty" in their chunk text.

**Bug 2 (fixed): BM25 had no stemming, so plural/singular queries missed
otherwise-exact keyword matches.** Even after Bug 1's fix, querying
`bm25_index.py` directly with *"What penalties were awarded and scored in
the match?"* (plural) returned zero penalty chunks in the top 10 - generic
Duel/Interception chunks outranked them. The same query with "penalty"
singular returned the penalty chunks at the very top (scores 5.28-5.85).
Root cause: `_tokenize()` did exact `[a-z0-9]+` tokenization with no
stemming, so `"penalties"` (query) and `"penalty"` (chunk text) were
completely different, non-matching tokens to BM25. **Fixed**: `_tokenize()`
now runs every token through `nltk`'s `PorterStemmer` (works standalone, no
corpus download needed - added `nltk` as an explicit direct dependency
since it was previously only pulled in transitively via `rouge-score`).
Verified live: the plural query now returns the identical top result as
the singular query.

**Finding 3 (not fixed - a real limitation, flagged for the ablation
study): the cross-encoder reranker itself demotes correct penalty chunks.**
With both bugs fixed, the pre-rerank RRF-fused pool for the plural penalty
query already has penalty chunks at roughly half of its top 10 positions -
hybrid retrieval alone does reasonably well. But after
`retrieval/reranker.py`'s cross-encoder re-scores that same pool, **zero**
penalty chunks survive into the final top 6 - generic "Foul Won" chunks
end up on top instead. `cross-encoder/ms-marco-MiniLM-L-6-v2` is trained on
general web/Q&A relevance (MS MARCO), not football-specific text, and
doesn't reliably recognize that a penalty shot is what "penalties...scored"
is asking about. This is exactly the kind of gap the ablation study
(`scripts/run_ablation.py`, `no_reranker` config) exists to quantify - not
something patched ad hoc here.

**Reproduce this investigation** (all quota-free, no Gemini calls):

```bash
uv run python -c "
from pathlib import Path
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.chunking import build_hierarchical_chunks
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.retrieval.bm25_index import BM25Index
from football_rag.agents.retriever import RetrieverAgent

events = load_match_events(3869685, cache_dir=Path('data/statsbomb_cache'))
chunks = build_hierarchical_chunks(events)
store = ChunkVectorStore(persist_dir=Path('data/chroma'))
store.ingest(chunks)

query = 'What penalties were awarded and scored in the match?'
print('--- BM25 alone (should surface penalty chunks near the top) ---')
bm25 = BM25Index(chunks)
for cid, score in bm25.query(query, top_n=5):
    text = next(c.text for c in chunks if c.chunk_id == cid)
    print(f'  ({score:.3f}) {text[:90]}')

print()
print('--- Full hybrid + rerank (penalty chunks currently do NOT survive) ---')
for h in RetrieverAgent(store, chunks).retrieve(query, top_n=5):
    score = h.get('rerank_score', h.get('fusion_score'))
    print(f\"  ({score:.3f}) {h['text'][:90]}\")
"
```

**Expect:** the BM25-alone section shows penalty chunks at the top with
scores ~5.2-5.9; the full hybrid+rerank section still shows generic
Foul/Duel chunks on top (Finding 3, unresolved). If BM25 alone *also*
misses the penalty chunks, Bug 2's stemming fix has regressed - check
`bm25_index.py::_tokenize()` still stems via `PorterStemmer`.

## 13. "Ask anything" gap analysis: StatsAgent, lineups, ensure_salient, Duel/Block fixes

Triggered by the goal to let a user "ask anything about the game or the
match" rather than just narrow demo queries. Auditing the pipeline against
that bar (starting from the concrete observation that `StructuredQuery.intent`
was extracted by Query Understanding but never read anywhere downstream —
confirmed via a full-codebase grep, zero hits outside its own definition)
found five gaps. Four are fixed here; two (multi-turn conversational memory,
spatial/GIS queries over x/y coordinates) are explicitly out of scope for
this pass, along with cross-match/historical analytics (this system is
single-match by design per the proposal) and real SoccerNet/report
integration (already deferred in `evaluation/dataset.py`).

**Gap 1+2 (fixed): counting/aggregate questions had no answer path at all.**
"How many yellow cards", "how many shots did Messi take" are structurally
the wrong shape for RAG retrieval — it samples a top-N, it can never
reliably enumerate "all X". **Fixed** by wiring `StructuredQuery.intent`
(finally) into a new `intent="statistic"` value, extracted by
`QueryUnderstandingAgent` for counting-style questions, with a new `outcome`
field (e.g. "Goal", "Yellow Card") for scoping. `graph.py` adds a
conditional edge routing `intent=="statistic"` straight to a new
`agents/stats.py::StatsAgent` → END, bypassing retriever/aggregator/
reasoning/generator/evaluator entirely. `StatsAgent` makes no LLM call — it
filters the full `list[Event]` (not a retrieved sample) by whatever
`StructuredQuery` fields are set, and returns a deterministic count +
team/player breakdown + evidence list. This is **a deliberate addition
beyond the proposal's original 6-agent design**.

**A real bug found live while verifying this**: the first end-to-end test
of "How many yellow cards were shown in the match?" returned *"No matching
events found"* even though this match has 8 real yellow cards. Cause:
`QueryUnderstandingAgent`'s prompt never told the model what StatsBomb
`event_type` strings actually exist, so it extracted a plausible-but-fake
`event_types=["Card"]` — there is no `"Card"` event type in StatsBomb data;
cards are the `outcome` of a `Foul Committed` or `Bad Behaviour` event.
**Fixed**: `SYSTEM_PROMPT` now enumerates the real event type vocabulary
explicitly and calls out the card/`Foul Committed`/`Bad Behaviour` mapping
by name. Verified live after the fix: the same query now extracts
`event_types=['Foul Committed', 'Bad Behaviour'], outcome='Yellow Card'`
and `StatsAgent` returns exactly 8 matches (Argentina 5, France 3),
matching this match's raw data exactly.

**Gap 3 (fixed): broad summary queries still missed goals/cards/subs.**
Flagged as an open, unfixed problem back in §9 above — chunking and
re-ranking improvements helped queries with a topical anchor ("Messi",
"goals") but did nothing for the default, anchor-free "Summarize the match
and provide key tactical insights." query. **Fixed**:
`RetrieverAgent.retrieve()` gained `ensure_salient: bool = False`. When
true, chunks for Goal-outcome shots, Own Goals, Red/Yellow Cards, and
Substitutions are force-included (capped at half of `top_n`) regardless of
similarity/BM25/rerank score, with the rest filled normally.
`graph.py`'s `retriever_node` sets it exactly when `intent=="match_summary"`
and no metadata filter is active — i.e. a genuinely broad, unscoped
request, not a query that's already anchored to an entity. Verified live,
offline (no LLM call, just `RetrieverAgent` directly): without
`ensure_salient`, the top 8 hits for the default broad query were 100%
generic Duel/Pass chunks (seq-163, seq-125, seq-46, seq-171, seq-76, tw-19,
seq-123, seq-13) with zero goals anywhere. With `ensure_salient=True`, the
first four hits became Messi's 22' penalty goal (`ev-318`), Di María's 35'
shot (`ev-491`), and Mbappé's 79'/80' shots (`ev-1202`, `ev-1226`) — the
four real goals of the match, in the right order — with the remaining four
slots still filled by the normal hybrid-ranked pool.

**Gap 4 (fixed): lineups/formations were structurally unanswerable.**
"What formation did Argentina play", "who started for France" had zero
underlying data — `Starting XI` events are filtered out of `Event` entirely
by `LOW_SIGNAL_EVENT_TYPES` (they carry no per-event tactical detail in the
existing sentence template), and `fetch_lineups()` existed but was never
called anywhere. **Fixed**: new `chunking/lineup_chunker.py::chunk_lineups()`
reads the `Starting XI` events directly from the pre-filter `raw_events`
list (no new HTTP call — `raw_events` is already fetched at every ingestion
call site) and produces one chunk per team with formation + all 11 starting
players (position, jersey number). New `chunk_level="lineup"`, a fourth
chunking tier. `build_hierarchical_chunks()` gained an optional
`raw_events` param to wire this in; the three CLI/API call sites were
updated from the bundled `load_match_events()` to its two constituent calls
(`fetch_raw_events()` + `parse_events()`) so `raw_events` stays available
alongside `events`. Verified live, offline: exactly 2 lineup chunks
generated for match 3869685 — "Argentina started in a 4-3-3 formation: ..."
and "France started in a 4-2-3-1 formation: ...", both matching the real,
publicly known formations from that final.

**Gap 5 (fixed): two common event types were silently losing sub-fields.**
Same class of bug as the earlier `shot.type` fix (§ above/§12) — auditing
which kept event types extract zero detail beyond their generic name found
`Duel` (98 occurrences in this match) and `Block` (50 occurrences).
**Fixed**: `_describe_event()` now also reads `duel.type.name`
("Tackle"/"Aerial Lost") into `detail`, and `block.deflection`/
`block.offensive` (booleans) into `detail` as "deflected"/"offensive"
flags. Verified live: a sample Duel now has `detail="Aerial Lost"`, a
sample Block now has `detail="deflected"`.

**Reproduce the offline checks** (no Gemini calls):

```bash
uv run python -c "
from pathlib import Path
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.chunking import build_hierarchical_chunks
from football_rag.chunking.lineup_chunker import chunk_lineups
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.agents.retriever import RetrieverAgent
from football_rag.agents.stats import StatsAgent
from football_rag.agents.state import StructuredQuery

raw_events = fetch_raw_events(3869685, cache_dir=Path('data/statsbomb_cache'))
events = parse_events(raw_events, 3869685)

# Gap 5: Duel/Block detail
duel = next(e for e in events if e.event_type == 'Duel' and e.detail)
block = next(e for e in events if e.event_type == 'Block' and e.detail)
print('Duel detail:', duel.detail, '| Block detail:', block.detail)

# Gap 4: lineups
lineups = chunk_lineups(3869685, raw_events)
print(len(lineups), 'lineup chunks')
for c in lineups:
    print(' ', c.text[:70])

# Gap 1+2: StatsAgent, bypassing the LLM entirely
cards_query = StructuredQuery(intent='statistic', event_types=['Foul Committed', 'Bad Behaviour'], outcome='Yellow Card')
print(StatsAgent().answer('how many yellow cards', cards_query, events).splitlines()[0])

# Gap 3: ensure_salient
chunks = build_hierarchical_chunks(events, raw_events=raw_events)
store = ChunkVectorStore(persist_dir=Path('data/chroma'))
store.ingest(chunks)
retriever = RetrieverAgent(store, chunks)
query = 'Summarize the match and provide key tactical insights.'
without = retriever.retrieve(query, top_n=8, ensure_salient=False)
with_ = retriever.retrieve(query, top_n=8, ensure_salient=True)
print('goals without ensure_salient:', sum(1 for h in without if 'outcome: Goal' in h['text']))
print('goals with ensure_salient:', sum(1 for h in with_ if 'outcome: Goal' in h['text']))
"
```

**Expect:** `Duel detail: Aerial Lost` (or "Tackle" depending on which
sample is found first) and `Block detail: deflected` (or "offensive"); `2
lineup chunks` each starting with "<Team> started in a <formation>
formation:"; `8 matching event(s) found.` for the cards query; `goals
without ensure_salient: 0` and `goals with ensure_salient:` a positive
number (at least 1, typically more within the forced-salient budget).

**One live end-to-end check** (uses Query Understanding's Gemini quota,
one call): `uv run python scripts/run_agentic_pipeline.py 3869685 "How
many yellow cards were shown in the match?"` — confirms the graph's
conditional routing and `format_pipeline_result()` correctly render the
stats-only response shape (empty retrieved-chunks/aggregated-context/
reasoning-insights sections) without breaking, and that the extracted
`event_types` match real StatsBomb vocabulary rather than an invented one.

**Two more crash bugs found via a live run of the broad summary query
(only surfaced once `ensure_salient` actually fires — user-reported after
seeing the naive baseline's "Not enough information." output on the same
query and asking whether it was broken):**

**Bug A: `TypeError: 'NoneType' object is not iterable` in
`build_agentic_user_prompt()`.** Gemini's structured-output API treats a
Pydantic field with a Python default (`supporting_chunk_ids: list[str] =
Field(default_factory=list)`) as nullable, and returned an explicit `null`
for it on at least one `ReasoningInsight` — which bypasses the
`default_factory` via the SDK's own parsing path, so downstream code saw
`None` instead of `[]`. **Fixed**: `ReasoningAgent.reason()` now normalizes
any `None` back to `[]` right after the Gemini call (the single point where
this should be caught, same pattern as the existing `response.text or ""`
fix); `prompts.py::build_agentic_user_prompt()` and
`formatting.py::format_pipeline_result()` also guard with `or []` as
defense-in-depth.

**Bug B: `TypeError: unsupported format string passed to NoneType.__format__`
in `format_pipeline_result()`** (and the equivalent `KeyError` in
`scripts/app.py`'s Streamlit retriever table). `RetrieverAgent._force_salient()`
builds forced hit dicts with only `chunk_id`/`text`/`metadata` — no
`rerank_score` or `fusion_score` key at all, since these chunks bypass
ranking entirely by design. Both renderers assumed every hit has a score.
**Fixed**: both now display `"forced"` in place of a numeric score when
neither key is present.

**Reproduce (live, uses ~4 Gemini calls — query understanding, reasoning,
generator, evaluator):**

```bash
uv run python scripts/run_agentic_pipeline.py 3869685 "Summarize the match and provide key tactical insights."
```

**Expect:** section 2 shows five `(forced)` entries at the top — Messi's
22'/108' goals, Di María's 35' goal, and Mbappé's 79'/80' goals — followed
by the normal similarity-ranked pool; section 5 is a complete, accurate
summary covering all five goals with correct minutes/scorers; section 6
(Evaluator) passes. No traceback.

## Troubleshooting

- **`RuntimeError: GEMINI_API_KEY is not set`** — `.env` missing or not at
  repo root; `load_dotenv()` in `agents/_llm.py` only looks in the current
  working directory tree.
- **`429` / rate-limit error from Gemini** — free tier has per-minute and
  per-day request caps (current numbers: https://ai.google.dev/gemini-api/docs/rate-limits).
  Confirmed live: the free tier's `GenerateRequestsPerDayPerProjectPerModel`
  quota is only 20 requests/day for the current default model - easy to
  exhaust with a single testing session across all four entry points, since
  the agentic pipeline alone makes up to 4 calls/run. The error message
  itself includes a `retryDelay`, but that hint is unreliable for a
  per-day quota; if retrying immediately doesn't help, it means the day's
  quota is actually gone, not that a few seconds will fix it. Through the
  API this now surfaces as a clean `429` with the real Gemini message in
  `detail` (via `api/main.py`'s `except genai_errors.APIError` handler,
  converted to a `RuntimeError` by `_api_client._raise_for_status`, then
  shown as `st.error` in the UI) rather than an opaque `500 Internal Server
  Error` - if you see the latter instead, the running API process is out of
  date (see "No `--reload`" above) and needs restarting to pick up the fix.
  Workarounds: wait for the daily reset, use `enable_evaluator=False` to
  cut one call per run, or switch to a different free model in the agent
  model constants (`agents/query_understanding.py`, `reasoning.py`,
  `generator.py`, `evaluator.py`, `generation/generator.py`).
- **Schema/validation error on `response.parsed`** — means Gemini's
  structured-output mode failed to fill a `StructuredQuery` /
  `ReasoningInsight` / `EvaluationResult` field. Print `response.text`
  instead of `.parsed` in `agents/_llm.py::call_structured` to see the raw
  JSON Gemini returned and compare against the Pydantic model in
  `agents/state.py`.
- **`404 NOT_FOUND ... no longer available to new users`** — a pinned model
  name (e.g. `gemini-2.5-flash`) has been deprecated for new API keys.
  Current model constants use the `-latest` aliases (`gemini-flash-latest`,
  `gemini-flash-lite-latest`), which Google keeps pointed at whatever the
  current recommended Flash model is specifically to avoid this. If it
  recurs, list what your key can actually call:
  `uv run python -c "from football_rag.agents._llm import get_client; [print(m.name) for m in get_client().models.list() if 'generateContent' in (m.supported_actions or [])]"`
- **`ZeroDivisionError` in `rank_bm25`** — fixed: `BM25Index` used to crash
  when a metadata filter narrowed the BM25 candidate corpus to zero chunks
  (`BM25Okapi` divides by corpus size). It now returns no hits instead.
- **Empty `response_text`, no traceback** — fixed (see §9): was caused by
  oversized time-window/sequence chunks blowing up the evidence prompt
  enough to hit `finish_reason=MAX_TOKENS` before any visible text.
  `render_events()` now caps chunk size and `max_output_tokens` was
  doubled to 4096. If it recurs, diagnose with:
  `response = client.models.generate_content(...); print(response.candidates[0].finish_reason)`
  - `MAX_TOKENS` means the output budget is still too tight for the prompt
  (bump `max_output_tokens` further, or reduce `top_n`/evidence size);
  other `finish_reason` values (e.g. `SAFETY`) point elsewhere entirely.
- **Slow first run** — expected (StatsBomb fetch + embedding model
  download + embedding ~3500+ chunks across all three tiers on CPU, plus a
  one-time `cross-encoder/ms-marco-MiniLM-L-6-v2` download the first time
  anything calls `RetrieverAgent`). Subsequent runs reuse
  `data/statsbomb_cache/` and `data/chroma/`.
- **`bert_score` triggers a huge download** — expected only if you actually
  call `evaluation/generation_metrics.py::bertscore()` (not used by
  `run_evaluation.py`/`run_ablation.py` by default); its default model is
  ~1GB+. `rouge_scores()`/`bleu_score()` never trigger this.
- **Different match** — pass any StatsBomb open-data match id, e.g.
  `uv run python scripts/run_agentic_pipeline.py <match_id> "your query"`.
- **Streamlit terminal spam like `ModuleNotFoundError: No module named
  'torchvision'`** — harmless. Streamlit's file watcher probes every
  installed package for hot-reload paths, including `transformers`' optional
  vision submodules that `sentence-transformers` never touches here. Not a
  real error; the app runs fine alongside it.
- **`streamlit run` fails with a port already in use** — pass
  `--server.port <other_port>`, or stop whatever else is bound to 8501.
- **Mangled accented characters (e.g. "Mbapp�") in CLI output** — a
  Windows console codepage display issue only, not a data or pipeline bug;
  the same names render correctly with proper accents in the Streamlit UI
  (browser output isn't limited by the console's codepage).
- **Streamlit UI shows "Couldn't reach the API"** — `scripts/run_api.py`
  isn't running, crashed, or is on a different port/host than
  `scripts/app.py` expects. Check `FOOTBALL_RAG_API_URL` (default
  `http://localhost:8000`) matches wherever `run_api.py` actually started.
- **API and UI on different machines/environments** — `GEMINI_API_KEY` only
  needs to be set where `scripts/run_api.py` runs; `scripts/app.py` never
  touches Gemini directly and doesn't need the key at all.
