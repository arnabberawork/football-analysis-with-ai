# Testing guide: Gemini-backed RAG pipelines

Manual test plan for the four runnable entry points in this repo, all
Gemini-backed: the naive baseline (`scripts/run_vertical_slice.py`), the
6-agent pipeline CLI (`scripts/run_agentic_pipeline.py`), a FastAPI service
over the same agentic pipeline (`scripts/run_api.py`,
`src/football_rag/api/main.py`), and a Streamlit UI (`scripts/app.py`) that
talks to that API over HTTP rather than importing pipeline code itself -
plus the evaluation harness (`scripts/run_evaluation.py`,
`scripts/run_ablation.py`). There's no automated test suite yet, so this is
a run-it-and-read-the-output checklist.

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

## 2. Naive baseline (single Gemini call)

Cheapest real test — one retrieval + one `gemini-flash-latest` call.

```bash
uv run python scripts/run_vertical_slice.py
```

First run will be slow (~1-2 min): downloads the StatsBomb match JSON,
caches it to `data/statsbomb_cache/`, downloads the `all-MiniLM-L6-v2`
embedding model, embeds ~1500+ event chunks into `data/chroma/`. Re-running
afterwards should be much faster (cache hits + idempotent Chroma upsert).

**Expect:**
- `N events parsed`, `N event-level chunks generated`, `vector store now
  holds N chunks` (chunk count should match after any re-run).
- A "Retrieved evidence" list of `[chunk_id] (distance) text` lines.
- A "Generated response" section: a coherent ~150-200 word match summary
  mentioning specific minutes/players, grounded in the retrieved evidence.

**Fail signs:** `RuntimeError: GEMINI_API_KEY is not set` (check `.env`),
a `429`/quota error from the Gemini call (see Troubleshooting), or an empty/
garbled response.

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
2. **Retriever** — a list of `[chunk_id] (fusion_score) text` lines, ranked
   by descending fusion_score.
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
from football_rag.chunking.event_chunker import chunk_events
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.retrieval.vector_store import ChunkVectorStore

events = load_match_events(3869685, cache_dir=Path('data/statsbomb_cache'))
chunks = chunk_events(events)
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
2. Click **Run pipeline** with the default query. First click ingests +
   embeds the match server-side (~1 min) before the pipeline runs. **Expect**
   the same six numbered stages as the CLI script, rendered as expandable
   panels, a sortable table (Retriever), and metric cards (Evaluator)
   instead of print statements.
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
   again without restarting it. **Expect** a friendly `st.error` naming the
   API URL it couldn't reach, not an unhandled exception/blank page —
   confirms `scripts/app.py`'s `except (RuntimeError, ConnectionError)`
   handling around every `_api_client` call.

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
