# Experiments log

This is a status/findings log for the actual research-scale experimental
campaign (the `experiment/` directory), kept separate from `TESTING.md`
(which documents manual test procedures for the codebase itself). This
file answers "what has actually been run, what did it find, and what's
still outstanding" — update it as each experiment progresses rather than
only at the very end, since results and known issues discovered along the
way are exactly the kind of thing that's easy to lose track of otherwise.

## Status at a glance

| Experiment | Status | Script | Result file |
|---|---|---|---|
| Embedding comparison (MiniLM vs E5) | ✅ Done | `compare_embeddings.py` | `results/embedding_comparison.csv` |
| Naive vs Agentic comparison | ✅ Done (122/122 rows) | `run_full_comparison.py` | `results/main_comparison.csv` |
| Error analysis | ✅ Done (against full 122 rows) | `error_analysis.py` | `results/worst_cases.csv` |
| Ablation study | ✅ Done (75/75 rows) | `run_full_ablation.py` | `results/ablation.csv` |
| Human evaluation | ✅ Done (3 raters × 20 pairs) | `human_eval_sample.py` / `human_eval_analyze.py` | `results/human_eval_summary.csv` |

**All 5 experiments are now complete.** See §5 below for a real inter-rater-agreement analysis that wasn't part of the original single-rater tooling.

Aggregated tables for everything above: `experiment/results/thesis_tables.md` (regenerate via `uv run python experiment/make_tables.py`).

## 1. Embedding comparison (MiniLM vs E5)

Dense-retrieval-only comparison (no LLM calls, quota-free) over the 4-match
representative subset, 15 queries.

| Embedder | P@5 | R@5 | P@10 | R@10 |
|---|---|---|---|---|
| MiniLM (`all-MiniLM-L6-v2`) | 0.440 | 0.283 | 0.307 | **0.339** |
| E5 (`e5-small-v2`) | 0.440 | 0.273 | 0.287 | 0.316 |

**Reading:** effectively tied, not decisively separated — Precision@5 is
identical to three decimal places; MiniLM edges ahead on every other
metric, but by margins (0.01–0.02) well within what 15 queries can
distinguish reliably. Both models are affected equally by whatever
retrieval-design choices are held constant (BM25, reranker), so this
isolates the embedding model's own contribution as intended. Full
discussion: `docs/8.2_embedding_model_comparison.md`.

**Caveat:** scoped to a 4-match subset, not the full 18-match corpus —
embedding the full ~32,000-chunk corpus twice (once per model) was measured
to take multiple hours on CPU-only hardware, so this is a deliberate,
documented scope reduction (see the script's own module docstring), not a
silent one.

**Run-to-run variance found while updating this section — not a
methodological bug, but worth being upfront about.** Re-running
`compare_embeddings.py` with no code or dataset change produced slightly
different numbers across attempts (e.g. MiniLM P@5 read 0.467 in an
earlier run, 0.440 in the final one now reported above; `query_dataset.py`
was unchanged across these runs, ruling out a ground-truth-relevance
explanation). Root cause: `ChunkVectorStore` uses Chroma's default HNSW
index, which is an *approximate* nearest-neighbor search, not exact
brute-force cosine ranking — combined with `store.ingest(..., skip_existing=True)`
building the collection incrementally across separate process runs, the
graph's approximate top-K result near the similarity-score boundary can
shift slightly run to run even though the embeddings themselves are
deterministic. The table above is the most recent run and matches the
committed `experiment/results/embedding_comparison.csv` /
`thesis_tables.md` exactly; treat single-run P@K/R@K numbers from an HNSW
index as accurate to roughly ±0.02-0.03, not exact, when citing this
comparison.

## 2. Naive vs Agentic comparison (the headline result)

Full sweep: 61 queries × 2 systems = 122 rows, across all 18 frozen
matches.

| Metric | Naive | Agentic |
|---|---|---|
| Precision@5 | 0.357 | **0.763** |
| Recall@5 | 0.303 | **0.458** |
| Precision@10 | 0.231 | **0.482** |
| Recall@10 | 0.349 | **0.461** |
| ROUGE-1 | 0.140 | **0.181** |
| ROUGE-2 | 0.036 | 0.039 |
| ROUGE-L | 0.115 | **0.132** |
| BLEU | 1.511 | **2.099** |
| BERTScore F1 | **0.828** | 0.818 |

**Reading:** agentic wins on every metric except BERTScore, where the two
are within noise of each other. Strong, clean support for RQ1 (agentic RAG
improves over naive) and RQ3 (reasoning/evaluator stages measurably help).

**Caveat baked into the numbers already:** 36 of the 61 agentic queries are
statistic-intent (routed to `StatsAgent`, which does not retrieve). Their
retrieval metrics are recorded as `N/A` and excluded from the P@K/R@K
average shown above (they're still included in the generation-metric
average and in `n_queries`) — see `run_full_comparison.py::_metrics_row()`.
Averaging them in as `0.0` would have unfairly penalised the agentic system
for doing exactly what it's designed to do.

### Bugs found and fixed during this run (not before it — genuinely discovered live)

These are worth keeping in the thesis's methodology/limitations discussion,
since they were caught *during* data collection, not assumed away in
advance:

1. **Cross-match contamination.** The first version of the agentic
   pipeline had no mandatory `match_id` filter on dense retrieval — a
   generic query like "summarize the match" (no team/player names for the
   LLM to extract) could retrieve chunks from *other* matches in the
   shared vector store. Confirmed live: an Argentina-France summary
   narrated England and Croatia's tactics. Fixed in `agents/graph.py`
   (`_with_match_filter()`) — every retrieval call now always constrains
   to the current match, regardless of what the LLM extracted.
2. **StatsAgent counted penalty-shootout kicks as match goals.**
   StatsBomb encodes a shootout kick as an ordinary `Shot`/`outcome=Goal`
   event; `StatsAgent` had no period-5 exclusion, so "how many goals did
   each team score" for the 2022 World Cup Final returned 7-5 instead of
   the real 3-3. Fixed in `agents/stats.py`
   (`_is_goal_count_query()`/`SHOOTOUT_PERIOD`). The same ambiguity was
   also fixed at the chunk-text level (`chunking/event_chunker.py` now
   tags period-5 events `(Penalty Shootout)` in the rendered sentence),
   since it was independently causing a naive-RAG answer to substitute a
   real shootout kick for a real in-match goal it never retrieved.
3. **Naive's fixed task template caused false "Not enough information"
   answers.** `NaiveRAGGenerator` used one hardcoded instruction
   ("write a 150-200 word match summary...") regardless of what was
   actually asked. Given evidence that fully answered a narrow question
   ("how many cards were shown" — all 8 real cards were in the retrieved
   evidence), the model still declined, apparently because the fixed
   instruction didn't match the narrower question. Fixed in
   `generation/prompts.py::TASK_INSTRUCTIONS` — now instructs the model to
   directly answer the actual request, falling back to the full
   summary/key-moments/insights structure only when a full summary is
   what's actually being asked.
4. **`EvaluatorAgent`/`ReasoningAgent` crashed under a different Gemini
   model.** After switching `GEMINI_MODEL`, both agents' structured-output
   calls started returning `None` (the model exhausted its token budget
   on internal reasoning before finishing valid JSON), crashing
   `agents/graph.py` with an `AttributeError`/`TypeError`. Fixed by raising
   both agents' `max_tokens` and adding safe fallback handling (treat a
   missing evaluation as a pass-through, a missing reasoning result as an
   empty insight list) — logged as a `WARNING` in `agents/_llm.py` now, not
   silent.
5. **Query-understanding extraction reliability.** Observed live that the
   *same* "how many yellow or red cards" query sometimes correctly
   extracted `outcome="Yellow Card"` (giving the right 8/5/3 count) and
   sometimes didn't (giving a wildly wrong 50-event "all fouls" count,
   confirmed reproducible with a hand-built `StructuredQuery` missing the
   `outcome` field). This is inherent LLM non-determinism in the query
   understanding step, not a code bug — no safe code-level fix exists
   without risking breaking a legitimate "how many fouls did each team
   commit" query the same way. Worth an error-analysis example, not a
   silent patch.

**Data collection note:** results were collected in two batches under two
different `GEMINI_MODEL` values (`gemini-flash-latest`, then
`gemini-3.5-flash` after switching to Tier 1 billing to clear the free-tier
daily quota wall). The working dataset was wiped and restarted from zero
after the model switch specifically so all 122 rows are under one
consistent model — do not mix in any earlier snapshot folder
(`results/main_comparison_<timestamp>/`) predating that reset when citing
final numbers.

## 3. Error analysis

Refreshed against the full 122-row dataset (`results/worst_cases.csv`, 40
rows: worst-10-by-P@5 and worst-10-by-ROUGE-L, per system).

**Notable finding:** most of agentic's "worst ROUGE-L" rows are
statistic-intent queries answered correctly by `StatsAgent` but scored
very low on ROUGE-L anyway. `StatsAgent`'s output format ("N matching
event(s) found. By team: ... By player: ... Details: ...") is stylistically
very different from the human-written reference sentence ("8 card(s) were
shown: 5 for Argentina, 3 for France."), so ROUGE penalises correct answers
for wording, not for being wrong. Worth reporting as a generation-metric
limitation in the thesis, not as an agentic-system weakness.

## 4. Ablation study

Complete: 75/75 (query, config) pairs, 5 configs (full, no-hybrid,
no-reranker, no-reasoning, no-evaluator) over the 15-query representative
subset (4 matches).

| Config | Precision@5 | ROUGE-L |
|---|---|---|
| full | 0.427 | 0.122 |
| no_evaluator | 0.427 | 0.131 |
| no_hybrid | 0.427 | **0.142** |
| no_reasoning | 0.427 | 0.129 |
| no_reranker | 0.427 | 0.122 |

Interrupted twice mid-run by a transient `httpx.RemoteProtocolError`
(dropped connection, most likely the machine going idle/sleep between
calls) - both times resumed cleanly from the checkpoint with zero data
lost. This surfaced a real gap that's now fixed: the script previously only
treated Gemini 429 quota errors as "recoverable, stop gracefully" and
crashed with a full traceback on anything else. Broadened in
`experiment/_checkpoint.py::is_transient_network_error()` (catches
`httpx.TransportError` and its subclasses) so this class of error now logs
a clean warning and exits gracefully instead, same as a quota hit.

**Two things worth being upfront about in the thesis, not glossed over:**

1. **Precision@5 is identical across all 5 configs.** Expected for
   `no_reasoning`/`no_evaluator` - those toggles only affect what happens
   *after* retrieval, so a retrieval metric can't differ there by
   construction. Less expected for `no_hybrid`/`no_reranker`, which *do*
   change retrieval - most likely explanation is that for these 15
   queries, the genuinely-relevant chunks already rank in the top-5 under
   dense-only retrieval regardless of whether BM25 fusion or reranking is
   additionally applied, so hybrid/reranking's effect shows up in ordering
   and generation quality rather than in raw top-5 set membership here.
   Worth stating this explicitly rather than presenting P@5 as "unaffected
   by ablation" without explanation.
2. **`full` is not the best config on ROUGE-L in the raw 15-query
   average** (0.142 for no_hybrid vs. full's 0.122) - **but this is
   substantially explained away, not a real ablation effect.** Splitting
   the 15 queries into the 8 statistic-intent ones (routed to
   `StatsAgent`, which cannot be affected by any of the 5 toggles - it's
   deterministic and doesn't touch hybrid/reranker/reasoning/evaluator at
   all) and the 7 non-statistic ones (where the toggles genuinely can
   affect the output) shows the statistic-query rows vary by
   config anyway (0.056-0.091) - this can only be noise from Query
   Understanding's independent, non-deterministic re-extraction per
   config (see finding #5 above), not a real effect of what's being
   ablated. Restricted to just the 7 queries where the ablation has an
   actual causal path, the spread collapses to 0.193-0.201 and `full` is
   no longer the worst config - it's mid-pack. Full write-up with the
   numbers: `docs/8.5_ablation_study.md`. The corrected, defensible
   conclusion is "no component appears harmful, but this sample size
   can't attribute the naive-vs-agentic gap to any individual piece" -
   not "full is worse than removing components," which was the
   surface-level, uncorrected reading.

## 5. Human evaluation

Complete: **3 raters × 20 pairs** (blind, paired comparison — each rater
saw the same 20 sampled query pairs, randomised naive/agentic into
"Response A"/"Response B" per pair, not told which was which). Re-sampled
against the final, complete 122-row `main_comparison.csv` (verified by
timestamp and a byte-for-byte content cross-check of every sampled
response against the source file — the original 3-pair sample predated
the full sweep and was discarded).

The original single-rater tooling (`human_eval_analyze.py`) only reported
one rater's means and win-rate. Rebuilt to handle multiple raters properly
now that there are 3 real ones: per-rater breakdown, an averaged-across-
raters summary (same shape as before, so `make_tables.py` still reads it
unmodified), a pooled win-rate, and — new — **Krippendorff's alpha**
inter-rater agreement per dimension, which the original proposal (§6.11.4)
called for but the tooling never actually implemented until now.

### Mean scores (averaged across 3 raters, 1-5 scale)

| Dimension | Naive | Agentic |
|---|---|---|
| Relevance | 2.62 | **4.40** |
| Coherence | 4.30 | **4.63** |
| Factual consistency | 2.77 | **4.20** |
| Fluency | **4.65** | 4.53 |

### Pooled win rate (60 paired judgments = 3 raters × 20 pairs)

| Dimension | Agentic wins | Naive wins | Ties |
|---|---|---|---|
| Relevance | 42 | 8 | 10 |
| Coherence | 25 | 9 | 26 |
| Factual consistency | 39 | 9 | 12 |
| Fluency | 12 | **16** | 32 |

### Inter-rater agreement (Krippendorff's alpha, ordinal)

| Dimension | α | Interpretation |
|---|---|---|
| Relevance | 0.92 | Excellent agreement |
| Factual consistency | 0.86 | Strong agreement |
| Fluency | 0.46 | Moderate agreement |
| Coherence | 0.22 | Weak agreement |

**Reading:** a strong, consistent result for RQ1/RQ3 — agentic wins
decisively on relevance and factual consistency (the two dimensions most
directly tied to grounding/correctness), and on coherence. **Fluency is
the one dimension where naive edges ahead** (16 vs. 12 wins), which is
plausible rather than concerning: naive's fixed, simpler prose template
produces a more uniform writing style, while agentic's richer,
reasoning-informed output has more surface area for occasional awkward
phrasing.

**Report the agreement numbers honestly, not selectively.** Relevance and
factual consistency have very strong agreement (raters consistently agree
on what's correct and on-topic), which is reassuring for those two
dimensions' numbers above. Coherence's weak agreement (α=0.22) means
raters genuinely disagreed on what counts as "well-organised" for this
task — worth flagging explicitly as a limitation of that dimension's
definition/subjectivity in the thesis, not silently presented alongside
the higher-confidence dimensions as if it had the same evidentiary weight.

**The pooled averages above mask a real split by query type - don't cite
only the aggregate.** Breaking the 20 sampled pairs down by query type
(full numbers and discussion: `docs/8.6_human_evaluation.md`): agentic
wins decisively on match-summary (n=21 ratings) and, most importantly,
on goals-count (n=21) - naive averages a startling 1.00/5 on relevance
there, the worst score anywhere in this evaluation, while `StatsAgent`
scores 4.71. That goals-count result is exactly the independent check
§8.4.3's generation-metric limitation needed: ROUGE/BLEU penalised
`StatsAgent` for wording, but human judges are not fooled by naive's
fluent-but-wrong answers on the same queries. Conversely, **naive rated
higher on entity-focus and cards-count** (n=9 each) - real observations
in this sample, but resting on only 3 sampled pairs per type, too small
to generalise from confidently. Report the query-level nuance, not just
the clean aggregate win.

## Open items before treating any of this as final thesis numbers

- Consider spot-checking the other three query types (summary/goals/player)
  the same way the cards-criteria bug was found (§2, bug list) — that bug
  was only caught because someone happened to look; the others haven't had
  the same scrutiny.
- Decide how to present the ablation study's two counter-intuitive
  findings (§4) in the thesis — as a genuine limitation worth its own
  discussion, not smoothed over.
- Decide how to present coherence's weak inter-rater agreement (§5) — as a
  noted limitation of that dimension, not at face value alongside the
  higher-agreement dimensions.
- `thesis_tables.md` is now fully populated (all 5 experiments) — safe to
  paste into the results chapter, but cross-reference the discussion
  points above rather than reporting the tables alone.
- Re-run `make_tables.py` after human eval is complete to fold it into
  `thesis_tables.md` (already includes the completed ablation results).
