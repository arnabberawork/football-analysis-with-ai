# Thesis experiment

Everything needed for stages 8-13 of the supervisor's sequence (stages
1-7 are the core `src/football_rag/` package - see the repo root
`CLAUDE.md`). This folder is the research-scale scaffolding on top of
that package; it does not duplicate any pipeline logic.

**This file documents the directory and how to run it. For what's
actually been run and what it found (status, headline numbers, bugs
discovered during data collection), see `../experiments.md` at the repo
root — keep that updated as experiments progress, not just this reference.**

| File | Supervisor stage | What it does |
|---|---|---|
| `matches.py` | 1 | The frozen 18-match list (2022 World Cup) everything else imports |
| `freeze_dataset.py` | 1, 2 | Downloads/caches all 18 matches' raw events, runs hierarchical chunking, writes `results/dataset_eda.csv` |
| `compare_embeddings.py` | 3 | MiniLM vs E5 retrieval-only comparison (no LLM calls) |
| `build_query_dataset.py` | 8 | Generates the ~48-56 query evaluation set into `query_dataset.py` |
| `query_dataset.py` | 8 | Generated output of the above (review before treating as frozen) |
| `_checkpoint.py` | - | Shared resumable-CSV helper used by the two `run_full_*` scripts |
| `run_full_comparison.py` | 9, 10, 13 | Resumable naive-vs-agentic sweep over the full query set |
| `run_full_ablation.py` | 12 | Resumable ablation (5 configs) over a representative query subset |
| `error_analysis.py` | 12 | Surfaces worst P@K/ROUGE cases from the comparison results |
| `human_eval_sample.py` | 11 | Builds a blinded rating sheet from a sample of comparison results |
| `human_eval_analyze.py` | 11 | Aggregates your filled-in ratings |
| `make_tables.py` | 13 | Aggregates every `results/*.csv` into final thesis-ready tables |
| `results/` | - | All CSV outputs land here (gitignored) - each script also saves a permanent, timestamped snapshot copy alongside its "latest" file, never overwritten |

Every script in this folder also logs its execution progress via
`football_rag.logging_config::setup_logging()` to a timestamped file under
the repo root's `logs/` (gitignored) - useful for checking on a long
sweep's progress independently while it's running (`tail` the log file),
separate from the CSV results themselves.

## Gemini quota reality

The model used (`GEMINI_MODEL`/`GEMINI_MODEL_LITE`, configurable in
`.env` - see the repo root `.env.example`) is shared by the naive
baseline's generator and the agentic pipeline's reasoning/generator/
evaluator agents - one naive-vs-agentic query pair costs ~4 calls (fewer
for statistic-intent queries, which route to `StatsAgent` with zero LLM
calls) against the **same** free-tier daily cap (20 requests/day/model).

Two ways to handle this:

- **Free tier (default):** `run_full_comparison.py` and
  `run_full_ablation.py` are both designed to be safely stopped and
  re-run - they checkpoint completed rows to CSV immediately and skip
  anything already done. **Run them once a day until they report "all
  done"** - expect roughly 8-12 days for the full comparison sweep at this
  pace, not a single sitting.
- **Paid Tier 1:** removes the daily request cap entirely (Flash-tier
  pricing is cheap - the full 122-row comparison cost about $1.50 in
  practice). Enable billing on the same Google Cloud project backing your
  `GEMINI_API_KEY` and the full sweep can complete in one sitting instead
  of over a week.

## Suggested order

See `../experiments.md` for current status before starting - most of the
steps below have already been run at least once; don't blindly re-run
from scratch without checking what exists in `results/` first.

1. `uv run python experiment/freeze_dataset.py` (quota-free)
2. `uv run python experiment/compare_embeddings.py` (quota-free)
3. `uv run python experiment/build_query_dataset.py` (quota-free) - then
   read `query_dataset.py` and tidy the auto-drafted narrative summaries
   (flagged `# TODO: review`) before relying on it for real numbers.
4. `uv run python experiment/run_full_comparison.py` - re-run daily (free
   tier) or once (Tier 1, see above).
5. `uv run python experiment/run_full_ablation.py` - same cadence, after 4.
6. `uv run python experiment/error_analysis.py`
7. `uv run python experiment/human_eval_sample.py`, then fill in
   `results/human_eval_sheet.csv` by hand, then
   `uv run python experiment/human_eval_analyze.py`
8. `uv run python experiment/make_tables.py`
