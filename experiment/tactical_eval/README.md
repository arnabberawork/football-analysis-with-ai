# Tactical-insight evaluation kit

A blind, paired evaluation of the tactical quality of match summaries produced by the
naive RAG baseline and the Agentic RAG framework. It addresses the remaining gap between
the thesis title ("tactical insights") and the evidence: the original human evaluation
rated relevance, coherence, factual consistency and fluency, but not tactical quality.

## What is in this folder

| File | Purpose | Give to raters? |
|---|---|---|
| `tactical_eval_sheet_rater1.xlsx`, `..._rater2.xlsx`, `..._rater3.xlsx` | Rating sheets (one per rater). Sheet "Instructions" holds the rubric; sheet "Ratings" holds the 13 pairs | **Yes** - one each |
| `tactical_eval_key.csv` | Which system is Response A / B for each pair | **No - keep private** |
| `tactical_eval_analyze.py` | Computes the results once sheets are returned | No |
| `make_kit.py` | Rebuilds the sheets from `experiment/results/main_comparison.csv` (fixed random seed) | No |

## The 13 pairs

All 13 match-summary queries ("Summarize the match and provide key tactical insights.")
for which both systems produced a full answer. The five matches where the baseline
answered only "Not enough information." are excluded, because there are no tactics to
compare. Response A/B order is randomised and balanced (6 agentic-as-A, 7 naive-as-A).
The responses are exactly those evaluated in Chapter 5 - nothing is regenerated.

## Dimensions (1-5 for each response)

1. **Tactical relevance** - does it discuss how the teams played, not just list events?
2. **Tactical correctness** - are the tactical claims plausible and consistent with the match?
3. **Evidence support** - are the claims tied to specific events (minutes, players, actions)?
4. **Depth of analysis** - does it explain causes and effects, beyond description?

Plus an overall preference (A / B / Tie) for tactical insight.

## Steps

1. Recruit three raters who follow football closely (ideally different people from the
   earlier evaluation; if not, note this in the thesis).
2. Send each rater their own sheet. Ask them to work independently (about 60-75 minutes).
3. When returned, save each as `tactical_eval_sheet_raterN_completed.xlsx` in this folder.
4. Run:
   ```
   uv add openpyxl
   uv run python experiment/tactical_eval/tactical_eval_analyze.py
   ```
5. The script prints a results table (means, wins/ties, sign test, Krippendorff's alpha)
   and saves `tactical_eval_results.csv`. Send these to be written into Section 5.6 /
   RQ3 of the thesis - report the result whatever it is.
