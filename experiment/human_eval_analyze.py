"""Stage 11 (human evaluation, part 2): aggregates your filled-in ratings.

Reads `results/human_eval_sheet.csv` (your ratings) and
`results/human_eval_key.csv` (the a/b -> naive/agentic mapping hidden
from you while rating), unmasks each row, and reports per-system mean
scores across relevance/coherence/factual_consistency/fluency plus a
paired comparison (how often agentic scored higher than naive on each
dimension, for the same query).

Skips any pair still missing a rating (partial progress is fine - fill
in more rows and re-run any time).

Usage:
    uv run python experiment/human_eval_analyze.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SHEET_PATH = RESULTS_DIR / "human_eval_sheet.csv"
KEY_PATH = RESULTS_DIR / "human_eval_key.csv"
SUMMARY_PATH = RESULTS_DIR / "human_eval_summary.csv"
RATING_COLUMNS = ["relevance", "coherence", "factual_consistency", "fluency"]


def main() -> None:
    if not SHEET_PATH.exists() or not KEY_PATH.exists():
        print("Run experiment/human_eval_sample.py first.")
        return

    sheet = pd.read_csv(SHEET_PATH)
    key = pd.read_csv(KEY_PATH).set_index("pair_id")

    rated_rows = []
    for _, row in sheet.iterrows():
        rating_cols_a = [f"{c}_a" for c in RATING_COLUMNS]
        rating_cols_b = [f"{c}_b" for c in RATING_COLUMNS]
        if row[rating_cols_a].isna().any() or row[rating_cols_b].isna().any():
            continue

        mapping = key.loc[row["pair_id"]]
        for suffix, system in [("_a", mapping["a_system"]), ("_b", mapping["b_system"])]:
            rated_rows.append({"pair_id": row["pair_id"], "system": system, **{c: row[f"{c}{suffix}"] for c in RATING_COLUMNS}})

    if not rated_rows:
        print(f"No fully-rated pairs yet in {SHEET_PATH} - fill in the *_a/*_b rating columns (1-5) and re-run.")
        return

    unmasked = pd.DataFrame(rated_rows)
    print(f"{unmasked['pair_id'].nunique()} of {len(sheet)} pairs fully rated.\n")

    summary = unmasked.groupby("system")[RATING_COLUMNS].mean()
    print("Mean scores by system:")
    print(summary.to_string())
    summary.to_csv(SUMMARY_PATH)

    naive_by_pair = unmasked[unmasked["system"] == "naive"].set_index("pair_id")
    agentic_by_pair = unmasked[unmasked["system"] == "agentic"].set_index("pair_id")
    common = naive_by_pair.index.intersection(agentic_by_pair.index)
    print("\nAgentic vs naive, per-dimension win rate (same query, paired comparison):")
    for col in RATING_COLUMNS:
        agentic_wins = (agentic_by_pair.loc[common, col] > naive_by_pair.loc[common, col]).sum()
        naive_wins = (naive_by_pair.loc[common, col] > agentic_by_pair.loc[common, col]).sum()
        ties = len(common) - agentic_wins - naive_wins
        print(f"  {col}: agentic wins {agentic_wins}, naive wins {naive_wins}, ties {ties} (n={len(common)})")

    print(f"\nSaved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
