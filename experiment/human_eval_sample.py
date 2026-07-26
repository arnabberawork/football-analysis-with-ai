"""Stage 11 (human evaluation, part 1): builds a blinded rating sheet.

Samples completed (match_id, query) pairs from
`experiment/results/main_comparison.csv` where *both* naive and agentic
responses exist, shuffles which one is shown as "response_a" vs
"response_b" per pair (so you rate blind, not knowing which system
produced which response), and writes:

- `results/human_eval_sheet.csv` - what you actually fill in: query,
  response_a, response_b, and empty rating columns
  (relevance/coherence/factual_consistency/fluency, 1-5) for each.
- `results/human_eval_key.csv` - the hidden a/b -> naive/agentic mapping.
  Don't open this until after rating, or the blinding is pointless.

Refuses to overwrite an existing sheet by default (ratings are real work,
not something to regenerate silently like the auto-drafted query
dataset) - pass --force to regenerate anyway.

Usage:
    uv run python experiment/human_eval_sample.py [n] [--force]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPARISON_CSV = RESULTS_DIR / "main_comparison.csv"
SHEET_PATH = RESULTS_DIR / "human_eval_sheet.csv"
KEY_PATH = RESULTS_DIR / "human_eval_key.csv"
RATING_COLUMNS = ["relevance", "coherence", "factual_consistency", "fluency"]
SEED = 42


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    n = int(args[0]) if args else 20

    if SHEET_PATH.exists() and not force:
        print(f"{SHEET_PATH} already exists - pass --force to regenerate (this discards any ratings already filled in).")
        return

    if not COMPARISON_CSV.exists():
        print(f"{COMPARISON_CSV} doesn't exist yet - run experiment/run_full_comparison.py first.")
        return

    df = pd.read_csv(COMPARISON_CSV)
    naive = df[df["system"] == "naive"].set_index(["match_id", "query"])
    agentic = df[df["system"] == "agentic"].set_index(["match_id", "query"])
    common_keys = naive.index.intersection(agentic.index).tolist()

    if not common_keys:
        print("No (match_id, query) pairs with both naive and agentic responses completed yet.")
        return

    rng = random.Random(SEED)
    sample_size = min(n, len(common_keys))
    sampled_keys = rng.sample(common_keys, sample_size)

    sheet_rows = []
    key_rows = []
    for i, key in enumerate(sampled_keys):
        match_id, query = key
        naive_text = naive.loc[key, "response_text"]
        agentic_text = agentic.loc[key, "response_text"]

        if rng.random() < 0.5:
            response_a, response_b = naive_text, agentic_text
            a_system, b_system = "naive", "agentic"
        else:
            response_a, response_b = agentic_text, naive_text
            a_system, b_system = "agentic", "naive"

        row = {"pair_id": i, "match_id": match_id, "query": query, "response_a": response_a, "response_b": response_b}
        for col in RATING_COLUMNS:
            row[f"{col}_a"] = ""
            row[f"{col}_b"] = ""
        sheet_rows.append(row)
        key_rows.append({"pair_id": i, "a_system": a_system, "b_system": b_system})

    pd.DataFrame(sheet_rows).to_csv(SHEET_PATH, index=False)
    pd.DataFrame(key_rows).to_csv(KEY_PATH, index=False)

    print(f"Sampled {sample_size} pairs from {len(common_keys)} available.")
    print(f"Fill in ratings (1-5) in: {SHEET_PATH}")
    print(f"Don't open until after rating: {KEY_PATH}")
    print(f"Then run: uv run python experiment/human_eval_analyze.py")


if __name__ == "__main__":
    main()
