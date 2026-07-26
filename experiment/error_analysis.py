"""Stage 12 (error analysis half): surfaces the worst-performing rows from
`run_full_comparison.py`'s results for qualitative discussion in the
thesis - the "why did it fail here" cases, not the aggregate numbers
(those are `make_tables.py`'s job).

Reads `experiment/results/main_comparison.csv` (whatever's completed so
far - doesn't need the full sweep to be done) and writes the bottom-N
rows by precision_at_5 and by rougeL, per system, to
`experiment/results/worst_cases.csv`.

Usage:
    uv run python experiment/error_analysis.py [n]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "main_comparison.csv"
OUTPUT_PATH = RESULTS_DIR / "worst_cases.csv"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not CSV_PATH.exists():
        print(f"{CSV_PATH} doesn't exist yet - run experiment/run_full_comparison.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"{len(df)} rows loaded ({df['system'].value_counts().to_dict()})")

    worst_rows = []
    for system in df["system"].unique():
        system_df = df[df["system"] == system]

        worst_precision = system_df.nsmallest(n, "precision_at_5").copy()
        worst_precision["reason"] = "worst_precision_at_5"
        worst_rows.append(worst_precision)

        worst_rouge = system_df.nsmallest(n, "rougeL").copy()
        worst_rouge["reason"] = "worst_rougeL"
        worst_rows.append(worst_rouge)

    combined = pd.concat(worst_rows, ignore_index=True).drop_duplicates(subset=["match_id", "query", "system", "reason"])
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWorst {n} by precision_at_5 and by rougeL, per system:")
    for _, row in combined.iterrows():
        print(f"  [{row['system']:7s}/{row['reason']:20s}] match={row['match_id']} P@5={row['precision_at_5']:.2f} rougeL={row['rougeL']:.2f} query={row['query']!r}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
