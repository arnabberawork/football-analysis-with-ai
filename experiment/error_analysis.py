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

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from football_rag.output_writer import tee_to_file

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "main_comparison.csv"
OUTPUT_PATH = RESULTS_DIR / "worst_cases.csv"


def main() -> None:
    with tee_to_file("error_analysis") as log_path:
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

        # Same auditing convention as run_full_comparison.py/run_full_ablation.py/
        # make_tables.py's save_snapshot(): OUTPUT_PATH stays a single "latest"
        # file, but every run also gets its own permanent, timestamped copy so
        # re-running this against a bigger main_comparison.csv later doesn't
        # silently erase what an earlier, smaller run's worst-cases looked like.
        snapshot_dir = RESULTS_DIR / f"worst_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, snapshot_dir / OUTPUT_PATH.name)
        print(f"Snapshot saved: {snapshot_dir / OUTPUT_PATH.name}")
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
