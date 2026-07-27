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

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from football_rag.logging_config import setup_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "main_comparison.csv"
OUTPUT_PATH = RESULTS_DIR / "worst_cases.csv"


def main() -> None:
    log_path = setup_logging("error_analysis")
    logger.info("Logging to %s", log_path)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not CSV_PATH.exists():
        logger.warning("%s doesn't exist yet - run experiment/run_full_comparison.py first.", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    logger.info("%d rows loaded (%s)", len(df), df["system"].value_counts().to_dict())

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

    logger.info("Worst %d by precision_at_5 and by rougeL, per system:", n)
    for _, row in combined.iterrows():
        logger.info(
            "  [%-7s/%-20s] match=%s P@5=%.2f rougeL=%.2f query=%r",
            row["system"], row["reason"], row["match_id"], row["precision_at_5"], row["rougeL"], row["query"],
        )

    logger.info("Saved: %s", OUTPUT_PATH)

    # Same auditing convention as run_full_comparison.py/run_full_ablation.py/
    # make_tables.py's save_snapshot(): OUTPUT_PATH stays a single "latest"
    # file, but every run also gets its own permanent, timestamped copy so
    # re-running this against a bigger main_comparison.csv later doesn't
    # silently erase what an earlier, smaller run's worst-cases looked like.
    snapshot_dir = RESULTS_DIR / f"worst_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, snapshot_dir / OUTPUT_PATH.name)
    logger.info("Snapshot saved: %s", snapshot_dir / OUTPUT_PATH.name)
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
