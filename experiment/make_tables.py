"""Stage 13: aggregates every `results/*.csv` into final, thesis-ready
summary tables - the last step, run once the earlier stages have
produced (at least some) data. Safe to run at any point - each section
is skipped with a note if its source CSV doesn't exist yet, so this
works equally well as a "how far along am I" check mid-experiment and as
the final aggregation pass.

Writes `experiment/results/thesis_tables.md` (plain markdown tables,
paste-ready into the results chapter) and prints the same to console.

Usage:
    uv run python experiment/make_tables.py
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from football_rag.output_writer import tee_to_file

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_DIR / "thesis_tables.md"


def _to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer - no `tabulate` dependency needed."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    with tee_to_file("make_tables") as log_path:
        _run()
    print(f"\nLog saved to {log_path}")


def _run() -> None:
    sections: list[str] = []

    comparison_csv = RESULTS_DIR / "main_comparison.csv"
    if comparison_csv.exists():
        df = pd.read_csv(comparison_csv, keep_default_na=False)
        # precision_at_k/recall_at_k are "N/A" (a string) for StatsAgent-routed
        # rows, which bypass retrieval by design (see run_full_comparison.py).
        # pd.to_numeric(..., errors="coerce") turns those into NaN, which
        # .mean() then correctly skips per-row - if left as "N/A" strings,
        # pandas would treat the whole column as non-numeric and silently
        # drop it entirely from every system's average, not just those rows.
        numeric_cols = ["precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10", "bertscore_f1"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        metric_cols = ["precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10", "rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
        summary = df.groupby("system")[metric_cols].mean(numeric_only=True).round(3)
        summary["n_queries"] = df.groupby("system").size()
        n_statistic = (df["precision_at_5"].isna() & (df["system"] == "agentic")).sum()
        sections.append(
            "## Naive vs Agentic - retrieval + generation metrics (stages 9/10/13)\n\n"
            + _to_markdown(summary.reset_index())
            + f"\n\n_Note: retrieval metrics (P@K/R@K) are averaged only over rows where retrieval actually ran - "
            f"{n_statistic} agentic row(s) routed to StatsAgent (exact counts, no retrieval by design) are excluded "
            f"from the retrieval-metric average but still included in generation-metric averages and n_queries._"
        )
    else:
        sections.append("## Naive vs Agentic\n\n_Not yet available - run `experiment/run_full_comparison.py`._")

    embedding_csv = RESULTS_DIR / "embedding_comparison.csv"
    if embedding_csv.exists():
        df = pd.read_csv(embedding_csv)
        sections.append("## Embedding model comparison - MiniLM vs E5 (stage 3)\n\n" + _to_markdown(df))
    else:
        sections.append("## Embedding model comparison\n\n_Not yet available - run `experiment/compare_embeddings.py`._")

    ablation_csv = RESULTS_DIR / "ablation.csv"
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        summary = df.groupby("config")[["precision_at_5", "rougeL"]].mean().round(3)
        summary["n_rows"] = df.groupby("config").size()
        sections.append("## Ablation study (stage 12)\n\n" + _to_markdown(summary.reset_index()))
    else:
        sections.append("## Ablation study\n\n_Not yet available - run `experiment/run_full_ablation.py`._")

    human_eval_csv = RESULTS_DIR / "human_eval_summary.csv"
    if human_eval_csv.exists():
        df = pd.read_csv(human_eval_csv)
        sections.append("## Human evaluation (stage 11)\n\n" + _to_markdown(df))
    else:
        sections.append("## Human evaluation\n\n_Not yet available - run `experiment/human_eval_sample.py` then `human_eval_analyze.py`._")

    worst_cases_csv = RESULTS_DIR / "worst_cases.csv"
    if worst_cases_csv.exists():
        df = pd.read_csv(worst_cases_csv)
        sections.append(f"## Error analysis (stage 12)\n\n{len(df)} worst-case rows saved to `worst_cases.csv` - see that file for qualitative discussion material (not reproduced here, response texts are long).")
    else:
        sections.append("## Error analysis\n\n_Not yet available - run `experiment/error_analysis.py`._")

    content = "# Thesis results tables\n\n" + "\n\n".join(sections) + "\n"
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(content)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Every run gets its own permanent, timestamped copy for auditing - same
    # convention as run_full_comparison.py/run_full_ablation.py's
    # save_snapshot(). OUTPUT_PATH itself stays a single, overwritten
    # "latest" file (useful to open without hunting for the newest folder),
    # but re-running this after collecting more data should never silently
    # erase what an earlier run's tables looked like.
    snapshot_dir = RESULTS_DIR / f"thesis_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, snapshot_dir / OUTPUT_PATH.name)
    print(f"Snapshot saved: {snapshot_dir / OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
