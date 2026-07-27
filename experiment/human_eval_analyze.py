"""Stage 11 (human evaluation, part 2): aggregates one or more raters'
filled-in ratings.

Auto-discovers every completed rating sheet matching
`results/human_eval_sheet_rater*_completed*.csv` (falls back to the
single-rater `results/human_eval_sheet.csv` if no rater-specific files
exist, so this stays backward compatible with a solo-rater run). Each
sheet is unmasked against the shared `results/human_eval_key.csv` (the
a/b -> naive/agentic mapping every rater rated blind), then reports:

- Per-rater x per-system mean scores (relevance/coherence/
  factual_consistency/fluency).
- Averaged-across-raters per-system means - saved to
  `results/human_eval_summary.csv` in the same shape as the original
  single-rater output, so `make_tables.py` keeps working unmodified.
- Agentic-vs-naive win rate, both per rater and pooled across all raters'
  judgments together.
- Inter-rater agreement (Krippendorff's alpha, ordinal) per dimension -
  only computed when 2+ raters are found, since agreement needs multiple
  judges by definition. This is what the original proposal (§6.11.4) asks
  for and the single-rater version of this script never implemented.

Confirmed live that a rater can accidentally save their sheet as HTML
with a `.csv` extension (e.g. from a spreadsheet's "export as HTML"),
which a plain `pd.read_csv` can't parse at all - `_load_sheet()` below
falls back to `pd.read_html()` automatically when that happens, so this
doesn't require manually pre-fixing every future occurrence.

Skips any pair a given rater left incomplete (partial progress from that
rater is fine - fill in more rows and re-run any time); a rater sheet
that's fully unrated is skipped entirely with a note.

Usage:
    uv run python experiment/human_eval_analyze.py
"""

from __future__ import annotations

import re
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
KEY_PATH = RESULTS_DIR / "human_eval_key.csv"
SUMMARY_PATH = RESULTS_DIR / "human_eval_summary.csv"
BY_RATER_PATH = RESULTS_DIR / "human_eval_summary_by_rater.csv"
AGREEMENT_PATH = RESULTS_DIR / "human_eval_agreement.csv"
RATING_COLUMNS = ["relevance", "coherence", "factual_consistency", "fluency"]


def _load_sheet(path: Path) -> pd.DataFrame:
    """Reads a rating sheet as CSV, falling back to HTML-table parsing if
    the file was accidentally saved as HTML with a .csv extension (a
    confirmed-live failure mode - see module docstring)."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        tables = pd.read_html(path)
        if not tables:
            raise ValueError(f"{path} isn't valid CSV and no HTML table was found in it either.")
        return tables[0]


def _discover_rater_sheets() -> dict[str, Path]:
    """Returns {rater_label: path}, preferring a "*_fixed.csv" version over
    the raw "*_completed.csv" for the same rater when both exist (the fixed
    version is a manually-recovered proper CSV from an HTML mishap)."""
    candidates = sorted(RESULTS_DIR.glob("human_eval_sheet_rater*_completed*.csv"))
    by_rater: dict[str, Path] = {}
    for path in candidates:
        m = re.match(r"human_eval_sheet_(rater\d+)_completed(_fixed)?\.csv", path.name)
        if not m:
            continue
        rater_label, is_fixed = m.group(1), m.group(2)
        if rater_label not in by_rater or is_fixed:
            by_rater[rater_label] = path

    if by_rater:
        return by_rater

    # Backward compatible: no rater-specific files, fall back to the
    # original single-sheet flow.
    solo_path = RESULTS_DIR / "human_eval_sheet.csv"
    return {"rater1": solo_path} if solo_path.exists() else {}


def _unmask(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair_id, system) with that rater's scores, skipping any
    pair this rater left incomplete."""
    rated_rows = []
    for _, row in sheet.iterrows():
        rating_cols_a = [f"{c}_a" for c in RATING_COLUMNS]
        rating_cols_b = [f"{c}_b" for c in RATING_COLUMNS]
        if row[rating_cols_a].isna().any() or row[rating_cols_b].isna().any():
            continue
        mapping = key.loc[row["pair_id"]]
        for suffix, system in [("_a", mapping["a_system"]), ("_b", mapping["b_system"])]:
            rated_rows.append({"pair_id": row["pair_id"], "system": system, **{c: row[f"{c}{suffix}"] for c in RATING_COLUMNS}})
    return pd.DataFrame(rated_rows)


def _win_rate_counts(unmasked: pd.DataFrame) -> dict[str, tuple[int, int, int, int]]:
    """Per-dimension (agentic_wins, naive_wins, ties, n) for one rater's
    unmasked ratings. `unmasked` must have a unique pair_id per system (true
    for any single rater's own sheet, since each pair appears exactly once)
    - do not call this on data pooled across multiple raters, since pair_id
    then repeats and .loc[] lookups silently misalign (confirmed live: this
    produced impossible negative tie counts before pooling was fixed to sum
    each rater's own counts instead, see _report_pooled_win_rate)."""
    naive_by_pair = unmasked[unmasked["system"] == "naive"].set_index("pair_id")
    agentic_by_pair = unmasked[unmasked["system"] == "agentic"].set_index("pair_id")
    common = naive_by_pair.index.intersection(agentic_by_pair.index)
    counts = {}
    for col in RATING_COLUMNS:
        agentic_wins = int((agentic_by_pair.loc[common, col] > naive_by_pair.loc[common, col]).sum())
        naive_wins = int((naive_by_pair.loc[common, col] > agentic_by_pair.loc[common, col]).sum())
        ties = len(common) - agentic_wins - naive_wins
        counts[col] = (agentic_wins, naive_wins, ties, len(common))
    return counts


def _print_win_rate(label: str, counts: dict[str, tuple[int, int, int, int]]) -> None:
    n = next(iter(counts.values()))[3]
    print(f"  {label} - agentic vs naive win rate (n={n} paired queries):")
    for col, (agentic_wins, naive_wins, ties, _n) in counts.items():
        print(f"    {col}: agentic {agentic_wins}, naive {naive_wins}, ties {ties}")


def _compute_agreement(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Krippendorff's alpha (ordinal) per dimension, over the raw (not
    system-unmasked) ratings - agreement is about whether raters judge the
    same item consistently, independent of which system produced it.
    Unit of analysis = (pair_id, side) - one item per response shown, not
    per pair, so a 20-pair sheet gives 40 items."""
    rater_labels = list(sheets.keys())
    rows = []
    for dim in RATING_COLUMNS:
        # matrix shape: (n_raters, n_items) - item = (pair_id, side)
        item_cols = []
        for _, sheet in sheets.items():
            sheet_sorted = sheet.sort_values("pair_id")
            item_cols.append(pd.concat([sheet_sorted[f"{dim}_a"], sheet_sorted[f"{dim}_b"]]).to_numpy(dtype=float))
        matrix = np.vstack(item_cols)
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
        rows.append({"dimension": dim, "krippendorff_alpha": alpha, "n_raters": len(rater_labels), "n_items": matrix.shape[1]})
    return pd.DataFrame(rows)


def main() -> None:
    if not KEY_PATH.exists():
        print("Run experiment/human_eval_sample.py first.")
        return

    rater_paths = _discover_rater_sheets()
    if not rater_paths:
        print("No completed rating sheets found (looked for results/human_eval_sheet_rater*_completed*.csv and results/human_eval_sheet.csv).")
        return

    key = pd.read_csv(KEY_PATH).set_index("pair_id")

    raw_sheets: dict[str, pd.DataFrame] = {}
    unmasked_by_rater: dict[str, pd.DataFrame] = {}
    for rater_label, path in rater_paths.items():
        sheet = _load_sheet(path)
        unmasked = _unmask(sheet, key)
        if unmasked.empty:
            print(f"{rater_label} ({path.name}): no fully-rated pairs yet - skipped.")
            continue
        print(f"{rater_label} ({path.name}): {unmasked['pair_id'].nunique()} of {len(sheet)} pairs fully rated.")
        raw_sheets[rater_label] = sheet
        unmasked_by_rater[rater_label] = unmasked

    if not unmasked_by_rater:
        print("\nNo rater has any fully-rated pairs yet - fill in ratings and re-run.")
        return

    print()

    # Per-rater x per-system means
    by_rater_rows = []
    for rater_label, unmasked in unmasked_by_rater.items():
        summary = unmasked.groupby("system")[RATING_COLUMNS].mean()
        summary.insert(0, "rater", rater_label)
        by_rater_rows.append(summary.reset_index())
        print(f"Mean scores by system - {rater_label}:")
        print(summary.to_string())
        print()
    by_rater_df = pd.concat(by_rater_rows, ignore_index=True)
    by_rater_df.to_csv(BY_RATER_PATH, index=False)

    # Averaged across raters (same shape as the original single-rater
    # output, so make_tables.py keeps reading it unmodified)
    pooled = pd.concat(unmasked_by_rater.values(), ignore_index=True)
    overall_summary = pooled.groupby("system")[RATING_COLUMNS].mean()
    print(f"Mean scores by system - averaged across {len(unmasked_by_rater)} rater(s):")
    print(overall_summary.to_string())
    overall_summary.to_csv(SUMMARY_PATH)
    print(f"\nSaved: {SUMMARY_PATH}")
    print(f"Saved: {BY_RATER_PATH}")

    # Win rate: per-rater, then pooled by summing each rater's own counts -
    # NOT by concatenating raw rows first, since pair_id repeats across
    # raters (each rated the same 20 pairs) and a naive concat-then-.loc[]
    # silently misaligns rows with duplicate index values (confirmed live -
    # see _win_rate_counts docstring).
    print()
    per_rater_counts = {rater_label: _win_rate_counts(unmasked) for rater_label, unmasked in unmasked_by_rater.items()}
    for rater_label, counts in per_rater_counts.items():
        _print_win_rate(rater_label, counts)

    pooled_counts = {
        col: (
            sum(c[col][0] for c in per_rater_counts.values()),
            sum(c[col][1] for c in per_rater_counts.values()),
            sum(c[col][2] for c in per_rater_counts.values()),
            sum(c[col][3] for c in per_rater_counts.values()),
        )
        for col in RATING_COLUMNS
    }
    _print_win_rate(f"pooled ({len(unmasked_by_rater)} raters combined)", pooled_counts)

    # Inter-rater agreement - only meaningful with 2+ raters who rated the
    # same items (i.e. more than one entry in raw_sheets)
    if len(raw_sheets) >= 2:
        print()
        agreement = _compute_agreement(raw_sheets)
        print("Inter-rater agreement (Krippendorff's alpha, ordinal - 0=chance, 1=perfect agreement):")
        print(agreement.to_string(index=False))
        agreement.to_csv(AGREEMENT_PATH, index=False)
        print(f"\nSaved: {AGREEMENT_PATH}")
    else:
        print("\nOnly one rater found - inter-rater agreement needs 2+ raters on the same items, skipped.")


if __name__ == "__main__":
    main()
