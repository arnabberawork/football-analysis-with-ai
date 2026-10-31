"""Tactical-insight evaluation (blind, paired): aggregates the raters'
completed sheets.

Reads every `tactical_eval_sheet_rater*_completed.xlsx` in this folder,
unmasks Response A/B with `tactical_eval_key.csv`, and reports:

- mean score per system for each dimension (tactical relevance, tactical
  correctness, evidence support, depth), averaged over raters and pairs;
- pooled wins / losses / ties per dimension (each rater x pair is one
  paired judgement);
- overall preference counts (A/B/Tie, unmasked);
- an exact two-sided sign test per dimension on the per-pair mean
  difference (agentic - naive), ignoring ties;
- inter-rater agreement per dimension (Krippendorff's alpha, ordinal),
  when two or more raters are found.

Results are written to `tactical_eval_results.csv` and printed as tables
ready to paste into the thesis.

Usage:
    uv add openpyxl          # once, if not already installed
    uv run python experiment/tactical_eval/tactical_eval_analyze.py
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = pd.read_csv(HERE / "tactical_eval_key.csv")
DIMS = ["tactical_relevance", "tactical_correctness", "evidence_support", "depth"]


def sign_test_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load() -> pd.DataFrame:
    sheets = sorted(HERE.glob("tactical_eval_sheet_rater*_completed.xlsx"))
    if not sheets:
        raise SystemExit("No completed sheets found (expected tactical_eval_sheet_rater*_completed.xlsx).")
    long_rows = []
    for path in sheets:
        rater = path.stem.split("rater")[1].split("_")[0]
        df = pd.read_excel(path, sheet_name="Ratings").merge(KEY, on="pair_id")
        for _, r in df.iterrows():
            if any(pd.isna(r[f"{side}_{d}"]) for side in "AB" for d in DIMS):
                continue  # skip incomplete rows
            pref = str(r.get("preferred (A/B/Tie)", "")).strip().upper()
            for side in "AB":
                system = r[f"{side.lower()}_system"]
                rec = {"rater": rater, "pair_id": r["pair_id"], "system": system}
                rec.update({d: float(r[f"{side}_{d}"]) for d in DIMS})
                rec["preferred"] = (pref == side)
                rec["tie"] = (pref == "TIE")
                long_rows.append(rec)
        print(f"loaded {path.name}")
    return pd.DataFrame(long_rows)


def main() -> None:
    df = load()
    raters = sorted(df["rater"].unique())
    print(f"\nRaters: {len(raters)}; rated pairs: {df.groupby('rater')['pair_id'].nunique().to_dict()}\n")

    means = df.groupby("system")[DIMS].mean().T[["naive", "agentic"]]
    wide = df.pivot_table(index=["rater", "pair_id"], columns="system", values=DIMS)
    rows = []
    for d in DIMS:
        diff = wide[(d, "agentic")] - wide[(d, "naive")]
        per_pair = (df.pivot_table(index="pair_id", columns="system", values=d, aggfunc="mean"))
        pdiff = per_pair["agentic"] - per_pair["naive"]
        alpha = float("nan")
        if len(raters) >= 2:
            mat = df.pivot_table(index="rater", columns=["pair_id", "system"], values=d).to_numpy()
            alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")
        rows.append({
            "dimension": d,
            "naive_mean": round(means.loc[d, "naive"], 2),
            "agentic_mean": round(means.loc[d, "agentic"], 2),
            "agentic_wins": int((diff > 0).sum()),
            "naive_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "sign_test_p": round(sign_test_p(int((pdiff > 0).sum()), int((pdiff < 0).sum())), 4),
            "krippendorff_alpha": round(alpha, 2) if not np.isnan(alpha) else None,
        })
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "tactical_eval_results.csv", index=False)

    pref = df[df["preferred"]].groupby("system").size().to_dict()
    ties = int(df[df["tie"]].groupby(["rater", "pair_id"]).ngroups)
    print(res.to_string(index=False))
    print(f"\nOverall preference: agentic {pref.get('agentic', 0)}, naive {pref.get('naive', 0)}, tie {ties}")
    print("\nMarkdown for the thesis:\n")
    print("| Dimension | Naive RAG | Agentic RAG | Agentic wins | Naive wins | Ties | Sign test p | Krippendorff's alpha |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dimension']} | {r['naive_mean']} | {r['agentic_mean']} | {r['agentic_wins']} | {r['naive_wins']} | {r['ties']} | {r['sign_test_p']} | {r['krippendorff_alpha']} |")


if __name__ == "__main__":
    main()
