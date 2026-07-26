"""The frozen match set for the thesis experiment (supervisor's stage 1).

18 matches from the 2022 FIFA World Cup (`competition_id=43, season_id=106`
in StatsBomb open-data - confirmed live against the real 64-match list),
selected for stage variety (group/R16/QF/SF/3rd-place/final), scoreline
variety (high-scoring blowouts alongside tight 0-0/1-1 games), and
penalty-shootout coverage (useful for "statistic"-intent queries). Frozen
here as a plain list so every experiment script imports the same set and
none of it is picked ad hoc.
"""

from __future__ import annotations

from pydantic import BaseModel


class FrozenMatch(BaseModel):
    match_id: int
    stage: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    date: str
    went_to_penalties: bool = False


FROZEN_MATCHES: list[FrozenMatch] = [
    FrozenMatch(match_id=3869685, stage="Final", home_team="Argentina", away_team="France", home_score=3, away_score=3, date="2022-12-18", went_to_penalties=True),
    FrozenMatch(match_id=3869519, stage="Semi-finals", home_team="Argentina", away_team="Croatia", home_score=3, away_score=0, date="2022-12-13"),
    FrozenMatch(match_id=3869552, stage="Semi-finals", home_team="France", away_team="Morocco", home_score=2, away_score=0, date="2022-12-14"),
    FrozenMatch(match_id=3869684, stage="3rd Place Final", home_team="Croatia", away_team="Morocco", home_score=2, away_score=1, date="2022-12-17"),
    FrozenMatch(match_id=3869321, stage="Quarter-finals", home_team="Netherlands", away_team="Argentina", home_score=2, away_score=2, date="2022-12-09", went_to_penalties=True),
    FrozenMatch(match_id=3869486, stage="Quarter-finals", home_team="Morocco", away_team="Portugal", home_score=1, away_score=0, date="2022-12-10"),
    FrozenMatch(match_id=3869354, stage="Quarter-finals", home_team="England", away_team="France", home_score=1, away_score=2, date="2022-12-10"),
    FrozenMatch(match_id=3869420, stage="Quarter-finals", home_team="Croatia", away_team="Brazil", home_score=1, away_score=1, date="2022-12-09", went_to_penalties=True),
    FrozenMatch(match_id=3869117, stage="Round of 16", home_team="Netherlands", away_team="United States", home_score=3, away_score=1, date="2022-12-03"),
    FrozenMatch(match_id=3869118, stage="Round of 16", home_team="England", away_team="Senegal", home_score=3, away_score=0, date="2022-12-04"),
    FrozenMatch(match_id=3869151, stage="Round of 16", home_team="Argentina", away_team="Australia", home_score=2, away_score=1, date="2022-12-03"),
    FrozenMatch(match_id=3869152, stage="Round of 16", home_team="France", away_team="Poland", home_score=3, away_score=1, date="2022-12-04"),
    FrozenMatch(match_id=3869219, stage="Round of 16", home_team="Japan", away_team="Croatia", home_score=1, away_score=1, date="2022-12-05", went_to_penalties=True),
    FrozenMatch(match_id=3869220, stage="Round of 16", home_team="Morocco", away_team="Spain", home_score=0, away_score=0, date="2022-12-06", went_to_penalties=True),
    FrozenMatch(match_id=3869253, stage="Round of 16", home_team="Brazil", away_team="South Korea", home_score=4, away_score=1, date="2022-12-05"),
    FrozenMatch(match_id=3869254, stage="Round of 16", home_team="Portugal", away_team="Switzerland", home_score=6, away_score=1, date="2022-12-06"),
    FrozenMatch(match_id=3857271, stage="Group Stage", home_team="England", away_team="Iran", home_score=6, away_score=2, date="2022-11-21"),
    FrozenMatch(match_id=3857291, stage="Group Stage", home_team="Spain", away_team="Costa Rica", home_score=7, away_score=0, date="2022-11-23"),
]

MATCH_IDS: list[int] = [m.match_id for m in FROZEN_MATCHES]

# A representative subset spanning every major stage (Final, Quarter-final,
# Round of 16, Group Stage), used wherever running against the full 18-match
# corpus is computationally impractical - confirmed live that embedding the
# full ~32,000-chunk corpus twice (once per embedding model, in
# compare_embeddings.py) would take hours on CPU-only hardware, and the
# ablation study's 5-configs-per-query multiplier has the same problem
# against Gemini's daily quota. Both scripts use this same subset so the
# comparison and the ablation are drawn from identical matches.
REPRESENTATIVE_SUBSET_MATCH_IDS: list[int] = [
    3869685,  # Final: Argentina 3-3 France (pens)
    3869420,  # Quarter-final: Croatia 1-1 Brazil (pens)
    3869254,  # Round of 16: Portugal 6-1 Switzerland
    3857291,  # Group Stage: Spain 7-0 Costa Rica
]
