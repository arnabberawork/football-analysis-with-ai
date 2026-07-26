"""Stage 8: generate the ~48-56 query evaluation set across the 18 frozen
matches (`experiment/matches.py`).

Everything here is templated and derived directly from real `Event` data -
not hand-written per query - so the *facts* in every reference summary
(score, scorer, minute, cards) are correct by construction. Three query
families per match, added only when applicable to that match:

- 1 `match_summary` query, with an auto-drafted reference summary. Flagged
  `# TODO: review` in the generated file - read it over and tighten the
  phrasing, this is a draft, not a finished thesis-quality reference.
- A "goals per team" `statistic` query, always added (even 0-0 games -
  the answer is just "0-0"). Ground truth is computed from events
  (Shot events with outcome=="Goal", period 1-4 only - period 5 is the
  penalty shootout, which is conventionally excluded from "goals scored
  in the match"), not from StatsBomb's match-level score field, so the
  ground truth is derived from exactly the same data the pipeline sees.
- A "cards" `statistic` query, added only if the match actually had a
  booking.
- An entity-focused query on a multi-goal scorer, added only if the match
  has one (most 1-goal-each games don't get an entity query - there's no
  natural standout to ask about).

Writes `experiment/query_dataset.py` (overwritten each run - review your
edits before re-running this, or move them elsewhere first).

Quota-free - no Gemini calls, pure ingestion + templating.

Usage:
    uv run python experiment/build_query_dataset.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matches import FROZEN_MATCHES  # noqa: E402

from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.schema import Event

CACHE_DIR = Path("data/statsbomb_cache")
OUTPUT_PATH = Path(__file__).resolve().parent / "query_dataset.py"


def _goal_moments(events: list[Event], home_team: str, away_team: str) -> list[dict]:
    """Goal-scoring moments, period 1-4 only (period 5 is the penalty
    shootout, conventionally excluded from "goals scored in the match").

    Handles own goals correctly: StatsBomb tags an own goal as a *pair* of
    events - "Own Goal Against" on the conceding team (carries the actual
    player who scored into their own net) and "Own Goal For" on the
    benefiting team (no player attached) - confirmed live on match
    3869151 (Enzo Fernandez o.g. credited to Australia), which is exactly
    why a naive `event_type == "Shot"` filter alone undercounts: it misses
    own goals entirely. Only "Own Goal Against" is used here (with the
    credited team flipped to the opponent) to avoid double-counting the
    same real-world goal from both tagged events.
    """
    moments: list[dict] = []
    for e in events:
        if e.period > 4:
            continue
        if e.event_type == "Shot" and e.outcome == "Goal":
            moments.append({"minute": e.minute, "second": e.second, "team": e.team_name, "player": e.player_name, "own_goal": False})
        elif e.event_type == "Own Goal Against":
            benefiting_team = away_team if e.team_name == home_team else home_team
            moments.append({"minute": e.minute, "second": e.second, "team": benefiting_team, "player": e.player_name, "own_goal": True})
    moments.sort(key=lambda m: (m["minute"], m["second"]))
    return moments


def _card_events(events: list[Event]) -> list[Event]:
    return [
        e
        for e in events
        if e.event_type in ("Foul Committed", "Bad Behaviour") and e.outcome and "Card" in e.outcome
    ]


def _time_str(minute: int, second: int) -> str:
    return f"{minute}'" + (f"{second:02d}" if second else "")


def _draft_match_summary(match, moments: list[dict], cards: list[Event]) -> str:
    if match.home_score > match.away_score:
        verb = f"{match.home_team} beat {match.away_team}"
    elif match.away_score > match.home_score:
        verb = f"{match.away_team} beat {match.home_team}"
    else:
        verb = f"{match.home_team} drew with {match.away_team}"
    line = f"{verb} {max(match.home_score, match.away_score)}-{min(match.home_score, match.away_score)} in the {match.stage} ({match.date})"
    if match.went_to_penalties:
        line += ", with the match going to a penalty shootout"
    line += ". "

    if moments:
        scorer_bits = []
        for m in moments:
            tag = f" (own goal, credited to {m['team']})" if m["own_goal"] else ""
            scorer_bits.append(f"{m['player']}{tag} ({_time_str(m['minute'], m['second'])})")
        line += f"Goal-scorers: {', '.join(scorer_bits)}. "
    else:
        line += "Neither side scored in regulation or extra time. "

    if cards:
        card_bits = ", ".join(f"{e.player_name} ({e.team_name}, {e.outcome})" for e in cards)
        line += f"Cards shown: {card_bits}."
    else:
        line += "No cards were shown."
    return line


def _draft_entity_summary(player: str, player_moments: list[dict], team: str) -> str:
    bits = ", ".join(f"a goal in the {_time_str(m['minute'], m['second'])} minute" for m in player_moments)
    return f"{player} scored {len(player_moments)} goals for {team}: {bits}."


def build_dataset_source() -> str:
    lines = [
        '"""Generated by experiment/build_query_dataset.py - review before treating as frozen.',
        "",
        "Reuses `LabeledQuery`/`RelevanceCriterion` from `football_rag.evaluation.dataset`",
        "(the small STARTER_DATASET there is untouched - this is the larger,",
        "research-scale set used by experiment/run_full_comparison.py and friends).",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from football_rag.evaluation.dataset import LabeledQuery, RelevanceCriterion",
        "",
        "EXPERIMENT_DATASET: list[LabeledQuery] = [",
    ]

    total = 0
    for match in FROZEN_MATCHES:
        raw_events = fetch_raw_events(match.match_id, cache_dir=CACHE_DIR)
        events = parse_events(raw_events, match.match_id)
        moments = _goal_moments(events, match.home_team, match.away_team)
        cards = _card_events(events)

        computed_home = sum(1 for m in moments if m["team"] == match.home_team)
        computed_away = sum(1 for m in moments if m["team"] == match.away_team)
        if (computed_home, computed_away) != (match.home_score, match.away_score):
            print(
                f"  NOTE match {match.match_id}: computed goals {computed_home}-{computed_away} "
                f"!= recorded score {match.home_score}-{match.away_score} - check this match by hand"
            )

        # 1. match_summary
        summary = _draft_match_summary(match, moments, cards)
        lines.append("    LabeledQuery(")
        lines.append(f"        match_id={match.match_id},")
        lines.append('        query="Summarize the match and provide key tactical insights.",')
        lines.append('        relevance_criteria=[RelevanceCriterion(event_type="Shot", outcome="Goal")],')
        lines.append("        # TODO: review - auto-drafted from event data, tighten the phrasing")
        lines.append(f"        reference_summary=({summary!r}),")
        lines.append("    ),")
        total += 1

        # 2. goals-per-team statistic (always added)
        goal_summary = f"{match.home_team} scored {computed_home} goal(s) and {match.away_team} scored {computed_away} goal(s)."
        lines.append("    LabeledQuery(")
        lines.append(f"        match_id={match.match_id},")
        lines.append('        query="How many goals did each team score in this match?",')
        lines.append('        relevance_criteria=[RelevanceCriterion(event_type="Shot", outcome="Goal")],')
        lines.append(f"        reference_summary=({goal_summary!r}),  # exact, computed from event data")
        lines.append("    ),")
        total += 1

        # 3. cards statistic (only if the match had a booking)
        if cards:
            by_team: dict[str, int] = defaultdict(int)
            for e in cards:
                by_team[e.team_name] += 1
            card_summary = f"{len(cards)} card(s) were shown: " + ", ".join(f"{count} for {team}" for team, count in by_team.items()) + "."
            lines.append("    LabeledQuery(")
            lines.append(f"        match_id={match.match_id},")
            lines.append('        query="How many yellow or red cards were shown in the match?",')
            # Relevant chunks must actually mention a card, not just any foul -
            # most fouls in a match aren't carded, so filtering on event_type
            # alone massively over-labels "relevant" (confirmed live: 40-99
            # "relevant" chunks per match for this query vs. single digits for
            # every other query type). StatsBomb's card outcome names are
            # "Yellow Card", "Red Card", and "Second Yellow" - the last of
            # which doesn't contain the literal word "Card", so match on
            # "Yellow"/"Red Card" rather than "Card" to catch all three.
            lines.append(
                '        relevance_criteria=[\n'
                '            RelevanceCriterion(event_type="Foul Committed", outcome="Yellow"),\n'
                '            RelevanceCriterion(event_type="Foul Committed", outcome="Red Card"),\n'
                '            RelevanceCriterion(event_type="Bad Behaviour", outcome="Yellow"),\n'
                '            RelevanceCriterion(event_type="Bad Behaviour", outcome="Red Card"),\n'
                '        ],'
            )
            lines.append(f"        reference_summary=({card_summary!r}),  # exact, computed from event data")
            lines.append("    ),")
            total += 1

        # 4. entity-focused query (only if a multi-goal scorer exists).
        # Own goals are excluded from a player's own scoring tally - crediting
        # an own goal as "Messi scored" would be wrong.
        by_player: dict[str, list[dict]] = defaultdict(list)
        for m in moments:
            if not m["own_goal"]:
                by_player[m["player"]].append(m)
        multi_scorers = {p: ms for p, ms in by_player.items() if len(ms) >= 2}
        if multi_scorers:
            player, player_moments = max(multi_scorers.items(), key=lambda kv: len(kv[1]))
            team = player_moments[0]["team"]
            entity_summary = _draft_entity_summary(player, player_moments, team)
            lines.append("    LabeledQuery(")
            lines.append(f"        match_id={match.match_id},")
            lines.append(f'        query="Describe {player}\'s goals in the match.",')
            lines.append(f'        relevance_criteria=[RelevanceCriterion(player_name={player.split()[0]!r}, event_type="Shot", outcome="Goal")],')
            lines.append("        # TODO: review - auto-drafted from event data, tighten the phrasing")
            lines.append(f"        reference_summary=({entity_summary!r}),")
            lines.append("    ),")
            total += 1

    lines.append("]")
    print(f"\n{total} queries generated across {len(FROZEN_MATCHES)} matches.")
    return "\n".join(lines) + "\n"


def main() -> None:
    source = build_dataset_source()
    OUTPUT_PATH.write_text(source, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
