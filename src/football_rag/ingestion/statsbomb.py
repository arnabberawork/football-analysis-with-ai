"""Loader for the StatsBomb open-data repository (proposal §6.4.1, §6.5).

Fetches raw JSON over HTTP from the public GitHub mirror and normalises it
into the `Event` schema defined in `football_rag.schema`. No StatsBomb
credentials are required — the data is public and free for research use
(StatsBomb, 2023).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

from football_rag.schema import Event

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Event types that carry little independent narrative value on their own and
# are dropped from the textual knowledge base (they still count toward raw
# event totals, they are just not turned into chunks). Kept configurable so
# the cutoff is a single, visible list rather than buried logic.
LOW_SIGNAL_EVENT_TYPES = {
    "Ball Receipt*",
    "Carry",
    "Pressure",
    "Miscontrol",
    "Dispossessed",
    "Goal Keeper",
    "50/50",
    "Referee Ball-Drop",
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Camera On",
    "Camera off",
}


def _cache_path(cache_dir: Path, relative: str) -> Path:
    path = cache_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _get_json(relative_path: str, cache_dir: Optional[Path] = None) -> list | dict:
    """Fetch a JSON file from the open-data repo, with optional disk caching."""
    if cache_dir is not None:
        cached = _cache_path(cache_dir, relative_path)
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/{relative_path}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if cache_dir is not None:
        cached = _cache_path(cache_dir, relative_path)
        cached.write_text(json.dumps(data), encoding="utf-8")

    return data


def fetch_competitions(cache_dir: Optional[Path] = None) -> list[dict]:
    return _get_json("competitions.json", cache_dir)


def fetch_matches(competition_id: int, season_id: int, cache_dir: Optional[Path] = None) -> list[dict]:
    return _get_json(f"matches/{competition_id}/{season_id}.json", cache_dir)


def fetch_raw_events(match_id: int, cache_dir: Optional[Path] = None) -> list[dict]:
    return _get_json(f"events/{match_id}.json", cache_dir)


def fetch_lineups(match_id: int, cache_dir: Optional[Path] = None) -> list[dict]:
    return _get_json(f"lineups/{match_id}.json", cache_dir)


def _describe_event(raw: dict) -> tuple[Optional[str], Optional[str]]:
    """Build a short (detail, outcome) pair capturing what happened, beyond
    the generic event type. Only the fields useful for a human-readable
    sentence are extracted — StatsBomb events carry far more attributes than
    we need for narrative chunking.
    """
    event_type = raw["type"]["name"]
    detail: Optional[str] = None
    outcome: Optional[str] = None

    if event_type == "Pass":
        p = raw.get("pass", {})
        recipient = p.get("recipient", {}).get("name")
        outcome = p.get("outcome", {}).get("name", "Complete")
        height = p.get("height", {}).get("name")
        detail = f"to {recipient}" if recipient else None
        if height:
            detail = f"{detail} ({height})" if detail else f"({height})"
    elif event_type == "Shot":
        s = raw.get("shot", {})
        outcome = s.get("outcome", {}).get("name")
        technique = s.get("technique", {}).get("name")
        body_part = s.get("body_part", {}).get("name")
        xg = s.get("statsbomb_xg")
        parts = [p for p in (technique, body_part) if p]
        detail = ", ".join(parts) if parts else None
        if xg is not None:
            detail = f"{detail}; xG {xg:.2f}" if detail else f"xG {xg:.2f}"
    elif event_type in ("Foul Committed", "Foul Won"):
        card = raw.get("foul_committed", {}).get("card", {}).get("name")
        outcome = card
    elif event_type == "Dribble":
        outcome = raw.get("dribble", {}).get("outcome", {}).get("name")
    elif event_type == "Duel":
        outcome = raw.get("duel", {}).get("outcome", {}).get("name")
    elif event_type == "Interception":
        outcome = raw.get("interception", {}).get("outcome", {}).get("name")
    elif event_type == "Substitution":
        replacement = raw.get("substitution", {}).get("replacement", {}).get("name")
        detail = f"replaced by {replacement}" if replacement else None
    elif event_type == "Bad Behaviour":
        outcome = raw.get("bad_behaviour", {}).get("card", {}).get("name")

    return detail, outcome


def parse_events(
    raw_events: list[dict],
    match_id: int,
    include_low_signal: bool = False,
) -> list[Event]:
    """Convert raw StatsBomb event JSON into normalised `Event` objects.

    Only events with a `location` and a recognised team/player are kept —
    purely administrative events (kickoff markers, half start/end, etc.) are
    filtered out by default since they add no tactical narrative value
    (see LOW_SIGNAL_EVENT_TYPES).
    """
    events: list[Event] = []
    for raw in raw_events:
        event_type = raw["type"]["name"]
        if not include_low_signal and event_type in LOW_SIGNAL_EVENT_TYPES:
            continue

        team = raw.get("team")
        if team is None:
            continue

        location = raw.get("location")
        player = raw.get("player")
        detail, outcome = _describe_event(raw)

        events.append(
            Event(
                match_id=str(match_id),
                team_id=str(team["id"]),
                team_name=team["name"],
                player_id=str(player["id"]) if player else None,
                player_name=player["name"] if player else None,
                minute=raw["minute"],
                second=raw.get("second", 0),
                period=raw["period"],
                x=location[0] if location else None,
                y=location[1] if location else None,
                event_type=event_type,
                possession_id=str(raw["possession"]) if "possession" in raw else None,
                outcome=outcome,
                detail=detail,
            )
        )

    events.sort(key=lambda e: (e.period, e.minute, e.second))
    return events


def load_match_events(
    match_id: int,
    cache_dir: Optional[Path] = None,
    include_low_signal: bool = False,
) -> list[Event]:
    """Convenience one-shot: fetch + parse a single match's events."""
    raw_events = fetch_raw_events(match_id, cache_dir=cache_dir)
    return parse_events(raw_events, match_id, include_low_signal=include_low_signal)
