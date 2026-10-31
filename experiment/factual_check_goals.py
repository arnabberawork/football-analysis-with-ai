"""Automatic factual check of goal claims (no API calls, no new runs).

Checks every goal claim in the saved answers against the StatsBomb event
data, in the spirit of FActScore (Min et al., 2023) but restricted to the
claim type that can be verified exactly: who scored, and when.

For each answer to the 25 retrieval-based queries (18 match summaries and
7 "Describe <player>'s goals" questions):

1. The answer is split into units (bullet lines and sentences). A unit is a
   goal claim if it uses goal language (scored, goal, equaliser, converted,
   netted, own goal, ...) and contains a match minute (22', 22'24, 23rd
   minute, minute 23).
2. Each minute in such a unit is paired with the nearest player name from
   that match (accent- and case-insensitive surname or given-name tokens).
3. A claim is correct when the named player scored (or conceded an own goal)
   within the minute tolerance: StatsBomb minute m is reported either as m
   (timestamp style, 22'24) or m+1 (ordinal style, "23rd minute"), so a
   claimed minute c matches a goal at minute m when m-1 <= c <= m+2.
   Penalty-shootout kicks (period 5) are not goals of the match.

Reported per system:
- goal-claim precision: correct claims / claims naming a player;
- goal recall: true goals in scope that the answer states correctly
  (all match goals for a summary; the player's goals for a player question);
- unsupported claims: claims naming a player who did not score at that time.

Reads experiment/results/main_comparison.csv and, if present,
experiment/results/hybrid_baseline.csv. Writes
experiment/results/factual_check_goals.csv (one row per answer) and
experiment/results/factual_check_goals_claims.csv (one row per claim, for
manual auditing).

Usage:
    uv run python experiment/factual_check_goals.py
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVENTS = HERE.parent / "data" / "statsbomb_cache" / "events"
SHOOTOUT_PERIOD = 5

GOAL_WORDS = re.compile(r"\b(goals?|scor\w*|equali[sz]\w*|netted|nets|converted|converts|convert|brace|hat-?trick|own goal|levell?ed|levels|doubled|opened the scoring|restored|winner|winning goal|headed home|volleyed home|fired home|slotted|tapped in)\b", re.I)
SHOOTOUT = re.compile(r"shoot-?out", re.I)
MISS = re.compile(r"\b(off target|wide|blocked|saved|save|post|crossbar|bar|missed|misses|miss)\b", re.I)
SPAN = re.compile(r"\b(?:within|in|during|after|before|over)\s+the\s+(?:opening|first|last|final|closing)\s+\d{1,3}\s+minutes?|\bafter\s+\d{1,3}\s+minutes?|\b\d{1,3}\s+minutes?\s+(?:of|and)\b", re.I)
NON_GOAL = re.compile(r"(goal-?less|no goals?|goalkeeper\w*|goal kick\w*|(?:goal-?)?scoring (?:chance|opportunit)\w*|attempts? on goal|shots? on goal|towards goal|at goal|on target|penalty area|penalty box|penalty shoot-?out|shoot-?out|xg)", re.I)
MINUTE = re.compile(r"(?<![\d.])(\d{1,3})\s*(?:['’′]|(?:st|nd|rd|th)[\s-]+minute\b|\s*minutes?\b|\s*min\b)|\bminute\s+(\d{1,3})\b", re.I)
STOP = {"de", "da", "di", "do", "dos", "das", "del", "van", "von", "der", "den", "la", "le", "el", "al", "jr", "junior", "filho", "neto", "ben", "bin", "mac"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def match_facts(match_id: str):
    events = json.load(open(EVENTS / f"{match_id}.json", encoding="utf-8"))
    goals, players = [], set()
    for e in events:
        p = e.get("player", {}).get("name")
        if p:
            players.add(p)
        if e.get("period") == SHOOTOUT_PERIOD:
            continue
        t = e["type"]["name"]
        if t == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            goals.append({"player": p, "minute": e["minute"], "own_goal": False})
        elif t == "Own Goal Against":
            goals.append({"player": p, "minute": e["minute"], "own_goal": True})
    token_map: dict[str, set] = {}
    for p in players:
        for tok in re.findall(r"[a-z]+", norm(p)):
            if len(tok) >= 3 and tok not in STOP:
                token_map.setdefault(tok, set()).add(p)
    return goals, token_map


def units(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip(" *#-•\t")
        if not line:
            continue
        out.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip())
    return out


def claims_in(text: str, token_map: dict, scope_players: set) -> list[dict]:
    """Goal claims: (minute, players named in the same statement)."""
    found = []
    for u in units(text):
        if SHOOTOUT.search(u):
            continue
        cleaned = NON_GOAL.sub(" ", u)
        if not GOAL_WORDS.search(cleaned):
            continue
        nu = norm(u)
        players = set()
        for m in re.finditer(r"[a-z]+", nu):
            if m.group(0) in token_map:
                players |= token_map[m.group(0)]
        if not players and scope_players:
            players = set(scope_players)  # player question: the answer is about that player
        spans = [(m.start(), m.end()) for m in SPAN.finditer(nu)]
        for clause in re.finditer(r"[^,;]+", nu):
            ctext = clause.group(0)
            if MISS.search(ctext) and not GOAL_WORDS.search(NON_GOAL.sub(" ", ctext)):
                continue  # a miss described in this clause, not a goal
            for m in MINUTE.finditer(ctext):
                pos = clause.start() + m.start()
                if any(a <= pos < b for a, b in spans):
                    continue
                minute = int(m.group(1) or m.group(2))
                if minute > 130:
                    continue
                found.append({"unit": u, "minute": minute, "players": players})
    return found


def score(text: str, goals: list, token_map: dict, scope_player: str | None):
    in_scope = [g for g in goals if scope_player is None or (not g["own_goal"] and scope_player in norm(g["player"]))]
    scope_players = {g["player"] for g in in_scope} if scope_player is not None else set()
    raw = claims_in(text, token_map, scope_players)
    seen, claims = set(), []
    for c in raw:
        hit = None
        if c["players"]:
            used = {cl["goal_index"] for cl in claims}
            cands = [i for i, g in enumerate(goals)
                     if g["player"] in c["players"] and g["minute"] - 1 <= c["minute"] <= g["minute"] + 2]
            if cands:
                # closest goal by minute (a claim of m or m+1 fits a goal at m), preferring one not yet credited
                hit = min(cands, key=lambda i: (i in used, abs(c["minute"] - goals[i]["minute"] - 0.5)))
        key = (frozenset(c["players"]), hit if hit is not None else ("x", c["minute"]))
        if key in seen:
            continue
        seen.add(key)
        c["goal_index"] = hit
        c["status"] = "unattributed" if not c["players"] else ("correct" if hit is not None else "unsupported")
        claims.append(c)
    attributed = [c for c in claims if c["players"]]
    correct = [c for c in attributed if c["status"] == "correct"]
    stated = {c["goal_index"] for c in correct}
    recalled = sum(1 for i, g in enumerate(goals) if g in in_scope and i in stated)
    return {
        "claims": len(attributed), "correct": len(correct), "unsupported": len(attributed) - len(correct),
        "unattributed": len(claims) - len(attributed), "goals_in_scope": len(in_scope), "goals_stated": recalled,
        "precision": (len(correct) / len(attributed)) if attributed else "", "recall": (recalled / len(in_scope)) if in_scope else "",
    }, claims


def main() -> None:
    rows = list(csv.DictReader(open(RESULTS / "main_comparison.csv", encoding="utf-8")))
    keys = {(r["match_id"], r["query"]) for r in rows if r["system"] == "agentic" and r["precision_at_5"] not in ("N/A", "")}
    answers = [r for r in rows if (r["match_id"], r["query"]) in keys]
    hybrid = RESULTS / "hybrid_baseline.csv"
    if hybrid.exists():
        answers += [r for r in csv.DictReader(open(hybrid, encoding="utf-8")) if (r["match_id"], r["query"]) in keys]
    facts = {}
    out, claim_rows = [], []
    for r in answers:
        if r["match_id"] not in facts:
            facts[r["match_id"]] = match_facts(r["match_id"])
        goals, token_map = facts[r["match_id"]]
        m = re.match(r"Describe (.+?)'s goals", r["query"])
        scope = None
        if m:
            toks = [t for t in re.findall(r"[a-z]+", norm(m.group(1))) if len(t) >= 3 and t not in STOP]
            scope = toks[-1] if toks else None
            for g in goals:
                if not g["own_goal"] and all(t in norm(g["player"]) for t in toks[:2]):
                    scope = norm(g["player"])
                    break
        s, claims = score(r["response_text"], goals, token_map, scope)
        out.append({"match_id": r["match_id"], "query": r["query"], "system": r["system"], **s})
        for c in claims:
            claim_rows.append({"match_id": r["match_id"], "query": r["query"], "system": r["system"], "minute": c["minute"],
                               "players": "; ".join(sorted(c["players"])), "status": c["status"], "unit": c["unit"][:300]})
    with open(RESULTS / "factual_check_goals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    with open(RESULTS / "factual_check_goals_claims.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(claim_rows[0].keys())); w.writeheader(); w.writerows(claim_rows)
    for system in sorted({o["system"] for o in out}):
        rs = [o for o in out if o["system"] == system]
        cl = sum(o["claims"] for o in rs); co = sum(o["correct"] for o in rs)
        gi = sum(o["goals_in_scope"] for o in rs); gs = sum(o["goals_stated"] for o in rs)
        print(f"{system:13} answers {len(rs):2}  claims {cl:3}  correct {co:3}  unsupported {cl - co:2}  "
              f"precision {co / cl if cl else float('nan'):.3f}  goals stated {gs}/{gi} = recall {gs / gi:.3f}  "
              f"unattributed {sum(o['unattributed'] for o in rs)}")


if __name__ == "__main__":
    main()
