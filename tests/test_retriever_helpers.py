import unittest

from football_rag.agents.retriever import RetrieverAgent, _matches_where, _normalize, _salience_priority
from football_rag.schema import Chunk, ChunkMetadata


def _chunk(chunk_id, text, event_type=None, period=1, chunk_level="event", **metadata_overrides):
    metadata = ChunkMetadata(
        match_id="1",
        minute_start=0,
        minute_end=0,
        period=period,
        event_type=event_type,
        chunk_level=chunk_level,
        **metadata_overrides,
    )
    return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)


def _hit(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": "already ranked", "metadata": {}}


def _agent(chunks: list[Chunk]) -> RetrieverAgent:
    # vector_store is never touched by _force_salient/_fuse; enable_reranker=False
    # avoids downloading the cross-encoder model for what's otherwise a pure-logic test.
    return RetrieverAgent(vector_store=None, chunks=chunks, enable_hybrid=False, enable_reranker=False)


class SaliencePriorityTests(unittest.TestCase):
    def test_goal_shot_is_priority_zero(self):
        chunk = _chunk("1", "22' [Shot] Argentina - Messi took a shot; outcome: Goal.", event_type="Shot")
        self.assertEqual(_salience_priority(chunk), 0)

    def test_own_goal_is_priority_zero(self):
        chunk = _chunk("1", "50' [Own Goal Against] France conceded.", event_type="Own Goal Against")
        self.assertEqual(_salience_priority(chunk), 0)

    def test_shootout_period_goal_is_not_salient(self):
        # The exact bug this was added to fix: a penalty-shootout kick is
        # an ordinary Shot/outcome=Goal event, indistinguishable by
        # event_type/text from a real in-match goal - period 5 must be
        # excluded so it never gets forced in alongside real goals.
        chunk = _chunk("1", "120' [Shot] France - Mbappe took a shot; outcome: Goal.", event_type="Shot", period=5)
        self.assertIsNone(_salience_priority(chunk))

    def test_red_card_is_priority_one(self):
        chunk = _chunk("1", "60' [Foul Committed] outcome: Red Card.", event_type="Foul Committed")
        self.assertEqual(_salience_priority(chunk), 1)

    def test_yellow_card_via_bad_behaviour_is_priority_two(self):
        chunk = _chunk("1", "60' [Bad Behaviour] outcome: Yellow Card.", event_type="Bad Behaviour")
        self.assertEqual(_salience_priority(chunk), 2)

    def test_yellow_card_via_foul_committed_is_priority_two(self):
        chunk = _chunk("1", "60' [Foul Committed] outcome: Yellow Card.", event_type="Foul Committed")
        self.assertEqual(_salience_priority(chunk), 2)

    def test_second_yellow_dismissal_is_priority_one(self):
        # Regression test: a second-booking sending-off is recorded with
        # outcome "Second Yellow", not "Red Card" - confirmed live (2003
        # Arsenal vs Everton, Tie Li 86') that this outcome string matched
        # neither the "Red Card" nor "Yellow Card" substring check, so the
        # dismissal was excluded from salience entirely (priority None) and
        # never reached the generated summary's key moments.
        chunk = _chunk("1", "86' [Foul Committed] outcome: Second Yellow.", event_type="Foul Committed")
        self.assertEqual(_salience_priority(chunk), 1)

    def test_red_card_via_bad_behaviour_is_priority_one(self):
        # Bad Behaviour was previously hardcoded to priority 2 regardless of
        # its actual card outcome - a straight red recorded as Bad
        # Behaviour (rather than Foul Committed) was misranked alongside
        # yellow cards instead of outranking them.
        chunk = _chunk("1", "60' [Bad Behaviour] outcome: Red Card.", event_type="Bad Behaviour")
        self.assertEqual(_salience_priority(chunk), 1)

    def test_substitution_is_priority_three(self):
        chunk = _chunk("1", "70' [Substitution] Player off.", event_type="Substitution")
        self.assertEqual(_salience_priority(chunk), 3)

    def test_ordinary_event_is_not_salient(self):
        chunk = _chunk("1", "10' [Pass] Argentina played a pass.", event_type="Pass")
        self.assertIsNone(_salience_priority(chunk))

    def test_non_event_level_chunk_is_never_salient(self):
        # Even if the text happens to mention a goal, time-window/sequence
        # chunks mix multiple events together and aren't 1:1 with a single
        # event, so they're excluded regardless of content.
        chunk = _chunk("1", "20'-24': ... outcome: Goal ...", event_type=None, chunk_level="time_window")
        self.assertIsNone(_salience_priority(chunk))


class MatchesWhereTests(unittest.TestCase):
    def test_equality_match(self):
        metadata = ChunkMetadata(match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="event", event_type="Shot")
        self.assertTrue(_matches_where(metadata, {"event_type": "Shot"}))
        self.assertFalse(_matches_where(metadata, {"event_type": "Pass"}))

    def test_and_clause_requires_all(self):
        metadata = ChunkMetadata(match_id="1", minute_start=20, minute_end=20, period=1, chunk_level="event")
        where = {"$and": [{"period": 1}, {"minute_start": {"$gte": 10}}]}
        self.assertTrue(_matches_where(metadata, where))
        self.assertFalse(_matches_where(metadata, {"$and": [{"period": 2}, {"minute_start": {"$gte": 10}}]}))

    def test_gte_lte_operators(self):
        metadata = ChunkMetadata(match_id="1", minute_start=45, minute_end=45, period=1, chunk_level="event")
        self.assertTrue(_matches_where(metadata, {"minute_start": {"$gte": 45}}))
        self.assertFalse(_matches_where(metadata, {"minute_start": {"$gte": 46}}))
        self.assertTrue(_matches_where(metadata, {"minute_start": {"$lte": 45}}))
        self.assertFalse(_matches_where(metadata, {"minute_start": {"$lte": 44}}))

    def test_unsupported_operator_raises(self):
        metadata = ChunkMetadata(match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="event")
        with self.assertRaises(ValueError):
            _matches_where(metadata, {"minute_start": {"$ne": 1}})

    def test_player_name_matches_accent_and_case_insensitive_substring(self):
        metadata = ChunkMetadata(
            match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="event", player_name="Kylian Mbappé Lottin"
        )
        self.assertTrue(_matches_where(metadata, {"player_name": "mbappe"}))
        self.assertFalse(_matches_where(metadata, {"player_name": "messi"}))

    def test_non_fuzzy_field_requires_exact_equality(self):
        # Only team_name/player_name get substring/accent-insensitive
        # matching; every other field (e.g. event_type) still needs an
        # exact match.
        metadata = ChunkMetadata(
            match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="event", event_type="Foul Committed"
        )
        self.assertFalse(_matches_where(metadata, {"event_type": "Foul"}))


class NormalizeTests(unittest.TestCase):
    def test_strips_accents_and_lowercases(self):
        self.assertEqual(_normalize("Kylian Mbappé Lottin"), "kylian mbappe lottin")


class ForceSalientTests(unittest.TestCase):
    def test_all_goals_included_even_beyond_half_of_top_n(self):
        # The exact scenario that surfaced the bug: 6 real goals with
        # top_n=10 (old budget was top_n // 2 == 5, dropping the 6th).
        goal_chunks = [
            _chunk(f"g{i}", f"{i}' [Shot] took a shot; outcome: Goal.", event_type="Shot") for i in range(6)
        ]
        agent = _agent(goal_chunks)
        result = agent._force_salient(ranked=[], top_n=10)
        forced_ids = {h["chunk_id"] for h in result}
        self.assertEqual(forced_ids, {c.chunk_id for c in goal_chunks})

    def test_shootout_goals_never_forced(self):
        real_goal = _chunk("real", "22' [Shot] outcome: Goal.", event_type="Shot", period=1)
        shootout_goal = _chunk("shootout", "120' [Shot] outcome: Goal.", event_type="Shot", period=5)
        agent = _agent([real_goal, shootout_goal])
        result = agent._force_salient(ranked=[], top_n=10)
        forced_ids = {h["chunk_id"] for h in result}
        self.assertEqual(forced_ids, {"real"})

    def test_cards_and_subs_stay_budget_limited(self):
        cards = [_chunk(f"c{i}", "outcome: Yellow Card.", event_type="Bad Behaviour") for i in range(10)]
        agent = _agent(cards)
        result = agent._force_salient(ranked=[], top_n=10)
        # Budget for non-goal salience is max(1, top_n // 2) == 5.
        self.assertEqual(len(result), 5)

    def test_already_ranked_chunks_are_not_duplicated(self):
        goal = _chunk("g1", "22' [Shot] outcome: Goal.", event_type="Shot")
        agent = _agent([goal])
        already_ranked = [_hit("g1")]
        result = agent._force_salient(ranked=already_ranked, top_n=10)
        self.assertEqual([h["chunk_id"] for h in result].count("g1"), 1)

    def test_remaining_slots_filled_from_ranked_pool(self):
        goal = _chunk("g1", "22' [Shot] outcome: Goal.", event_type="Shot")
        agent = _agent([goal])
        ranked = [_hit("r1"), _hit("r2")]
        result = agent._force_salient(ranked=ranked, top_n=3)
        self.assertEqual([h["chunk_id"] for h in result], ["g1", "r1", "r2"])

    def test_forced_goals_exceeding_top_n_leave_no_room_for_ranked_pool(self):
        goals = [_chunk(f"g{i}", f"{i}' [Shot] outcome: Goal.", event_type="Shot") for i in range(4)]
        agent = _agent(goals)
        ranked = [_hit("r1")]
        result = agent._force_salient(ranked=ranked, top_n=3)
        self.assertEqual(len(result), 4)
        self.assertNotIn("r1", [h["chunk_id"] for h in result])


if __name__ == "__main__":
    unittest.main()
