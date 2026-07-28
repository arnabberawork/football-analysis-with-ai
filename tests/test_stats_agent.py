import unittest

from football_rag.agents.state import StructuredQuery
from football_rag.agents.stats import StatsAgent
from football_rag.schema import Event


def _event(**overrides) -> Event:
    defaults = dict(
        match_id="1",
        team_id="10",
        team_name="Argentina",
        player_id="100",
        player_name="Lionel Messi",
        minute=22,
        period=1,
        event_type="Shot",
        outcome="Goal",
    )
    defaults.update(overrides)
    return Event(**defaults)


def _query(**overrides) -> StructuredQuery:
    defaults = dict(intent="statistic")
    defaults.update(overrides)
    return StructuredQuery(**defaults)


class StatsAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = StatsAgent()

    def test_no_matches_returns_explicit_message(self):
        result = self.agent.answer("how many red cards?", _query(outcome="Red Card"), [])
        self.assertIn("No matching events found", result)

    def test_counts_by_team_and_player(self):
        events = [
            _event(player_name="Lionel Messi", minute=22),
            _event(player_name="Lionel Messi", minute=45, team_name="Argentina"),
            _event(player_name="Kylian Mbappe", team_name="France", team_id="20", minute=80),
        ]
        result = self.agent.answer("how many goals were scored?", _query(event_types=["Shot"], outcome="Goal"), events)
        self.assertIn("3 matching event(s) found.", result)
        self.assertIn("Argentina: 2", result)
        self.assertIn("France: 1", result)
        self.assertIn("Lionel Messi: 2", result)

    def test_player_filter_is_case_and_substring_insensitive_via_matches(self):
        events = [
            _event(player_name="Lionel Messi"),
            _event(player_name="Kylian Mbappe", team_name="France", team_id="20"),
        ]
        result = self.agent.answer("how many shots did Messi take?", _query(players=["messi"]), events)
        self.assertIn("1 matching event(s) found.", result)
        self.assertIn("Lionel Messi", result)
        self.assertNotIn("Mbappe", result)

    def test_goal_count_excludes_penalty_shootout_period(self):
        events = [
            _event(minute=22, period=1),  # regulation goal
            _event(minute=120, period=5, player_name="Lionel Messi"),  # shootout kick
        ]
        result = self.agent.answer(
            "how many goals did Argentina score?",
            _query(teams=["Argentina"], outcome="Goal"),
            events,
        )
        self.assertIn("1 matching event(s) found.", result)

    def test_explicit_period_5_query_is_not_filtered(self):
        events = [_event(minute=120, period=5)]
        result = self.agent.answer(
            "how many goals in the shootout?",
            _query(outcome="Goal", period=5),
            events,
        )
        self.assertIn("1 matching event(s) found.", result)

    def test_non_goal_statistic_is_not_shootout_filtered(self):
        # A "how many shots" query (no outcome="Goal", no period set) should
        # not silently drop a period-5 event - the shootout exclusion only
        # applies to goal-counting queries.
        events = [_event(minute=120, period=5, outcome=None)]
        result = self.agent.answer("how many shots were taken?", _query(event_types=["Shot"]), events)
        self.assertIn("1 matching event(s) found.", result)

    def test_minute_range_filter(self):
        events = [_event(minute=10), _event(minute=50), _event(minute=90)]
        result = self.agent.answer(
            "how many second half events?",
            _query(minute_min=46, minute_max=90),
            events,
        )
        self.assertIn("2 matching event(s) found.", result)

    def test_detail_list_capped_but_count_stays_exact(self):
        events = [_event(minute=m, outcome=None, event_type="Pass") for m in range(30)]
        result = self.agent.answer("how many passes?", _query(event_types=["Pass"]), events)
        self.assertIn("30 matching event(s) found.", result)
        self.assertIn("... and 10 more", result)


if __name__ == "__main__":
    unittest.main()
