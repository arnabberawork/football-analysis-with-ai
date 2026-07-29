import unittest

from football_rag.agents.context_aggregator import ContextAggregatorAgent


def _hit(chunk_id: str, text: str, minute_start: int | None = None) -> dict:
    return {"chunk_id": chunk_id, "text": text, "metadata": {"minute_start": minute_start}}


class ContextAggregatorTests(unittest.TestCase):
    def setUp(self):
        self.agent = ContextAggregatorAgent()

    def test_keeps_distinct_chunks(self):
        hits = [
            _hit("1", "Messi scored a penalty in the 22nd minute."),
            _hit("2", "Mbappe scored in the 80th minute."),
        ]
        kept = self.agent.aggregate(hits)
        self.assertEqual([h["chunk_id"] for h in kept], ["1", "2"])

    def test_drops_near_duplicate_keeping_first_occurrence(self):
        hits = [
            _hit("1", "23' [Shot] Argentina - Messi took a shot; outcome: Goal."),
            _hit("2", "23' [Shot] Argentina - Messi took a shot; outcome: Goal!"),
        ]
        kept = self.agent.aggregate(hits)
        self.assertEqual([h["chunk_id"] for h in kept], ["1"])

    def test_custom_threshold_is_respected(self):
        hits = [_hit("1", "abcdefghij"), _hit("2", "abcdefghix")]
        strict_agent = ContextAggregatorAgent(similarity_threshold=0.99)
        self.assertEqual(len(strict_agent.aggregate(hits)), 2)

        loose_agent = ContextAggregatorAgent(similarity_threshold=0.5)
        self.assertEqual(len(loose_agent.aggregate(hits)), 1)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(self.agent.aggregate([]), [])

    def test_textually_similar_chunks_far_apart_in_time_are_kept_distinct(self):
        # Regression test: two of Mbappe's real penalties in the 2022 WC
        # Final (79' and 117') render almost identically under the fixed
        # event-sentence template (same location/technique/xG), scoring a
        # 0.97 similarity ratio - well above the default 0.9 threshold -
        # so the second (his decisive extra-time equalizer) was silently
        # dropped as a "duplicate" of the first before this fix.
        hits = [
            _hit("1", "79'24 [Shot] France - Mbappe took a shot at (108.0, 40.0); Penalty; xG 0.78; outcome: Goal.", minute_start=79),
            _hit(
                "2",
                "117'05 [Shot] France - Mbappe took a shot at (108.0, 40.0); Penalty; xG 0.78; outcome: Goal.",
                minute_start=117,
            ),
        ]
        kept = self.agent.aggregate(hits)
        self.assertEqual([h["chunk_id"] for h in kept], ["1", "2"])

    def test_similar_chunks_close_in_time_are_still_deduped(self):
        # The minute-gap guard shouldn't disable dedup entirely - two
        # chunks describing the same moment (e.g. overlapping time-window
        # and sequence chunks) should still be recognised as duplicates.
        hits = [
            _hit("1", "23' [Shot] Argentina - Messi took a shot; outcome: Goal.", minute_start=23),
            _hit("2", "23' [Shot] Argentina - Messi took a shot; outcome: Goal!", minute_start=24),
        ]
        kept = self.agent.aggregate(hits)
        self.assertEqual([h["chunk_id"] for h in kept], ["1"])

    def test_missing_minute_metadata_falls_back_to_text_similarity_only(self):
        hits = [
            _hit("1", "23' [Shot] Argentina - Messi took a shot; outcome: Goal."),
            _hit("2", "23' [Shot] Argentina - Messi took a shot; outcome: Goal!"),
        ]
        kept = self.agent.aggregate(hits)
        self.assertEqual([h["chunk_id"] for h in kept], ["1"])


if __name__ == "__main__":
    unittest.main()
