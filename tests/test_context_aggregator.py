import unittest

from football_rag.agents.context_aggregator import ContextAggregatorAgent


def _hit(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text, "metadata": {}}


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


if __name__ == "__main__":
    unittest.main()
