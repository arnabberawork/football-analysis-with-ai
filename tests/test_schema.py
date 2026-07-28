import unittest

from pydantic import ValidationError

from football_rag.schema import Chunk, ChunkMetadata, Event


class EventTests(unittest.TestCase):
    def test_optional_fields_default_to_none_or_zero(self):
        event = Event(
            match_id="1",
            team_id="10",
            team_name="Argentina",
            minute=23,
            period=1,
            event_type="Shot",
        )
        self.assertEqual(event.second, 0)
        self.assertIsNone(event.player_id)
        self.assertIsNone(event.player_name)
        self.assertIsNone(event.x)
        self.assertIsNone(event.y)
        self.assertIsNone(event.possession_id)
        self.assertIsNone(event.outcome)
        self.assertIsNone(event.detail)

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValidationError):
            Event(team_id="10", team_name="Argentina", minute=23, period=1, event_type="Shot")


class ChunkTests(unittest.TestCase):
    def test_chunk_round_trips_metadata(self):
        metadata = ChunkMetadata(
            match_id="1",
            minute_start=20,
            minute_end=24,
            period=1,
            chunk_level="time_window",
        )
        chunk = Chunk(chunk_id="1-tw-0", text="20'-24': something happened.", metadata=metadata)
        self.assertEqual(chunk.metadata.chunk_level, "time_window")
        self.assertEqual(chunk.metadata.minute_end, 24)

    def test_chunk_level_is_free_text_not_enum(self):
        # chunk_level is a plain str field (documented as one of four
        # values) - pydantic won't reject an unexpected value, callers are
        # responsible for only ever constructing the four known tiers.
        metadata = ChunkMetadata(match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="anything")
        self.assertEqual(metadata.chunk_level, "anything")


if __name__ == "__main__":
    unittest.main()
