import unittest

from football_rag.retrieval.bm25_index import BM25Index
from football_rag.schema import Chunk, ChunkMetadata


def _chunk(chunk_id: str, text: str) -> Chunk:
    metadata = ChunkMetadata(match_id="1", minute_start=0, minute_end=0, period=1, chunk_level="event")
    return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)


class BM25IndexTests(unittest.TestCase):
    def test_empty_corpus_returns_no_hits_instead_of_crashing(self):
        index = BM25Index([])
        self.assertEqual(index.query("penalty"), [])

    def test_exact_keyword_match_ranks_first(self):
        chunks = [
            _chunk("1", "Messi scored a penalty in the 22nd minute."),
            _chunk("2", "France won a corner in the 40th minute."),
        ]
        index = BM25Index(chunks)
        results = index.query("penalty")
        self.assertEqual(results[0][0], "1")

    def test_stemming_matches_plural_query_to_singular_text(self):
        # Confirmed in CLAUDE.md/TESTING.md: without stemming, "penalties"
        # would completely miss a chunk that only contains "Penalty". Uses
        # 3 chunks (not 2) since BM25Okapi's IDF term degenerates to 0 for
        # a word present in exactly half of a 2-document corpus.
        chunks = [
            _chunk("1", "Messi converted the Penalty."),
            _chunk("2", "Nothing relevant happened here."),
            _chunk("3", "France won a corner kick."),
        ]
        index = BM25Index(chunks)
        results = index.query("penalties")
        self.assertGreater(len(results), 0)
        top_id, top_score = results[0]
        self.assertEqual(top_id, "1")
        self.assertGreater(top_score, 0.0)

    def test_top_n_is_respected(self):
        chunks = [_chunk(str(i), f"event number {i} pass") for i in range(5)]
        index = BM25Index(chunks)
        self.assertEqual(len(index.query("pass", top_n=2)), 2)


if __name__ == "__main__":
    unittest.main()
