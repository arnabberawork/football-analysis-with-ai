import unittest

from football_rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k


class PrecisionAtKTests(unittest.TestCase):
    def test_all_relevant(self):
        self.assertEqual(precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3), 1.0)

    def test_partial_relevance(self):
        self.assertAlmostEqual(precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4), 0.5)

    def test_k_smaller_than_retrieved_list(self):
        self.assertEqual(precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=1), 1.0)
        self.assertEqual(precision_at_k(["x", "b", "c"], {"a", "b", "c"}, k=1), 0.0)

    def test_empty_retrieved_list(self):
        self.assertEqual(precision_at_k([], {"a"}, k=5), 0.0)


class RecallAtKTests(unittest.TestCase):
    def test_all_relevant_captured(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "b"}, k=3), 1.0)

    def test_only_some_captured_within_k(self):
        self.assertAlmostEqual(recall_at_k(["a", "x", "y"], {"a", "b"}, k=3), 0.5)

    def test_relevant_beyond_k_not_counted(self):
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], {"a", "b"}, k=1), 0.5)

    def test_no_relevant_ids_returns_zero(self):
        self.assertEqual(recall_at_k(["a", "b"], set(), k=5), 0.0)


if __name__ == "__main__":
    unittest.main()
