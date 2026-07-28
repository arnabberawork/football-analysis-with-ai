import unittest

from football_rag.evaluation.generation_metrics import bleu_score, rouge_scores


class RougeScoresTests(unittest.TestCase):
    def test_identical_text_scores_near_one(self):
        text = "Messi scored a penalty in the 22nd minute."
        scores = rouge_scores(text, text)
        for value in scores.values():
            self.assertAlmostEqual(value, 1.0, places=5)

    def test_unrelated_text_scores_low(self):
        scores = rouge_scores(
            "The weather today is sunny and warm.",
            "Mbappe scored twice in the World Cup final.",
        )
        self.assertLess(scores["rougeL"], 0.3)

    def test_returns_all_three_rouge_types(self):
        scores = rouge_scores("a b c", "a b d")
        self.assertEqual(set(scores.keys()), {"rouge1", "rouge2", "rougeL"})


class BleuScoreTests(unittest.TestCase):
    def test_identical_text_scores_near_max(self):
        text = "Argentina won the match on penalties."
        self.assertGreater(bleu_score(text, text), 99.0)

    def test_unrelated_text_scores_low(self):
        score = bleu_score("The weather today is sunny.", "Messi scored a penalty.")
        self.assertLess(score, 20.0)


if __name__ == "__main__":
    unittest.main()
