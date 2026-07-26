import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ImportSmokeTests(unittest.TestCase):
    def test_hf_api_key_is_exposed_to_hf_token(self):
        os.environ["HF_API_KEY"] = "test-token"
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HF_API_TOKEN", None)

        import importlib

        import football_rag.embeddings.embedder as embedder_module

        importlib.reload(embedder_module)

        self.assertEqual(os.environ["HF_TOKEN"], "test-token")
        self.assertEqual(os.environ["HF_API_TOKEN"], "test-token")
    def test_agent_graph_and_generator_import(self):
        from football_rag.agents.graph import build_agentic_graph
        from football_rag.generation.generator import NaiveRAGGenerator

        self.assertTrue(callable(build_agentic_graph))
        self.assertEqual(NaiveRAGGenerator.__name__, "NaiveRAGGenerator")


if __name__ == "__main__":
    unittest.main()
