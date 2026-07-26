"""The 6-agent Agentic RAG pipeline (proposal §6.2.3, §6.3).

Query Understanding -> Retriever -> Context Aggregator -> Reasoning ->
Generator -> Evaluator (optional). Built on top of the same ingestion,
chunking, embedding and vector-store modules as the naive baseline in
`football_rag.generation.generator`, which this pipeline does not replace.
"""
