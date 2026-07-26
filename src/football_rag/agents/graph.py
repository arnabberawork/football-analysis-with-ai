"""LangGraph wiring for the 6-agent pipeline (proposal §6.2.3, §6.3), plus
one deliberate addition beyond the original design - see `stats_node`
below:

    query_understanding -[conditional]-> retriever -> context_aggregator
        -> (reasoning) -> generator -> (evaluator) -[conditional]->
        generator | END
                        \\-[conditional, intent=="statistic"]-> stats_node -> END

The Evaluator Agent is optional (§6.3.6); when disabled, `generator` routes
straight to END. When enabled and it rejects the response, its feedback is
looped back into `generator` for up to `max_retries` additional attempts.
The Reasoning Agent, hybrid retrieval, and re-ranking are also each
individually toggleable (`enable_reasoning`/`enable_hybrid`/
`enable_reranker`) - all four flags together are what
`scripts/run_ablation.py` flips one at a time for the ablation study
(proposal §6.11.6).

`stats_node` is a fifth branch, not in the proposal's original 6-agent
design: counting/aggregate questions ("how many shots did Messi take") are
routed straight to `agents/stats.py::StatsAgent` - a deterministic
computation over the full event list, no LLM call - since RAG retrieval
structurally cannot answer these reliably (it samples top-N chunks, never
"all" of anything). This bypasses retriever/aggregator/reasoning/generator/
evaluator entirely for that one intent.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from football_rag.agents.context_aggregator import ContextAggregatorAgent
from football_rag.agents.evaluator import EvaluatorAgent
from football_rag.agents.generator import GeneratorAgent
from football_rag.agents.query_understanding import QueryUnderstandingAgent, to_where_filter
from football_rag.agents.reasoning import ReasoningAgent
from football_rag.agents.retriever import RetrieverAgent
from football_rag.agents.state import AgentState
from football_rag.agents.stats import StatsAgent
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.schema import Chunk, Event

DEFAULT_TOP_N = 10


def build_agentic_graph(
    vector_store: ChunkVectorStore,
    chunks: list[Chunk],
    events: list[Event],
    enable_evaluator: bool = True,
    enable_reasoning: bool = True,
    enable_hybrid: bool = True,
    enable_reranker: bool = True,
    max_retries: int = 1,
    top_n: int = DEFAULT_TOP_N,
):
    """Compile the LangGraph pipeline. Agents are instantiated once here and
    closed over by the node functions, so each Gemini-backed agent creates
    at most one client per pipeline run rather than one per node invocation.
    """
    query_agent = QueryUnderstandingAgent()
    retriever_agent = RetrieverAgent(vector_store, chunks, enable_hybrid=enable_hybrid, enable_reranker=enable_reranker)
    aggregator_agent = ContextAggregatorAgent()
    reasoning_agent = ReasoningAgent() if enable_reasoning else None
    generator_agent = GeneratorAgent()
    evaluator_agent = EvaluatorAgent() if enable_evaluator else None
    stats_agent = StatsAgent()

    def query_understanding_node(state: AgentState) -> dict:
        structured = query_agent.understand(state["raw_query"])
        return {"structured_query": structured, "where_filter": to_where_filter(structured)}

    def route_after_query_understanding(state: AgentState) -> str:
        return "stats" if state["structured_query"].intent == "statistic" else "retrieve"

    def stats_node(state: AgentState) -> dict:
        response_text = stats_agent.answer(state["raw_query"], state["structured_query"], state["events"])
        return {
            "response_text": response_text,
            "retrieved_chunks": [],
            "aggregated_context": [],
            "reasoning_insights": [],
        }

    def retriever_node(state: AgentState) -> dict:
        structured = state["structured_query"]
        ensure_salient = structured.intent == "match_summary" and not state.get("where_filter")
        hits = retriever_agent.retrieve(
            state["raw_query"], where=state.get("where_filter"), top_n=top_n, ensure_salient=ensure_salient
        )
        return {"retrieved_chunks": hits}

    def context_aggregator_node(state: AgentState) -> dict:
        return {"aggregated_context": aggregator_agent.aggregate(state["retrieved_chunks"])}

    def reasoning_node(state: AgentState) -> dict:
        insights = reasoning_agent.reason(state["raw_query"], state["aggregated_context"])
        return {"reasoning_insights": insights}

    def generator_node(state: AgentState) -> dict:
        evaluation = state.get("evaluation")
        feedback = evaluation.feedback if evaluation is not None and not evaluation.passed else None
        response_text = generator_agent.generate(
            state["raw_query"], state["aggregated_context"], state.get("reasoning_insights", []), feedback=feedback
        )
        return {"response_text": response_text}

    def evaluator_node(state: AgentState) -> dict:
        evaluation = evaluator_agent.evaluate(state["raw_query"], state["response_text"], state["aggregated_context"])
        return {"evaluation": evaluation, "retry_count": state.get("retry_count", 0) + 1}

    def route_after_evaluator(state: AgentState) -> str:
        evaluation = state["evaluation"]
        if evaluation.passed or state.get("retry_count", 0) > max_retries:
            return "end"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("stats_node", stats_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("context_aggregator", context_aggregator_node)
    graph.add_node("generator", generator_node)

    graph.add_edge(START, "query_understanding")
    graph.add_conditional_edges(
        "query_understanding", route_after_query_understanding, {"stats": "stats_node", "retrieve": "retriever"}
    )
    graph.add_edge("stats_node", END)
    graph.add_edge("retriever", "context_aggregator")

    if enable_reasoning:
        graph.add_node("reasoning", reasoning_node)
        graph.add_edge("context_aggregator", "reasoning")
        graph.add_edge("reasoning", "generator")
    else:
        graph.add_edge("context_aggregator", "generator")

    if enable_evaluator:
        graph.add_node("evaluator", evaluator_node)
        graph.add_edge("generator", "evaluator")
        graph.add_conditional_edges("evaluator", route_after_evaluator, {"end": END, "retry": "generator"})
    else:
        graph.add_edge("generator", END)

    return graph.compile()


def run_agentic_pipeline(
    match_id: str,
    query: str,
    vector_store: ChunkVectorStore,
    chunks: list[Chunk],
    events: list[Event],
    enable_evaluator: bool = True,
    enable_reasoning: bool = True,
    enable_hybrid: bool = True,
    enable_reranker: bool = True,
    max_retries: int = 1,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """Compile the graph and run it once end-to-end, returning the full
    final state (structured_query, retrieved/aggregated evidence, reasoning
    insights, response_text, and the evaluator's verdict if enabled).
    """
    app = build_agentic_graph(
        vector_store,
        chunks,
        events,
        enable_evaluator=enable_evaluator,
        enable_reasoning=enable_reasoning,
        enable_hybrid=enable_hybrid,
        enable_reranker=enable_reranker,
        max_retries=max_retries,
        top_n=top_n,
    )
    initial_state: AgentState = {"match_id": match_id, "raw_query": query, "events": events, "retry_count": 0}
    return app.invoke(initial_state)
