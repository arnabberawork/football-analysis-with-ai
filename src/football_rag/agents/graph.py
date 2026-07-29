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

import logging

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

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 10


def _with_match_filter(where: dict | None, match_id: str) -> dict:
    """Always constrains retrieval to the current match, regardless of what
    (if anything) the LLM-derived query filter contains. `to_where_filter()`
    only ever reflects entities the LLM extracted from the query text - a
    query like "summarize the match" has no match name in it to extract, so
    match scoping can never be made to depend on that extraction.

    Confirmed live: without this, running the evaluation harness (which
    ingests every match's chunks into one shared vector store collection)
    produced a "summarize the match" response for the 2022 World Cup Final
    that narrated England and Croatia's tactics, because the unfiltered
    dense query returned zero chunks from the actual match and ten chunks
    from five unrelated matches instead.
    """
    match_clause = {"match_id": match_id}
    if where is None:
        return match_clause
    if "$and" in where:
        return {"$and": where["$and"] + [match_clause]}
    return {"$and": [where, match_clause]}


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
        logger.debug("[query_understanding] match=%s query=%r", state["match_id"], state["raw_query"])
        structured = query_agent.understand(state["raw_query"])
        logger.debug("[query_understanding] intent=%s", structured.intent)
        return {"structured_query": structured, "where_filter": to_where_filter(structured)}

    def route_after_query_understanding(state: AgentState) -> str:
        return "stats" if state["structured_query"].intent == "statistic" else "retrieve"

    def stats_node(state: AgentState) -> dict:
        logger.debug("[stats_node] match=%s query=%r (no LLM call)", state["match_id"], state["raw_query"])
        response_text = stats_agent.answer(state["raw_query"], state["structured_query"], state["events"])
        return {
            "response_text": response_text,
            "retrieved_chunks": [],
            "aggregated_context": [],
            "reasoning_insights": [],
        }

    def retriever_node(state: AgentState) -> dict:
        llm_where = state.get("where_filter")
        where = _with_match_filter(llm_where, state["match_id"])
        # Always on: RetrieverAgent._force_salient() now scopes forced
        # goal/card/sub candidates to this same `where`, so it's a no-op
        # for a query already narrowed to something salience can't match
        # (e.g. event_type="Pass") and correctly still fires for a
        # player/team-scoped query - confirmed live that gating this to
        # intent=="match_summary" only left player-focused queries (e.g.
        # "How was Neymar's performance?") with no guarantee his own goals
        # would be retrieved at all.
        logger.debug("[retriever] where=%s ensure_salient=True", where)
        hits = retriever_agent.retrieve(state["raw_query"], where=where, top_n=top_n, ensure_salient=True)
        logger.debug("[retriever] %d chunks retrieved", len(hits))
        return {"retrieved_chunks": hits}

    def context_aggregator_node(state: AgentState) -> dict:
        aggregated = aggregator_agent.aggregate(state["retrieved_chunks"])
        logger.debug("[context_aggregator] %d chunks -> %d after dedup", len(state["retrieved_chunks"]), len(aggregated))
        return {"aggregated_context": aggregated}

    def reasoning_node(state: AgentState) -> dict:
        logger.debug("[reasoning] analysing %d evidence chunks", len(state["aggregated_context"]))
        insights = reasoning_agent.reason(state["raw_query"], state["aggregated_context"])
        logger.debug("[reasoning] %d insights produced", len(insights))
        return {"reasoning_insights": insights}

    def generator_node(state: AgentState) -> dict:
        evaluation = state.get("evaluation")
        feedback = evaluation.feedback if evaluation is not None and not evaluation.passed else None
        logger.debug("[generator] retry=%s feedback=%s", state.get("retry_count", 0), bool(feedback))
        response_text = generator_agent.generate(
            state["raw_query"], state["aggregated_context"], state.get("reasoning_insights", []), feedback=feedback
        )
        return {"response_text": response_text}

    def evaluator_node(state: AgentState) -> dict:
        evaluation = evaluator_agent.evaluate(state["raw_query"], state["response_text"], state["aggregated_context"])
        logger.debug("[evaluator] passed=%s", evaluation.passed if evaluation is not None else None)
        return {"evaluation": evaluation, "retry_count": state.get("retry_count", 0) + 1}

    def route_after_evaluator(state: AgentState) -> str:
        evaluation = state["evaluation"]
        # Confirmed live: call_structured() can return None instead of a
        # parsed EvaluationResult if the model exhausts its output budget
        # before finishing valid JSON (crashed here with
        # AttributeError: 'NoneType' object has no attribute 'passed' under
        # GEMINI_MODEL=gemini-3.5-flash). There's no usable feedback to
        # retry with in that case, and retrying blind risks looping without
        # ever getting a verdict, so treat a missing evaluation as an
        # unscored pass-through rather than crashing or looping.
        if evaluation is None or evaluation.passed or state.get("retry_count", 0) > max_retries:
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
    logger.info("Running agentic pipeline: match=%s query=%r", match_id, query)
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


def stream_agentic_pipeline(
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
):
    """Same pipeline as `run_agentic_pipeline()`, but yields `(node_name,
    state)` after each agent completes instead of blocking until the whole
    run finishes - `state` is the full accumulated state so far (same shape
    `run_agentic_pipeline()` returns at the end). Powers `api/main.py`'s
    streaming endpoint so a UI can show live, agent-by-agent progress
    (`scripts/app.py`) instead of a single opaque wait. `run_agentic_pipeline()`
    itself is left untouched - the CLI script, evaluation harness, and
    ablation study all call it directly and don't need streaming.
    """
    logger.info("Streaming agentic pipeline: match=%s query=%r", match_id, query)
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
    state: dict = dict(initial_state)
    for update in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            state.update(node_output or {})
            yield node_name, dict(state)
