"""Human-readable rendering of `graph.run_agentic_pipeline()`'s result
dict - shared by `scripts/run_agentic_pipeline.py`'s console output and
`api/main.py`'s saved-output-file text, so both stay in sync with one
source of truth for the format instead of drifting apart.
"""

from __future__ import annotations

NODE_LABELS = {
    "ingest": "Ingesting match data",
    "query_understanding": "Query Understanding",
    "stats_node": "StatsAgent",
    "retriever": "Retriever",
    "context_aggregator": "Context Aggregator",
    "reasoning": "Reasoning",
    "generator": "Generator",
    "evaluator": "Evaluator",
}
"""Human-readable label per LangGraph node name (`agents/graph.py`), for
live per-stage progress display - shared by `api/main.py`'s streaming
endpoint and read by `scripts/app.py`."""


def summarize_stage(node_name: str, state: dict) -> str:
    """One-line summary of what a just-completed node produced, for live
    agent-by-agent progress display. Reads the same raw-pydantic-object
    state shape as `format_pipeline_result()` below (see its docstring) -
    this is called from `api/main.py`'s streaming endpoint on the same
    accumulated state `stream_agentic_pipeline()` yields, before it's ever
    JSON-serialized into a `PipelineRunResponse`.
    """
    if node_name == "query_understanding":
        structured = state["structured_query"]
        parts = [f"intent: {structured.intent}"]
        if structured.teams:
            parts.append(f"teams: {', '.join(structured.teams)}")
        if structured.players:
            parts.append(f"players: {', '.join(structured.players)}")
        return " · ".join(parts)
    if node_name == "stats_node":
        return "answered directly from match events - no retrieval needed"
    if node_name == "retriever":
        return f"{len(state.get('retrieved_chunks', []))} evidence chunks retrieved"
    if node_name == "context_aggregator":
        return f"{len(state.get('aggregated_context', []))} chunks kept after removing near-duplicates"
    if node_name == "reasoning":
        return f"{len(state.get('reasoning_insights', []))} tactical insight(s) generated"
    if node_name == "generator":
        return "draft response generated"
    if node_name == "evaluator":
        evaluation = state.get("evaluation")
        if evaluation is None:
            return "no verdict returned - treated as a pass-through"
        return "passed review" if evaluation.passed else "needs revision - regenerating with feedback"
    return "done"


def next_stage_hint(node_name: str, state: dict, enable_evaluator: bool = True, enable_reasoning: bool = True) -> str | None:
    """The node expected to run right after `node_name`, or `None` if the
    graph ends there - mirrors the conditional routing in `agents/graph.py`
    (`route_after_query_understanding`, `route_after_evaluator`) closely
    enough for progress-hinting purposes, without duplicating it as actual
    control flow (this never decides what the graph does, only what a UI
    should highlight next). Lets `api/main.py`'s streaming endpoint tell a
    client which branch was taken (retrieval vs. `StatsAgent`) and whether
    an evaluator failure is about to loop back to the Generator, instead of
    the client trying to infer that from `detail`'s free text.
    """
    if node_name == "ingest":
        return "query_understanding"
    if node_name == "query_understanding":
        return "stats_node" if state["structured_query"].intent == "statistic" else "retriever"
    if node_name == "stats_node":
        return None
    if node_name == "retriever":
        return "context_aggregator"
    if node_name == "context_aggregator":
        return "reasoning" if enable_reasoning else "generator"
    if node_name == "reasoning":
        return "generator"
    if node_name == "generator":
        return "evaluator" if enable_evaluator else None
    if node_name == "evaluator":
        evaluation = state.get("evaluation")
        if evaluation is not None and not evaluation.passed:
            return "generator"
        return None
    return None


def format_pipeline_result(match_id: str, query: str, result: dict) -> str:
    lines = [f"Match: {match_id}", f"Query: {query}", ""]

    lines.append("1. Query Understanding:")
    lines.append(f"   {result['structured_query']!r}")
    lines.append(f"   where filter: {result.get('where_filter')}")

    lines.append("")
    lines.append("2. Retriever (fused dense + BM25 hits, re-ranked):")
    for hit in result["retrieved_chunks"]:
        score = hit.get("rerank_score", hit.get("fusion_score"))
        # ensure_salient-forced hits (agents/retriever.py::_force_salient) carry
        # no score at all - they bypassed ranking entirely, so there's nothing
        # numeric to show.
        score_str = f"{score:.4f}" if score is not None else "forced"
        lines.append(f"   [{hit['chunk_id']}] ({score_str}) {hit['text']}")

    lines.append("")
    lines.append(f"3. Context Aggregator ({len(result['aggregated_context'])} deduplicated chunks kept)")

    lines.append("")
    lines.append("4. Reasoning insights:")
    for insight in result.get("reasoning_insights", []):
        lines.append(f"   - {insight.insight} (evidence: {', '.join(insight.supporting_chunk_ids or [])})")

    lines.append("")
    lines.append("5. Generated response:")
    lines.append("")
    lines.append(result["response_text"])

    evaluation = result.get("evaluation")
    if evaluation is not None:
        lines.append("")
        lines.append("6. Evaluator verdict:")
        lines.append(
            f"   passed={evaluation.passed} relevance={evaluation.relevance:.2f} "
            f"coherence={evaluation.coherence:.2f} factual_consistency={evaluation.factual_consistency:.2f}"
        )
        lines.append(f"   feedback: {evaluation.feedback}")

    return "\n".join(lines)
